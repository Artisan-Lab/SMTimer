import argparse
import collections

import json
import os
import time
import random
import numpy as np
# from matplotlib import pyplot
from dataset_filename_seperation import get_dataset_seperation

np.set_printoptions(suppress=True)
import torch as th
import torch.nn.init as INIT
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import dgl
from check_time import process_data, z3fun, getlogger
from dgl.data.tree import SSTBatch
from dgl_treelstm.trainer import Trainer,LSTM_Trainer
from dgl_treelstm.nn_models import TreeLSTM, LSTM, RNN, DNN
from dgl_treelstm.metric import Metrics
from dgl_treelstm.util import extract_root
from preprocessing import dgl_dataset,Tree_Dataset,Vocab,Constants,Vector_Dataset
import torch.nn.utils.rnn as rnn_utils
from sklearn.metrics import roc_curve

SSTBatch = collections.namedtuple('SSTBatch', ['graph', 'wordid', 'label', 'filename'])
FTBatch = collections.namedtuple('FTBatch', ['feature', 'label', 'filename', 'data_len'])
def batcher(device):
    def batcher_dev(batch):
        tree_batch = [x[0] for x in batch]
        try:
            batch_trees = dgl.batch(tree_batch, node_attrs=["y", "x"])
        except:
            for i in tree_batch:
                print(i.ndata['x'])

        return SSTBatch(graph=batch_trees,
                        wordid=batch_trees.ndata['x'].to(device),
                        label=batch_trees.ndata['y'].to(device),
                        filename=[x[1] for x in batch])
    return batcher_dev

def pad_feature_batcher(device, time_selection="origin", task="regression", threshold=60):
    def batcher_dev(batch):
        # x = th.Tensor([item.logic_tree for item in batch])
        x = [th.Tensor(item.logic_tree) for item in batch]
        data_length = [len(sq) for sq in x]
        if time_selection == "origin":
            y = th.Tensor([item.origin_time for item in batch])
            # y = th.Tensor([item.origin_time / 300 for item in batch])
        else:
            # y = th.Tensor([item.adjust_time / 300 for item in batch])
            y = th.Tensor([item.adjust_time for item in batch])
        if task != "regression":
            y = th.LongTensor([1 if item > threshold else 0 for item in y])
        x = rnn_utils.pad_sequence(x, batch_first=True)
        return FTBatch(feature=x,
                        label=y,
                        filename=[item.filename for item in batch],
                       data_len=data_length)
    return batcher_dev

def feature_batcher(device, time_selection="origin", task="regression", threshold=60):
    def batcher_dev(batch):
        x = th.Tensor([item.logic_tree for item in batch])
        # x = [th.Tensor(item.logic_tree) for item in batch]
        # data_length = [len(sq) for sq in x]
        if time_selection == "origin":
            y = th.Tensor([item.origin_time for item in batch])
        else:
            y = th.Tensor([item.adjust_time for item in batch])
        if task != "regression":
            y = th.LongTensor([1 if item > threshold else 0 for item in y])
        # x = rnn_utils.pad_sequence(x, batch_first=True)
        return FTBatch(feature=x,
                        label=y,
                        filename=[item.filename for item in batch],
                       data_len=None)
    return batcher_dev

def main(args):
    parallel = True
    np.random.seed(args.seed)
    th.manual_seed(args.seed)
    th.cuda.manual_seed(args.seed)

    best_epoch = -1

    cuda = args.gpu >= 0
    device = th.device('cuda:{}'.format(args.gpu)) if cuda else th.device('cpu')
    if cuda:
        th.cuda.set_device(args.gpu)

    smt_vocab_file = './data/gnucore/smt.vocab'
    smt_vocab = Vocab(filename=smt_vocab_file,
                      data=[Constants.UNK_WORD])

    try:
        pretrained_emb = th.load('./data/gnucore/smt.pth')
    except:
        pretrained_emb = th.zeros(smt_vocab.size(), 150)
        for word in smt_vocab.labelToIdx.keys():
            pretrained_emb[smt_vocab.getIndex(word), smt_vocab.getIndex(word)] = 1
        th.save(pretrained_emb, './data/gnucore/smt.pth')
    if args.model == "lstm":
        model = LSTM(args.h_size, args.regression, args.attention)
    elif args.model == "rnn":
        model = RNN(args.h_size, args.regression)
    elif args.model == "dnn":
        model = DNN(args.regression)
    else:
        model = TreeLSTM(smt_vocab.size(),
                         150,
                         args.h_size,
                         args.num_classes,
                         args.dropout,
                         args.regression,
                         args.attention,
                         cell_type='childsum' if args.child_sum else 'childsum',
                         pretrained_emb = pretrained_emb)
    # if parallel:
    #     model = th.nn.DataParallel(model, device_ids=[0, 1])
    model.to(device)
    print(model)

    metrics = Metrics(args.num_classes)
    if args.regression:
        metric_name = "Mse"
        criterion = nn.MSELoss()
        metric = metrics.msereducebysum
        best_dev_metric = float("inf")
        task = "regression"
        metric_list = [metrics.mse, metrics.mae, metrics.pearson]
    else:
        metric_name = "Acc"
        criterion = nn.CrossEntropyLoss(reduction='sum')
        metric = metrics.f1_score
        best_dev_metric = -1
        task = "classification"
        metric_list = [metrics.right_num, metrics.confusion_matrix, metrics.f1_score]

    optimizer = optim.Adagrad(filter(lambda p: p.requires_grad,
                                        model.parameters()), lr=args.lr, weight_decay=args.weight_decay)


    train_dataset, test_dataset = None, None

    if not args.single_test:
        train_dataset, test_dataset = load_dataset(args)
    # test
    # elif args.single_test:
    #     input = os.path.join('./data', args.data_source)
    #     # logger = getlogger()
    #     start = time.time()
    #     if True:
    #         if os.path.isdir(input):
    #             smt_scripts = []
    #             for root, dirs, files in os.walk(input):
    #                 for file in files:
    #                     # print(file)
    #                     with open(os.path.join(input, file), "r") as f:
    #                         data = f.read()
    #                         try:
    #                             data = json.loads(data)["smt_script"]
    #                         except:
    #                             pass
    #                         smt_scripts.append(data)
    #                         if len(smt_scripts) > 6000:
    #                             break
    #         elif os.path.isfile(input):
    #             with open(input, "r") as f:
    #                 data = f.read()
    #             smt_scripts = data.split("\n\n")
    #
    #         dl = []
    #         tl = []
    #         bigcount = 0
    #         smallcount = 0
    #         for script in smt_scripts:
    #             data, atime = process_data(script)
    #             if atime > 1:
    #             # if z3fun(data, None, 1000) == "unknown":
    #                 dl.append(data)
    #                 tl.append(atime)
    #                 bigcount += atime
    #             else:
    #                 # z3fun(data, None)
    #                 smallcount += atime
    #         end = time.time()
    #         print("solve time",end - start)
    #         dataset = Dataset().generate_feature_dataset(dl)
    #         print("> 10 num", len(dl))
    #         print("< 10", smallcount)
    #         print("> 10", bigcount)
    #     else:
    #         dataset = load_file(args)
    #     dataset = dgl_dataset(dataset, pretrained_emb, smt_vocab, task, args.time_selection)
    #     test_loader = DataLoader(dataset=dataset,
    #                              batch_size=1, collate_fn=batcher(device), shuffle=False, num_workers=0)
    #
    #     checkpoint = th.load('checkpoints/{}.pkl'.format(args.load_file))
    #     model.load_state_dict(checkpoint['model'])
    #     start = end
    #     error = 0
    #     save = 0
    #     t1 = 250
    #     model.eval()
    #     for step, batch in enumerate(test_loader):
    #         g = batch.graph
    #         n = g.number_of_nodes()
    #         with th.no_grad():
    #             h = th.zeros((n, args.h_size)).to(device)
    #             c = th.zeros((n, args.h_size)).to(device)
    #             logits = model(batch, h, c)
    #         batch_label, logits = extract_root(batch, device, logits)
    #         if args.regression:
    #             logits = logits.reshape(-1)
    #             pred = logits
    #             skip = (pred >= t1)
    #         else:
    #             pred = th.argmax(F.log_softmax(logits), 1)
    #             skip = pred == 1
    #         if not skip:
    #             if tl[step] > t1:
    #                 error += 1
    #             # data = dl[step]
    #             # print("sat")
    #             pass
    #             # print(step, pred)
    #             # data = process_data(smt_scripts[step])
    #             # z3fun(data, batch.filename)
    #         else:
    #             if tl[step] < 10:
    #                 error += 500
    #                 save += tl[step] * 500
    #             elif tl[step] < t1:
    #                 error += 1
    #                 save += tl[step]
    #             else:
    #                 save += tl[step]
    #             pass
    #             # print("skip")
    #     end = time.time()
    #     print("error", error)
    #     print("save time", save)
    #     print(end - start)
    #     return

    # return

    # test
    if args.load:
        checkpoint = th.load('checkpoints/{}.pkl'.format(args.load_file))
        model.load_state_dict(checkpoint['model'])
        optimizer = checkpoint['optim']

        # dataset = dgl_dataset(train_dataset + test_dataset, smt_vocab, task)

        qd = Tree_Dataset()
        qd.qt_list = test_dataset + train_dataset
        test_dataset = qd.qt_list
        test_filename = set([i.filename for i in qd.qt_list])
        # test_filename = ["echo", "ginstall", "expr", "tail", "seq", "split", "test", "yes", "chgrp", "date", "expand",
        #                  "head", "nohup", "printf", "sha1sum", "stat", "timeout", "uniq", "nice", "pr"]
        test_filename = checkpoint["args"].test_filename
        # total_mean = np.mean(np.array([x.feature for x in train_dataset]), axis=0)
        # total_std = np.std(np.array([x.feature for x in train_dataset]), axis=0)
        # total_time_std = np.std(np.array([x.adjust_time for x in train_dataset]), axis=0)
        # output = np.append(total_std,total_time_std)
        # print(time_var)
        # print(train_op)
        loss_list = []
        for fn in test_filename:
            print(fn)
            _, test_dataset = qd.split_with_filename([fn])
        # if True:
        #     _,test_dataset = qd.split_with_filename(test_filename)
            # test_dataset = qd.qt_list
            # program_std = np.std(np.array([x.feature for x in test_dataset]), axis=0)
            # program_mean = np.mean(np.array([x.feature for x in test_dataset]), axis=0)
            # program_time_std = np.std(np.array([x.adjust_time for x in test_dataset]), axis=0)
            # program_std = np.append(program_std, program_time_std)
            # output = np.row_stack([output, program_std])
            # print(time_var)
            # if len(test_dataset) == 0:
            #     continue
            if args.model == "lstm" or args.model == "rnn":
                dataset = test_dataset
                test_loader = DataLoader(dataset=test_dataset,
                                         batch_size=100, collate_fn=pad_feature_batcher(device, args.time_selection, task, args.threshold),
                                         shuffle=False, num_workers=0)
            elif args.model == "tree-lstm":
                dataset = dgl_dataset(test_dataset, pretrained_emb, smt_vocab, task, args.time_selection)
                test_loader = DataLoader(dataset=dataset,
                                         batch_size=100, collate_fn=batcher(device), shuffle=False, num_workers=0)
            print("here")
            none_op = []
            # for i in range(len(total_mean)):
            #     if total_mean[i] == 0 and program_mean[i] != 0:
            #         none_op.append(True)
            #     else:
            #         none_op.append(False)
            # print(test_op)
            # print(none_op)
            print("train data:", len(train_dataset), "test data:", len(test_dataset))

            total_result = 0
            total_loss = 0
            model.eval()
            pred_tensor = None
            label_tensor = None
            if args.model == "tree-lstm":
                for step, batch in enumerate(test_loader):
                    g = batch.graph
                    n = g.number_of_nodes()
                    with th.no_grad():
                        h = th.zeros((n, args.h_size)).to(device)
                        c = th.zeros((n, args.h_size)).to(device)
                        logits = model(batch, h, c)
                    batch_label, logits = extract_root(batch, device, logits)
                    if args.regression:
                        logits = logits.reshape(-1)
                        loss = criterion(logits, batch_label)
                        total_loss += loss * g.batch_size
                        pred = logits
                    else:
                        loss = criterion(logits, batch_label)
                        total_loss += loss
                        pred = th.argmax(F.log_softmax(logits), 1)
                    metric_result = metric(pred, batch_label)
                    total_result += metric_result
                    if pred_tensor == None:
                        pred_tensor = pred
                        label_tensor = batch_label
                    else:
                        pred_tensor = th.cat([pred_tensor, pred], dim= -1)
                        label_tensor = th.cat([label_tensor, batch_label], dim= -1)
            elif args.model == "lstm" or "rnn":
                for step, batch in enumerate(test_loader):
                    batch_feature = batch.feature.to(device)
                    batch_label = batch.label.to(device)
                    n = batch.feature.shape[0]
                    batch_feature = rnn_utils.pack_padded_sequence(batch_feature, batch.data_len, enforce_sorted=False, batch_first=True)
                    with th.no_grad():
                        logits = model(batch_feature).to(device)
                    if args.regression:
                        logits = logits.reshape(-1)
                        loss = criterion(logits, batch_label)
                        total_loss += loss * n
                        pred = logits
                    else:
                        loss = criterion(logits, batch_label)
                        total_loss += loss
                        pred = th.argmax(F.log_softmax(logits), 1)
                    metric_result = metric(pred, batch_label)
                    total_result += metric_result
                    if pred_tensor == None:
                        pred_tensor = pred
                        label_tensor = batch_label
                    else:
                        pred_tensor = th.cat([pred_tensor, pred], dim= -1)
                        label_tensor = th.cat([label_tensor, batch_label], dim= -1)

            print("==> Test Loss {:.4f} | {:s} {:.4f}".format(
                total_loss / len(dataset), metric_name, total_result / len(dataset)))

            dev_metric = total_result / len(dataset)

            metric_dic = {}
            # metric_list = [metrics.accuracy, metrics.confusion_matrix, metrics.f1_score]
            for m in metric_list:
                metric_dic[m.__name__] = m(pred_tensor, label_tensor)

            pred_tensor = [i.item() for i in pred_tensor]
            label_tensor = [i.item() for i in label_tensor]
            results = list(zip(pred_tensor, label_tensor))
            # print(results)

            # def modify_threshold(result, truth):
            #     threshold = 250
            #     time_out_setting = 200
            #     time_record = {}
            #     for ind, _ in enumerate(result):
            #         if result > threshold or truth > time_out_setting:
            #             time_record["timeout"].append(result)
            #         else:
            #             time_record["solvable"].append(result)
            #         if result < threshold and truth > time_out_setting:
            #             timeout_list = np.array(time_record["timeout"])
            #             solvable_list = time_record["solvable"]
            #             try:
            #                 solvable_limit = max(np.percentile(solvable_list, 95), np.mean(solvable_list), 60)
            #                 suitable_min_timeout = min(filter(lambda x: x > solvable_limit, timeout_list))
            #                 if isinstance(suitable_min_timeout, th.Tensor):
            #                     suitable_min_timeout = suitable_min_timeout.item()
            #                 max_solvable = max(filter(lambda x: x < suitable_min_timeout, time_record["solvable"]))
            #                 if isinstance(max_solvable, th.Tensor):
            #                     max_solvable = max_solvable.item()
            #                 threshold = max(suitable_min_timeout - 1, (suitable_min_timeout + max_solvable) / 2,
            #                                      threshold - 50)
            #             except (IndexError, ValueError):
            #                 pass
            #             print("decrease threshold to ", str(threshold))
            #     return


            checkpoint = {
                'model': model.state_dict(),
                'optim': optimizer,
                'metric': metric_name,
                'metric_value': metric_dic,
                'args': args,
                'result': results
            }
            loss_list.append(total_loss / len(dataset))
            print("------------------------------------------")
        # loss_l = np.array([8915.2061] + [x.item() for x in loss_list]).tolist()
        # data = {"filename": ['total'] + list(test_filename), "array": output.tolist(), 'loss': loss_l}
        # with open("feature_array_loss.json", 'w') as f:
        #     json.dump(data, f)
        # print(metric_dic)
        # print(results)
        th.save(checkpoint, 'checkpoints/{}.pkl'.format('_'.join([args.input, 'evaluation',
               None if args.regression else "c" , None if args.cross_index < 0 else str(args.cross_index + 1)])))
        return

    # find high quality test dataset
#     qd = Dataset()
#     qd.qt_list = train_dataset + test_dataset
#     test_filename = ["uname","shuf","ln","du","runcon","ginstall","uniq","unlink","mkfifo","ls","cp","who","pathchk","mktemp","vdir","shred",
# "chcon","nice","tty","comm","printf","touch","env","chroot","sleep","rmdir","od","sha512sum","factor","sha256sum",
# "id","tr","arch","sha224sum","pinky","users","date","md5sum","dirname","paste","readlink","dircolors","nl","sha1sum",
# "cksum","base64","fold","tail","wc","fmt","su","link","csplit","dir","unexpand","yes","df","join","hostname","head",
# "ptx","expand","basename","stty","mv","mknod","pwd","split","sum","cut","tee","rm","uptime","setuidgid","tsort","mkdir",
# "chown","pr","seq","chmod","tac","stat","sort","sha384sum","chgrp","cat"]
#     for i in range(10):
#         if i / 10 == 0:
#             random.shuffle(test_filename)
#         with th.no_grad():
#             for param in trainer.model.parameters():
#                 param.normal_(0, 0.4)
#         trainer.epoch = 0
#         slice = i % 10
#         train_dataset, test_dataset = qd.split_with_filename(test_filename[10 * slice: 10 * (slice + 1)])
#         train_op = np.sum(np.array([x.feature for x in train_dataset]), axis=0)
#         test_op = np.sum(np.array([x.feature for x in test_dataset]), axis=0)

    if args.model == "tree-lstm":
        trainer = Trainer(args, model, criterion, optimizer, device, metric, metric_name)
        datalen = int(len(train_dataset) * 9 / 10)
        random.shuffle(train_dataset)
        # dev_dataset = dgl_dataset(train_dataset[datalen:], pretrained_emb, smt_vocab, task, args.time_selection)
        # train_dataset = dgl_dataset(train_dataset[:datalen], pretrained_emb, smt_vocab, task, args.time_selection)
        train_dataset = dgl_dataset(train_dataset, pretrained_emb, smt_vocab, task, args.time_selection)
        test_dataset = dgl_dataset(test_dataset, pretrained_emb, smt_vocab, task, args.time_selection)

        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=args.batch_size,
                                  collate_fn=batcher(device),
                                  shuffle=True,
                                  num_workers=0)
        # dev_loader = DataLoader(dataset=dev_dataset,
        #                          batch_size=100, collate_fn=batcher(device), shuffle=False, num_workers=0)
        test_loader = DataLoader(dataset=test_dataset,
                                 batch_size=100, collate_fn=batcher(device), shuffle=False, num_workers=0)
    elif args.model == "lstm" or args.model == "rnn":
        trainer = LSTM_Trainer(args, model, criterion, optimizer, device, metric, metric_name)
        datalen = int(len(train_dataset) * 9 / 10)
        # random.shuffle(train_dataset)
        # dev_dataset = train_dataset[datalen:]
        # train_dataset = train_dataset[:datalen]
        # pos = [i.adjust_time > args.threshold for i in train_dataset]
        # neg_num = len(train_dataset) - len(pos)
        # train_dataset.append(pos * (neg_num // len(pos) - 1))

        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=args.batch_size,
                                  collate_fn=pad_feature_batcher(device, args.time_selection, task, args.threshold),
                                  shuffle=True,
                                  num_workers=0)
        # dev_loader = DataLoader(dataset=dev_dataset,
        #                          batch_size=100,
        #                          collate_fn=pad_feature_batcher(device, args.time_selection, task, args.threshold),
        #                          shuffle=False, num_workers=0)
        test_loader = DataLoader(dataset=test_dataset,
                                 batch_size=100, collate_fn=pad_feature_batcher(device, args.time_selection, task, args.threshold), shuffle=False, num_workers=0)
    else:
        trainer = LSTM_Trainer(args, model, criterion, optimizer, device, metric, metric_name)
        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=args.batch_size,
                                  collate_fn=feature_batcher(device, args.time_selection, task, args.threshold),
                                  shuffle=True,
                                  num_workers=0)
        test_loader = DataLoader(dataset=test_dataset,
                                 batch_size=100, collate_fn=feature_batcher(device, args.time_selection, task, args.threshold), shuffle=False, num_workers=0)

    checkpoint = {}
    for epoch in range(args.epochs):
        t_epoch = time.time()

        total_result, total_loss = trainer.train(train_loader)

        print("==> Epoch {:05d} | Train Loss {:.4f} | {:s} {:.4f} | Time {:.4f}s".format(
            epoch, total_loss / len(train_dataset), metric_name, total_result / len(train_dataset), time.time() - t_epoch))


        total_result, total_loss = trainer.test(test_loader)

        print("==> Epoch {:05d} | Dev Loss {:.4f} | {:s} {:.4f}".format(
            epoch, total_loss / len(test_dataset), metric_name, total_result / len(test_dataset)))

        # inspect data seperation's factors on result
        # for fn in test_filename[10 * slice: 10 * (slice + 1)]:
        #     _, test_dataset = qd.split_with_filename([fn])
        #     if len(test_dataset) == 0:
        #         continue
        #     dataset = dgl_dataset(test_dataset, smt_vocab, task)
        #     test_loader = DataLoader(dataset=dataset,
        #                              batch_size=100, collate_fn=batcher(device), shuffle=False, num_workers=0)
        #
        #     test_op = np.mean(np.array([x.feature for x in test_dataset]), axis=0)
        #     none_op = []
        #     for i in range(len(train_op)):
        #         if train_op[i] == 0 and test_op[i] != 0:
        #             none_op.append(True)
        #         else:
        #             none_op.append(False)
        #     print(test_op[-4:])
        #     # print(none_op)
        #     print("train data:", len(train_dataset), "test data:", len(test_dataset))
        #
        #     total_result = 0
        #     total_loss = 0
        #     model.eval()
        #     pred_tensor = None
        #     label_tensor = None
        #     for step, batch in enumerate(test_loader):
        #         g = batch.graph
        #         n = g.number_of_nodes()
        #         with th.no_grad():
        #             h = th.zeros((n, args.h_size)).to(device)
        #             c = th.zeros((n, args.h_size)).to(device)
        #             logits = model(batch, h, c)
        #         batch_label, logits = extract_root(batch, device, logits)
        #         if args.regression:
        #             logits = logits.reshape(-1)
        #             loss = criterion(logits, batch_label)
        #             total_loss += loss * g.batch_size
        #             pred = logits
        #         else:
        #             loss = criterion(logits, batch_label)
        #             total_loss += loss
        #             pred = th.argmax(F.log_softmax(logits), 1)
        #         metric_result = metric(pred, batch_label)
        #         total_result += metric_result
        #         if pred_tensor == None:
        #             pred_tensor = pred
        #             label_tensor = batch_label
        #         else:
        #             pred_tensor = th.cat([pred_tensor, pred], dim=-1)
        #             label_tensor = th.cat([label_tensor, batch_label], dim=-1)
        #
        #     print("==> Test Loss {:.4f} | {:s} {:.4f}".format(
        #         total_loss / len(dataset), metric_name, total_result / len(dataset)))
        #
        #     dev_metric = total_result / len(dataset)
        #
        #     metric_dic = {}
        #     # metric_list = [metrics.accuracy, metrics.confusion_matrix, metrics.f1_score]
        #     for m in metric_list:
        #         metric_dic[str(m)] = m(pred_tensor, label_tensor)
        #
        #     pred_tensor = [i.item() for i in pred_tensor]
        #     label_tensor = [i.item() for i in label_tensor]
        #     results = list(zip(pred_tensor, label_tensor))
        #
        #     checkpoint = {
        #         'model': model.state_dict(),
        #         'optim': optimizer,
        #         'metric': metric_name,
        #         'metric_value': metric_dic,
        #         'args': args,
        #         'result': results
        #     }
        #     print(fn)
        #     print("------------------------------------------")

        dev_metric = total_result / len(test_dataset)

        if (args.regression and dev_metric < best_dev_metric) or (not args.regression and dev_metric > best_dev_metric):
            best_dev_metric = dev_metric
            best_epoch = epoch
            checkpoint = {
                'model': model.state_dict(),
                'optim': optimizer,
                'metric': metric_name,
                'metric_value': dev_metric,
                'args': args, 'epoch': epoch
            }
            checkpoint_name = args.input
            if "/" in args.input:
                name_list = args.input.split("/")
                dataset_name = name_list[-2][0]
                checkpoint_name = name_list[-1]
            mt = "r" if args.regression else "c"
            ts = str(args.threshold)
            th.save(checkpoint, 'checkpoints/{}.pkl'.format('_'.join([dataset_name, checkpoint_name, args.model[0],
                            args.time_selection[0], mt, ts, None if args.cross_index < 0 else str(args.cross_index + 1)])))
        else:
            if best_epoch <= epoch - 20:
                break
            pass

        # lr decay
        for param_group in optimizer.param_groups:
            # if (epoch + 1) % 10 == 0:
            #     param_group['lr'] = max(1e-5, param_group['lr'] * 0.8)  # 10
            # else:
            param_group['lr'] = max(1e-5, param_group['lr']*0.99) #10
            # print(param_group['lr'])

    print('------------------------------------------------------------------------------------')
    print("Epoch {:05d} | Test {:s} {:.4f}".format(
        best_epoch, metric_name, best_dev_metric))

    # model.load_state_dict(checkpoint['model'])
    # pred_tensor, label_tensor = trainer.pred_tensor, trainer.label_tensor
    # batch_label = label_tensor.cpu().numpy()
    # batch_label[batch_label < 300] = 0
    # batch_label[batch_label >= 300] = 1
    # fpr, tpr, thresholds = roc_curve(batch_label, pred_tensor.cpu().numpy())
    # pyplot.plot(fpr, tpr, lw=1, label="lstm")
    #
    # pyplot.xlim([0.00, 1.0])
    # pyplot.ylim([0.00, 1.0])
    # pyplot.xlabel("False Positive Rate")
    # pyplot.ylabel("True Positive Rate")
    # pyplot.title("ROC")
    # pyplot.legend(loc="lower right")
    # pyplot.savefig(r"./ROC.png")

    # total_result, total_loss = trainer.test(test_loader)
    #
    # print("==> Epoch {:05d} | Test Loss {:.4f} | {:s} {:.4f}".format(
    #     epoch, total_loss / len(test_dataset), metric_name, total_result / len(test_dataset)))

def load_file(args):
    dataset = Tree_Dataset().generate_feature_dataset(os.path.join('./data', args.data_source))
    return dataset


def load_dataset(args):
    if args.model == "tree-lstm":
        dataset_type = Tree_Dataset
        feature_limit = 100
    else:
        dataset_type = Vector_Dataset
        feature_limit = 50
    output_dir = os.path.join('./data', args.input)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    train_file = os.path.join(output_dir, 'gnucore_train')
    test_file = os.path.join(output_dir, 'gnucore_test')
    # if "gnucore/" in train_file:
    #     test_filename = ["echo", "ginstall", "expr", "tail", "seq", "split", "test", "yes", "chgrp", "date", "expand",
    #                  "head", "nohup", "printf", "sha1sum", "stat", "timeout", "uniq", "nice", "pr"]
    #     # test_filename = ["arch", "chgrp", "csplit", "dirname", "fmt", "id", "md5sum", "mv", "pinky", "readlink", "seq",
    #     #                  "sleep", "tac", "tsort", "uptime", "base64", "chmod", "cut", "du", "fold", "join", "mkdir",
    #     #                  "nice", "pr", "rm"]
    #     # test_filename = ["setuidgid", "sort", "tail", "tty", "users", "basename", "chroot", "date", "expand", "ginstall",
    #     #                  "link", "mkfifo", "nl", "printenv", "rmdir", "sha1sum", "split", "test", "uname", "vdir",
    #     #                  "cat", "comm", "df", "expr"]
    #     test_filename = ["head", "ln", "mknod", "od", "printf", "runcon", "shred", "stat", "touch", "unexpand", "wc",
    #                      "chcon", "cp", "dir", "factor", "hostname", "ls", "mktemp", "pathchk", "ptx", "shuf", "su",
    #                      "tr", "unlink", "who"]
    # elif "klee" in train_file:
    #     test_filename = ['split', 'cp', 'base64', 'fmt', 'vdir', 'csplit', 'tr', 'join', 'shred']
    #     test_filename = ['tail', 'nice', 'sleep', 'ginstall', 'ls', 'du', 'expr', 'date', 'stat', 'df']
    #     test_filename = ['factor', 'chgrp', 'fold', 'head', 'nl', 'expand', 'setuidgid', 'mv', 'dir', 'tac']
    # elif "busybox" in train_file:
    #     # test_filename = ['raidautorun', 'smemcap', 'klogd', 'fstrim', 'cksum', 'killall5', 'mkswap', 'mt', 'mesg',
    #     #                  'chroot', 'fbsplash', 'insmod', 'nice', 'ionice', 'mkfs.vfat', 'stty', 'volname', 'sulogin']
    #     test_filename = ["adjtimex", "conspy", "fgconsole", "init", "linux", "makemime", "mv", "rmdir", "setconsole",
    #                      "swapon", "uevent", "arp", "devmem", "ipcalc", "loadkmap", "netstat", "route","setpriv", "sync",
    #                      "umount", "bootchartd", "dmesg", "fsync", "ipneigh", "login", "mkdir", "rpm","setserial",
    #                      "sysctl", "usleep", "cat", "dnsdomainname", "getopt", "iprule", "logread", "mkdosfs"]
    #     # test_filename = ["nohup", "rpm2cpio", "start-stop-daemon", "timeout", "watchdog", "chattr", "dumpkmap", "gnuzip",
    #     #                  "kbd_mode", "losetup", "mkfs.ext", "ping", "runlevel", "stat", "touch", "zcat", "chmod",
    #     #                  "hostname", "kill", "lsattr", "printenv", "run-parts", "udhcpc", "fatattr", "ifconfig", "false",
    #     #                  "lsmod", "readahead", "setarch", "swapoff", "udhcpd"]
    # elif "smt-comp" in train_file:
    #     test_filename = ['core', 'app5', 'app2', 'catchconv', 'gulwani-pldi08', 'bmc-bv', 'app10', 'app8', 'pspace',
    #                      'RWS', 'fft', 'tcas', 'ecc', 'HamiltonianPath', 'ConnectedDominatingSet',
    #                      '20170501-Heizmann-UltimateAutomizer', 'GeneralizedSlitherlink', '2019-Mann', 'mcm', 'zebra',
    #                      'uclid', 'samba', 'wget']
    #     test_filename = ['cvs', 'MazeGeneration', 'stp', 'float', '20190311-bv-term-small-rw-Noetzli',
    #                      'GraphPartitioning', 'Sage2', 'app9', 'tacas07', '2017-BuchwaldFried',
    #                      'WeightBoundedDominatingSet', 'app7', 'log-slicing', 'Commute', 'brummayerbiere2', 'VS3',
    #                      '2018-Mann', 'bench', 'Distrib', 'lfsr', 'brummayerbiere', 'openldap', 'inn']
    #     test_filename = ['galois', 'rubik', '20170531-Hansen-Check', 'ChannelRouting', 'Booth', 'app6', 'app1',
    #                      '2018-Goel-hwbench', 'bmc-bv-svcomp14', '20190429-UltimateAutomizerSvcomp2019', 'check2',
    #                      'brummayerbiere4', 'crafted', 'calypto', 'challenge', 'app12', 'simple', 'uum', 'pipe',
    #                      'xinetd', 'dwp', 'KnightTour', 'brummayerbiere3']
    # else:
    #     test_filename = None
    if args.cross_index < 0:
        fn_index = 0
    else:
        fn_index = args.cross_index
    test_filename = get_dataset_seperation(output_dir)[fn_index]

    dataset = []
    if os.path.isfile(train_file):
        train_dataset = th.load(train_file)
        try:
            if os.path.exists(train_file + "_1"):
                train_dataset = train_dataset.extend(th.load(train_file + "_1"))
        except IOError:
            pass
        if os.path.isfile(test_file):
            test_dataset = th.load(test_file)
        else:
            qd = dataset_type(feature_number_limit=feature_limit)
            qd.qt_list = train_dataset
            dataset = train_dataset
            # qd.qt_list = list(filter(lambda x:x.adjust_time > 1, qd.qt_list))
            if "smt-comp" in train_file:
                if args.random_test:
                    test_filename = list(set([x.filename for x in train_dataset]))
                    test_filename = list(set(x.split("_")[0] for x in test_filename))
                    l = int(len(test_filename) / 3)
                    test_filename = test_filename[:l]

                # if "smt-comp" in train_file and not args.load:
                #     qd.qt_list = list(filter(lambda x:x.adjust_time > 1, qd.qt_list))
                print("select program:", test_filename)
                test_filename1 = [x.filename for x in train_dataset]
                test_filename = list(filter(lambda x:x.split("_")[0] in test_filename, test_filename1))
            else:
                if args.random_test:
                    test_filename = list(set([x.filename for x in train_dataset]))
                    random.shuffle(test_filename)
                    l = int(len(test_filename) / 3)
                    test_filename = test_filename[:l]
                print("select program:", test_filename)
            train_dataset, test_dataset = qd.split_with_filename(test_filename)
            # train_dataset = train_dataset + test_dataset
    else:
        treeforassert = "tree+feature" in args.input
        qd = dataset_type(feature_number_limit=feature_limit, treeforassert=treeforassert)
        dataset = qd.generate_feature_dataset(os.path.join('./data', args.data_source), args.time_selection)
        train_dataset, test_dataset = qd.split_with_filename(test_filename)

    if args.augment:
        qd = dataset_type(feature_number_limit=feature_limit)
        augment_path = os.path.join(args.augment_path, 'gnucore_train')
        if os.path.isfile(augment_path):
            aug_dataset = th.load(augment_path)
            aug_dataset = list(filter(lambda x:x.adjust_time > 1, aug_dataset))
        else:
            print("augment data not found through the path")
            aug_dataset = []
        train_dataset = train_dataset + aug_dataset

    if not os.path.isfile(train_file):
        # if len(dataset) > 10000:
        #     th.save(dataset[:10000], train_file)
        #     del dataset[:10000]
        #     th.save(dataset[10000:20000], train_file + "_1")
        #     del dataset[10000:20000]
        #     th.save(dataset[20000:], train_file + "_2")
        # else:
        th.save(dataset, train_file)
        # th.save(test_dataset, test_file)
    print("train data:", len(train_dataset), "test data:", len(test_dataset))
    args.test_filename = test_filename
    # del qd
    return train_dataset, test_dataset


def parse_arg():
    # global args
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=41)
    parser.add_argument('--model', default='tree-lstm')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--child-sum', action='store_true')
    parser.add_argument('--x-size', type=int, default=300)
    parser.add_argument('--h-size', type=int, default=150)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--log-every', type=int, default=40)
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--weight-decay', type=float, default=1e-3)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--num_classes', type=float, default=2)
    parser.add_argument('--data_source', default='gnucore/script_dataset/training')
    parser.add_argument('--input', default='gnucore/training')
    parser.add_argument('--regression', action='store_false')
    parser.add_argument('--attention', action='store_true')
    parser.add_argument('--load', action='store_true')
    parser.add_argument('--load_file', default='regression2_0.05')
    parser.add_argument('--single_test', action='store_true')
    parser.add_argument('--time_selection', default='origin')
    parser.add_argument('--augment', action='store_true')
    parser.add_argument('--augment_path', default='data/gnucore/augment/crosscombine')
    parser.add_argument('--random_test', action='store_true')
    parser.add_argument('--threshold', type=int, default=60)
    parser.add_argument('--cross_index', type=int, default=-1)
    args = parser.parse_args()
    print(args)
    return args


if __name__ == '__main__':
    args = parse_arg()
    main(args)