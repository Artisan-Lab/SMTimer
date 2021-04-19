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
        # x = th.Tensor([item.feature for item in batch])
        x = [th.Tensor(item.feature) for item in batch]
        data_length = [len(sq) for sq in x]
        if time_selection == "origin":
            y = th.Tensor([item.origin_time for item in batch])
        else:
            y = th.Tensor([item.adjust_time for item in batch])
        if task != "regression":
            y = th.LongTensor([1 if item > threshold else 0 for item in y])
        try:
            x = rnn_utils.pad_sequence(x, batch_first=True)
        except:
            print("error")
        return FTBatch(feature=x,
                        label=y,
                        filename=[item.filename for item in batch],
                       data_len=data_length)
    return batcher_dev

def feature_batcher(device, time_selection="origin", task="regression", threshold=60):
    def batcher_dev(batch):
        x = th.Tensor([item.logic_tree for item in batch])
        # x = [th.Tensor(item.feature) for item in batch]
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
    np.random.seed(args.seed)
    th.manual_seed(args.seed)
    th.cuda.manual_seed(args.seed)

    best_epoch = -1

    cuda = args.gpu >= 0
    device = th.device('cuda:{}'.format(args.gpu)) if cuda else th.device('cpu')
    if cuda:
        th.cuda.set_device(args.gpu)

    smt_vocab_file = 'smt.vocab'
    smt_vocab = Vocab(filename=smt_vocab_file,
                      data=[Constants.UNK_WORD])

    try:
        pretrained_emb = th.load('smt.pth')
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
    if args.load:
        checkpoint = th.load('checkpoints/{}.pkl'.format(args.load_file))
        model.load_state_dict(checkpoint['model'])
        optimizer = checkpoint['optim']

        # dataset = dgl_dataset(train_dataset + test_dataset, smt_vocab, task)

        qd = Tree_Dataset()
        qd.fs_list = test_dataset + train_dataset
        test_dataset = qd.fs_list
        test_filename = set([i.filename for i in qd.fs_list])
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
            # test_dataset = qd.fs_list
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
        # print(metric_dic)
        # print(results)
        dir = args.input[5:]
        th.save(checkpoint, 'checkpoints/{}.pkl'.format('_'.join([dir, 'evaluation',
               None if args.regression else "c" , None if args.cross_index < 0 else str(args.cross_index + 1)])))
        return

    if args.model == "tree-lstm":
        trainer = Trainer(args, model, criterion, optimizer, device, metric, metric_name)
        random.shuffle(train_dataset)
        train_dataset = dgl_dataset(train_dataset, pretrained_emb, smt_vocab, task, args.time_selection)
        test_dataset = dgl_dataset(test_dataset, pretrained_emb, smt_vocab, task, args.time_selection)

        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=args.batch_size,
                                  collate_fn=batcher(device),
                                  shuffle=True,
                                  num_workers=0)
        test_loader = DataLoader(dataset=test_dataset,
                                 batch_size=100, collate_fn=batcher(device), shuffle=False, num_workers=0)
    elif args.model == "lstm" or args.model == "rnn":
        trainer = LSTM_Trainer(args, model, criterion, optimizer, device, metric, metric_name)

        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=args.batch_size,
                                  collate_fn=pad_feature_batcher(device, args.time_selection, task, args.threshold),
                                  shuffle=True,
                                  num_workers=0)
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

    # total_result, total_loss = trainer.test(test_loader)
    #
    # print("==> Epoch {:05d} | Test Loss {:.4f} | {:s} {:.4f}".format(
    #     epoch, total_loss / len(test_dataset), metric_name, total_result / len(test_dataset)))

def load_file(args):
    dataset = Tree_Dataset().generate_feature_dataset(args.data_source)
    return dataset


def load_dataset(args):
    if args.model == "tree-lstm":
        dataset_type = Tree_Dataset
        feature_limit = 100
    else:
        dataset_type = Vector_Dataset
        feature_limit = 50
    output_dir = args.input
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    train_file = os.path.join(output_dir, 'gnucore_train')
    test_file = os.path.join(output_dir, 'gnucore_test')
    if args.cross_index < 0:
        fn_index = 0
    else:
        fn_index = args.cross_index
    test_filename = get_dataset_seperation(output_dir)[fn_index]

    dataset = []
    if os.path.isfile(train_file):
        train_dataset = th.load(train_file)
        try:
            ind = 0
            while(os.path.exists(train_file + str(ind))):
                train_dataset = train_dataset.extend(th.load(train_file + str(ind)))
        except IOError:
            pass
        if os.path.isfile(test_file):
            test_dataset = th.load(test_file)
        else:
            qd = dataset_type(feature_number_limit=feature_limit)
            qd.fs_list = train_dataset
            dataset = train_dataset
            # qd.fs_list = list(filter(lambda x:x.adjust_time > 1, qd.fs_list))
            if "smt-comp" in train_file:
                if args.random_test:
                    test_filename = list(set([x.filename for x in train_dataset]))
                    test_filename = list(set(x.split("_")[0] for x in test_filename))
                    l = int(len(test_filename) / 3)
                    test_filename = test_filename[:l]

                # if "smt-comp" in train_file and not args.load:
                #     qd.fs_list = list(filter(lambda x:x.adjust_time > 1, qd.fs_list))
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
        dataset = qd.generate_feature_dataset(args.data_source, args.time_selection)
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
    parser.add_argument('--data_source', default='data/gnucore/script_dataset/training')
    parser.add_argument('--input', default='data/gnucore/training')
    parser.add_argument('--regression', action='store_false')
    parser.add_argument('--attention', action='store_true')
    parser.add_argument('--load', action='store_true')
    parser.add_argument('--load_file', default='regression2_0.05')
    parser.add_argument('--single_test', action='store_true')
    parser.add_argument('--time_selection', default='origin')
    parser.add_argument('--augment', action='store_true')
    parser.add_argument('--augment_path', default='data/gnucore/augment/crosscombine')
    parser.add_argument('--random_test', action='store_true')
    parser.add_argument('--threshold', type=int, default=200)
    parser.add_argument('--cross_index', type=int, default=-1)
    args = parser.parse_args()
    print(args)
    return args


if __name__ == '__main__':
    args = parse_arg()
    main(args)