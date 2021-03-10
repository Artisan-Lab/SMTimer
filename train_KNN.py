import argparse
import collections

import json
import os
import time
import random
import numpy as np
np.set_printoptions(suppress=True)
import torch as th
import torch.nn as nn

from check_time import process_data, z3fun, getlogger
from dgl_treelstm.KNN import KNN
from dgl_treelstm.metric import Metrics
from preprocessing import dgl_dataset,Tree_Dataset,Vocab,Constants,Vector_Dataset,op
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
import time

warnings.filterwarnings('ignore')

def main(args):
    data = load_dataset(args)

    if args.model_selection == "all":
        sknn = True
        iknn = True
    elif args.model_selection == "knn":
        sknn = True
        iknn = False
    else:
        sknn = False
        iknn = True

    # knn classifier
    test_dataset = None
    train_dataset = None
    if args.cross_project:
        train_dataset = data
        output_dir = os.path.join('./data', args.eva_input, 'gnucore_train')
        test_dataset = th.load(output_dir)
        data = test_dataset
    test_filename = list(set([x.filename for x in data]))
    if "smt-comp" in args.input:
        test_filename = list(set([x.filename for x in data]))
        test_filename = list(set(x.split("_")[0] for x in test_filename))
    dataset = Vector_Dataset()
    dataset.qt_list = data
    # output = {
    #     "x" : [i.feature.tolist() for i in data], "adjust" : [i.gettime("z3") for i in data],
    #     "origin" : [i.gettime("origin") for i in data],"filename" : [i.filename for i in data]
    # }
    # with open("data/KNN_training_data/gnucore.json", "w") as f:
    #     json.dump(output, f)
    # test_filename = ["expand"]
    total_num = 0
    incremental_total_result = []
    sklearn_total_result = []
    truth = []
    s = time.time()
    print(len(data))
    # cor(data)
    odds_ratio_test(data)
    # return
    # cl = KMeans(n_clusters=2)
    # pred = cl.fit_predict([i.feature for i in data])
    # truth = [1 if i.gettime(args.time_selection) > 200 else 0 for i in data]
    # a = np.array([pred, truth])
    for fn in test_filename:
        if "smt-comp" in args.input:
            # if fn != "Sage2":
            #     continue
            fn = list(map(lambda x:x.filename, filter(lambda x: x.filename.split("_")[0] == fn, data)))
            train_slice, test_dataset = dataset.split_with_filename(fn)
            fn = fn[0].split("_")[0]
        else:
            train_slice, test_dataset = dataset.split_with_filename([fn])
        if not args.cross_project:
            train_dataset = train_slice
        y_test = np.array([1 if i.gettime(args.time_selection) > args.time_limit_setting else 0 for i in test_dataset])
        if sum(y_test) == 0 or len(y_test) < 10:
            continue
        print(fn, len(y_test), sum(y_test))
        # continue
        total_num += len(y_test)
        if iknn:
            incremental_predict = simple_KNN(args, test_dataset, train_dataset, args.model_selection)
            incremental_total_result.extend(incremental_predict)

        if sknn:
            sklearn_predict = sklearn_KNN(args, test_dataset, train_dataset)
            sklearn_total_result.extend(sklearn_predict)

        truth.extend(y_test)
    e = time.time()
    print("time", e - s, "data number", len(truth))
    print("total result:")
    if iknn:
        acc = accuracy_score(truth, incremental_total_result)
        pre = precision_score(truth, incremental_total_result)
        rec = recall_score(truth, incremental_total_result)
        f1 = f1_score(truth, incremental_total_result)
        print('incremental test accuracy: {:.3}, precision: {:.3}, recall: {:.3}, f1 score: {:.3}'.format(acc, pre, rec, f1))

    if sknn:
        acc = accuracy_score(truth, sklearn_total_result)
        acc = accuracy_score(sklearn_total_result, truth)
        pre = precision_score(truth, sklearn_total_result)
        rec = recall_score(truth, sklearn_total_result)
        f1 = f1_score(truth, sklearn_total_result)
        print('test accuracy: {:.3}, precision: {:.3}, recall: {:.3}, f1 score: {:.3}'.format(acc, pre, rec, f1))

def cor(train_dataset):
    train_dataset = list(filter(lambda x: sum(x.feature) != 0, train_dataset))
    x = np.array([i.feature for i in train_dataset])
    x = np.power(10, x) - 1
    x = x[:,:150] + x[:,150:]
    x = np.log10(x + 1)
    y = np.array([1 if i.gettime(args.time_selection) > args.time_limit_setting else 0 for i in train_dataset])
    for i in range(73):
        if sum(x[:,i]) == 0:
            continue
        data = np.corrcoef(x[:,i], y)
        print(i, op[i], data[0,1])

def odds_ratio_test(train_dataset):
    # train_dataset = list(filter(lambda x:sum(x.feature) != 0, train_dataset))
    x = np.array([i.feature for i in train_dataset])
    y = np.array([1 if i.gettime(args.time_selection) > args.time_limit_setting else 0 for i in train_dataset])
    x = x[:, :150] + x[:, 150:]
    for i in range(150):
        # x_with = np.array(list(map(lambda x: x[i] == 0 and x[i + 150] == 0, x)))
        x_with = np.array(list(map(lambda x: x[i] == 0, x)))
        index = np.argwhere(x_with == False).reshape(-1)
        xp = x[index]
        y_wp, y_w = sum(y[index]), len(index)
        index = np.argwhere(x_with == True).reshape(-1)
        y_wop, y_wo = sum(y[index]), len(index)
        try:
            # print(op[i])
            # print(y_wp, y_w)
            # print(y_wop, y_wo)
            if y_w == 0:
                print("absent")
                continue
            if y_wo < 10:
                if i < len(op):
                    print(i, op[i], "little without")
                elif i >= 111:
                    print(i, "var", "unsuitable")
                continue
            if y_w < 10:
                if i < len(op):
                    print(i, op[i], "little with")
                elif i >= 111:
                    print(i, "var", "unsuitable")
                continue
            if i < len(op):
                print(i, op[i], (y_wp / y_w) / (y_wop / y_wo))
            elif i >= 111:
                break
                # print(i, "var", (y_wp / y_w) / (y_wop / y_wo))
            else:
                print(i, (y_wp / y_w) / (y_wop / y_wo))
        except ZeroDivisionError:
            pass

def sklearn_KNN(args, test_dataset, train_dataset):
    clf = KNeighborsClassifier(3, algorithm="ball_tree")
    y_test = np.array([1 if i.gettime(args.time_selection) > args.time_limit_setting else 0 for i in test_dataset])
    x_train = np.array([i.feature for i in train_dataset])
    y_train = np.array([1 if i.gettime(args.time_selection) > args.time_limit_setting else 0 for i in train_dataset])
    clf.fit(x_train, y_train)
    x_test = np.array([i.feature for i in test_dataset])
    y_test_pred = clf.predict(x_test)
    acc = accuracy_score(y_test, y_test_pred)
    pre = precision_score(y_test, y_test_pred)
    rec = recall_score(y_test, y_test_pred)
    f1 = f1_score(y_test, y_test_pred)
    print('test accuracy: {:.3}, precision: {:.3}, recall: {:.3}, f1 score: {:.3}'.format(acc, pre, rec, f1))
    return y_test_pred

def simple_KNN(args, test_dataset, train_dataset, model_selection):
    clf = KNN(k=7)
    y_test = np.array([1 if i.gettime(args.time_selection) > args.time_limit_setting else 0 for i in test_dataset])
    x_train = np.array([i.feature for i in train_dataset])
    y_train = np.array([1 if i.gettime(args.time_selection) > args.time_limit_setting else 0 for i in train_dataset])
    x_test = np.array([i.feature for i in test_dataset])
    # tf = TfidfTransformer()
    # x_train = np.power(10, x_train) - 1
    # x_train = tf.fit_transform(x_train)
    # x_train.todense()
    # x_train = x_train.toarray()
    # x_train = np.log(x_train[:,:150] + x_train[:,150:] + 1)
    # x_test = np.power(10, x_test) - 1
    # x_test = tf.transform(x_test)
    # x_test.todense()
    # x_test = x_test.toarray()
    # x_test = np.log(x_test[:,:150] + x_test[:,150:] + 1)
    clf.fit(x_train, y_train)
    clf.filename = np.array([i.filename for i in train_dataset])
    filename = np.array([i.filename for i in test_dataset])
    if "smt2" in clf.filename[0]:
        index = np.argsort(filename)
        x_test = x_test[index]
        y_test = y_test[index]
        reverse_index = [0] * len(index)
        for ind,i in enumerate(index):
            reverse_index[i] = ind
    if "fast" in args.model_selection:
        y_test_pred = clf.fast_incremental_predict(x_test, y_test)
    else:
        if "mask" in args.model_selection:
            clf.mask = True
        if "error" in args.model_selection:
            clf.accept_error = True
        y_test_pred = clf.incremental_predict(x_test, y_test)

    acc, pre, rec, fls = clf.score(y_test, y_test_pred)
    print('incremental test accuracy: {:.3}, precision: {:.3}, recall: {:.3}, f1 score: {:.3}'.format(acc, pre, rec, fls))
    if "smt2" in clf.filename[0]:
        y_test_pred = y_test_pred[reverse_index]
    return y_test_pred


def load_dataset(args):
    dataset_type = Vector_Dataset
    output_dir = os.path.join('./data', args.input)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    train_file = os.path.join(output_dir, 'gnucore_train')
    test_file = os.path.join(output_dir, 'gnucore_test')
    dataset = []
    if os.path.isfile(train_file):
        train_dataset = th.load(train_file)
    else:
        qd = dataset_type(feature_number_limit=2)
        train_dataset = qd.generate_feature_dataset(os.path.join('./data', args.data_source), args.time_selection)

    if args.augment:
        qd = dataset_type(feature_number_limit=2)
        augment_path = os.path.join(args.augment_path, 'gnucore_train')
        if os.path.isfile(augment_path):
            aug_dataset = th.load(augment_path)
            aug_dataset = list(filter(lambda x:x.adjust_time > 1, aug_dataset))
        else:
            print("augment data not found through the path")
            aug_dataset = []
        train_dataset = train_dataset + aug_dataset

    if not os.path.isfile(train_file):
        th.save(train_dataset, train_file)
    # del qd
    return train_dataset


def parse_arg():
    # global args
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=float, default=2)
    parser.add_argument('--data_source', default='gnucore/fv2')
    parser.add_argument('--input', default='gnucore/training')
    parser.add_argument('--single_test', action='store_true')
    parser.add_argument('--time_selection', default='origin')
    parser.add_argument('--augment', action='store_true')
    parser.add_argument('--augment_path', default='data/gnucore/augment/crosscombine')
    parser.add_argument('--cross_project', action='store_true')
    parser.add_argument('--eva_input', default='busybox/fv2')
    parser.add_argument('--time_limit_setting', type=int, default=300)
    parser.add_argument('--model_selection', default="all")
    args = parser.parse_args()
    print(args)
    return args


if __name__ == '__main__':
    args = parse_arg()
    main(args)