import json
import time
import os
import random
import sys
import gc
import signal

from .feature_extraction import Script_Info, feature_extractor, QT
from preprocessing.pysmt_tree import pysmt_query_tree

b = ["[","chmod","dd","expr","hostid","md5sum","nproc","ptx","sha224sum","stdbuf","touch","unlink","b2sum","chown","df",
"factor","id","mkdir","numfmt","pwd","sha256sum","stty","tr","uptime","base32","chroot","dir","false","join","mkfifo",
"od","readlink","sha384sum","sum","true","users","base64","cksum","dircolors","fmt","kill","mknod","paste","realpath",
"sha512sum","sync","truncate","vdir","basename","comm","dirname","fold","link","mktemp","pathchk","rm","shred","tac",
"tsort","wc","basenc","cp","du","getlimits","ln","mv","pinky","rmdir","shuf","tail","tty","who","cat","csplit","echo",
"ginstall","logname","nice","pr","runcon","sort","tee","uname","whoami","chcon","cut","env","groups","ls","nl",
"printenv","seq","split","test","unexpand","yes","chgrp","date","expand","head","make-prime-list","nohup","printf",
"sha1sum","stat","timeout","uniq"]

test_filename = ["echo", "ginstall", "expr", "tail", "seq", "split", "test", "yes", "chgrp", "date", "expand", "head",
            "nohup", "printf", "sha1sum", "stat", "timeout", "uniq", "nice", "pr"]

def handler(signum, frame):
    signal.alarm(1)
    raise TimeoutError

# input all kinds of scripts and return expression tree
class Dataset:
    def __init__(self, feature_number_limit=100, treeforassert=False):
        self.str_list = []
        self.script_list = []
        self.qt_list = []
        self.is_json = True
        self.filename_list = []
        self.treeforassert = treeforassert
        self.feature_number_limit = feature_number_limit
        self.klee = False
        self.selected_file = False

    # read data from file directory or script, preprocess scripts into expression trees
    def generate_feature_dataset(self, input, time_selection=None):
        self.str_list = []
        if isinstance(input, list):
            self.str_list = input
        elif isinstance(input, str) and '\n' in input:
            self.str_list = [input]
        else:
            self.load_from_directory(input)
            if "klee" in input:
                self.klee = True
        self.judge_json(self.str_list[0])
        output_ind = 0
        selected_filename = []
        for ind, string in enumerate(self.str_list):
            script = Script_Info(string, self.is_json)
            # self.script_list.append(script)
            s = time.time()
            # try:
            try:
                if script.solving_time_dic["z3"][0] < 0:
                    continue
                # if not self.selected_file:
                #     if script.solving_time < 20 and script.solving_time_dic["z3"][0] < 10:
                #         if len(self.str_list) > 20000 and ind % 10 != 0:
                #             continue
                selected_filename.append(self.filename_list[ind])
            except:
                pass
            gettrace = getattr(sys, 'gettrace', None)
            # if gettrace is None and not gettrace():
            # signal.alarm(1)
            signal.signal(signal.SIGALRM, handler)
            try:
                ret = self.parse_data(script, time_selection)
                self.qt_list.append(ret)
            except TimeoutError:
                signal.alarm(0)
                print("preprocess over time", len(self.qt_list))
            except (KeyError,AttributeError):
                continue
            finally:
                signal.alarm(0)
            if len(self.qt_list) % 500 == 0:
                print(len(self.qt_list))
                gc.collect()
                # print(qt.feature, e-s)
                # break
                # if len(self.qt_list) % 4000 == 0:
                #     th.save(self.qt_list, "/home/lsc/treelstm.pytorch/data/mid" + str(output_ind) + ".pkl")
                #     output_ind += 1
                #     del self.qt_list
                #     gc.collect()
                #     self.qt_list = []
        # if not self.selected_file:
        #     with open(os.path.dirname(input) + "/selected_file.txt", "w") as f:
        #         for i in selected_filename:
        #             f.write(i + "\n")
        del self.str_list
        return self.qt_list

    def parse_data(self, script, time_selection):
        if not self.treeforassert and not self.klee:
        # # if not self.klee:
        #     # my own parse for angr and smt-comp has been abandoned,to construct tree for asserts,please refer to pysmt
            querytree = feature_extractor(script, time_selection, self.feature_number_limit)
            querytree.treeforassert = self.treeforassert
        else:
            querytree = pysmt_query_tree(script, time_selection, self.feature_number_limit)
            querytree.treeforassert = self.treeforassert
        querytree.script_to_feature()
        qt = QT(querytree)
        del querytree.logic_tree, querytree.feature_list
        del querytree
        del script
        return qt

    def augment_scripts_dataset(self, input):
        if isinstance(input, list):
            self.str_list = input
        elif isinstance(input, str) and '\n' in input:
            self.str_list = [input]
        else:
            self.load_from_directory(input)
        self.judge_json(self.str_list[0])
        for string in self.str_list:
            script = Script_Info(string, self.is_json)
            # self.script_list.append(script)
        return self.script_list

    # only accept files with single script
    def load_from_directory(self, input):
        if not input or input == "":
            return
        if os.path.isdir(input):
            try:
                with open(os.path.dirname(input) + "/selected_file.txt") as f:
                    selected_file = f.read().split("\n")
                self.selected_file = True
            except:
                selected_file = None
            selected_file = None
            for root, dirs, files in os.walk(input):
                files.sort(key=lambda x: (len(x), x))
                for file in files:
                    if selected_file and file not in selected_file:
                        continue
                    # if os.path.getsize(os.path.join(root, file)) > 512 * 1024:
                    #     continue
                    self.read_from_file(file, os.path.join(root, file))
                    # if len(self.str_list) == 500:
                    #     return
        elif os.path.exists(input):
            self.read_from_file(None, input)

    def read_from_file(self, file, input):
        with open(input) as f:
            # if os.path.getsize(input) > 512 * 1024 or "klee" in input:
            if "klee" in input and "single_test" not in input:
                next = False
                start = False
                script = ""
                while(True):
                    try:
                        text_line = f.readline()
                        if text_line == "":
                            break
                    except:
                        continue
                    if "(set-logic QF_AUFBV )" in text_line:
                        start = True
                    if start:
                        script = script + text_line
                    if next == True:
                        self.str_list.append(script)
                        self.filename_list.append(file)
                        start = False
                        next = False
                        script = ""
                        if len(self.str_list) % 200 == 0:
                            print(len(self.str_list))
                    if "(exit)" in text_line:
                        next = True
            else:
                data = f.read()
                if data != "":
                    self.str_list.append(data)
                    self.filename_list.append(file)
                else:
                    data = ""


    def judge_json(self, data):
        try:
            json.loads(data)
            self.is_json = True
        except:
            pass

    def split_with_filename(self, test_filename=None):
        if not test_filename:
            random.shuffle(b)
            test_filename = b[:10]
        train_dataset = []
        test_dataset = []
        trt = 0
        tet = 0
        for qt in self.qt_list:
            if qt.filename in test_filename:
                test_dataset.append(qt)
                if qt.gettime() >= 300:
                    tet += 1
            else:
                train_dataset.append(qt)
                if qt.gettime() >= 300:
                    trt += 1
        return train_dataset,test_dataset
