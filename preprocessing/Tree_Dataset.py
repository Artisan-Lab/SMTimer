import time
import gc
import signal

from .feature_extraction import Script_Info, feature_extractor, QT
from preprocessing.pysmt_tree import pysmt_query_tree
from Dataset import Dataset, handler

# input all kinds of scripts and return expression tree
class Tree_Dataset(Dataset):

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
            signal.signal(signal.SIGALRM, handler)
            signal.alarm(1)
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
