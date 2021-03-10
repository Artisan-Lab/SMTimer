import warnings

from preprocessing.pysmt_tree import *
import numpy as np

class FV:
    def __init__(self, query_tree, filename=None):
        self.logic_tree = query_tree.feature_list
        self.origin_time = query_tree.origin_time
        self.adjust_time = query_tree.adjust_time
        try:
            self.filename = query_tree.script_info.filename
        except (KeyError,IndexError):
            self.filename = filename
        # self.feature = [math.log(x + 1) for x in query_tree.feature]
        self.feature = query_tree.feature

    def gettime(self, time_selection="origin"):
        try:
            if time_selection == "origin":
                return self.origin_time
            else:
                return self.adjust_time
        except (KeyError,IndexError):
            return self.timeout

class FV2(FV):
    def __init__(self, query_tree, filename=None):
        FV.__init__(self, query_tree, filename)
        feature_list = self.logic_tree
        if len(feature_list) == 2:
            self.feature = feature_list.flatten()
            # self.feature = np.log(self.feature + 1)
        else:
            warnings.warn("the feature vector sum up should be done during parsing", DeprecationWarning)
            self.feature = np.zeros(300)
            self.feature[:150] = np.sum(feature_list[:-1], axis=0)
            self.feature[150:] = feature_list[-1]
            self.feature = np.log(self.feature+1)


class feature_vectors(pysmt_query_tree):
    # def __init__(self, script_info, time_selection="origin"):
    #     query_tree.__init__(script_info, time_selection)

    def script_to_feature(self):
        data = self.script_info.script
        self.cal_training_label()
        assertions = self.get_variable(data)

        # for var_name in self.val_list:
        #     data = data.replace(var_name, self.val_dic[var_name])
        try:
            # parse assertion stack into expression trees
            self.assertions_to_feature_list(assertions)
            # summing up sub tree features
            self.standardlize()
        except (KeyError,IndexError) as e:
            # print(e)
            self.logic_tree = np.zeros((self.feature_number_limit, 150))

    def assertions_to_feature_list(self, assertions):
        limit = self.feature_number_limit
        if "QF_AUFBV" in assertions[0]:
            self.parse_klee_smt(assertions)
            self.feature = np.sum(self.feature_list, axis=0).tolist()
            return
        asserts = assertions[1:]
        if len(asserts) > limit:
            asserts[-limit] = "\n".join(asserts[:-limit + 1])
            asserts = asserts[-limit:]
        try:
            for assertion in asserts:
                feature = self.count_feature(assertion)
                self.feature_list.append(feature)
        except (KeyError,IndexError):
            traceback.print_exc()
            return

    def standardlize(self):
        limit = self.feature_number_limit
        if len(self.feature_list) == 0:
            self.feature_list = np.zeros((limit,150))
            return
        feature_list = np.array(self.feature_list)
        if len(feature_list) < limit:
            padding_num = limit - len(feature_list)
            feature_list = np.row_stack([feature_list, np.zeros([padding_num, 150])])
            self.feature_list = feature_list
        elif len(feature_list) > limit:
            feature_list[-limit] = np.sum(feature_list[:-limit + 1], axis=0)
            self.feature_list = feature_list[-limit:]
        self.feature = np.sum(feature_list, axis=0).tolist()
        if self.feature_number_limit == 2:
            self.feature_list = np.log10(np.array(self.feature_list)+1)
        else:
            self.feature_list = np.log(np.array(self.feature_list)+1)