from preprocessing.Dataset import Dataset, handler
from preprocessing.feature_vectors import *

import signal

class Vector_Dataset(Dataset):

    def generate_feature_dataset(self, input, time_selection=None):
        self.str_list = []
        if isinstance(input, list):
            self.str_list = input
        elif isinstance(input, str) and '\n' in input:
            self.str_list = [input]
        else:
            self.load_from_directory(input)
        if not len(self.str_list):
            return
        self.judge_json(self.str_list[0])
        selected_filename = []
        for ind, string in enumerate(self.str_list):
            script = Script_Info(string, self.is_json)
            try:
                if script.solving_time_dic["z3"][0] < 0:
                    continue
                # if not self.selected_file:
                #     if float(script.solving_time) < 20 and float(script.solving_time_dic["z3"][0]) < 10:
                #         if len(self.str_list) > 20000 and ind % 10 != 0:
                #             continue
                selected_filename.append(self.filename_list[ind])
            except:
                continue
            self.script_list.append(script)
            signal.signal(signal.SIGALRM, handler)
            # signal.alarm(1)
            try:
                fv = self.parse_data(script, time_selection)
                self.qt_list.append(fv)
            except TimeoutError:
                signal.alarm(0)
                print("preprocess over time", len(self.qt_list))
                continue
            except (KeyError,IndexError):
                continue
            finally:
                signal.alarm(0)
            if len(self.qt_list) % 500 == 0:
                print(len(self.qt_list))
                # break
        # if not self.selected_file:
        #     with open(os.path.dirname(input) + "/selected_file.txt", "w") as f:
        #         for i in selected_filename:
        #             f.write(i + "\n")
        return self.qt_list

    def parse_data(self, script, time_selection):
        featurevectors = feature_vectors(script, time_selection, self.feature_number_limit)
        featurevectors.script_to_feature()
        if self.feature_number_limit == 2:
            fv = FV2(featurevectors)
        else:
            fv = FV(featurevectors)
        return fv