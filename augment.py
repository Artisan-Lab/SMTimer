# all implemented data augment doesn't provide various SMT scripts improve training result, you could use more thorough
# changes than pruning, combination and operator replacement. Like refactoring some reasoning logic.
import json
import random
from preprocessing import op
from preprocessing import Tree_Dataset

bv_op = ["+", "-", "*", "<=", "<", "=", "ite", "toreal", "bv_constant", "bvnot", "bvand",
      "bvor", "bvxor", "extract", "bvult", "bvule", "bvneg", "bvadd", "bvsub", "bvmul", "bvudiv", "bvurem",
      "bvshl", "bvlshr", "bvrol", "bvror", "bvslt", "bvsle", "bvcomp", "bvsdiv", "bvsrem",
      "bvashr", "unknown", "distinct", ">=", ">", "bvuge", "bvugt", "bvsge", "bvsgt"]

class Script:
    def __init__(self, script):
        self.define = None
        self.asserts = None
        self.script = script
        self.split_script()

    def split_script(self):
        sl = self.script.split("(assert")
        define = sl[0]
        asserts = ["(assert" + x for x in sl[1:]]
        self.define = define
        self.asserts = asserts
        return define, asserts

    def generate_cut(self, k):
        return "".join([self.define] + self.asserts[-k:])

    def random_combine(self, script2):
        define = self.define + script2.define
        assert1 = random.sample(self.asserts, min(5, len(self.asserts)))
        assert2 = random.sample(script2.asserts, min(5, len(script2.asserts)))
        asserts = "".join(assert1 + assert2)
        asserts = asserts.replace("(check-sat)", "")
        return define + asserts + "(check-sat)"

    def change_op(self):
        new_script = self.script
        for i in range(5):
            ind1 = random.randint(0, 84)
            ind2 = random.randint(0, 84)
            while op[ind1] not in bv_op:
                ind1 = random.randint(0, 84)
            while op[ind2] not in bv_op:
                ind2 = random.randint(0, 84)
            new_script = new_script.replace(op[ind1] + " ", op[ind2] + " ", 10)
        return new_script

filename = ["echo", "ginstall", "expr", "tail", "seq", "split", "test", "yes", "chgrp", "date", "expand", "head",
                    "nohup", "printf", "sha1sum", "stat", "timeout", "uniq", "nice", "pr"]
dataset = Tree_Dataset().augment_scripts_dataset('./data/gnucore/raw_data/timeout')
ind = 0

for s in dataset:
    scr = Script(s.script)
    out = {"filename": s.filename, "smt_script": scr.generate_cut(5), "time": 0}
    with open('./data/gnucore/augment/last4/timeout' + str(ind), "w") as f:
        f.write(json.dumps(out, indent=4))
    ind += 1

last = None
last_filename = "no"
for s in dataset:
    scr = Script(s.script)
    if last != None:
        if last_filename in filename:
            output_filename = last_filename
        else:
            output_filename = s.filename
        out = {"filename": output_filename, "smt_script": scr.random_combine(last), "time": 0}
        with open('./data/gnucore/augment/crosscombine/timeout' + str(ind), "w") as f:
            f.write(json.dumps(out, indent=4))
        ind += 1
    if ind == 1:
        break
    last = scr
    last_filename = s.filename

def get_scr(sd, ind):
    try:
        scr = sd[i]
    except:
        scr = Script(s.script)
        sd[i] = scr
    return scr

sd = {}
for i, s in enumerate(dataset):
    scr = get_scr(sd, i)
    for j in range(2):
        k = random.randint(0,2741)
        output_filename = dataset[k].filename if dataset[k].filename in filename else s.filename
        out = {"filename": output_filename, "smt_script": scr.random_combine(get_scr(sd, k)), "time": 0}
        with open('./data/gnucore/augment/crosscombine/timeout' + str(ind), "w") as f:
            f.write(json.dumps(out, indent=4))
        ind += 1

for j in range(3):
    for s in dataset:
        scr = Script(s.script)
        out = {"filename": s.filename, "smt_script": scr.change_op(), "time": 0}
        with open('./data/gnucore/augment/changeop/changeop' + str(ind), "w") as f:
            f.write(json.dumps(out, indent=4))
        ind += 1

last_filename = "no"
for s in range(len(dataset)):
    if s != 0:
        with open('./data/gnucore/augment/crosscombine/timeout' + str(ind), "r") as f:
            string = f.read()
        string = json.loads(string)
        if last_filename in filename:
            output_filename = last_filename
        else:
            output_filename = dataset[s].filename
        string["filename"] = output_filename
        with open('./data/gnucore/augment/crosscombine/timeout' + str(ind), "w") as f:
            f.write(json.dumps(string, indent=4))
        ind += 1
    last_filename = dataset[s].filename

import os
for root, dirs, files in os.walk('./data/gnucore/script_dataset/training'):
    for file in files:
        with open(os.path.join('./data/gnucore/script_dataset/training', file), "r") as f:
            string = f.read()
        string = json.loads(string)
        string["filename"] = string["filename"].split("/")[-1]
        with open(os.path.join('./data/gnucore/script_dataset/training', file), "w") as f:
            f.write(json.dumps(string, indent=4))