SMT solving is a bottleneck for the symbolic execution. SMTimer provides a time prediction for SMT script solving for a certain solver(in our case `z3`). With the predicted solving time, symbolic execution can choose the path with lower time to explore first, or directly skip the timeout cases without wasting time on it.

Still try to make a small dataset for get started.

# Get started
The meaning for output are not very detail in this part, for more information, please read the `Detail description`.

## Collected data sharing
Our collected data is available on <https://drive.google.com/drive/folders/1fiYNM4EymKbAjBFGwInHQXXb2y5mJ15N?usp=sharing>, which including four datasets. We use the constraint model generated with GNU Coreutils(angr) as example. 
First put the data in certain path.
```
mkdir data/gnucore
cd data/gnucore
# put the data under the above directory
tar -xcvf gnu-angr.tar.gz .
``` 

After you get the data, we recommand you to put the SMT files in the `data/{dataset}/single_test` directory, this is not forced, but the scripts are based on fixed relative path.

## SMT solving(optional, time-consuming, about one day long for each dataset)

 The data we provided have already done this part for you. But you can use this command to solve the script and add the solved time into your script by your own.
```
pysmt-install --check
pysmt-install --z3
source SMTsolving.sh
```

After using this command you can see that the solving time of the scripts, also saved in the `adjustment.log`, and the time after adjustment has been added into script with the key of `solving_time_dic`. There is a item named `z3` in `solving_time_dic`.

## neural network (non-adaptive approach, cost about 15 minutes)
As example, we use our GNU coreutil dataset to train LSTM classification model, You can run the script. 
```
mkdir checkpoints, simulation_result
source run_NN_model.sh
```
After using this command you can see the the processed data number first, then the training process on your screen for the first command in the script. Then the prediction of test dataset and more measurement result for second command. And a simulation result for program `arch` for last command. The model and evaluation result are saved in `checkpoints` if you want to examine it and use the model. The simulation result are saved in `simulation_result`.

## increment-KNN (adaptive approach, cost about 10 minutes)

As example, we use our GNU coreutil dataset to train incremental-KNN classification model. You can run the script. 

`source run_KNN.sh`

After using this command you can see the the processed data number first, then the prediction measurement result on your screen for the first command in the script. And a simulation result for program `arch` for second command. Similarly, the simulation result are saved in `simulation_result`.

# Detail description
The following instruction would tell you how our most modules work. So you can change the setting or replay the experiment result. 

## Data collection
The data collection is not constructed in current project environment, but we still give out our SMT constraint model collection module, you may construct it by your own. All the files are located in `data_collection` directory.

## Collected data sharing
Our collected data is available on <https://drive.google.com/drive/folders/1fiYNM4EymKbAjBFGwInHQXXb2y5mJ15N?usp=sharing>, which including four datasets, which are constraint models generated with GNU Coreutils using angr and KLEE, BusyBox using angr and from SMT-comp. You may get more information about the SMT competitions from <https://smt-comp.github.io/2020/>, we use this dataset(non-increment,QF_BV) in our experiment. To download original data, the address is <https://www.starexec.org/starexec/secure/explore/spaces.jsp?id=404954>. Our script specifically handle its file names so it's hard to use original data. So we release data of our version with solving time for this dataset so you can replay our prediction result. We may refactor the processing to match the name setting for this dataset if we are free.

#### data selection
The training data is heavily imbalanced, we conduct a random undersampling to contain fewer fast-solving cases. So the result is a little different for different selection.

## SMT solving(optional, time-consuming)
This step is optional since it is time-consuming. You may directly use the solving time we solve with z3. If you want to use other SMT solvers or make sure the label is consistent with your setting, you can solve it by yourself. Because the time would be affected by many factors like runtime environment and parallel situation. We use the pySMT to solve so make sure you install the solver first.

```
pysmt-install --check
pysmt-install --z3
```

You can use this command to solve the script and add the solved time into your script. You may find the detail setting in `check_time.py`, `solve.py`, you need to change the logic in `solve` if you want to use other reasoning theories.

`source SMTsolving.sh`

## neural network (non-adaptive approach)

The neural network is the non-adaptive way. You may skip this part as well if you are only interested in the adaptive approach(incremental_KNN). We mainly present this result for comparision.

The neural network models train with features after extraction, first make some directory that is needed,

```
mkdir checkpoints, simulation_result, data
``` 

As example, we use our GNU coreutil dataset to train LSTM classification model. The script in `get started` does three work. First it train the model, secondly it represents more results for the model, thirdly you can see the simulation result of program `arch` with the model you used.

Next, we introduce the command separatedly for better explanation.

The `train.py` includes the feature extraction, neural network training and some evaluation. The result of feature extraction would be saved in the path of `input`, so you can reuse the feature of SMT scripts. We do not provide this middle result because the data size is too large. After training, the model would be saved in `checkpoints` directory, the model name is composed of `dataset_input_model_time-type_task_threshold_index.pkl`. For tree-LSTM, our model use the feature vector instead of abstract tree by default, which makes it functionally works like LSTM. But you may use "tree+feature" as input directory to use abstract tree. To make it more practicable, we make some induction, which replaces oversized abstract trees into vectors. Our results suggest the improvement does not worth the cost. If you want to further use the tree structure, you could work on it in `preprocessing/abstract_tree_extraction.py`.

`python train.py --data_source data/gnucore/single_test --input data/gnucore/pad_feature --model lstm --time_selection z3 --threshold 200 --regression`

 To further use other models or see the regression result, you may use different command line argument. The help information should guide you the usage. We explain some of them for experiment replay.

+ --model lstm/tree-lstm/dnn, we support three models including LSTM, tree-LSTM, DNN.
+ --regression, we support the time prediction(regression) and timeout constraint classification(classification)

You can check the trained model with the same file using `load_file` options, the other setting should be the same as the last command. The result would also be saved in `checkpoints` directory, the result name is composed of `dataset_evaulation_task_index.pkl`. The result includes more evaluation measurement results amd the predict result for you selected test dataset.

`python train.py --input data/gnucore/pad_feature --load_file g_pad_feature_l_z_r_200_1 --model lstm --time_selection z3 --regression`

Then, we evaluate the solving time with our purposed system in simulation. You can see the data that predicted wrongly, original solving time, solving time after boosting(total_time), timeout constraints number with your setting threshold(pos_num), and tp, fn, fp cases numbers, also the classification measurement results for prediction(regression result would be classified with you setting threshold).

`python simulation.py --model_name lstm --load_file checkpoints/g_pad_feature_l_z_r_200_1.pkl --test_directory data/gnucore/single_test/arch --time_selection adjust --regression`

## increment-KNN (adaptive approach)

Like the neural network model, the KNN extracts features, we support two main setting including KNN implemented in sklearn(non-adaptive so poor in performance,better in efficiency), incremental-KNN. As example, we use our GNU coreutil dataset to train incremental-KNN classification model. The script in `get started` does two jobs. First it represents the prediction results for every program, next you can see the simulation result of program `arch`.

Next, we introduce the command separatedly for better explanation.

The `train_KNN.py` includes the feature extraction and KNN evaluation. The result of feature extraction would be saved in the path of `input`, so you can reuse the feature of scripts. We only provide this result for GNU(angr) in json structure in `KNN_training_data` directory, which is also used in our KNN predictor for symbolic execution tools. You can use it with `--input KNN_training_data/gnucore.json`.

`python train_KNN.py --data_source data/gnucore/single_test --input data/gnucore/fv2_serial --time_selection z3 --time_limit_setting 200 --model_selection increment-knn`

The following description are the same as the neural network simulation. Just in case you skip neural network part, we repeat the description.

Then, we evaluate the solving time with our purposed system in simulation. You can see the data that predicted wrongly, original solving time, solving time after boosting(total_time), timeout contraints number with your setting threshold(pos_num), and tp, fn, fp cases numbers, also the classification metric for prediction.

`python simulation.py --model_name KNN --load_file data/gnucore/fv2_serial/train --test_directory data/gnucore/single_test/arch --time_selection adjust`

## use of the prediction
To use the tool, you can use the `KNN_Predictor.py` from the home directory, take `angr` with `z3` solver backend as example, you need to replace the solving `result = solver.check()` like:
```
# set the solve time with t1=1s
solver.set('timeout', 1000)
# solve for the first time
result = solver.check()
# get script
query_smt2 = solver.to_smt2()
# timeout for t1
if result == z3.unknown:
    # predict the solve time, predictor is a KNN Predictor define above
    predicted_solvability = predictor.predict(query_smt2)
    # predicted result is solvable, else skip
    if predicted_solvability == 0:
        # set the solve time with 300s, the default solve time of angr
        solver.set('timeout', 3000000)
        #solve again
        result = solver.check()
        # update the model
        if result == z3.unknown:
            # prediction wrong, but solving corrects it
            predictor.increment_KNN_data(1)
        else:
            # prediction right
            predictor.increment_KNN_data(0)
```
