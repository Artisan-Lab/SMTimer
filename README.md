SMT solving is a bottleneck for symbolic execution. SMTimer provides a time prediction for SMT script solving for a certain solver(in our case `z3`). With the predicted solving time, symbolic execution can choose the path with a lower time to explore first, or directly skip the timeout cases without wasting time on them. This repository is the open-source project for `Boosting Symbolic Execution via Constraint Solving Time Prediction`. We hope the guidance and the explanation would help you better understand the paper and reproduce the experiments.

# Get Started
To get started, we demonstrate the process based on a small example. For detail output meaning, please read the `Detail description`.

## Build environment 
You can build the environment with pip. For detail, the building dependencies include numpy, torch, sklearn, keras, dgl, pysmt, EasyProcess, matplotlib.

`pip install -r requirements.txt`


## Collected data
Our collected data is available on <https://doi.org/10.5281/zenodo.4722615> and <https://drive.google.com/drive/folders/1fiYNM4EymKbAjBFGwInHQXXb2y5mJ15N?usp=sharing>, which including four datasets. For getting started, we use a small subset of the constraint models generated with GNU Coreutils(angr) . You can find the SMT files in `data/example/single_test`. 

If you want to use the whole dataset, please download it and put the data in certain paths.
```
cd data/gnucore
# put the data under the above directory
tar -zxvf gnu-angr.tar.gz
``` 

After you get the data, we recommend you to put the SMT files in the `data/{dataset}/single_test` directory, this is not forced, but the scripts are based on fixed relative paths.

## SMT solving(optional, time-consuming, about hours long for example)

 Considering the time budget, the data we provided have already done this part for you. But you can use this command to solve the SMT script and add the solved time into your files on your own.
```
pysmt-install --check
pysmt-install --z3
source SMTsolving.sh example
```

Screen outputs: 
+ the solving time of the SMT scripts.

Saved results: 
+ the solving time is also saved in the `adjustment.log`. 
+ the time after the adjustment has been added into SMT scripts in the field `solving_time_dic`. You can get it from the dictionary with the key `z3`.

## Neural network (non-adaptive approach, cost about 2-3 minutes)
For example, we use our GNU coreutil dataset subset to train the LSTM regression model, You can run the script. 
```
mkdir checkpoints simulation_result
source run_NN_models.sh example
```
Screen outputs: 
+ the processed SMT scripts number first, then the neural network training process on your screen
+ the prediction of test dataset and more measurement results
+ a simulation result for program `arch`

Saved results: 
+ the model and evaluation results are saved in the `checkpoints` directory if you want to examine them and use the model. 
+ the simulation results are saved in the `simulation_result` directory.

## Increment-KNN (adaptive approach, cost about 1 minutes)

For example, we use our GNU coreutil dataset subset to train the Incremental-KNN classification model. You can run the script. 

`source run_KNN.sh example`

Screen outputs: 
+ the processed data number first, then the prediction measurement result on your screen
+ a simulation result for program `arch`

Saved results:
+ the feature vector dataset for reuse in `data/example/pad_feature`
+ the simulation results are saved in the `simulation_result` directory.

# Detail description
The following instruction would tell you how our most modules work. So you can change the setting or reproduce the experiment result. The reference of the figures, the tables, the equations in our explanation are from the above paper.

## Experiment reproduction

With the project, you can reproduce three experiments. You can find the detail description in the quoted part.

+ the classification of timeout constraint models (Section 5.2.1, Table 2)

     [a](####Evaluation)
     Neural network - Evaluation
     Increment-KNN - Training and evaluation
+ time prediction (Section 5.2.3, Table 4)

    > Neural network - Evaluation
+ simulation for solving time (Section 5.3, Table 6)

    > Neural network - Simulation <br> Increment-KNN - Simulation

To reproduce experiments for different dataset, you need to change the dataset name in the path-related options. Just in case the hard coded problem, our dataset name is gnucore(for GNU coreutils(angr)), busybox(Busybox utilities(angr)), smt-comp(for SMT-COMP), klee(for GNU coreutils(KLEE)).

## Data collection
The data collection is not constructed in the current project environment, but we still give out our SMT constraint model collection module, you may construct it on your own. All the files are located in the `data_collection` directory.

## Collected data sharing
Our collected data is available on <https://doi.org/10.5281/zenodo.4722615> and <https://drive.google.com/drive/folders/1fiYNM4EymKbAjBFGwInHQXXb2y5mJ15N?usp=sharing>, which including four datasets, which are constraint models generated with GNU Coreutils using angr and KLEE, BusyBox using angr, and from SMT-comp. You may get more information about the SMT competitions from <https://smt-comp.github.io/2020/>, we use this dataset(non-increment, QF_BV) in our experiment. To download original data, the address is <https://www.starexec.org/starexec/secure/explore/spaces.jsp?id=404954>. Our code specifically handles its file names so it's hard to use original data. So we release data of our version with solving time for this dataset so you can replay our prediction result. We may refactor the processing to match the name setting for this dataset if we are free.

#### Data selection
The training data is heavily imbalanced, we conduct a random under-sampling to contain fewer fast-solving cases. So the result is a little different for different selections.

## SMT solving(optional, time-consuming, about one day long for each dataset)
This step is optional since it is time-consuming. You may directly use the solving time we solve with z3. If you want to use other SMT solvers or make sure that the label is consistent with your setting, you can solve it by yourself. Because the time would be affected by many factors like runtime environment and parallel situation. We use the pySMT to solve them so make sure you install the solver first.

```
pysmt-install --check
pysmt-install --z3
```

You can use this script to solve the SMT scripts and add the solving time into your SMT files. You may find the detail setting in `check_time.py`, `solve.py`. You need to change the logic in `solve.py` if you want to use other reasoning theories.

`source SMTsolving.sh`

## Neural network (non-adaptive approach)

The neural network is a non-adaptive way for the solving time inference problem. You may skip this part as well if you are only interested in the adaptive approach(incremental_KNN). We mainly present this result for comparison and replay experiment results.

First, make some directories that are needed,

```
mkdir checkpoints simulation_result
``` 

For example, we use GNU coreutil dataset to train the LSTM regression model. The script in `get started` does three work. 
 
 + train the model
 + evaluate the model
 + runs a simulation for program `arch` with the model you used.

Next, we introduce the command separately for a better explanation.
#### Training
The `train.py` does the feature extraction, neural network training, and some evaluations. The result of feature extraction would be saved in the path of `input`, so you can reuse the feature of SMT scripts. We do not provide this middle result because the data size is too large. After training, the model would be saved in the `checkpoints` directory, the model name is in the format of `dataset_input_model_time-type_task_threshold_index.pkl`.

`python train.py --data_source data/gnucore/single_test --input data/gnucore/pad_feature --model lstm --time_selection z3 --threshold 200 --regression --cross_index 0`

 To further use other models or see the classification result, you may use different command-line arguments. The help information should guide you. We explain some of them for experiment replay.

+ --model lstm/tree-lstm/dnn, we support three models including LSTM, tree-LSTM, DNN.
+ --regression, we support the time prediction(regression) and timeout constraint classification(classification)
+ --cross_index 0/1/2, we experiment with three-fold cross-validation, you may find the result difference is huge, which is a motivation to use the adaptive approach
+ --tree_for_assert, true for using abstract trees as tree nodes in tree-LSTM model

 For tree-LSTM, our model uses the feature vector instead of the abstract tree by default, which makes it functionally works like LSTM. But you may use `--tree_for_assert` option to use the abstract tree. To make it more practicable, we make some inductions, which replace oversized abstract trees with vectors. Our results suggest the improvement does not worth the cost. If you want to further research the tree structure, you could work on it in `preprocessing/abstract_tree_extraction.py`.
 
#### Evaluation
You can check the trained model with the same file using `load_file` options, the other setting should be the same as the last command. The evaluation result includes more measurement results (e.g. MAE for regression, precision,recall,f1_score,confusion_matrix for classification) and the predicted result for selected test dataset. The result would also be saved in the `checkpoints` directory, the result name is in the format of `dataset_evaulation_task_index.pkl`.

`python train.py --input data/gnucore/pad_feature --load_file g_pad_feature_l_z_r_200_0 --model lstm --time_selection z3 --regression`

#### Simulation
Then, we simulate the solving time with our purposed system. The time is calculated with the Eq.(1),(2) in the paper. The screen outputs show the data that predicted wrongly, and the simulation results shown in Table.6. The results include original solving time, solving time after boosting(total_time), timeout constraints number with your setting threshold, and tp, fn, fp cases numbers, also the classification measurement results for prediction(regression result would be classified with you setting threshold).

`python simulation.py --model_name lstm --load_file checkpoints/g_pad_feature_l_z_r_200_0.pkl --test_directory data/gnucore/single_test/arch --time_selection adjust --regression`

## Increment-KNN (adaptive approach)

The increment-KNN is based on the concept of online learning. It adaptively updates the model with test constraints.

For example, we use our GNU coreutil dataset to train incremental-KNN classification model. The script in `get started` does two jobs. 

+ represent the prediction results for every program
+ run a simulation for program `arch`.

Next, we introduce the command separately for a better explanation.

[nn_eva](#### Training and evaluation)
The `train_KNN.py` does the feature extraction and KNN evaluation. The result of feature extraction would be saved in the path of `input`, so you can reuse the feature of scripts. We only provide the result for GNU(angr) in the JSON structure in the `KNN_training_data` directory, which is also used in our KNN predictor for symbolic execution tools. You can use it with `--input KNN_training_data/gnucore.json`. You should get the result in Table.2 and get a more detailed result for single programs.

`python train_KNN.py --data_source data/gnucore/single_test --input data/gnucore/fv2_serial --time_selection z3 --time_limit_setting 200 --model_selection increment-knn`

We support two main settings including KNN implemented in sklearn(non-adaptive so poor in performance, better in efficiency, mainly for comparison), incremental-KNN. You may use different command-line arguments. 
 
#### Simulation
The following description is the same as the neural network simulation. Just in case you skip the neural network part, we repeat the description.

Then, we simulate the solving time with our purposed system. The time is calculated with the Eq.(1),(2) in the paper. The screen outputs show the data that predicted wrongly, and the simulation results shown in Table.6. The results include original solving time, solving time after boosting(total_time), timeout constraints number with your setting threshold, and tp, fn, fp cases numbers, also the classification measurement results for prediction.

`python simulation.py --model_name KNN --load_file data/gnucore/fv2_serial/train --test_directory data/gnucore/single_test/arch --time_selection adjust`

## The use of the prediction
To use our KNN predictor as a tool, you can use the `KNN_Predictor.py` from the home directory. Taking `angr` with `z3` solver backend as an example, you need to replace the solving `result = solver.check()` like:
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
This is a code representation of our predictor architecture interacting with symbolic execution tool angr.
