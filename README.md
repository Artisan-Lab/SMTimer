SMT solving is a bottleneck for the symbolic execution. SMTimer provides a time prediction for SMT script solving for a certain solver. With the predicted solving time, symbolic execution can choose the path with lower time to explore first, or directly skip the timeout cases without wasting time on it.

The project is still under refactoring and construction

# Data collection
The data collection is not included in current project environment, but we also give out our SMT script collection script, you may construct it by your own. The files are located in `data_collection` directory.

# Collected data share
Our collected data is available on <a href="https://???" target="_blank"></a>, which including three datasets, which are constraint models generated with GNU Coreutils using angr and KLEE, BusyBox using angr, you may also use the collections of problems in SMT competitions <a href="https://smt-comp.github.io/2020/" target="_blank"></a>, we also use this dataset(non-increment BV) in our experiment. We also release our solving time file for this dataset so you can replay the result.

# SMT solving
This step is optional since it is time-consuming. You may directly use the solving time we solve with z3. If you want to use other SMT solvers or make sure the label is precious with your setting. The time would be affected by many factors like runtime environment and parallel number. We use the pySMT to solve so make sure you install the solver.

```
pysmt-install --check
pysmt-install --z3
```

You can use this command to solve the script and add the solved time into your script. You may find the detail setting in `check_time.py`, `solve.py`, you need to change the logic in `solve` if you want to use other reasoning theories.

`source SMTsolving.sh`

# neural network (non-adaptive approach)

The neural network is the non-adaptive way. You may skip this part as well if you are only interested in the adaptive approach(incremental_KNN). We mainly present this result for comparision.

The neural network models train with features after extraction, we support three models including LSTM, tree-LSTM, DNN,
first make some directory that is needed, the data should be placed in the `data` directory
```
mkdir checkpoints, simulation_result, data
```
As example, we use our GNU coreutil dataset to train LSTM classification model, You can run the script. The file does three work. First it train the model, secondly it represents more results for the model, thirdly you can see the simulation result of `arch` with the model you used. To further use other models or see the regression result, you may use different command line argument. The help information should guide you the usage.
> source run_NN_model.sh

Next, we introduce the command separatedly for better explanation
The `train.py` includes the feature extraction, neural network training and some evaluation. The result of feature extraction would be saved in the path of `input`, so you can reuse the feature of scripts. We do not provide this result because the data size is too large. After training, the model would be saved in `checkpoints` directory, the model name is composed of dataset_input_model_time-type_task_threshold_index.pkl. For tree-LSTM, our model just use the feature vector instead of abstract tree, which makes it functionally works like LSTM. But you may use "tree+feature" as input directory to use abstract tree, the efficiency is not ensured for use that, so be careful.
> python train.py --data_source data/gnucore/single_test --input data/gnucore/pad_feature --model lstm --time_selection z3 --threshold 200 --regression

You can check the trained model with the same file with the `load, load_file` option, the other setting should be the same as the last command. The result would also be saved in `checkpoints` directory, the result name is composed of dataset_evaulation_task_index.pkl. The result includes more evaluation metrics amd the predict result for you selected test dataset.
> python train.py --input data/gnucore/pad_feature --load --load_file g_pad_feature_l_z_r_200_1 --model lstm --time_selection z3 --regression

Then, we evaluate the solving time with our purposed system in simulation. You can see the data that predicted wrongly, original solving time, solving time after boosting(total_time), timeout contraints number with your setting threshold(pos_num), and tp, fn, fp cases numbers, also the classification metric for prediction(regression result would be classified with you setting threshold).
> python simulation --model_name lstm --load_file checkpoints/g_pad_feature_l_z_r_200_1.pkl --test_directory data/gnucore/single_test/arch --time_selection adjust

# increment-KNN (adaptive approach)

Like the neural network model, the KNN extracts features, we support two main setting including KNN implemented in sklearn(non-adaptive so poor in performance,better in efficiency), incremental-KNN. As example, we use our GNU coreutil dataset to train incremental-KNN classification model, You can run the script. The file does two jobs. First it represents the prediction results for every program, next you can see the simulation result of `arch`.
> source run_KNN.sh

Next, we introduce the command separatedly for better explanation
The `train_KNN.py` includes the feature extraction and KNN evaluation. The result of feature extraction would be saved in the path of `input`, so you can reuse the feature of scripts. We do not provide this result directly, but in json structure in `KNN_training_data`, because we want to easily use our KNN model for symbolic execution tools. Still under refactor:to do change the data saving to avoid the use of torch in KNN.
> python train_KNN.py --data_source data/gnucore/single_test --input data/gnucore/fv2_serial --time_selection z3 --time_limit_setting 200 --model_selection increment-knn

The following description are the same as the neural network simulation. Just in case you skip neural network part, we repeat the description.
Then, we evaluate the solving time with our purposed system in simulation. You can see the data that predicted wrongly, original solving time, solving time after boosting(total_time), timeout contraints number with your setting threshold(pos_num), and tp, fn, fp cases numbers, also the classification metric for prediction.
> python simulation --model_name KNN --load_file data/gnucore/fv2_serial/train --test_directory data/gnucore/single_test/arch --time_selection adjust

# use of the prediction
To use the tool, you can use the `predictor.py` from the home directory, take `angr` solve with `z3` as example, you replace the solving `result = solver.check()` like:
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
    predicted_solving_time = predictor.predict(query_smt2)
    # predicted result is solvable with our prediction timeout t2, else skip
    if predicted_solving_time < 200:
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