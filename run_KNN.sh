# run the incremental KNN, here represent the experiment for GNU coreutils,
# for more setting of the model, please refer to train_KNN.py help and readme
python train_KNN.py --data_source data/gnucore/single_test --input data/gnucore/fv2_serial --time_selection z3 --time_limit_setting 200 --model_selection increment-knn

# you may further run the simulation for a tested program to see the solving in the order of data collection
python simulation.py --model_name KNN --input_index 8 --input data/gnucore/fv2_serial/train