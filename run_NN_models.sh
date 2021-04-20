# run the nerual network model, here only represent one model,
# using lstm model,
# to use other models, please refer to train.py help and readme
python train.py --data_source data/gnucore/single_test --input data/gnucore/pad_feature --model lstm --

# evaluate the trained model with single program's result
python train.py --input data/gnucore/pad_feature --load --load_file g_pad_feature_l_z_c_200_1

# you may further run the simulation for a tested program to see the solving in the order of data collection
python simulation.py --model_name lstm --input_index 0 --adapt --input data/gnucore/fv2_serial/train
