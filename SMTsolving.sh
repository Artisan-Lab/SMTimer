# the data is an example for showing the process, for full experiments you need to download the whole dataset from not decided

# get the solving time for single SMT scripts from the input directory, this step is mainly because the solving time is unstable inside symbolic execution, we need the label to be accurate for later prediction
# noting that: this step is extramely time-consuming, so if you just want to see the result of prediction, you could skip this file and use the data we collected
python check_time.py --input data/example/arch

# add the solving time to the SMT script in json structure for later use
python merge_check_time.py --script_input data/example/arch

