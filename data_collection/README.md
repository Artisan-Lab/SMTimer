### angr
We provide our SMT script collection script, but you have to build the environment by your own, you need to change `angr` dependency `claripy` in `backends/backend_z3.py`, we modify its SMT solving which in the form of `solver.check()` with a data collection function to record hard solving constraints, you may find the realization in this file under current directory. The data collection file is `output_query_data_struct.py`. For specific, our angr version is 8.19.10.30.
To run the symbolic execution, you need to compile the files first, then you can use the script
> source run_gnu.sh

### KLEE
For KLEE, we use the script from the PathConstraintClassifier. you may use `-use-query-log=solver:smt2` to get the constraint model, but the data require some further work, since they all save in single file and all assertions are in one command.