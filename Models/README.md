# Welcome to the TRUHiC repository.

Benchmark methods are placed under the "benchmark" directory.

## Data Preparation:

The data should be placed in a structured format similar to the sample data provided. You can check "sample_input_directory_structure.txt" for the tree view of the sample data directory.

## Environment

You will need a tensorflow 2.14 environment to run TRUHiC and DFHiC scripts, which can be found as "tensorflow-gpu.yml". 

For the rest of the benchmark methods, you need to use "torch-gpu.yml" environment.

Using miniconda, you can install these environments as follows:

> conda env create -f [environment-name].yml


## Running the code

TRUHiC and DFHiC are implemented in Tensorflow and you can run them on a slurm job queue using the provided "sbatch-tensorflow.job" script.

The other benchmark methods are implemented in pytorch 2.2. You can run them using the provided "sbatch-torch.job" slurm script or similar commands to that.

## Calculating vision metrics

To calculate the vision metrics on human cell lines, you need to use "Evaluate_Metrics_args.py" and modify the test chromosomes defined inside it as needed.

For cross cell-line experiments, you will need to use "Evaluate_Metrics_args_cross.py" and similarly adjust the test chromosomes manually inside this script.

Both of the mentioned scripts use the torch environment.
