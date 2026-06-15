# TRUHiC
## Introduction
TRUHiC is a Hi-C data resolution enhancement method that integrates a customized and lightweight transformer architecture embedded into a U-2 Net architecture to augment low-resolution Hi-C data for the characterization of 3D chromatin structure. 

This repository contains the source code, trained models, example datasets, and analysis scripts used in the manuscript entitled *"TRUHiC: A TRansformer-embedded U-2 Net to enhance Hi-C data for 3D chromatin structure characterization."*(https://www.biorxiv.org/content/10.1101/2025.03.29.646133v2)

## Getting started
The source code for model training, inference, and downstream biological analyses is organized in the `Models/` and `Experiments/` directories. Detailed instructions are provided in the README files within each folder. Example input files required for reproducing the demonstration analyses are available in the Data/ directory.

### Repository Structure
```
TRUHiC/
├── Models/        Training and inference scripts
├── Experiments/   Chromatin features and validation analyses
├── Data/          Example datasets and reference files
└── README.md
```
Detailed descriptions of the contents of each folder are provided in:
```
Models/README.md
Experiments/README.md
Data/README.md
```
A legacy archive (Models.zip) is retained for compatibility with earlier releases.  

### Installation
Clone the repository:
```
git clone https://github.com/shilab/TRUHiC
cd TRUHiC
```
### Prerequisites:
Python >= 3.7.3  
Jupyterlab >= 4.2.3

Install the required dependencies 
```
pip3 install pandas==1.2.4 numpy==1.20.2 scipy==1.7.3 matplotlib==3.5.3 statsmodels==0.13.5 seaborn==0.11.1 scikit_posthocs==0.8.1 jupyterlab
```
The code was developed and tested using the following package versions:  
Pandas 1.2.x, Numpy 1.20.x, SciPy 1.7.x, Matplotlib 3.5.x, statsmodels 0.13.x, seaborn 0.11.x, scikit_posthocs 0.8.x. 

### Quick Start
After extracting the example dataset and organizing the files according to `Data/sample_input_directory_structure.txt`:
1. Navigate to the `Models/` directory.
2. Open `TRUHiC_main.py`.
3. Update the input and output paths to match your local environment.
4. Run:
```
python TRUHiC_main.py
```
Example SLURM job submission scripts are also provided in:
```
Models/sbatch-tensorflow.job
```
for cluster-based execution

## Reproducibility
Please note that the scripts and workflows provided in this repository are organized to reproduce the analyses presented in the TRUHiC study. Input file formats and example datasets are included to facilitate replication of the reported results. Users are welcome to run the provided scripts on their own systems and adapt them for use with their own datasets. While the repository has been tested in our computing environment, minor modifications to file paths, software versions, or system-specific configurations may be required when deploying the workflow in different environments. Users wishing to apply TRUHiC to their own datasets may need to update directory paths, input configurations, and environment-specific settings accordingly.

## Contact
We welcome your questions, bug reports, suggestions, requests for additional information, or collaboration interests. Please feel free to reach out to us via the following email addresses and we will respond as soon as possible:  
:email: Dr. Chong Li:   tun53987@temple.edu or lichong0710@gmail.com (personal email)  
:email: Dr. Mohammad Erfan Mowlaei:   mohammad.erfan.mowlaei@temple.edu  
:email: Dr. Mindy Shi:   mindyshi@temple.edu

## References
#### If you find our results useful in your research, please cite our work as:
Chong Li, Mohammad Erfan Mowlaei, Human Genome Structural Variation Consortium (HGSVC), HGSVC Functional Analysis Working Group, Vincenzo Carnevale, Sudhir Kumar, Xinghua Shi. “TRUHiC: A Transformer-embedded U-2 Net to enhance Hi-C data for chromatin structure characterization.”

