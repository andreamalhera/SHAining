# SHAining 

Codebase for the paper "SHAining a Light on Feature Impact for Automated Process Discovery Benchmarks" 

## Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Installation](#installation)
- [Pipeline](#pipeline)
    - [Event Log Generation](#event-log-generation)
    - [Process Discovery](#process-discovery)
    - [Feature Impact Calculation](#feature-impact-calculation)
- [Paper Figures and Experimental Results](#experiments)
- [References](#references)

## Overview
Process mining is a powerful technique for automatically monitoring and enhancing real-world processes by analyzing event data (ED). Automated process discovery (PD) is often the first step in this analysis. SHAining is a novel pipeline that aims to provide a comprehensive understanding of the impact of features on the performance of PD algorithms.

The SHAining pipeline consists of three main steps: ED generation, Benchmarking, and Feature impact calculation. In the ED generation step, logs are generated based on specific feature values. The benchmarking step evaluates the performance of PD algorithms on the generated logs, while the last step analyzes the impact of features on the algorithm's performance. The pipeline is designed to be modular, allowing for easy integration of new PD algorithms and feature sets. SHAining is implemented in Python and is available as an open-source repository.

Following data artefacts are included in this paper: 
```
data/
├── bpm_25
│   ├── 8fts_3miners_ind_ilp_sm1_benchmark.csv # Contains benchmarking results from paper
│   ├── 8fts_3miners_ind_ilp_sm1_shapley.csv # Contains shapley value results from paper
│   ├── 8fts_genEL_features.csv # Contains features of generated logs
│   └── bpic_features.csv
...
```

## Requirements
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
-  <details>
      <summary><a href="https://www.java.com/en/">Java</a> (Click to see Java installation steps)</summary>
    If your server does not have java installed, install it using the following commands:
    
    1. Install Java on the server:
    ```bash
    sudo apt update
    sudo apt install default-jre
    sudo apt install default-jdk
    ```
    2. Verify the installation:
    ```bash
    java -version
    ```
    
    3. Ensure java executable is in the PATH:
    
        - Locate Java's installation path
        ```bash
        sudo update-alternatives --config java
        ```
        - Add the path to the environment variables
        ```bash
        export PATH=$PATH:/path/to/java/bin
        ```
        - Refresh the shell
        ```bash
        source ~/.bashrc
        ```
    4. Ensure that the Java executable has the correct permissions:
    ```bash
    sudo chmod a+x /path/to/java/bin/java
    ```
    </details>

## Installation
To run SHAining, follow the steps below:

#### Note: For Unix-based servers: 
Install virtual framebuffer, a display server:
```
sudo apt-get install -y xvfb
Xvfb :99 -screen 0 1024x768x16 & export DISPLAY=:99
```

#### Startup
```bash
conda env create -f .conda.yml
conda activate shaining
export PYTHONPATH=$PYTHONPATH:/path/to/shaining
```
This step will create a new conda environment with the necessary dependencies specified in ```.conda.yml```.

#### Test
To run a test of the complete pipeline: 
```
python main.py -a config_files/test/SHAining.json
```
The resulting output will start looking like: 
```
=========================== SHAining ==========================
For layer 1 with 3 features and 2 values per feature we expect 6 possible logs.
For layer 2 with 3 features and 2 values per feature we expect 12 possible logs.
For layer 3 with 3 features and 2 values per feature we expect 8 possible logs.
Total number of possible logs: 26
```
Completion can take a few minutes. Data and results of experiments from the paper can be found in the [data](data) directory.

## Pipeline 
Our framework includes several steps: [Event Log Generation](#event-log-generation), [Process Discovery](#process-discovery), and [Feature Impact Calculation](#feature-impact-calculation).
## Event Log Generation
After deciding on the EL features, the next step is to generate synthetic event logs that closely math the specified features. The event logs for this study are generated with the help of [GEDI](https://github.com/lmu-dbs/gedi/tree/bpm24). The logs are generated based on the features specified in the configuration file. 

A sample code for generating event logs for a given config file is shown below:
```
python main.py -a config_files/test/generation.json
```
The JSON file consists of the following key-value pairs:
- pipeline_step: denotes the current step in the pipeline (here: event_logs_generation)
- output_path: the path where the generated event logs are stored.
- generator_params: defines the configuration of the generator itself. For the generator, we can set values for the general 'experiment', 'config_space', 'n_trials', and a 'similarity_threshold'.
    - experiment: defines the objectives of the experiment.
    - similarity_threshold: defines how similar the generated logs should be to the original targets.
    - config_space: here, we define the configuration of the generator module (here: process tree generator). The process tree generator can process input information which defines characteristics for the generated data. Please refer to the [GEDI documentation](https://github.com/lmu-dbs/gedi/tree/bpm24?tab=readme-ov-file#experiments) for more information on the configuration options.
    - n_trials: the maximum number of trials for the hyperparameter optimization to find a feasible solution to the specific configuration being used as the target

## Extracting evaluation metrics with PD algorithms
Extracting evaluation metrics is a downstream task which is used for evaluating the goodness of the synthesized event log datasets with the metrics of real-world datasets. This repository supports the following PD algorithms:
- [Inductive Miner](https://pm4py.fit.fraunhofer.de/documentation)
- [Heuristics Miner](https://pm4py.fit.fraunhofer.de/documentation)
- [ILP Miner](https://pm4py.fit.fraunhofer.de/documentation)
- [Inductive Miner Infrequent](https://pm4py.fit.fraunhofer.de/documentation)
- [Split-Miner-2.0](https://link.springer.com/article/10.1007/s10115-018-1214-x)
    - [Source Code](https://figshare.com/articles/software/Split_Miner_2_0/12910139)
    - [Additional dependencies](https://mvnrepository.com/artifact/javax.xml.bind/jaxb-api/2.3.1)

Before running the benchmarking, make sure you have the necessary requirements.

### Process Discovery
```console
python main.py -a config_files/test/benchmark.json
```

The JSON file consists of the following key-value pairs:
- pipeline_step: denotes the current step in the pipeline (here: benchmarking)
- output_path: the path where the generated evaluation metrics are stored.
- input_path: the path where the generated event logs are stored.
- timeout: the maximum time (in seconds) allowed for the execution of the PD algorithms.
- miners: the list of PD algorithms to be used for the benchmarking.

## Feature Impact Calculation
The feature impact calculation is done by using [Shapely values](https://proceedings.neurips.cc/paper_files/paper/2017/file/8a20a8621978632d76c43dfd28b67767-Paper.pdf). Shapely values are calculated for each feature and the impact is visualized using various intuitive plots. The feature impact calculation is done by executing the following command:
```markdown
python main.py -a config_files/test/shapley.json
```

The JSON file consists of the following key-value pairs:
- pipeline_step: denotes the current step in the pipeline (here: shapley_computation)
- miners: the list of PD algorithms to be used for the benchmarking.
- feature_names: the list of features to be used for the feature impact calculation.
- input_path: the path where the generated evaluation metrics are stored.
- output_path: the path where the generated feature impact values should be stored.

## Experiments
To produce the experiment visualizations, we employ [jupyter notebooks](https://jupyter.org/install) and [add the installed environment to the jupyter notebook](https://medium.com/@nrk25693/how-to-add-your-conda-environment-to-your-jupyter-notebook-in-just-4-steps-abeab8b8d084). We then start all visualizations by running e.g.: `jupyter noteboook`. In the following, we describe the `.ipynb`-files in the folder `\notebooks` to reproduce the figures from our paper. 

#### [Feature Selection via Greedy Algorithm:](notebooks/section5_greedy_feature_selection.ipynb) 
As mentioned in section 5.1 set-up and implementation details.
#### [Table 3:](notebooks/section5_RQ1_Table3_Figures3atog_rankings.ipynb) 
Mean Shapley values and standard deviation of normalized values across evaluation metrics, ordered by feature impact variability.
#### [Figures 3 (a)-(g):](notebooks/section5_RQ1_Table3_Figures3atog_rankings.ipynb)
Feature impact ranking per metric and process discovery algorithm.
#### [Figure 4:](notebooks/section5_RQ2_Figure4_beeswarm_plots.ipynb)
To visualise the trends between features and the Shapley values, we use a beeswarm plot. The beeswarm plot is a one-dimensional scatter plot that arranges data points in a single line. The plot is used to show the distribution of Shapley values for each feature. The beeswarm plot is created using matplotlib and Seaborn library in Python. The plot is used to identify the features that have the most significant impact on the performance of the PD algorithms.
#### [Fitness Analysis for Inductive and ILP miner:](notebooks/section5_RQ2_fitness_ilp_ind_analysis.ipynb)
Includes measurements boxplots, shapley value boxplots and dedicated beeswarm plots.

## References
Anonymous
