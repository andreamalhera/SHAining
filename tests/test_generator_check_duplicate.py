import json
import pandas as pd
import pytest
import os
import re
import shutil

from pathlib import Path
from shaining import GenerateLogs
from shaining.utils.io_helpers import get_output_key_value_location, delete_list_of_dirs

"""
Purpose of this test is to see if generator skips existing logs.
"""

def clear_output():
    delete_list_of_dirs([os.path.join("data", "test", "generation", "duplicate_logs"),
                         os.path.join("data", "generated", "duplicate_logs"),
                         os.path.join("data", "features", "test", "generation", "duplicate_logs"),
                         ])

def test_duplicate_EL_jsons_GenerateEventLogs():
    INPUT_PARAMS =  {   "pipeline_step": "event_logs_generation",
                        "output_path": "data/test/generation/duplicate_logs",
                        "generator_params": {
                            "config_space": { "mode": [ 5, 20], "sequence": [ 0.01, 1], "choice": [ 0.01, 1],
                                             "parallel": [ 0.01, 1], "loop": [ 0.01, 1], "silent": [ 0.01, 1],
                                             "lt_dependency": [ 0.01, 1], "num_traces": [ 10, 100],
                                             "duplicate": [ 0], "or": [ 0]},
                                             "n_trials": 50,
                            "experiment": [
                            { "n_traces": 99.0 },
                            { "n_traces": 998.4 },
                            { "ratio_variants_per_number_of_traces": 0.99 },
                            { "ratio_variants_per_number_of_traces": 0.5 },
                            { "trace_len_min": 1.0 },
                            { "trace_len_min": 1.2 },
                            { "n_traces": 99.0, "ratio_variants_per_number_of_traces": 0.99 },
                            { "n_traces": 998.4, "ratio_variants_per_number_of_traces": 0.5, "trace_len_min": 1.2 }
                            ]}
                     }

    EXPECTED_OUTPUT_NAMES = ['genEL1_990_nan_nan', 'genEL1_9984_nan_nan', 'genEL2_nan_05_nan', 'genEL3_990_099_nan', 'genEL3_nan_099_nan', 'genEL4_9984_05_12', 'genEL5_nan_nan_10', 'genEL6_nan_nan_12']
    DUPLICATED_DATA  = {
    'log': ['genEL1_990_nan_nan', 'genEL3_nan_099_nan', 'genEL5_nan_nan_10', 'genEL6_nan_nan_12'],
    'target_similarity': [1.000000, 0.990099, 1.000000, 0.833333]}
    DUPLICATED_NAMES = DUPLICATED_DATA['log']
    df = pd.DataFrame(DUPLICATED_DATA)
    DUPL_LIST_OF_DICTS = df.to_dict(orient='records')

    save_path = get_output_key_value_location(INPUT_PARAMS["generator_params"]["experiment"][-1], INPUT_PARAMS["output_path"], "TEST")
    save_path = os.path.split(save_path)[0]
    feat_save_path = os.path.join(*Path(save_path).parts[:1],"features",*Path(save_path).parts[1:])

    delete_list_of_dirs([os.path.split(save_path)[0], #feat_save_path within os.path.split(save_path)[0]
                         ])
    clear_output()

    os.makedirs(save_path, exist_ok=True)
    os.makedirs(feat_save_path, exist_ok=True)

    for idx, dup in enumerate(DUPLICATED_NAMES):
        log_path = os.path.join(save_path, dup)
        feat_path = os.path.join(feat_save_path, dup)
        with open(log_path+".xes", 'w') as f:
            pass
        with open(feat_path+".json", 'w') as f:
            json.dump(DUPL_LIST_OF_DICTS[idx], f, indent=4)
            pass

    if os.path.exists(save_path):
        existing_logs = os.listdir(save_path)

    genED = GenerateLogs(INPUT_PARAMS)
    genED.run()
    output_logs = os.listdir(save_path)

    assert len(output_logs) > len(existing_logs)
    df = pd.read_csv(save_path + "_features.csv")
    assert df.shape == (8,5), f"Expected (8, 5) but got {df.shape}"
    assert df['log'].to_list() == EXPECTED_OUTPUT_NAMES

def test_duplicate_json_GenerateEventLogs():
    INPUT_PARAMS =  {   "pipeline_step": "event_logs_generation",
                        "output_path": "data/test/generation/duplicate_logs",
                        "generator_params": {
                            "config_space": { "mode": [ 5, 20], "sequence": [ 0.01, 1], "choice": [ 0.01, 1],
                                             "parallel": [ 0.01, 1], "loop": [ 0.01, 1], "silent": [ 0.01, 1],
                                             "lt_dependency": [ 0.01, 1], "num_traces": [ 10, 100],
                                             "duplicate": [ 0], "or": [ 0]},
                                             "n_trials": 50,
                            "experiment": [
                            { "n_traces": 99.0 },
                            { "n_traces": 998.4 },
                            { "ratio_variants_per_number_of_traces": 0.99 },
                            { "ratio_variants_per_number_of_traces": 0.5 },
                            ]}
                     }

    EXPECTED_OUTPUT_NAMES = ['genEL1_990_nan', 'genEL2_9984_nan',
                             'genEL4_nan_05', 'genEL3_nan_099']
    DUPLICATED_DATA  = {
    'log': ['genEL1_990_nan', 'genEL3_nan_099', 'genEL4_nan_05'],
    'target_similarity': [3.000000, 2.990099, 3.000000]}
    DUPLICATED_NAMES = DUPLICATED_DATA['log']
    df = pd.DataFrame(DUPLICATED_DATA)
    DUPL_LIST_OF_DICTS = df.to_dict(orient='records')

    mock_tasks = {k: v for d in INPUT_PARAMS["generator_params"]["experiment"] for k, v in d.items()}
    save_path = get_output_key_value_location(mock_tasks, INPUT_PARAMS["output_path"], "TEST")
    save_path = os.path.split(save_path)[0]
    feat_save_path = os.path.join(*Path(save_path).parts[:1],"features",*Path(save_path).parts[1:])

    delete_list_of_dirs([os.path.split(save_path)[0], #feat_save_path within os.path.split(save_path)[0]
                         ])
    clear_output()

    os.makedirs(save_path, exist_ok=True)
    os.makedirs(feat_save_path, exist_ok=True)

    for idx, dup in enumerate(DUPLICATED_NAMES):
        feat_path = os.path.join(feat_save_path, dup)
        with open(feat_path+".json", 'w') as f:
            json.dump(DUPL_LIST_OF_DICTS[idx], f, indent=4)
            pass

    genED = GenerateLogs(INPUT_PARAMS)
    genED.run()
    output_logs = os.listdir(save_path)

    assert len(output_logs) == len(EXPECTED_OUTPUT_NAMES)
    df = pd.read_csv(save_path + "_features.csv")
    assert df.shape == (4,4), f"Expected (4, 4) but got {df.shape}"
    assert set(df['log'].to_list()) == set(EXPECTED_OUTPUT_NAMES)
    assert not set(DUPLICATED_DATA['target_similarity']).issubset(set(df['target_similarity'].to_list()))

def test_duplicate_csv_GenerateEventLogs():
    INPUT_PARAMS =  {   "pipeline_step": "event_logs_generation",
                        "output_path": "data/test/generation/duplicate_logs",
                        "generator_params": {
                            "config_space": { "mode": [ 5, 20], "sequence": [ 0.01, 1], "choice": [ 0.01, 1],
                                             "parallel": [ 0.01, 1], "loop": [ 0.01, 1], "silent": [ 0.01, 1],
                                             "lt_dependency": [ 0.01, 1], "num_traces": [ 10, 100],
                                             "duplicate": [ 0], "or": [ 0]},
                                             "n_trials": 50,
                            "experiment": [
                            { "n_traces": 99.0 },
                            { "n_traces": 998.4 },
                            { "ratio_variants_per_number_of_traces": 0.99 },
                            { "ratio_variants_per_number_of_traces": 0.5 },
                            ]}
                     }

    EXPECTED_OUTPUT_NAMES = ['genEL1_990_nan', 'genEL2_9984_nan',
                             'genEL4_nan_05', 'genEL3_nan_099']
    DUPLICATED_DATA  = {'log': {0: 'genEL1_990_nan', 1: 'genEL2_9984_nan',
                                2: 'genEL3_nan_099', 3: 'genEL4_nan_05'},
                        'n_traces': {0: 99.0, 1: 100.0, 2: None, 3: None},
                        'ratio_variants_per_number_of_traces': {0: None, 1: None, 2: 1.0, 3: 0.42},
                        'target_similarity': {0: 3.0, 1: 4.0, 2: 7.99, 3: 6.92}}
    df = pd.DataFrame(DUPLICATED_DATA)

    mock_tasks = {k: v for d in INPUT_PARAMS["generator_params"]["experiment"] for k, v in d.items()}
    save_path = get_output_key_value_location(mock_tasks, INPUT_PARAMS["output_path"], "TEST")
    save_path = os.path.split(save_path)[0]
    feat_save_path = os.path.join(*Path(save_path).parts[:1],"features",*Path(save_path).parts[1:])

    delete_list_of_dirs([os.path.split(save_path)[0], #feat_save_path within os.path.split(save_path)[0]
                         ])
    clear_output()

    os.makedirs(save_path, exist_ok=True)
    os.makedirs(feat_save_path, exist_ok=True)

    df.to_csv(save_path + "_features.csv", index=False)

    genED = GenerateLogs(INPUT_PARAMS)
    genED.run()
    output_logs = os.listdir(save_path)

    assert len(output_logs) == len(EXPECTED_OUTPUT_NAMES)
    df = pd.read_csv(save_path + "_features.csv")
    assert df.shape == (4,4), f"Expected (4, 4) but got {df.shape}"
    assert set(df['log'].to_list()) == set(EXPECTED_OUTPUT_NAMES)
    assert not set(DUPLICATED_DATA['target_similarity'].values()).issubset(set(df['target_similarity'].to_list()))

def test_duplicate_EL_csv_all_GenerateEventLogs():
    INPUT_PARAMS =  {   "pipeline_step": "event_logs_generation",
                        "output_path": "data/test/generation/duplicate_logs",
                        "generator_params": {
                            "config_space": { "mode": [ 5, 20], "sequence": [ 0.01, 1], "choice": [ 0.01, 1],
                                             "parallel": [ 0.01, 1], "loop": [ 0.01, 1], "silent": [ 0.01, 1],
                                             "lt_dependency": [ 0.01, 1], "num_traces": [ 10, 100],
                                             "duplicate": [ 0], "or": [ 0]},
                                             "n_trials": 50,
                            "experiment": [
                            { "n_traces": 99.0 },
                            { "n_traces": 998.4 },
                            { "ratio_variants_per_number_of_traces": 0.99 },
                            { "ratio_variants_per_number_of_traces": 0.5 },
                            ]}
                     }

    EXPECTED_OUTPUT_NAMES = ['genEL1_990_nan', 'genEL2_9984_nan',
                             'genEL4_nan_05', 'genEL3_nan_099']
    DUPLICATED_DATA  = {'log': {0: 'genEL1_990_nan', 1: 'genEL2_9984_nan',
                                2: 'genEL3_nan_099', 3: 'genEL4_nan_05'},
                        'n_traces': {0: 99.0, 1: 100.0, 2: None, 3: None},
                        'ratio_variants_per_number_of_traces': {0: None, 1: None, 2: 1.0, 3: 0.42},
                        'target_similarity': {0: 3.0, 1: 4.0, 2: 70.99, 3: 6.92}}
    df = pd.DataFrame(DUPLICATED_DATA)
    DUPLICATED_NAMES = df['log'].to_list()

    mock_tasks = {k: v for d in INPUT_PARAMS["generator_params"]["experiment"] for k, v in d.items()}
    save_path = get_output_key_value_location(mock_tasks, INPUT_PARAMS["output_path"], "TEST")
    save_path = os.path.split(save_path)[0]
    feat_save_path = os.path.join(*Path(save_path).parts[:1],"features",*Path(save_path).parts[1:])

    delete_list_of_dirs([os.path.split(save_path)[0], #feat_save_path within os.path.split(save_path)[0]
                         ])
    clear_output()

    os.makedirs(save_path, exist_ok=True)
    os.makedirs(feat_save_path, exist_ok=True)

    df.to_csv(save_path + "_features.csv", index=False)

    for idx, dup in enumerate(DUPLICATED_NAMES):
        log_path = os.path.join(save_path, dup)
        feat_path = os.path.join(feat_save_path, dup)
        with open(log_path+".xes", 'w') as f:
            pass

    if os.path.exists(save_path):
        existing_logs = os.listdir(save_path)

    genED = GenerateLogs(INPUT_PARAMS)
    genED.run()
    output_logs = os.listdir(save_path)

    assert len(output_logs) == len(existing_logs)
    df = pd.read_csv(save_path + "_features.csv")
    assert df.shape == (4,4), f"Expected (4, 4) but got {df.shape}"
    assert set(df['log'].to_list()) == set(EXPECTED_OUTPUT_NAMES)
    assert set(df['target_similarity'].to_list())==set(DUPLICATED_DATA['target_similarity'].values())
