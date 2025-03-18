import pandas as pd
import pytest
from shaining import GenerateLogs
from shaining.utils.io_helpers import get_output_key_value_location, delete_directory_contents
from shaining.utils.io_helpers import delete_list_of_dirs
import os
import re
import shutil

def test_naming_scheme_GenerateEventLogs():
    """
    Checks if the experiments are named correctly

    Description:    The aim of the test is to check if the experiments are named correctly.
                    Many logs with increasing number of features are generated to cover all
                    the possible combinations of the features.
                    The naming scheme is checked to ensure that the logs are named correctly.
    """
    delete_list_of_dirs([os.path.join('data','test', 'generation', 'naming_scheme_logs'),
                    os.path.join('data', 'generated', '4_aq1_eav_ekbr3_nusa'),
                    os.path.join('data', 'features', 'generated', '4_aq1_eav_ekbr3_nusa'),
                    ])
    INPUT_PARAMS =  {   "pipeline_step": "event_logs_generation",
                        "output_path": "data/test/generation/naming_scheme_logs",
                        "generator_params": { "similarity_threshold": 0.0,
                            "config_space": { "mode": [ 5, 20], "sequence": [ 0.01, 1], "choice": [ 0.01, 1],
                                             "parallel": [ 0.01, 1], "loop": [ 0.01, 1], "silent": [ 0.01, 1],
                                             "lt_dependency": [ 0.01, 1], "num_traces": [ 10, 100],
                                             "duplicate": [ 0], "or": [ 0]},
                                             "n_trials": 50,
                            "experiment": [
                                        {  "activities_q1": 4.0 },

                                        { "activities_q1": 4.0, "end_activities_variance": 0.0 },

                                        { "end_activities_variance": 0.0, "eventropy_k_block_ratio_3": 4.37, "n_unique_start_activities": 1.0 },

                                        { "activities_q1": 4.0, "end_activities_variance": 0.0, "eventropy_k_block_ratio_3": 4.37, "n_unique_start_activities": 1.0 }

                            ]}
                     }

    LOG_NAMES = ['genEL2_40_00_nan_nan.xes', 'genEL3_nan_00_437_10.xes',
                 'genEL4_40_00_437_10.xes', 'genEL1_40_nan_nan_nan.xes']
    save_path = get_output_key_value_location(INPUT_PARAMS["generator_params"]["experiment"][-1], INPUT_PARAMS["output_path"], "TEST")
    save_path = os.path.split(save_path)[0]

    genED = GenerateLogs(INPUT_PARAMS)
    genED.run()

    output_logs = os.listdir(save_path)

    assert len(output_logs) == len(LOG_NAMES)
    assert sorted(output_logs) == sorted(LOG_NAMES)
