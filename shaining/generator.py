import itertools
import json
import gc
import numpy as np
import os
import pandas as pd
import random
import re
import shutil
import subprocess
from gedi import GenerateEventLogs
from gedi.utils.column_mappings import column_mappings
import math
from itertools import combinations, product
from pathlib import Path
from shaining.pruning import prune_ELnames_per_level
from shaining.utils.compute_similarity_posteriously import compute_feats_posteriously
from shaining.utils.io_helpers import get_output_key_value_location
from shaining.utils.merge_jsons import json_to_csv
from shaining.utils.param_keys import OUTPUT_PATH, INPUT_PATH
from shaining.utils.param_keys.generator import GENERATOR_PARAMS, EXPERIMENT, N_TRIALS
from shaining.utils.param_keys.generator import SIMILARITY_THRESHOLD, CONFIG_SPACE, LOG
from shaining.utils.param_keys.generator import FEATURE_KEYS

#TODO: Test timer issue
random.seed(42)
"""
    def clear(self):
        self.config_space.clear()
        self.existing_files.clear()
        self.experiment_groups.clear()
        self.tasks.clear()
        self.feature_keys.clear()
        self.generated_final_file_paths.clear()
        self.generated_possible_file_paths.clear()
        del self.destination_features
        del self.max_non_nan
        del self.num_trials
        del self.output_path
        del self.similarity_threshold
        gc.collect()
"""

def get_keys_abbreviation(obj_keys):
    abbreviated_keys = []
    for obj_key in obj_keys:
        key_slices = obj_key.split("_")
        chars = []
        for key_slice in key_slices:
            for idx, single_char in enumerate(key_slice):
                if idx == 0 or single_char.isdigit():
                    chars.append(single_char)
        abbreviated_key = ''.join(chars)
        abbreviated_keys.append(abbreviated_key)
    return '_'.join(abbreviated_keys)

def get_output_key_value_location(obj, output_path, identifier, obj_keys=None):
    obj_sorted = dict(sorted(obj.items()))
    if obj_keys is None:
        obj_keys = [*obj_sorted.keys()]

    obj_values = [round(x, 4) for x in [*obj_sorted.values()]]

    if len(obj_keys) > 9:
        folder_path = os.path.join(output_path, f"{len(obj_keys)}_features")
        generated_file_name = f"{identifier}"
    else:
        folder_path = os.path.join(output_path, f"{len(obj_keys)}_{get_keys_abbreviation(obj_keys)}")
        obj_values_joined = '_'.join(map(str, obj_values)).replace('.', '')
        generated_file_name = f"{identifier}_{obj_values_joined}"


    os.makedirs(folder_path, exist_ok=True)
    save_path = os.path.join(folder_path, generated_file_name)
    return save_path

def get_expected_log_names(targets, output_path, feature_keys=None):
    log_names = []
    #If log in output_path and features in feature_path remove from targets
    if targets is not None:
        tasks = pd.DataFrame.from_dict(data=targets)
        columns_to_rename = {col: column_mappings()[col] for col in tasks.columns if col in column_mappings()}
        tasks = tasks.rename(columns=columns_to_rename)

    experiment_keys = {k: v for d in targets for k, v in d.items()}.keys()
    experiment_keys = feature_keys if feature_keys is not None else list(experiment_keys)

    for feature in experiment_keys:
        if feature not in tasks.columns.tolist():
            tasks[feature] = np.nan
    if tasks is not None:
        feature_keys = sorted([feature for feature in tasks.columns.tolist() if feature != "log"])
    for task in [(index, row) for index, row in tasks.iterrows()]:
        try:
            identifier = [x for x in task[1] if isinstance(x, str)][0]
        except IndexError:
            identifier = "genEL"+str(task[0]+1)
        task = task[1].drop('log', errors='ignore')

        save_path = get_output_key_value_location(task.to_dict(),
                                        output_path, identifier, feature_keys)+".xes"

        # Extract feature set (the part after the first underscore)
        feature_set = "_".join(os.path.basename(save_path).split("_")[1:])
        log_names.append(feature_set)
    return log_names

class GenerateLogs:
    def __init__(self, params):
        self._parse_params(params)

    def _parse_params(self, params):
        self.output_path = params.get(OUTPUT_PATH)
        self.tasks = params.get(GENERATOR_PARAMS).get(EXPERIMENT)

        self.feature_keys = params.get(GENERATOR_PARAMS).get(FEATURE_KEYS)
        if self.feature_keys is None:
            feature_keys = {k: v for d in self.tasks for k, v in d.items()}
            self.feature_keys = list(feature_keys.keys())

        log_names = get_expected_log_names(self.tasks, self.output_path, self.feature_keys)
        self.tasks = {log_name:task for task, log_name in zip(self.tasks, log_names)}

        self.config_space = params.get(GENERATOR_PARAMS).get(CONFIG_SPACE)
        self.num_trials = params.get(GENERATOR_PARAMS).get(N_TRIALS)
        
        level_set = set([len(task) for task in self.tasks.values()])
        self.level = next(iter(level_set)) if len(level_set)==1 else 'Mixed' 
        return

    def retrieve_genEL_output_paths(self):
        mock_target = {obj_key: obj_value for obj_value, obj_key in enumerate(self.feature_keys)}
        written_path = os.path.split(get_output_key_value_location(mock_target, self.output_path, "TEST"))[0]
        features_path = os.path.join(*Path(written_path).parts[:1],'features',* Path(written_path).parts[1:])
        similarities_path = written_path+'_features.csv'
        #similarities_path = os.path.join(*(Path(written_path).parts[:1] + ('features',) + Path(written_path).parts[1:]))+'_features.csv'

        self.original_output_path = written_path
        return written_path, features_path, similarities_path

    def _still_needed(self, log_name, existing_logs, json_files, similarities):
        if log_name in existing_logs and log_name.split('.xes',1)[0] in similarities:
            print(f"    SUCCESS: {log_name} and its target similarities (in csv) already exist"
            +f" in {self.output_path}. Skipping.")
            return False
        elif log_name in existing_logs and log_name.replace('.xes', '.json') in json_files:
            print(f"    SUCCESS: {log_name} and its target similarities (in json) already exist"
            +f" in {self.output_path}. Skipping.")
            return False
        elif log_name in existing_logs:
            self.pending_features.append(log_name)
            print(f"    WARNING: {log_name} already exists in {self.output_path}."
                  +"Features will be computed.")
            return False
        return True

    def prune_existing_logs(self):
        existing_logs = []
        json_files = []
        featured_logs = []

        log_names = list(self.tasks.keys())
        logs_path, features_json_path, similarities_path = self.retrieve_genEL_output_paths()

        if logs_path is not None and os.path.exists(logs_path):
            existing_files = [os.path.join(root, file) for root, _, files in os.walk(logs_path) for file in files]
            existing_logs = [os.path.basename(filename).split('_',1)[1] for filename in existing_files if filename.endswith('.xes')]
            #existing_logs = [os.path.basename(filename) for filename in existing_files if filename.endswith('.xes')]

        if features_json_path is not None and os.path.exists(features_json_path):
            feature_files = [os.path.join(root, file) for root, _, files in os.walk(features_json_path) for file in files]
            json_files = [os.path.basename(filename).split('_',1)[1] for filename in feature_files if filename.endswith('.json')]

        if similarities_path is not None and os.path.exists(similarities_path):
            featured_logs = pd.read_csv(similarities_path)[LOG].tolist()
            featured_logs = [log.split('_',1)[1] for log in featured_logs]

        self.pending_features = []
        if len(existing_logs) > 0:
            new_tasks = {task: value for task,
                         value in self.tasks.items() if self._still_needed(task,
                                                                           existing_logs,
                                                                           json_files,
                                                                           featured_logs)}
            self.all_tasks = self.tasks
            self.tasks = new_tasks
        return

    def run(self):
        print(f"=========================== GenerateLogs {self.level} ===========================")
        self.prune_existing_logs()

        if len(self.tasks) > 0:
            print(f"INFO: Generating {len(self.tasks)} logs with targets:", self.tasks)
            new_params = { GENERATOR_PARAMS: {
                    EXPERIMENT: list(self.tasks.values()),
                    CONFIG_SPACE: self.config_space,
                    N_TRIALS: self.num_trials},
                    OUTPUT_PATH: self.output_path
                    }
            # Generate logs using the gedi function
            lvl_ged = GenerateEventLogs(new_params)
            ged_output_path = lvl_ged.output_path
            ged_features = [config['metafeatures'] for config in lvl_ged.generated_features]
            lvl_ged.clear()

            exp_all_keys = {k: v for d in list(self.tasks.values()) for k, v in d.items()}
            written_path = os.path.split(get_output_key_value_location(exp_all_keys, ged_output_path, "TEST"))[0]
        else:
            print("INFO: No logs to generate. Using existing logs.")
            written_path = self.original_output_path

        if len(self.pending_features) > 0:
            compute_feats_posteriously(self.original_output_path, list(self.all_tasks.values()))
            print(f"INFO: Computing missing {len(self.pending_features)} features of existing logs: ", self.pending_features)

        feeed_output_dir = os.path.join(*(Path(written_path).parts[:1] + ('features',) + Path(written_path).parts[1:]))

        if os.path.exists(feeed_output_dir) and len(os.listdir(feeed_output_dir)) > 0:
            json_to_csv(feeed_output_dir, written_path+'_features.csv')

        print(f"========================= ~ GenerateLogs {self.level} ===========================")
        # TODO: See if SHAiningTask breaks if uncomment next line
        #self.clear()
        return written_path+'_features.csv'
