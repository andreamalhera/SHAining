import itertools
import json
import os
import numpy as np
import re

from pathlib import Path
from shaining.utils.param_keys.coalition import LOG_NAMES, LEVEL, FEATURE_NAMES, EXPERIMENT_LIST

class Coalitions:
    def __init__(self, params = None):
        self._parse_params(params)
        print(f"=========================== CoalitionTask {self.level-1}-->{self.level} =======================")
        file_path = Path("data/level3_targets_3x2x3_Feb27.json")
        self.log_names = {log_name for log_name in self.log_names if log_name.startswith("genEL")}
        if self.level == 3 and file_path.exists():
            with open(file_path, "r") as file:
                self.next_level_targets = json.load(file)
                print(f"INFO: Loaded {len(self.next_level_targets)} targets from {file_path}")
                return 

        new_params = self.coalition_wrapper()

        new_params = [dict(t) for t in {frozenset(d.items()) for d in new_params}]
        print(f"INFO: {len(new_params)} possible coalitions for next level {self.level}.")
        self.next_level_targets = new_params
        print(f"========================= ~ CoalitionTask {self.level-1}-->{self.level} =======================")

    def _parse_params(self, params):
        self.log_names = params[LOG_NAMES]
        if self.log_names is None:
            raise ValueError(f'The log names are missing from the input parameters.')
        self.level = params[LEVEL]
        if self.level is None:
            raise ValueError(f'The level to be pruned is missing from the input parameters.')
        self.feature_names = params[FEATURE_NAMES]
        if self.feature_names is None:
            raise ValueError(f'The feature names are missing from the input parameters.')
        self.experiment_list = params[EXPERIMENT_LIST]
        if self.experiment_list is None:
            raise ValueError(f'The experiment list is missing from the input parameters.')

    def generate_next_level_logs(self, log_names, current_level):
        # Extract feature values from log names
        feature_values = []
        for log in log_names:
            # Split the log name into parts
            parts = log.split('_')
            # Extract the feature values (ignore the first part which is the log name)
            features = parts[1:]
            feature_values.append(features)
        
        print(f"INFO: {len(self.log_names)}/{len(self.experiment_list)} ELs from layer {self.level-1}"
             +f" will be used to form coalitions on layer {self.level}. Continuing.")
        # Determine the number of features
        num_features = len(feature_values[0])
        
        # Generate all combinations of logs to fill in the missing features
        next_level_logs = set()  # Use a set to avoid duplicates
        

        # Iterate over all possible pairs (or more, depending on the level) of logs
        for combination in itertools.combinations(feature_values, current_level + 1):
            # Initialize a list to hold the new feature values
            new_features = ['nan'] * num_features
            
            # Fill in the new_features list by taking non-nan values from the combination
            for features in combination:
                for i in range(num_features):
                    if features[i] != 'nan':
                        new_features[i] = features[i]
            
            # Check if the new_features list has exactly (current_level + 1) non-nan values
            if new_features.count('nan') == num_features - (current_level + 1):
                # Generate the new log name
                log_name = f"genEL{len(next_level_logs) + 1}_{'_'.join(new_features)}"
                next_level_logs.add(log_name)
        
        return sorted(next_level_logs)
    
    def map_feature_names_to_values(self, log_name, feature_names, unique_pairs_dictionary):
        
        # Split the log name into parts
        parts = log_name.split('_')
        
        # Extract the feature values (ignore the first part, which is the log name)
        feature_values = parts[1:]
        
        # Create a dictionary mapping feature names to their values
        feature_dict = {}
        for feature_name, feature_value in zip(feature_names, feature_values):
            if feature_value != 'nan':  # Only include non-nan values
                # feature_dict[feature_name] = np.double(feature_value)
                key = "_".join([str(feature_name), str(feature_value)])
                if key not in unique_pairs_dictionary:
                    print(f"ERROR: {key} not found in unique_pairs_dictionary")
                    print(f"Available keys: {list(unique_pairs_dictionary.keys())}")
                    raise KeyError(f"The feature : value combination {key} not found in the level one targets given to the class.")
                feature_dict[feature_name] = unique_pairs_dictionary[key]
        
        return feature_dict
    
    def coalition_wrapper(self):
        self.log_names = list(self.log_names)
        #determine the current level of the logs
        current_level = max(
                    sum(
                        part.lower() != 'nan'
                        for part in re.sub(r'^genEL\d+_', '', os.path.basename(file))
                                    .split('.')[0]
                                    .split('_')
                    )
                    for file in self.log_names
                )
        
        if current_level != self.level - 1: 
            print(f'Not sufficient logs yielded results while benchmarking, causing a lvl missmatch {current_level}!={self.level}.')
            return []
            raise ValueError(f'The current level of the logs is {current_level}, but the level to be pruned is {self.level}.')
        
        
        # Preprocessing
        unique_log_names = list({re.sub(r'^genEL\d+_', '', x).split('.')[0]: x for x in self.log_names}.values())
        
        # next level logs
        next_level_logs = self.generate_next_level_logs(unique_log_names, current_level)
        
        ## Generate the experiment list for next level
        ### Generate the unique key value pairs from self.config

        unique_pairs_dictionary = {
                                        "_".join([str(k), str(v).replace('.', '') ]): v
                                        for d in self.experiment_list
                                        for item in [d] 
                                        for _ in d 
                                        for k, v in [tuple(item.items())[0]]
                                    }
        experiment = [self.map_feature_names_to_values(log,self.feature_names,unique_pairs_dictionary) for log in next_level_logs]
        
        return experiment

        
