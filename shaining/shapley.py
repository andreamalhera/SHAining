import itertools
import numpy as np
import os
import pandas as pd
import random
import shap

from datetime import datetime as dt
from itertools import permutations, combinations
from math import factorial
from shaining.utils.describe_shapley_experiments import describe_shapley_experiments
from shaining.utils.param_keys import OUTPUT_PATH, INPUT_PATH
from shaining.utils.param_keys.coalition import FEATURE_NAMES
from shaining.utils.param_keys.benchmark import MINERS

from tabulate import tabulate

class ShapleyTask:
    def __init__(self, params=None):
        start = dt.now()
        print("=========================== ShapleyTask =============================")
        print(f"INFO: Running with {params}")

        self._parse_params(params)
        shapley_values_df = self.shapley_wrapper()
        self.results = shapley_values_df
        print(f"INFO: Saving Shapley values to {self.output_path}")

        shapley_values_df.to_csv(os.path.join(self.output_path, 'shapley_values.csv'), index=False)
        print(f"SUCCESS: SHApley task took {dt.now()-start} sec")
        print(f"Successfully saved the Shapley values to the {os.path.join(self.output_path, 'shapley_values.csv')}")
        print("========================= ~ ShapleyTask =============================")

    def _parse_params(self, params):
        self.input_path = params.get(INPUT_PATH)
        if not self.input_path:
            raise ValueError("Input path is required")
        self.output_path = params.get(OUTPUT_PATH)
        if not self.output_path:
            raise ValueError("Output path is required")
        self.feature_names = params.get(FEATURE_NAMES)
        self.level = 3 #TODO len(self.feature_names)
        if not self.feature_names:
            raise ValueError("Feature names are required for shapley value computation")
        self.miners = params.get(MINERS)
        if not self.miners:
            raise ValueError("At least one miner is required")
        return

    ## Helper functions

    def get_combinations(self, players: [str], length: int) -> list:
        """Get all combinations of a list of players of a given length"""
        return list(combinations(players, length))

    def get_permutations(self, players: [str], length: int) -> list:
        """Get all permutations of a list of players of a given length"""
        return list(permutations(players, length))

    def get_all_combinations(self, players: [str]) -> list:
        """Get a list of all combinations of all lengths of a list of players"""
        combinations = []
        for i in range(len(players)):
            combinations += self.get_combinations(players=players, length=i+1)
        return combinations

    def get_all_permutations(self, players:[str]) -> int:
        """Get all permutations that contain all players"""
        return self.get_permutations(players=players, length=len(players))

    def assign_random_value(self, combinations: list, value_range:int = 20) -> dict:
        """To each of the given player combinations assign a random value."""
        comb_dict = {}
        for comb in combinations:
            comb_dict[comb] = random.randint(value_range*len(comb)-value_range, value_range*len(comb))
        return comb_dict

    def create_player_list(self, no_players: int) -> list:
        """Create a list with consecutive letters/characters of given length"""
        players = []
        for i in range(no_players):
            players += chr(ord('A') + i)
        return list(players)

    def table(self, valued_sub_sets: dict, head: list) -> str:
        """Create a table for better representation"""
        data = []
        for key, value in valued_sub_sets.items():
            data.append([key, value])
        return tabulate(data, headers=head, tablefmt="grid")

    def is_matching_subset(self, sub_set1: list, sub_set2: list) -> bool:
        """Checks if two sub sets are matching"""
        return len(sub_set1) is len(sub_set2) is len(set(sub_set1).intersection(sub_set2))

    def get_sub_set_value(self, comb: dict, sub_set: list) -> int:
        """Get a sub sets value out of the given combination dictionary"""
        for c in comb:
            if self.is_matching_subset(c, sub_set):
                return comb[c]

    def generate_permutations(self, log):
        parts = log.split('_')
        permutations = []
        for i in range(1, len(parts)+1):
            for combo in itertools.combinations(range(len(parts)), i):
                perm = ['nan'] * len(parts)
                for idx in combo:
                    perm[idx] = parts[idx]
                permutations.append('_'.join(perm))
        return permutations

    def clean_log(self, entry):
        return '_'.join([part for part in entry.split('_') if part != 'nan'])


    def map_features(self, log_identifier):
        return tuple(log_identifier.split('_'))

    def extract_features(self,log_entry, feature_names):
        values = log_entry.split('_')
        # Match values to feature names, keeping the order
        features_in_log = [feature for value, feature in zip(values, feature_names) if value != 'nan']
        return features_in_log

    ## Shapely functions
    def shapley_sequence_definition(self, valued_sub_sets:dict, players: list) -> dict:
        """Calculate Shapley values by sequence definition"""
        shapley_dict = {players[i]: 0 for i in range(0, len(players))}
        permutations = self.get_all_permutations(players)

        for perm in permutations:
            perm_players = []
            perm_value = 0
            for player in perm:
                perm_players.append(player)
                sub_set_value = self.get_sub_set_value(valued_sub_sets, perm_players)
                shapley_dict[player] += sub_set_value - perm_value
                perm_value = sub_set_value
        return {player: value/factorial(len(players)) for player, value in shapley_dict.items()}


    def shapley_sub_set_definition(self, valued_sub_sets:dict, players: list) -> dict:
        """Calculate Shapley values by sub set definition"""
        shapley_dict = {players[i]: 0 for i in range(0, len(players))}   

        for player in players:
            # manual adding values for empty set
            freq = factorial(len(players) - 1)
            marg_contrib = self.get_sub_set_value(valued_sub_sets, (player,))
            shapley_dict[player] += freq*marg_contrib

            for sub_set in valued_sub_sets:
                if not player in sub_set:
                    freq = factorial(len(players) - 1 - len(sub_set)) * factorial(len(sub_set))
                    marg_contrib = self.get_sub_set_value(valued_sub_sets, sub_set + (player,)) - self.get_sub_set_value(valued_sub_sets, sub_set)
                    shapley_dict[player] += freq*marg_contrib
        return {player: value/factorial(len(players)) for player, value in shapley_dict.items()}

    #TODO: Remove prints and add shapley experiments description
    def shapley_wrapper(self):
        shapley_values_dict = {} # Stores the shapley values for each metric
        feature_names_values_dict = {} # Stores the feature names for each metric

        full_df = pd.read_csv(self.input_path)
        describe_shapley_experiments(full_df, self.feature_names)
        full_df = full_df[full_df.columns[full_df.columns.str.contains('|'.join(self.miners + ['log']))]] # filters the df if the column cointains the miner name
        metric_list = full_df.columns[1:] # extracts the metric names

        for metric_name in metric_list:
            #print(f'Calculating Shapley values for metric: {metric_name}')

            # Step 0: Read the file and do preprocessing
            data = full_df.copy()

            # Step 1: Preprocessing
            data.replace(-1,np.nan, inplace=True) # replace -1 with nan
            data.dropna(inplace=True) # remove columns with all nan values

            data['log_transformed'] = data['log'].apply(lambda x: x[x.find('_')+1:]) # remove the first part of the log so that only the features remain
            data = data.drop_duplicates(subset='log_transformed', keep='first').reset_index(drop=True) # remove duplicates, if any
            data.drop(columns = ["log_transformed"], inplace=True)
            data = data [['log',metric_name]]

            data['log'] = data['log'].str.replace(r'(genELtask)_(\d+)', r'\1\2', regex=True) # remove the number from the logm if any
            data['log'] = data['log'].apply(lambda x: x[x.find('_')+1:]) # remove the first part of the log so that only the features remain
            data = data.drop_duplicates(subset='log', keep='first').reset_index(drop=True) # remove duplicates, if any

            # # Step 2: Remove all the nan and its corresponding _ from the log

            data['features_in_log'] = data['log'].apply(lambda x: self.extract_features(x, self.feature_names)) # extracts features in the logs
            data['log'] = data['log'].str.replace(r'(_nan)', r'', regex=True).str.replace(r'(nan_)', r'', regex=True)

            # # Step 2.5: Map the features to integers

            ### Description: Each unique combination of a feature name and value is mapped to a unique integer. This is done to avoid the wrong shapley value calculation that occurs from same feature value being represented by different feature names.
            data['map_features'] = data['log'].apply(lambda x: str.split(x, '_'))
            data['map_features'] = data.apply(lambda row: [f"{x}_{y}" for x,y in zip(row['map_features'], row['features_in_log'])], axis=1)

            unique_keys = set(item for sublist in data['map_features'] for item in sublist)
            key_to_value = {key: idx for idx, key in enumerate(unique_keys, start=1)}
            data['map_features'] = data['map_features'].apply(lambda lst: {key: key_to_value[key] for key in lst})
            data["log"] = data["map_features"].apply(lambda x: '_'.join([str(i) for i in x.values()]))

            # # Step 3: Identifying all logs with features = level

            data_filtered = data[data['log'].str.count('_') == self.level-1]
            if data_filtered.empty:
                #print(f"ERROR: Lattice data insufficient for shapley experiments for {metric_name}")
                raise ValueError(f"Lattice data insufficient for shapley experiments for {metric_name}. Try rerunning BenchmarkingTask with longer timeout params or larger memory limitations. To get other results with those limitations, please delete the benchmark directories containing jsons beforehand.")
                return data_filtered

            data_filtered = data_filtered[~data_filtered['log'].str.contains('nan')].reset_index(drop=True)

            # Step 4: Find all permutations while maintaining order

            # Applying the permutation generation and storing results
            data_filtered['permutations'] = data_filtered['log'].apply(self.generate_permutations)
            data_filtered['permutations'] = data_filtered['permutations'].apply(lambda x: [self.clean_log(entry) for entry in x])

            # Step 5: find all the permutations of the eligible logs and calculate the shapley values

            shap_values_list = [] #Stores shapley values for each metric
            feature_values_list = [] #Stores the data values (feature values) for each metric
            feature_names_value_list = []

            #print(f"    INFO: {metric_name} has {len(data_filtered)} experiments with {self.level} features")
            for idx in range(len(data_filtered)):
                features_names = data_filtered.loc[idx, 'permutations'][:3]
                feature_values = [ {v:k for k,v in data_filtered.loc[idx, 'map_features'].items()}[int(map_id)].split('_')[0] for map_id in features_names]
                features_metric_data = data[data['log'].isin(data_filtered.iloc[idx]['permutations'])].reset_index(drop=True)

                # 2**self.level - 1 is the number of possible combinations of features for a certain level
                if len(features_metric_data) < (2**self.level - 1):
                    #print(f"Skipping due to lack of data for {features_names}")
                    continue

                # # Create the dictionary with feature tuples 
                features_metric_data_dict = {self.map_features(row['log']): row[metric_name] for index, row in features_metric_data.iterrows()}

                features_metric_data = features_metric_data.sort_values(by='features_in_log', key=lambda x: x.str.len(), ascending=False).reset_index(drop=True)
                feature_names_list = features_metric_data['features_in_log'][0]
                #print("Features:", feature_names_list,":", features_names)
                # calculate shap values
                #print(features_metric_data_dict)
                shap_values = self.shapley_sequence_definition(valued_sub_sets=features_metric_data_dict, players=features_names)
                #print(self.table(shap_values, ['feature', 'metric']))
                #print('---')

                shap_values_list.append(list(shap_values.values()))
                feature_values_list.append(feature_values)
                feature_names_value_list.append(feature_names_list)
                # break
            shapley_values = pd.DataFrame(
                    [(f, v, d) for features, values, feature_values in zip(feature_names_value_list, shap_values_list, feature_values_list) 
                            for f, v, d in zip(features, values, feature_values)], 
                    columns=["feature", "shap_value", "feature_value"]
                )
            shapley_values_dict[metric_name] = shapley_values
            feature_names_values_dict[metric_name] = feature_names_value_list

        shapley_values_df = pd.DataFrame(
                                [(metric, row["feature"], row["shap_value"], row["feature_value"])
                                for metric, sub_df in shapley_values_dict.items()
                                for _, row in sub_df.iterrows()],
                                columns=["metric", "feature_name", "shapley_value", "feature_value"]
                            )
        shapley_values_df[['metric', 'algorithm']] = shapley_values_df['metric'].str.split('_', n=1, expand=True)
        shapley_values_df = shapley_values_df[["algorithm", "metric", "feature_name", "feature_value", "shapley_value"]]
        return shapley_values_df
