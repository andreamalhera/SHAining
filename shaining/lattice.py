import os
import pandas as pd

from collections import Counter
from datetime import datetime as dt
from itertools import combinations
from math import comb
from shaining.benchmark import BenchmarkTest
from shaining.coalition import Coalitions
from shaining.generator import GenerateLogs
from shaining.shapley import ShapleyTask
from shaining.pruning import prune_ELnames_per_level, prune_per_benchmarking
from shaining.utils.param_keys import OUTPUT_PATH, TARGET_SIMILARITY, SIMILARITY_THRESHOLD
from shaining.utils.param_keys import FEAT_PATH, CURRENT_LEVEL, INPUT_PATH, EVENTLOGS_SELECTION
from shaining.utils.param_keys.benchmark import MINERS
from shaining.utils.param_keys.coalition import LOG_NAMES, LEVEL, FEATURE_NAMES, EXPERIMENT_LIST
from shaining.utils.param_keys.generator import GENERATOR_PARAMS, EXPERIMENT
from shaining.utils.param_keys.generator import CONFIG_SPACE, N_TRIALS, LOG
from shaining.utils.param_keys.generator import FEATURE_KEYS

def generate_feature_combinations(feature_names, values, layer=1):
    # Create a list of (feature, value) pairs
    feature_value_pairs = []
    for feature, values in zip(feature_names, values):
        feature_value_pairs.extend([{feature: value} for value in values])

    # Generate unique combinations of these pairs up to the specified layer
    all_combinations = []
    for combo in combinations(feature_value_pairs, layer):
        # Merge dictionaries only if no duplicate features exist
        merged_dict = {}
        valid_combo = True
        for item in combo:
            feature = next(iter(item))
            if feature in merged_dict:
                valid_combo = False
                break
            merged_dict.update(item)
        if valid_combo:
            all_combinations.append(merged_dict)
    return all_combinations

class Lattice():
    def __init__(self, feature_names: list, values: list, system_params: dict) -> None:
        if feature_names is None or values is None:
            raise ValueError("feature_names and values must not be None")
        if len(feature_names) != len(values):
            raise ValueError("feature_names and values must have the same length")
        if system_params is None:
            raise ValueError("system_params must not be None")

        self.feature_names = feature_names
        self.values = values
        self.system_params = system_params

    def describe_potential(self):
        feature_names = self.feature_names
        values = self.values
        checked = True
        total_logs = 0
        potential = {}
        for layer in range(1, len(feature_names)+1):
            num_logs_to_generate = comb(len(feature_names), layer) * (len(values[0]) ** layer)
            total_logs += num_logs_to_generate
            potential['layer_'+str(layer)] = num_logs_to_generate
            print("For layer", layer, "with", len(feature_names), "features and",
                  len(values[0]), "values per feature we expect", num_logs_to_generate, "possible logs.")
        potential['total'] = total_logs
        potential = dict(sorted(potential.items(),
                                key=lambda x: (x[0] != 'total', -x[0] if isinstance(x[0], int) else float('-inf'))))

        print("Total number of possible logs:", total_logs)
        return potential

    # TODO: If target_similarities and benchmark_results are found in lattice path csv skip lattice.form. Sepparate benchmark_results and lattice path
    def form(self, contestants):
        print(f"================================ Lattice ================================")
        start = dt.now()
        #Creates all coalitions/targets for initial layer
        targets = generate_feature_combinations(self.feature_names, self.values, 1)

        max_level = len(self.feature_names)
        layer = Layer(system_params = self.system_params , layer_one_targets = targets)
        highest_layer_reached = False
        self.targets = targets

        for level in range(1, max_level+1):
            if not highest_layer_reached:
                next_lvl_coalitions = layer.form(self.targets, level, contestants, self.feature_names)
                #TODO Last layer check should be function of lattice?
                highest_layer_reached = layer._reached_highest_level(next_lvl_coalitions)
                print(f"INFO: Created layer of layer {level}/{max_level} with {len(self.targets)} targets.")
                self.targets = next_lvl_coalitions
        self.reached_level = level

        print(f"SUCCESS: Level {self.reached_level} lattice with {len(layer.targets)} ELs was saved in {layer.output_path} "
        +f"took {dt.now()-start} sec.")
        print(f"============================== ~ Lattice ================================")

        return layer.output_path


class Layer():
    def __init__(self, system_params, layer_one_targets) -> None:
        self.targets = []
        self.system_params = system_params
        self.benchmark_results = pd.DataFrame()
        self.layer_one_targets = layer_one_targets
        #self.coalition_layer = 1
        return
    
    def retrieve_genEL_similarities_path(self):
        try:
            expected_path = self.system_params.get(GENERATOR_PARAMS, {}).get(OUTPUT_PATH, "")
            existing_files = [os.path.join(root, file) for root, _, files in os.walk(expected_path) for file in files]
            path_to_csv = next((filename for filename in existing_files if filename.endswith(".csv")), None)
            return path_to_csv
        except Exception as e:
            print(f"Error retrieving path: {e}")
            return None

    # TODO: If target_similarities and benchmark_results are found in lattice path csv skip lattice.form. Sepparate benchmark_results and lattice path
    def form(self, targets, layer, contestants, feature_names):
        print(f"================================ Layer {layer} ================================")
        #path_to_genEL_similarities = self.retrieve_genEL_similarities_path() or self.generator(targets)
        path_to_genEL_similarities = self.generator(targets, feature_names)
        self.targets = self.targets + targets

        benchmark_results, benchmark_output_path = self.benchmark(path_to_genEL_similarities, contestants, layer)
        #pruned_lvl_names = prune_ELnames_per_level(benchmark_results[LOG].to_list(), layer)
        #benchmark_results = benchmark_results[benchmark_results[LOG].isin(pruned_lvl_names)]
        #benchmark_results = prune_per_benchmarking(benchmark_results, contestants)
        self.benchmark_results = pd.concat([self.benchmark_results, benchmark_results], ignore_index=True)
        self.output_path = benchmark_output_path

        next_lvl_coalitions = self.form_coalitions(self.benchmark_results, layer+1, feature_names, self.layer_one_targets)

        all_keys = sorted({k for dic in next_lvl_coalitions for k in dic})
        next_lvl_coalitions = sorted(next_lvl_coalitions, key=lambda d: tuple(d.get(k, float('inf')) for k in all_keys))

        print(f"=============================== ~ Layer {layer} ===============================")
        return next_lvl_coalitions

    def _reached_highest_level(self, next_lvl_coalitions):
        pair_counter = Counter()
        for entry in next_lvl_coalitions:
            keys = list(entry.keys())
            for pair in combinations(keys, 2):
                pair_counter[pair] += 1
        value_counter = {}
        for entry in next_lvl_coalitions:
            for key, value in entry.items():
                if key not in value_counter:
                    value_counter[key] = Counter()
                value_counter[key][value] += 1
        value_counter = {key: len(val) for key, val in value_counter.items()}

        if len(pair_counter) == 0:
            print(f"WARNING: No pairs in the next layer. Will stop and store lattice now.")
            return True
        elif any(value < 2 for value in value_counter.values()):
            print("WARNING: Not all values have at least two entries in the next layer. Will stop and store lattice now.")
            return True
        else:
            return False

    def generator(self, targets, feature_keys):
        config_space = self.system_params.get(GENERATOR_PARAMS).get(CONFIG_SPACE)
        num_trials = self.system_params.get(GENERATOR_PARAMS).get(N_TRIALS)

        #path_to_similarities = GenerateLogs({GENERATOR_PARAMS: {CONFIG_SPACE: config_space,
        generator_task = GenerateLogs({GENERATOR_PARAMS: {CONFIG_SPACE: config_space,
                                                N_TRIALS: num_trials,
                                                EXPERIMENT: targets,
                                                FEATURE_KEYS: feature_keys},
                                            OUTPUT_PATH: self.system_params.get(GENERATOR_PARAMS).get(OUTPUT_PATH)
                                            })
        path_to_similarities = generator_task.run()
        return path_to_similarities

    def benchmark(self, path_to_genEL_similarities, contestants, layer):
        input_path = path_to_genEL_similarities.rsplit("_",1)[0]
        eventlogs_selection = {FEAT_PATH: path_to_genEL_similarities,
                               SIMILARITY_THRESHOLD: self.system_params.get(SIMILARITY_THRESHOLD),
                               CURRENT_LEVEL: layer}
        output_path = self.system_params.get(OUTPUT_PATH)

        benchpath_suffix = "benchmark_"+contestants[0] if len(contestants)==1 else "benchmark"
        benchmark_output_path = os.path.join(output_path, os.path.split(input_path)[-1].replace(".xes",""), benchpath_suffix) +'.csv'

        benchmark_results = BenchmarkTest({INPUT_PATH: input_path,
                                   EVENTLOGS_SELECTION: eventlogs_selection,
                                   OUTPUT_PATH: output_path,
                                   MINERS: contestants})

        benchmark_output_path = benchmark_results.filepath
        benchmark_results = benchmark_results.results
        pruned_lvl_names = prune_ELnames_per_level(benchmark_results[LOG].to_list(), layer)
        benchmark_results = benchmark_results[benchmark_results[LOG].isin(pruned_lvl_names)]
        benchmark_results = prune_per_benchmarking(benchmark_results, contestants)
        return benchmark_results, benchmark_output_path

    # TODO: Adds test
    def form_coalitions(self, benchmarked, layer, feature_names, layer_one_targets):
        log_names = set(benchmarked[LOG].to_list())
        coalitions = Coalitions({LOG_NAMES: log_names,
                                 LEVEL: layer,
                                 FEATURE_NAMES: feature_names,
                                 EXPERIMENT_LIST: layer_one_targets
                                 }).next_level_targets
        return coalitions
