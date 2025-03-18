import os
import pandas as pd

from pathlib import Path
from shaining.lattice import Layer, Lattice
from shaining.utils.io_helpers import delete_list_of_dirs

def clean_output():
    delete_list_of_dirs([os.path.join("data", "generated", "3_nt_rvpnot_tlm"),
                         os.path.join("data", "features", "generated", "3_nt_rvpnot_tlm"),
                         os.path.join("output", "layer_test", "3_nt_rvpnot_tlm"),
                         os.path.join("output", "3_nt_rvpnot_tlm"),])
def features_to_disk(features, test_type = "layer"):
    df = pd.DataFrame(features)
    output_path = os.path.join("output", test_type+"_test", "3_nt_rvpnot_tlm_features.csv")
    os.makedirs(os.path.join(*Path(output_path).parts)[:-1], exist_ok=True)
    df.to_csv(output_path, index=False)

def benchmark_to_disk(benchmark_results, miner="sm1"):
    df = pd.DataFrame(benchmark_results)
    miner_suffix = "_"+miner if type(miner) == str else ""
    output_path = os.path.join("output", "3_nt_rvpnot_tlm","benchmark"+miner_suffix+".csv")
    os.makedirs(os.path.join(*Path(output_path).parts)[:-1], exist_ok=True)
    df.to_csv(output_path, index=False)

def test_Layer():
    FEATURES_FOR_DISK = {'log': {0: 'genEL1_990_nan_nan', 1: 'genEL2_9984_nan_nan', 2: 'genEL3_nan_099_nan',
                        3: 'genEL4_nan_119_nan', 4: 'genEL5_nan_nan_10', 5: 'genEL6_nan_nan_12'},
                'n_traces': {0: 99.0, 1: 100.0, 2: None, 3: None, 4: None, 5: None},
                'ratio_variants_per_number_of_traces': {0: None, 1: None, 2: 1.0, 3: 1.0, 4: None, 5: None},
                'target_similarity': {0: 1.0, 1: 0.9964279466259522, 2: 0.99, 3: 0.81,
                                      4: 1.0, 5: 0.991304347826087},
                'trace_len_min': {0: None, 1: None, 2: None, 3: None, 4: 1.0, 5: 1.0}}

    BENCHMARK_RESULTS_FOR_DISK = {'log': {0: 'genEL3_nan_099_nan', 1: 'genEL4_nan_119_nan', 2: 'genEL5_nan_nan_10',
                                 3: 'genEL2_9984_nan_nan', 4: 'genEL6_nan_nan_12', 5: 'genEL1_990_nan_nan'},
                         'fitness_sm1': {0: 1.0, 1: 1.0, 2: 0.78, 3: 1.0, 4: 0.78, 5: 0.89},
                         'precision_sm1': {0: 0.47, 1: 0.47, 2: 0.92, 3: 1.0, 4: 0.92, 5: 0.58},
                         'fscore_sm1': {0: 0.64, 1: 0.64, 2: 0.85, 3: 1.0, 4: 0.85, 5: 0.71},
                         'size_sm1': {0: 36, 1: 36, 2: 31, 3: 11, 4: 31, 5: 33},
                         'cfc_sm1': {0: 13, 1: 13, 2: 12, 3: 4, 4: 12, 5: 15},
                         'pnsize_sm1': {0: 25, 1: 25, 2: 18, 3: 8, 4: 18, 5: 17},
                         'exectime_sm1': {0: 6.74, 1: 6.76, 2: 5.34, 3: 5.33, 4: 5.36, 5: 5.37},
                         'benchtime_sm1': {0: 130.25, 1: 131.46, 2: 5.49, 3: 5.36, 4: 5.52, 5: 8.86}}


    TARGETS = [{'n_traces': 99.0}, {'n_traces': 998.4},
               {'ratio_variants_per_number_of_traces': 0.99},
               {'ratio_variants_per_number_of_traces': 1.19},
               {'trace_len_min': 1.0}, {'trace_len_min': 1.2}]
    LEVEL = 1
    CONTESTANTS = ['sm1']
    FEATURE_NAMES = ['n_traces', 'ratio_variants_per_number_of_traces', 'trace_len_min']
    SYSTEM_PARAMS = {"generator_params": {
                        "config_space": { "mode": [ 5, 20], "sequence": [ 0.01, 1], "choice": [ 0.01, 1],
                                            "parallel": [ 0.01, 1], "loop": [ 0.01, 1], "silent": [ 0.01, 1],
                                            "lt_dependency": [ 0.01, 1], "num_traces": [ 10, 100],
                                            "duplicate": [ 0], "or": [ 0]},
                                            "n_trials": 50,
                                            "output_path": os.path.join('output', 'layer_test')
                                            },
                     "similarity_threshold": 0.8,
                     "output_path": "output"}
    EXPECTED_VALUES = {0.99, 1.19, 1.2, 99.0, 1.0, 998.4}
    EXPECTED_LAYER = [{'ratio_variants_per_number_of_traces': 0.99, 'n_traces': 99.0},
                        {'ratio_variants_per_number_of_traces': 1.19, 'n_traces': 99.0},
                        {'trace_len_min': 1.0, 'n_traces': 99.0},
                        {'trace_len_min': 1.2, 'n_traces': 99.0},
                        {'n_traces': 998.4, 'ratio_variants_per_number_of_traces': 0.99},
                        {'ratio_variants_per_number_of_traces': 1.19, 'n_traces': 998.4},
                        {'trace_len_min': 1.0, 'n_traces': 998.4},
                        {'trace_len_min': 1.2, 'n_traces': 998.4},
                        {'trace_len_min': 1.0, 'ratio_variants_per_number_of_traces': 0.99},
                        {'trace_len_min': 1.2, 'ratio_variants_per_number_of_traces': 0.99},
                        {'ratio_variants_per_number_of_traces': 1.19, 'trace_len_min': 1.0},
                        {'ratio_variants_per_number_of_traces': 1.19, 'trace_len_min': 1.2}]

    clean_output()
    features_to_disk(FEATURES_FOR_DISK)
    benchmark_to_disk(BENCHMARK_RESULTS_FOR_DISK)

    layer = Layer(SYSTEM_PARAMS, TARGETS).form(TARGETS, LEVEL,
                                                      CONTESTANTS, FEATURE_NAMES)
    feature_names = {key for d in layer for key in d.keys()}
    feature_values = {value for d in layer for value in d.values()}

    assert len(layer) == len(EXPECTED_LAYER)
    assert len(feature_names) == len(FEATURE_NAMES)
    assert set(feature_names) == set(FEATURE_NAMES)
    assert len(feature_values) == len(EXPECTED_VALUES)
    assert set(feature_values) == set(EXPECTED_VALUES)
    assert layer == EXPECTED_LAYER

def test_Layer_multiple_contestants():
    FEATURES_FOR_DISK = {'log': {0: 'genEL1_990_nan_nan', 1: 'genEL2_9984_nan_nan', 2: 'genEL3_nan_099_nan',
                        3: 'genEL4_nan_119_nan', 4: 'genEL5_nan_nan_10', 5: 'genEL6_nan_nan_12'},
                'n_traces': {0: 99.0, 1: 100.0, 2: None, 3: None, 4: None, 5: None},
                'ratio_variants_per_number_of_traces': {0: None, 1: None, 2: 1.0, 3: 1.0, 4: None, 5: None},
                'target_similarity': {0: 1.0, 1: 0.9964279466259522, 2: 0.99, 3: 0.81,
                                      4: 1.0, 5: 0.991304347826087},
                'trace_len_min': {0: None, 1: None, 2: None, 3: None, 4: 1.0, 5: 1.0}}

    BENCHMARK_RESULTS_FOR_DISK = {'log': {0: 'genEL1_990_nan_nan', 1: 'genEL2_9984_nan_nan', 2: 'genEL3_nan_099_nan',
                                3: 'genEL4_nan_119_nan', 4: 'genEL5_nan_nan_10', 5: 'genEL6_nan_nan_12'},
                                'fitness_sm1': {0: 0.89, 1: 1.0, 2: 1.0, 3: 1.0, 4: 0.78, 5: 0.78},
                                'precision_sm1': {0: 0.58, 1: 1.0, 2: 0.47, 3: 0.47, 4: 0.92, 5: 0.92},
                                'fscore_sm1': {0: 0.71, 1: 1.0, 2: 0.64, 3: 0.64, 4: 0.85, 5: 0.85},
                                'size_sm1': {0: 33, 1: 11, 2: 36, 3: 36, 4: 31, 5: 31},
                                'cfc_sm1': {0: 15, 1: 4, 2: 13, 3: 13, 4: 12, 5: 12},
                                'pnsize_sm1': {0: 17, 1: 8, 2: 25, 3: 25, 4: 18, 5: 18},
                                'exectime_sm1': {0: 5.37, 1: 5.33, 2: 6.74, 3: 6.76, 4: 5.34, 5: 5.36},
                                'benchtime_sm1': {0: 8.86, 1: 5.36, 2: 130.25, 3: 131.46, 4: 5.49, 5: 5.52},
                                'fitness_sm2': {0: 0.4, 1: 1.0, 2: 0.53, 3: 0.53, 4: -1.0, 5: -1.0},
                                'precision_sm2': {0: 0.81, 1: 1.0, 2: 0.79, 3: 0.79, 4: -1.0, 5: -1.0},
                                'fscore_sm2': {0: 0.54, 1: 1.0, 2: 0.64, 3: 0.64, 4: -1.0, 5: -1.0},
                                'size_sm2': {0: 32, 1: 11, 2: 33, 3: 33, 4: 35, 5: 35},
                                'cfc_sm2': {0: 14, 1: 4, 2: 12, 3: 12, 4: 16, 5: 16},
                                'pnsize_sm2': {0: 17, 1: 8, 2: 20, 3: 20, 4: 21, 5: 21},
                                'exectime_sm2': {0: 5.76, 1: 5.65, 2: 5.73, 3: 5.69, 4: 5.69, 5: 5.7},
                                'benchtime_sm2': {0: 8.62, 1: 5.68, 2: 72.15, 3: 71.8, 4: 5.7, 5: 5.7}}

    TARGETS = [{'n_traces': 99.0}, {'n_traces': 998.4},
               {'ratio_variants_per_number_of_traces': 0.99},
               {'ratio_variants_per_number_of_traces': 1.19},
               {'trace_len_min': 1.0}, {'trace_len_min': 1.2}]
    LEVEL = 1
    CONTESTANTS = ['sm1', 'sm2']
    FEATURE_NAMES = ['n_traces', 'ratio_variants_per_number_of_traces', 'trace_len_min']
    SYSTEM_PARAMS = {"generator_params": {
                        "config_space": { "mode": [ 5, 20], "sequence": [ 0.01, 1], "choice": [ 0.01, 1],
                                            "parallel": [ 0.01, 1], "loop": [ 0.01, 1], "silent": [ 0.01, 1],
                                            "lt_dependency": [ 0.01, 1], "num_traces": [ 10, 100],
                                            "duplicate": [ 0], "or": [ 0]},
                                            "n_trials": 50,
                                            "output_path": os.path.join('output', 'layer_test')
                                            },
                     "similarity_threshold": 0.8,
                     "output_path": "output"}
    EXPECTED_VALUES = {0.99, 1.19, 1.2, 99.0, 1.0, 998.4}
    EXPECTED_LAYER = [{'ratio_variants_per_number_of_traces': 0.99, 'n_traces': 99.0},
                        {'ratio_variants_per_number_of_traces': 1.19, 'n_traces': 99.0},
                        {'trace_len_min': 1.0, 'n_traces': 99.0},
                        {'trace_len_min': 1.2, 'n_traces': 99.0},
                        {'n_traces': 998.4, 'ratio_variants_per_number_of_traces': 0.99},
                        {'ratio_variants_per_number_of_traces': 1.19, 'n_traces': 998.4},
                        {'trace_len_min': 1.0, 'n_traces': 998.4},
                        {'trace_len_min': 1.2, 'n_traces': 998.4},
                        {'trace_len_min': 1.0, 'ratio_variants_per_number_of_traces': 0.99},
                        {'trace_len_min': 1.2, 'ratio_variants_per_number_of_traces': 0.99},
                        {'ratio_variants_per_number_of_traces': 1.19, 'trace_len_min': 1.0},
                        {'ratio_variants_per_number_of_traces': 1.19, 'trace_len_min': 1.2}]

    clean_output()
    features_to_disk(FEATURES_FOR_DISK)
    benchmark_to_disk(BENCHMARK_RESULTS_FOR_DISK)
    benchmark_to_disk(BENCHMARK_RESULTS_FOR_DISK, miner="sm2")

    layer = Layer(SYSTEM_PARAMS, TARGETS).form(TARGETS, LEVEL,
                                                      CONTESTANTS, FEATURE_NAMES)
    feature_names = {key for d in layer for key in d.keys()}
    feature_values = {value for d in layer for value in d.values()}

    assert len(layer) == len(EXPECTED_LAYER)
    assert len(feature_names) == len(FEATURE_NAMES)
    assert set(feature_names) == set(FEATURE_NAMES)
    assert len(feature_values) == len(EXPECTED_VALUES)
    assert set(feature_values) == set(EXPECTED_VALUES)
    assert layer == EXPECTED_LAYER

def test_Lattice():
    FEATURE_FOR_DISK = {'n_traces': 99.0, 'ratio_variants_per_number_of_traces': 0.99, 'trace_len_min': 1.0}
    BENCHMARK_RESULTS_FOR_DISK = {'log': {0: 'genEL12_nan_119_12', 1: 'genEL2_990_099_12', 2: 'genEL3_nan_099_nan',
                                          3: 'genEL10_nan_099_12', 4: 'genEL7_9984_119_10', 5: 'genEL4_990_nan_12',
                                          6: 'genEL3_990_119_10', 7: 'genEL5_9984_099_10', 8: 'genEL6_9984_099_12',
                                          9: 'genEL8_9984_nan_12', 10: 'genEL2_990_119_nan', 11: 'genEL11_nan_119_10',
                                          12: 'genEL9_nan_099_10', 13: 'genEL1_990_099_10', 15: 'genEL4_9984_119_nan',
                                          16: 'genEL5_990_nan_10', 17: 'genEL4_nan_119_nan', 18: 'genEL4_990_119_12',
                                          19: 'genEL5_nan_nan_10', 20: 'genEL2_9984_nan_nan', 21: 'genEL7_9984_nan_10',
                                          24: 'genEL6_nan_nan_12', 25: 'genEL8_9984_119_12', 27: 'genEL1_990_nan_nan'},
                                  'fitness_sm1': {0: 0.92, 1: 0.9, 2: 1.0, 3: 0.8, 4: 0.79, 5: 1.0, 6: 0.79, 7: 0.94,
                                                  8: 0.9, 9: 0.95, 10: 0.86, 11: 0.8, 12: 0.9, 13: 0.94, 15: 0.86, 16: 0.98,
                                                  17: 1.0, 18: 0.84, 19: 0.78, 20: 1.0, 21: 0.98, 24: 0.78, 25: 0.84, 27: 0.89},
                                  'precision_sm1': {0: 0.33, 1: 0.42, 2: 0.47, 3: 0.86, 4: 0.72, 5: 0.81, 6: 0.72, 7: 0.56,
                                                    8: 0.42, 9: 1.0, 10: 0.53, 11: 0.65, 12: 0.42, 13: 0.56, 15: 0.53, 16: 0.81,
                                                    17: 0.47, 18: 0.61, 19: 0.92, 20: 1.0, 21: 0.81, 24: 0.92, 25: 0.61, 27: 0.58},
                                  'fscore_sm1': {0: 0.48, 1: 0.57, 2: 0.64, 3: 0.83, 4: 0.75, 5: 0.89, 6: 0.75, 7: 0.7,
                                                 8: 0.57, 9: 0.97, 10: 0.66, 11: 0.72, 12: 0.57, 13: 0.7, 15: 0.66, 16: 0.89,
                                                 17: 0.64, 18: 0.7, 19: 0.85, 20: 1.0, 21: 0.89, 24: 0.85, 25: 0.7, 27: 0.71},
                                  'size_sm1': {0: 50, 1: 48, 2: 36, 3: 19, 4: 49, 5: 26, 6: 49, 7: 37, 8: 48, 9: 18, 10: 47, 11: 43, 12: 48,
                                               13: 37, 15: 47, 16: 23, 17: 36, 18: 48, 19: 31, 20: 11, 21: 23, 24: 31, 25: 48, 27: 33},
                                  'cfc_sm1': {0: 29, 1: 27, 2: 13, 3: 9, 4: 28, 5: 12, 6: 28, 7: 15, 8: 27, 9: 7, 10: 27, 11: 22,
                                              12: 27, 13: 15, 15: 27, 16: 10, 17: 13, 18: 27, 19: 12, 20: 4, 21: 10, 24: 12, 25: 27, 27: 15},
                                  'pnsize_sm1': {0: 31, 1: 29, 2: 25, 3: 12, 4: 31, 5: 15, 6: 31, 7: 23, 8: 29, 9: 12, 10: 30, 11: 25,
                                                 12: 29, 13: 23, 15: 30, 16: 13, 17: 25, 18: 29, 19: 18, 20: 8, 21: 13, 24: 18, 25: 29, 27: 17},
                                  'exectime_sm1': {0: 5.81, 1: 6.84, 2: 5.71, 3: 6.84, 4: 6.9, 5: 4.13, 6: 6.89, 7: 6.85,
                                                   8: 6.9, 9: 8.19, 10: 5.78, 11: 5.83, 12: 5.75, 13: 8.2, 15: 5.79, 16: 6.83,
                                                   17: 5.76, 18: 6.78, 19: 5.74, 20: 5.7, 21: 6.76, 24: 5.77, 25: 6.81, 27: 5.77},
                                  'benchtime_sm1': {0: 164.92, 1: 28.79, 2: 117.18, 3: 8.4, 4: 26.86, 5: 4.9, 6: 26.8, 7: 172.61,
                                                    8: 28.74, 9: 8.4, 10: 87.62, 11: 51.8, 12: 23.24, 13: 174.67, 15: 88.5, 16: 7.14,
                                                    17: 114.22, 18: 67.06, 19: 5.9, 20: 5.72, 21: 7.07, 24: 5.93, 25: 66.28, 27: 9.46}}
    BENCHMARK_RESULTS_FOR_DISK = {'log': {0: 'genEL3_nan_099_nan', 1: 'genEL3_nan_119_10', 2: 'genEL4_nan_119_12', 3: 'genEL4_nan_119_nan', 4: 'genEL1_990_119_nan', 5: 'genEL5_nan_nan_10', 6: 'genEL2_9984_nan_nan', 7: 'genEL6_nan_nan_12', 8: 'genEL1_990_nan_nan', 9: 'genEL2_9984_119_nan', 10: 'genEL12_nan_119_12', 11: 'genEL2_990_099_12', 12: 'genEL3_nan_099_nan', 13: 'genEL10_nan_099_12', 14: 'genEL7_9984_119_10', 15: 'genEL4_990_nan_12', 16: 'genEL3_990_119_10', 17: 'genEL5_9984_099_10', 18: 'genEL6_9984_099_12', 19: 'genEL8_9984_nan_12', 20: 'genEL2_990_119_nan', 21: 'genEL11_nan_119_10', 22: 'genEL9_nan_099_10', 23: 'genEL1_990_099_10', 24: 'genEL1_nan_099_nan', 25: 'genEL4_9984_119_nan', 26: 'genEL5_990_nan_10', 27: 'genEL4_nan_119_nan', 28: 'genEL4_990_119_12', 29: 'genEL5_nan_nan_10', 30: 'genEL2_9984_nan_nan', 31: 'genEL7_9984_nan_10', 32: 'genEL6_9984_119_nan', 33: 'genEL3_990_nan_10', 34: 'genEL6_nan_nan_12', 35: 'genEL8_9984_119_12', 36: 'genEL6_990_nan_12', 37: 'genEL1_990_nan_nan'}, 'fitness_sm1': {0: 1.0, 1: 0.8, 2: 0.92, 3: 1.0, 4: 0.86, 5: 0.78, 6: 1.0, 7: 0.78, 8: 0.89, 9: 0.86, 10: 0.92, 11: 0.9, 12: 1.0, 13: 0.8, 14: 0.79, 15: 1.0, 16: 0.79, 17: 0.94, 18: 0.9, 19: 0.95, 20: 0.86, 21: 0.8, 22: 0.9, 23: 0.94, 24: 1.0, 25: 0.86, 26: 0.98, 27: 1.0, 28: 0.84, 29: 0.78, 30: 1.0, 31: 0.98, 32: 0.86, 33: 0.98, 34: 0.78, 35: 0.84, 36: 1.0, 37: 0.89}, 'precision_sm1': {0: 0.47, 1: 0.65, 2: 0.33, 3: 0.47, 4: 0.53, 5: 0.92, 6: 1.0, 7: 0.92, 8: 0.58, 9: 0.53, 10: 0.33, 11: 0.42, 12: 0.47, 13: 0.86, 14: 0.72, 15: 0.81, 16: 0.72, 17: 0.56, 18: 0.42, 19: 1.0, 20: 0.53, 21: 0.65, 22: 0.42, 23: 0.56, 24: 0.47, 25: 0.53, 26: 0.81, 27: 0.47, 28: 0.61, 29: 0.92, 30: 1.0, 31: 0.81, 32: 0.53, 33: 0.81, 34: 0.92, 35: 0.61, 36: 0.81, 37: 0.58}, 'fscore_sm1': {0: 0.64, 1: 0.72, 2: 0.48, 3: 0.64, 4: 0.66, 5: 0.85, 6: 1.0, 7: 0.85, 8: 0.71, 9: 0.66, 10: 0.48, 11: 0.57, 12: 0.64, 13: 0.83, 14: 0.75, 15: 0.89, 16: 0.75, 17: 0.7, 18: 0.57, 19: 0.97, 20: 0.66, 21: 0.72, 22: 0.57, 23: 0.7, 24: 0.64, 25: 0.66, 26: 0.89, 27: 0.64, 28: 0.7, 29: 0.85, 30: 1.0, 31: 0.89, 32: 0.66, 33: 0.89, 34: 0.85, 35: 0.7, 36: 0.89, 37: 0.71}, 'size_sm1': {0: 36, 1: 43, 2: 50, 3: 36, 4: 47, 5: 31, 6: 11, 7: 31, 8: 33, 9: 47, 10: 50, 11: 48, 12: 36, 13: 19, 14: 49, 15: 26, 16: 49, 17: 37, 18: 48, 19: 18, 20: 47, 21: 43, 22: 48, 23: 37, 24: 36, 25: 47, 26: 23, 27: 36, 28: 48, 29: 31, 30: 11, 31: 23, 32: 47, 33: 23, 34: 31, 35: 48, 36: 26, 37: 33}, 'cfc_sm1': {0: 13, 1: 22, 2: 29, 3: 13, 4: 27, 5: 12, 6: 4, 7: 12, 8: 15, 9: 27, 10: 29, 11: 27, 12: 13, 13: 9, 14: 28, 15: 12, 16: 28, 17: 15, 18: 27, 19: 7, 20: 27, 21: 22, 22: 27, 23: 15, 24: 13, 25: 27, 26: 10, 27: 13, 28: 27, 29: 12, 30: 4, 31: 10, 32: 27, 33: 10, 34: 12, 35: 27, 36: 12, 37: 15}, 'pnsize_sm1': {0: 25, 1: 25, 2: 31, 3: 25, 4: 30, 5: 18, 6: 8, 7: 18, 8: 17, 9: 30, 10: 31, 11: 29, 12: 25, 13: 12, 14: 31, 15: 15, 16: 31, 17: 23, 18: 29, 19: 12, 20: 30, 21: 25, 22: 29, 23: 23, 24: 25, 25: 30, 26: 13, 27: 25, 28: 29, 29: 18, 30: 8, 31: 13, 32: 30, 33: 13, 34: 18, 35: 29, 36: 15, 37: 17}, 'exectime_sm1': {0: 5.82, 1: 7.41, 2: 7.4, 3: 5.8, 4: 7.33, 5: 5.7, 6: 5.7, 7: 5.66, 8: 5.72, 9: 7.31, 10: 5.81, 11: 6.84, 12: 5.71, 13: 6.84, 14: 6.9, 15: 4.13, 16: 6.89, 17: 6.85, 18: 6.9, 19: 8.19, 20: 5.78, 21: 5.83, 22: 5.75, 23: 8.2, 24: 6.34, 25: 5.79, 26: 6.83, 27: 5.76, 28: 6.78, 29: 5.74, 30: 5.7, 31: 6.76, 32: 4.26, 33: 4.13, 34: 5.77, 35: 6.81, 36: 6.83, 37: 5.77}, 'benchtime_sm1': {0: 155.49, 1: 63.32, 2: 174.94, 3: 155.46, 4: 105.28, 5: 5.86, 6: 5.73, 7: 5.82, 8: 9.37, 9: 102.38, 10: 164.92, 11: 28.79, 12: 117.18, 13: 8.4, 14: 26.86, 15: 4.9, 16: 26.8, 17: 172.61, 18: 28.74, 19: 8.4, 20: 87.62, 21: 51.8, 22: 23.24, 23: 174.67, 24: 89.45, 25: 88.5, 26: 7.14, 27: 114.22, 28: 67.06, 29: 5.9, 30: 5.72, 31: 7.07, 32: 78.7, 33: 4.44, 34: 5.93, 35: 66.28, 36: 7.61, 37: 9.46}}
    FEATURE_NAMES = ['n_traces','ratio_variants_per_number_of_traces','trace_len_min']
    VALUE_LIST = [[99.0, 998.4],
                  [0.99, 1.19],
                  [1.0, 1.2]]
    VALUE_LIST = [[round(item, 2) for item in sublist] for sublist in VALUE_LIST]
    CONTESTANTS = ['sm1']
    SYSTEM_PARAMS = {"generator_params": {
                        "config_space": { "mode": [ 5, 20], "sequence": [ 0.01, 1], "choice": [ 0.01, 1],
                                            "parallel": [ 0.01, 1], "loop": [ 0.01, 1], "silent": [ 0.01, 1],
                                            "lt_dependency": [ 0.01, 1], "num_traces": [ 10, 100],
                                            "duplicate": [ 0], "or": [ 0]},
                                            "n_trials": 50,
                                            "output_path": os.path.join('output', 'shaining_test')
                                            },
                     "similarity_threshold": 0.8,
                     "output_path": "output"}

    clean_output()
    benchmark_to_disk(BENCHMARK_RESULTS_FOR_DISK)
    lattice_path = Lattice(FEATURE_NAMES, VALUE_LIST, SYSTEM_PARAMS).form(contestants=CONTESTANTS)

    assert lattice_path is not None
    assert type(lattice_path) == str
    assert os.path.exists(lattice_path)
    results = pd.read_csv(lattice_path)
    #assert results.shape == (96, 5)
    #assert results.isna().any().sum() == 0
