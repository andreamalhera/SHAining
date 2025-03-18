import pandas as pd
import pytest
import os

from pathlib import Path
from scipy.spatial.distance import cosine
from shaining.benchmark import BenchmarkTest, get_dump_path
from shaining.utils.io_helpers import delete_list_of_dirs, dump_features_json
from shaining.utils.param_keys import INPUT_PATH, OUTPUT_PATH, EVENTLOGS_SELECTION

def clear_output():
    delete_list_of_dirs([os.path.join("output", "test", "benchmark_test"),
                         os.path.join("output", "test", "benchmark"),
                         os.path.join("output", "benchmark"),
                         os.path.join("output", "benchmark_test")])
    files_to_remove = [os.path.join("data", "test", "benchmark_test","benchmark_test_feat.csv"),
                       os.path.join("data", "test", "benchmark_test","genEL2_099_12.xes")]
    for filepath in files_to_remove:
        os.remove(filepath) if os.path.exists(filepath) else None

def mock_existing_EL(mock_EL_name, csv_file):
    similarity_row = {'log':mock_EL_name, 'target_similarity': 0.91}
    Path(os.path.join(csv_file.rsplit("_",1)[0],mock_EL_name+".xes")).touch()

    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file)

    new_csv_file = os.path.join(csv_file.rsplit("_",1)[0],"benchmark_modified.csv")
    # Append the new row
    new_df = pd.DataFrame([similarity_row], columns=df.columns)
    df = pd.concat([df, new_df], ignore_index=True)
    df.to_csv(new_csv_file, index=False)
    return new_csv_file


def test_single_BenchmarkTest():
    INPUT_PARAMS = {'benchmark_test': 'discovery',
                    'input_path': 'data/test/benchmark_test',
                    'output_path': 'output',
                    'miners': ['sm2']}
    VALIDATION_LOG_NAMES = ['genEL1_02_04', 'BPIC2013_open_problems']
    VALIDATION_OUTPUT = {'log': {0: 'genEL1_02_04', 1: 'BPIC2013_open_problems'}, 'fitness_sm2': {0: 0.42, 1: 0.0}, 'precision_sm2': {0: 0.98, 1: 1.0}, 'fscore_sm2': {0: 0.59, 1: 0.0}, 'size_sm2': {0: 20, 1: 2}, 'pnsize_sm2': {0: 13, 1: 2}, 'cfc_sm2': {0: 10, 1: 0}, 'exectime_sm2': {0: 4.38, 1: 4.59}, 'benchtime_sm2': {0: 5.45, 1: 4.69}}
    clear_output()

    benchmark = BenchmarkTest(params=INPUT_PARAMS)
    benchmark_results = benchmark.results.round(2)
    benchmark_results = benchmark_results.sort_values(by='log', ascending=False).reset_index(drop=True)
    results_dict = benchmark_results.to_dict()

    assert benchmark.filename == os.path.join(os.path.split(INPUT_PARAMS['input_path'])[1],"benchmark_sm2.csv")
    assert benchmark_results.shape == (2, 9), f"Unexpected shape: {benchmark_results.shape}"
    assert set(benchmark_results['log'].to_list()) == set(VALIDATION_LOG_NAMES)
    assert set(results_dict.keys()) == set(VALIDATION_OUTPUT.keys())

    prefixes = ['exectime_','benchtime_'] # Times may vary to the 2nd decimal place.
    filtered_d = {k: v for k, v in results_dict.items() if not any(k.startswith(p) for p in prefixes)}
    FILTERED_VALIDATION_OUTPUT = {k: v for k, v in VALIDATION_OUTPUT.items() if not any(k.startswith(p) for p in prefixes)}

    for key, value in FILTERED_VALIDATION_OUTPUT.items():
        assert filtered_d.get(key) == value, f"Mismatch at {key}: {value} != {results_dict.get(key)}"

def test_BenchmarkTest():
    INPUT_PARAMS = {'benchmark_test': 'discovery', 'input_path': 'data/test/benchmark_test',
                    'output_path': 'output',
                    'miners': ['ilp', 'heuristics', 'inductive', 'imf', 'sm1', 'sm2']}
    VALIDATION_LOG_NAMES = ['genEL1_02_04', 'BPIC2013_open_problems']
    VALIDATION_OUTPUT = {'log': {0: 'genEL1_02_04', 1: 'BPIC2013_open_problems'},
                         'fitness_ilp': {0: 1.0, 1: 1.0}, 'precision_ilp': {0: 0.35, 1: 0.91}, 'fscore_ilp': {0: 0.51, 1: 0.95}, 'size_ilp': {0: 28, 1: 10},
                         'pnsize_ilp': {0: 12, 1: 5}, 'cfc_ilp': {0: 9, 1: 3}, 'exectime_ilp': {0: 0.13, 1: 0.15}, 'benchtime_ilp': {0: 1.84, 1: 0.41},
                         'fitness_heuristics': {0: 0.82, 1: 0.99}, 'precision_heuristics': {0: 0.38, 1: 0.96}, 'fscore_heuristics': {0: 0.52, 1: 0.98}, 'size_heuristics': {0: 30, 1: 12},
                         'pnsize_heuristics': {0: 20, 1: 7}, 'cfc_heuristics': {0: 16, 1: 7}, 'exectime_heuristics': {0: 0.02, 1: 0.09}, 'benchtime_heuristics': {0: 7.54, 1: 0.45},
                         'fitness_inductive': {0: 1.0, 1: 1.0}, 'precision_inductive': {0: 0.42, 1: 0.91}, 'fscore_inductive': {0: 0.59, 1: 0.95}, 'size_inductive': {0: 37, 1: 23},
                         'pnsize_inductive': {0: 25, 1: 16}, 'cfc_inductive': {0: 21, 1: 14}, 'exectime_inductive': {0: 0.02, 1: 0.09}, 'benchtime_inductive': {0: 11.76, 1: 1.57},
                         'fitness_imf': {0: 0.9, 1: 0.85}, 'precision_imf': {0: 0.42, 1: 0.91}, 'fscore_imf': {0: 0.57, 1: 0.88}, 'size_imf': {0: 25, 1: 17},
                         'pnsize_imf': {0: 16, 1: 10}, 'cfc_imf': {0: 12, 1: 8}, 'exectime_imf': {0: 0.02, 1: 0.15}, 'benchtime_imf': {0: 2.99, 1: 0.6},
                         'fitness_sm1': {0: 0.8, 1: 0.94}, 'precision_sm1': {0: 0.86, 1: 0.96}, 'fscore_sm1': {0: 0.83, 1: 0.95}, 'size_sm1': {0: 19, 1: 10},
                         'pnsize_sm1': {0: 12, 1: 7}, 'cfc_sm1': {0: 9, 1: 5}, 'exectime_sm1': {0: 3.33, 1: 3.72}, 'benchtime_sm1': {0: 4.99, 1: 4.07},
                         'fitness_sm2': {0: 0.42, 1: 0.0}, 'precision_sm2': {0: 0.98, 1: 1.0}, 'fscore_sm2': {0: 0.59, 1: 0.0}, 'size_sm2': {0: 20, 1: 2},
                         'pnsize_sm2': {0: 13, 1: 2}, 'cfc_sm2': {0: 10, 1: 0}, 'exectime_sm2': {0: 5.36, 1: 3.38}, 'benchtime_sm2': {0: 6.36, 1: 3.47}}

    clear_output()

    results = BenchmarkTest(params=INPUT_PARAMS).results.round(2)
    results = results.sort_values(by='log', ascending=False).reset_index(drop=True)
    results_dict = results.to_dict()

    assert results.shape == (2,49), f"Unexpected shape: {results.shape}"
    assert set(results['log'].to_list()) == set(VALIDATION_LOG_NAMES)
    assert set(results_dict.keys()) == set(VALIDATION_OUTPUT.keys())

    prefixes = ['exectime_','benchtime_'] # Times may vary to the 2nd decimal place.
    filtered_d = {k: v for k, v in results_dict.items() if not any(k.startswith(p) for p in prefixes)}
    FILTERED_VALIDATION_OUTPUT = {k: v for k, v in VALIDATION_OUTPUT.items() if not any(k.startswith(p) for p in prefixes)}

    for key, value in FILTERED_VALIDATION_OUTPUT.items():
        current_values = filtered_d.get(key)
        assert len(set(current_values)) == len(set(value))
        if key != 'log':
            keys = sorted(set(value))
            v1 = [current_values.get(k, 0) for k in keys]
            v2 = [value.get(k, 0) for k in keys]
            assert cosine(v1, v2) <  0.1, f"Mismatch at {key}: {value} != {results_dict.get(key)}. Cosine distance over 0.1 tolerance: {cosine(v1, v2)}"
        else:
            assert filtered_d.get(key) == value, f"Mismatch at {key}: {value} != {results_dict.get(key)}"

def test_selection_BenchmarkTest():
    INPUT_PARAMS = {'benchmark_test': 'discovery',
                    'input_path': 'data/test/benchmark_test',
                    'eventlogs_selection':{'feat_path': 'data/test/benchmark_test_feat.csv',
                                           'similarity_threshold': 0.9,
                                           'current_level': 2},
                    'output_path': 'output', 'miners': ['sm2']}
    VALIDATION_LOG_NAMES = ['genEL1_02_04']
    VALIDATION_OUTPUT = {'log': {0: 'genEL1_02_04'},
                         'fitness_sm2': {0: 0.42},
                         'precision_sm2': {0: 0.98},
                         'fscore_sm2': {0: 0.59},
                         'size_sm2': {0: 20},
                         'pnsize_sm2': {0: 13},
                         'cfc_sm2': {0: 10},
                         'exectime_sm2': {0: 4.38},
                         'benchtime_sm2': {0: 5.45}}
    clear_output()

    benchmark = BenchmarkTest(params=INPUT_PARAMS)
    benchmark_results = benchmark.results.round(2)
    benchmark_results = benchmark_results.sort_values(by='log', ascending=False).reset_index(drop=True)
    results_dict = benchmark_results.to_dict()

    assert benchmark.filename == os.path.join(os.path.split(INPUT_PARAMS['input_path'])[1],"benchmark_sm2.csv")
    assert benchmark_results.shape == (1, 9), f"Unexpected shape: {benchmark_results.shape}"
    assert set(benchmark_results['log'].to_list()) == set(VALIDATION_LOG_NAMES)
    assert set(results_dict.keys()) == set(VALIDATION_OUTPUT.keys())

    prefixes = ['exectime_','benchtime_'] # Times may vary to the 2nd decimal place.
    filtered_d = {k: v for k, v in results_dict.items() if not any(k.startswith(p) for p in prefixes)}
    FILTERED_VALIDATION_OUTPUT = {k: v for k, v in VALIDATION_OUTPUT.items() if not any(k.startswith(p) for p in prefixes)}

    for key, value in FILTERED_VALIDATION_OUTPUT.items():
        assert filtered_d.get(key) == value, f"Mismatch at {key}: {value} != {results_dict.get(key)}"

def test_existing_json_single_miner_BenchmarkTest():
    SAVED_RESULTS = {'log': 'genEL2_099_12', 'fitness_sm2': 0.3679630542865878,
                     'precision_sm2': 0.9316628701594533, 'fscore_sm2': 0.5275633685368432,
                     'size_sm2': 43, 'cfc_sm2': 22, 'pnsize_sm2': 25, 'exectime_sm2': 3.55,
                     'benchtime_sm2': 13.33}
    INPUT_PARAMS = {'benchmark_test': 'discovery',
                    'input_path': 'data/test/benchmark_test',
                    'eventlogs_selection':{'feat_path': 'data/test/benchmark_test_feat.csv',
                                           'similarity_threshold': 0.8,
                                           'current_level': 2},
                    'output_path': 'output', 'miners': ['sm2']}
    # 'genEL2_099_12' dissapears due to existing results in disk
    # 'BPIC2013_open_problems' dissapears due to format of name being incompatible with level pruning
    VALIDATION_LOG_NAMES = ['genEL2_099_12', 'genEL1_02_04']
    VALIDATION_OUTPUT = {'log': {0: 'genEL2_099_12', 1: 'genEL1_02_04'},
                         'fitness_sm2': {0: 0.37, 1: 0.42},
                         'precision_sm2': {0: 0.93, 1: 0.98},
                         'fscore_sm2': {0: 0.53, 1: 0.59},
                         'size_sm2': {0: 43, 1: 20},
                         'cfc_sm2': {0: 22, 1: 10},
                         'pnsize_sm2': {0: 25, 1: 13},
                         'exectime_sm2': {0: 3.55, 1: 4.89},
                         'benchtime_sm2': {0: 13.33, 1: 5.93}}

    clear_output()

    # Simulate existing files in disk
    INPUT_PARAMS['eventlogs_selection']['feat_path'] = mock_existing_EL("genEL2_099_12", INPUT_PARAMS['eventlogs_selection']['feat_path'])

    dump_path = get_dump_path(INPUT_PARAMS[OUTPUT_PATH],
                              INPUT_PARAMS[INPUT_PATH])
                              #INPUT_PARAMS['miners'])
    dump_features_json(SAVED_RESULTS,
                       dump_path,
                       'genEL2_099_12',
                       content_type="benchmark_sm2")

    benchmark = BenchmarkTest(params=INPUT_PARAMS)
    benchmark_results = benchmark.results.round(2)
    results_dict = benchmark_results.to_dict()

    assert benchmark.num_tasks == {'sm2': 1}
    assert benchmark_results.shape == (2, 9), f"Unexpected shape: {benchmark_results.shape}"
    assert benchmark.filename == os.path.join(os.path.split(INPUT_PARAMS['input_path'])[1],"benchmark_sm2.csv")
    assert set(benchmark_results['log'].to_list()) == set(VALIDATION_LOG_NAMES)
    assert set(results_dict.keys()) == set(VALIDATION_OUTPUT.keys())

    prefixes = ['exectime_','benchtime_'] # Times may vary to the 2nd decimal place.
    filtered_d = {k: v for k, v in results_dict.items() if not any(k.startswith(p) for p in prefixes)}
    FILTERED_VALIDATION_OUTPUT = {k: v for k, v in VALIDATION_OUTPUT.items() if not any(k.startswith(p) for p in prefixes)}

    for key, value in FILTERED_VALIDATION_OUTPUT.items():
        assert filtered_d.get(key) == value, f"Mismatch at {key}: {value} != {results_dict.get(key)}"

def test_existing_jsons_multi_miner_BenchmarkTest():
    SAVED_RESULTS_SM2 = {'log': 'genEL2_099_12', 'fitness_sm2': 0.3679630542865878,
                        'precision_sm2': 0.9316628701594533, 'fscore_sm2': 0.5275633685368432,
                        'size_sm2': 43, 'cfc_sm2': 22, 'pnsize_sm2': 25, 'exectime_sm2': 3.55,
                        'benchtime_sm2': 13.33}
    SAVED_RESULTS_SM1 = {'log': 'genEL2_099_12', 'fitness_sm1': 0.9,
                         'precision_sm1':0.42, 'fscore_sm1': 0.57, 'size_sm1': 48,
                         'cfc_sm1': 27, 'pnsize_sm1': 29, 'exectime_sm1': 3.75,
                         'benchtime_sm1': 19.02}
    INPUT_PARAMS = {'benchmark_test': 'discovery',
                    'input_path': 'data/test/benchmark_test',
                    'eventlogs_selection':{'feat_path': 'data/test/benchmark_test_feat.csv',
                                           'similarity_threshold': 0.8,
                                           'current_level': 2},
                    'output_path': 'output', 'miners': ['sm2', 'sm1']}
    # 'genEL2_099_12' dissapears due to existing results in disk
    # 'BPIC2013_open_problems' dissapears due to format of name being incompatible with level pruning
    VALIDATION_LOG_NAMES = ['genEL1_02_04','genEL2_099_12']
    VALIDATION_OUTPUT = {'log': {0: 'genEL1_02_04', 1: 'genEL2_099_12'},
                         'fitness_sm1': {0: 0.8, 1: 0.9},
                         'precision_sm1': {0: 0.86, 1: 0.42},
                         'fscore_sm1': {0: 0.83, 1: 0.57},
                         'size_sm1': {0: 19, 1: 48},
                         'cfc_sm1': {0: 9, 1: 27},
                         'pnsize_sm1': {0: 12, 1: 29},
                         'exectime_sm1': {0: 3.56, 1: 3.75},
                         'benchtime_sm1': {0: 5.04, 1: 19.02},
                         'fitness_sm2': {0: 0.42, 1: 0.37},
                         'precision_sm2': {0: 0.98, 1: 0.93},
                         'fscore_sm2': {0: 0.59, 1: 0.53},
                         'size_sm2': {0: 20, 1: 43},
                         'cfc_sm2': {0: 10, 1: 22},
                         'pnsize_sm2': {0: 13, 1: 25},
                         'exectime_sm2': {1: 3.55, 0: 4.89},
                         'benchtime_sm2': {1: 13.33, 0: 5.93}}

    clear_output()

    # Simulate existing files in disk
    INPUT_PARAMS['eventlogs_selection']['feat_path'] = mock_existing_EL("genEL2_099_12", INPUT_PARAMS['eventlogs_selection']['feat_path'])
    dump_path = get_dump_path(INPUT_PARAMS[OUTPUT_PATH],
                              INPUT_PARAMS[INPUT_PATH])
                              #INPUT_PARAMS['miners'])
    dump_features_json(SAVED_RESULTS_SM1,
                       dump_path,
                       'genEL2_099_12',
                       content_type="benchmark_sm1")
    dump_path = get_dump_path(INPUT_PARAMS[OUTPUT_PATH],
                              INPUT_PARAMS[INPUT_PATH])
                              #INPUT_PARAMS['miners'])
    dump_features_json(SAVED_RESULTS_SM2,
                       dump_path,
                       'genEL2_099_12',
                       content_type="benchmark_sm2")

    benchmark = BenchmarkTest(params=INPUT_PARAMS)
    benchmark_results = benchmark.results.round(2)
    results_dict = benchmark_results.to_dict()

    assert benchmark.num_tasks == {'sm2': 1, 'sm1': 1}
    assert benchmark_results.shape == (2, 17), f"Unexpected shape: {benchmark_results.shape}"
    assert benchmark.filename == os.path.join(os.path.split(INPUT_PARAMS['input_path'])[1],"benchmark.csv")
    assert set(benchmark_results['log'].to_list()) == set(VALIDATION_LOG_NAMES)
    assert set(results_dict.keys()) == set(VALIDATION_OUTPUT.keys())

    prefixes = ['exectime_','benchtime_'] # Times may vary to the 2nd decimal place.
    filtered_d = {k: v for k, v in results_dict.items() if not any(k.startswith(p) for p in prefixes)}
    FILTERED_VALIDATION_OUTPUT = {k: v for k, v in VALIDATION_OUTPUT.items() if not any(k.startswith(p) for p in prefixes)}

    for key, value in FILTERED_VALIDATION_OUTPUT.items():
        assert filtered_d.get(key) == value, f"Mismatch at {key}: {value} != {results_dict.get(key)}"

def test_existing_csv_single_miner_BenchmarkTest():
    INPUT_PARAMS = {'benchmark_test': 'discovery',
                    'input_path': 'data/test/benchmark_test',
                    'eventlogs_selection':{'feat_path': 'data/test/benchmark_test_feat.csv',
                                           'similarity_threshold': 0.8,
                                           'current_level': 2},
                    'output_path': 'output', 'miners': ['sm2']}
    # 'genEL2_099_12' dissapears due to existing results in disk
    # 'BPIC2013_open_problems' dissapears due to format of name being incompatible with level pruning
    VALIDATION_LOG_NAMES = ['genEL2_099_12', 'genEL1_02_04']
    VALIDATION_OUTPUT = {'log': {0: 'genEL2_099_12', 1: 'genEL1_02_04'},
                         'fitness_sm2': {0: 0.37, 1: 0.42},
                         'precision_sm2': {0: 0.93, 1: 0.98},
                         'fscore_sm2': {0: 0.53, 1: 0.59},
                         'size_sm2': {0: 43, 1: 20},
                         'cfc_sm2': {0: 22, 1: 10},
                         'pnsize_sm2': {0: 25, 1: 13},
                         'exectime_sm2': {0: 3.55, 1: 4.89},
                         'benchtime_sm2': {0: 13.33, 1: 5.93}}

    clear_output()

    # Simulate existing files in disk
    INPUT_PARAMS['eventlogs_selection']['feat_path'] = mock_existing_EL("genEL2_099_12", INPUT_PARAMS['eventlogs_selection']['feat_path'])

    dump_path = os.path.join("output","benchmark_test")
                              #INPUT_PARAMS['miners'])
    df = pd.DataFrame(VALIDATION_OUTPUT)
    df.to_csv(os.path.join(dump_path, "benchmark_sm2.csv"), index=False)

    benchmark = BenchmarkTest(params=INPUT_PARAMS)
    benchmark_results = benchmark.results.round(2)
    results_dict = benchmark_results.to_dict()

    assert benchmark.num_tasks == {'sm2': 0}
    assert benchmark_results.shape == (2, 9), f"Unexpected shape: {benchmark_results.shape}"
    assert benchmark.filepath == os.path.join(INPUT_PARAMS['output_path'],
                                              os.path.split(INPUT_PARAMS['input_path'])[1],
                                              "benchmark_sm2.csv")
    assert set(benchmark_results['log'].to_list()) == set(VALIDATION_LOG_NAMES)
    assert set(results_dict.keys()) == set(VALIDATION_OUTPUT.keys())

    prefixes = ['exectime_','benchtime_'] # Times may vary to the 2nd decimal place.
    filtered_d = {k: v for k, v in results_dict.items() if not any(k.startswith(p) for p in prefixes)}
    FILTERED_VALIDATION_OUTPUT = {k: v for k, v in VALIDATION_OUTPUT.items() if not any(k.startswith(p) for p in prefixes)}

    for key, value in FILTERED_VALIDATION_OUTPUT.items():
        assert filtered_d.get(key) == value, f"Mismatch at {key}: {value} != {results_dict.get(key)}"

    SAVED_PARTIAL_RESULT = { 'log': {0: 'genEL2_099_12'},
                            'fitness_sm2': {0: 0.37},
                            'precision_sm2': {0: 0.93},
                            'fscore_sm2': {0: 0.53},
                            'size_sm2': {0: 43},
                            'cfc_sm2': {0: 22},
                            'pnsize_sm2': {0: 25},
                            'exectime_sm2': {0: 3.55},
                            'benchtime_sm2': {0: 13.33} }
    INPUT_PARAMS = {'benchmark_test': 'discovery',
                    'input_path': 'data/test/benchmark_test',
                    'eventlogs_selection':{'feat_path': 'data/test/benchmark_test_feat.csv',
                                           'similarity_threshold': 0.8,
                                           'current_level': 2},
                    'output_path': 'output', 'miners': ['sm2']}
    # 'genEL2_099_12' dissapears due to existing results in disk
    # 'BPIC2013_open_problems' dissapears due to format of name being incompatible with level pruning
    VALIDATION_LOG_NAMES = ['genEL2_099_12', 'genEL1_02_04']
    VALIDATION_OUTPUT = {'log': {0: 'genEL2_099_12', 1: 'genEL1_02_04'},
                         'fitness_sm2': {0: 0.37, 1: 0.42},
                         'precision_sm2': {0: 0.93, 1: 0.98},
                         'fscore_sm2': {0: 0.53, 1: 0.59},
                         'size_sm2': {0: 43, 1: 20},
                         'cfc_sm2': {0: 22, 1: 10},
                         'pnsize_sm2': {0: 25, 1: 13},
                         'exectime_sm2': {0: 3.55, 1: 4.89},
                         'benchtime_sm2': {0: 13.33, 1: 5.93}}

    clear_output()

    # Simulate existing files in disk
    INPUT_PARAMS['eventlogs_selection']['feat_path'] = mock_existing_EL("genEL2_099_12", INPUT_PARAMS['eventlogs_selection']['feat_path'])

    dump_path = os.path.join("output","benchmark_test")
                              #INPUT_PARAMS['miners'])
    df = pd.DataFrame(SAVED_PARTIAL_RESULT)
    df.to_csv(os.path.join(dump_path, "benchmark_sm2.csv"), index=False)

    benchmark = BenchmarkTest(params=INPUT_PARAMS)
    benchmark_results = benchmark.results.round(2)
    results_dict = benchmark_results.to_dict()

    assert benchmark.num_tasks == {'sm2': 1}
    assert benchmark_results.shape == (2, 9), f"Unexpected shape: {benchmark_results.shape}"
    assert benchmark.filepath == os.path.join(INPUT_PARAMS['output_path'],
                                              os.path.split(INPUT_PARAMS['input_path'])[1],
                                              "benchmark_sm2.csv")
    assert set(benchmark_results['log'].to_list()) == set(VALIDATION_LOG_NAMES)
    assert set(results_dict.keys()) == set(VALIDATION_OUTPUT.keys())

    prefixes = ['exectime_','benchtime_'] # Times may vary to the 2nd decimal place.
    filtered_d = {k: v for k, v in results_dict.items() if not any(k.startswith(p) for p in prefixes)}
    FILTERED_VALIDATION_OUTPUT = {k: v for k, v in VALIDATION_OUTPUT.items() if not any(k.startswith(p) for p in prefixes)}

    for key, value in FILTERED_VALIDATION_OUTPUT.items():
        assert filtered_d.get(key) == value, f"Mismatch at {key}: {value} != {results_dict.get(key)}"

def test_get_dump_path():
    INPUT_PARAMS = {'input_path': 'data/test/benchmark_test',
                    'output_path': 'output', 'miners': ['sm2']}
    dump_path = get_dump_path(INPUT_PARAMS['output_path'],
                              INPUT_PARAMS['input_path'],
                              INPUT_PARAMS['miners'])
    assert dump_path == os.path.join('output', 'test', 'benchmark_test', 'benchmark_sm2')

    multiple_miners = INPUT_PARAMS['miners']
    multiple_miners.append('sm1')
    dump_path = get_dump_path(INPUT_PARAMS['output_path'],
                              INPUT_PARAMS['input_path'],
                              multiple_miners)
    assert dump_path == os.path.join('output', 'test', 'benchmark_test', 'benchmark')

    dump_path = get_dump_path(INPUT_PARAMS['output_path'],
                              INPUT_PARAMS['input_path'])
    assert dump_path == os.path.join('output', 'test', 'benchmark_test')

    #TODO: I would actually expect some variation in the dump_path after integrating functionality in other parts of benchmark
    dump_path = get_dump_path(INPUT_PARAMS['output_path'],
                              os.path.join(INPUT_PARAMS['input_path'],'nonexistent.xes'),
                              INPUT_PARAMS['miners'])
    assert dump_path == 'output/test/benchmark_test'
