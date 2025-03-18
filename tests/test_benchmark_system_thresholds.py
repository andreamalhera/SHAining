import pandas as pd
import pytest
import os
import math
import time
from scipy.spatial.distance import cosine
from shaining.benchmark import BenchmarkTest
from shaining.utils.io_helpers import delete_list_of_dirs
from unittest.mock import patch

#TODO: This is a duplicate of tests.test_benchmark import clear_output. Difficulty getting import correct for locally pytest and CI.
def clear_output():
    delete_list_of_dirs([os.path.join("output", "test", "benchmark_test"),
                         os.path.join("output", "test", "benchmark"),
                         os.path.join("output", "benchmark"),
                         os.path.join("output", "benchmark_test")])
    files_to_remove = [os.path.join("data", "test", "benchmark_test","benchmark_test_feat.csv"),
                       os.path.join("data", "test", "benchmark_test","genEL2_099_12.xes")]
    for filepath in files_to_remove:
        os.remove(filepath) if os.path.exists(filepath) else None

def test_benchmark_timer():
    INPUT_PARAMS = {'benchmark_test': 'discovery', 'input_path': 'data/test/benchmark_test',
                    'output_path': 'output', 'miners': ['inductive'],
                    'system_params': {'timeout': 2}}
    # Validation output is None because none of the metrics could be calculated due to timeout.
    VALIDATION_OUTPUT = {'log': {}, 'fitness_inductive': {}, 'precision_inductive': {}, 'fscore_inductive': {}, 'size_inductive': {}, 'cfc_inductive': {}, 'pnsize_inductive': {}, 'exectime_inductive': {}, 'benchtime_inductive': {}}
    clear_output()
    results = BenchmarkTest(params=INPUT_PARAMS).results.round(2)
    results = results.sort_values(by='log', ascending=False).reset_index(drop=True)

    assert results.to_dict() == VALIDATION_OUTPUT

def test_benchmark_memory():
    INPUT_PARAMS = {'benchmark_test': 'discovery', 'input_path': 'data/test/benchmark_test',
                    'output_path': 'output', 'miners': ['sm2'],
                    'system_params': {'max_memory': 1024}}#1024 is 1KB. Both files are expected to fail.
    # Validation output is None because none of the metrics could be calculated due to memory overflow.
    VALIDATION_OUTPUT = {'log': {}, 'fitness_sm2': {}, 'precision_sm2': {}, 'fscore_sm2': {}, 'size_sm2': {}, 'cfc_sm2': {}, 'pnsize_sm2': {}, 'exectime_sm2': {}, 'benchtime_sm2': {}}
    clear_output()
    results = BenchmarkTest(params=INPUT_PARAMS).results.round(2)
    results = results.sort_values(by='log', ascending=False).reset_index(drop=True)

    assert results.to_dict() == VALIDATION_OUTPUT

def test_benchmark_overall_memory():
    # Test with a very low memory limit to force failure
    INPUT_PARAMS = {
        'benchmark_test': 'discovery',
        'input_path': 'data/test/benchmark_test',
        'output_path': 'output',
        'miners': ['sm2'],
        'system_params': {'overall_max_memory': 1 * 1024**2}  # 1MB limit (too low, forcing failure)
    }

    with patch("shaining.benchmark.check_memory_limit", side_effect=[True, True, False]) as mock_memory_check, \
         patch("shaining.benchmark.multiprocessing.Pool") as mock_pool:

        # Mock `map_async().ready()` to simulate running until memory exceeds
        mock_pool.return_value.__enter__.return_value.map_async.return_value.ready.side_effect = [False, False, True]
        mock_pool.return_value.__enter__.return_value.map_async.return_value.get.return_value = []

        results = BenchmarkTest(params=INPUT_PARAMS).results.round(2)

        # Ensure it returns an empty DataFrame (since memory was exceeded)
        assert isinstance(results, pd.DataFrame)
        assert len(results) == 0

        # Ensure memory check was called multiple times (indicating restart attempts)
        assert mock_memory_check.call_count >= 2, f"Expected at least 2 calls, got {mock_memory_check.call_count}"

        # Ensure the function restarted twice before proceeding
        assert mock_pool.call_count == 3

        # Ensure pool was properly closed and cleaned up
        assert mock_pool.return_value.__exit__.call_count == 3
