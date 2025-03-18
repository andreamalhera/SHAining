import os
from shaining.utils.io_helpers import _results_already_exists

def test_results_already_exists():
    TEST_PATH = os.path.join('data', 'test','benchmark_test_feat.csv')
    TEST_LOGNAMES = ['genEL1_02_04', 'BPIC2013_open_problems']
    assert _results_already_exists(TEST_PATH, TEST_LOGNAMES)

    TEST_LOGNAMES.append('genEL10_nan_099_564')
    assert not _results_already_exists(TEST_PATH, TEST_LOGNAMES)