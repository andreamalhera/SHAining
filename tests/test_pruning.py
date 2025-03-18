import pandas as pd
from shaining.pruning import prune_ELnames_per_level

def test_prune_ELnames_per_level():
    data = {
        "activities_q1": [54.5, 89.0, 315.0, None, 89.0],
        "eventropy_k_block_ratio_3": [None, None, None, None, None],
        "log": [
            "genEL10000_nan_nan_nan_nan_758_31382_nan_nan",
            "genEL10001_6413_nan_nan_nan_758_nan_-097_nan",
            "genEL10002_6413_nan_nan_nan_nan_nan_081_nan",
            "genEL10002_nan_nan_433_nan_nan_17479_97_nan",
            "genEL10003_6413_nan_nan_nan_758_nan_259_nan",
        ],
        "n_unique_start_activities": [None, None, None, 4.0, None],
        "ratio_top_5_variants": [None, None, None, None, None],
        "skewness_variant_occurrence": [1.699144, 9.054366, 11.753222, None, 9.054366],
        "start_activities_q1": [43.50, None, None, 70.75, None],
        "target_similarity": [0.968350, 0.992081, 0.976671, 0.999359, 0.991997],
        "trace_len_kurtosis_hist": [None, -1.833333, -1.833333, -1.833333, -1.833333],
        "trace_len_variance": [None, None, None, None, None],
    }
    EXPECTED_OUTPUT = [
            "genEL10000_nan_nan_nan_nan_758_31382_nan_nan",
            "genEL10002_6413_nan_nan_nan_nan_nan_081_nan",
        ]

    df = pd.DataFrame(data)
    result = prune_ELnames_per_level(df['log'].to_list(), 2)
    assert len(result) == len(EXPECTED_OUTPUT)
    assert result == EXPECTED_OUTPUT

