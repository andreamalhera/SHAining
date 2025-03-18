from shaining.coalition import Coalitions

def test_coalitions():
    INPUT_PARAMS =     {
      "pipeline_step": "coalition",
      "log_names":["genEL5_9984_nan_nan","genEL6_nan_119_nan","genEL7_nan_nan_10"],
      "level": 2,
      "feature_names": ["n_traces", "ratio_variants_per_number_of_traces", "trace_len_min"],
      "experiment_list" : [
                        { "n_traces": 99.0 },
                        { "n_traces": 998.4 },
                        { "ratio_variants_per_number_of_traces": 0.99 },
                        { "ratio_variants_per_number_of_traces": 1.19 },
                        { "trace_len_min": 1.0 },
                        { "trace_len_min": 1.2 }
                      ]
    }
    
    VALIDATION_RESULT = [{'n_traces': 998.4, 'trace_len_min': 1.0}, {'ratio_variants_per_number_of_traces': 1.19, 'trace_len_min': 1.0}, {'n_traces': 998.4, 'ratio_variants_per_number_of_traces': 1.19}]
    
    # Calculating the coalitions
    coalitions = Coalitions(params=INPUT_PARAMS)
    next_level_targets = coalitions.next_level_targets

    assert sorted(next_level_targets, key=lambda d: sorted(d.items())) == sorted(VALIDATION_RESULT, key=lambda d: sorted(d.items()))


    