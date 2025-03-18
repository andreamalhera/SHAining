import pandas as pd
import pytest
import os

from shaining.generator import GenerateLogs
from shaining.utils.io_helpers import delete_list_of_dirs

def test_GenerateEventLogs():
    INPUT_PARAMS =  {"pipeline_step": "event_logs_generation",
                        "output_path": "data/test/generation/3_complete",
                        "generator_params": {
                        "config_space": { "mode": [ 5, 20], "sequence": [ 0.01, 1], "choice": [ 0.01, 1],
                                            "parallel": [ 0.01, 1], "loop": [ 0.01, 1], "silent": [ 0.01, 1],
                                            "lt_dependency": [ 0.01, 1], "num_traces": [ 10, 100],
                                            "duplicate": [ 0], "or": [ 0]},
                                            "n_trials": 50,
                        "experiment": [
                        { "n_traces": 99.0 },
                        { "n_traces": 998.4 },
                        { "ratio_variants_per_number_of_traces": 0.99 },
                        { "ratio_variants_per_number_of_traces": 1.19 },
                        { "trace_len_min": 1.0 },
                        { "trace_len_min": 1.2 },
                        ]}
                    }
    VALIDATION_OUTPUT = [1.0, 1.0, 0.99, 0.81, 1.0, 0.99]
    VALIDATION_LOGS = ['genEL1_990_nan_nan', 'genEL2_9984_nan_nan', 'genEL3_nan_099_nan', 'genEL4_nan_119_nan', 'genEL5_nan_nan_10', 'genEL6_nan_nan_12']
    OUTPUT_PATH = os.path.join('data', 'test','generation', '3_complete', '3_nt_rvpnot_tlm_features.csv')

    delete_list_of_dirs([os.path.join('data', 'test','generation', '3_complete'),
                         os.path.join('data', 'generated', '3_nt_rvpnot_tlm'),
                         os.path.join('data', 'features', 'generated', '3_nt_rvpnot_tlm'),
                         ])
    genED = GenerateLogs(INPUT_PARAMS)
    genED.run()
    output = pd.read_csv(OUTPUT_PATH, index_col=None)
    #print(output)
    target_similarities = [round(similarity,2) for similarity in output.sort_values('log')['target_similarity'].to_list()]
    target_logs = [log for log in output.sort_values('log')['log'].to_list()]

    assert len(target_similarities) == len(VALIDATION_OUTPUT)

    # Check if results are similar enough, since they vary between machines
    AGREEMENT_THRESHOLD = 0.75
    agreement = sum(1 for o, u in zip(target_similarities, VALIDATION_OUTPUT) if o == u) / len(VALIDATION_OUTPUT) * 100
    assert agreement >= AGREEMENT_THRESHOLD
    assert len(target_logs) == len(VALIDATION_LOGS)
    assert target_logs == VALIDATION_LOGS


