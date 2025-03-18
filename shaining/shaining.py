import pandas as pd

from datetime import datetime as dt
from shaining.lattice import Lattice
from shaining.benchmark import BenchmarkTest
from shaining.shapley import ShapleyTask
from shaining.utils.param_keys import OUTPUT_PATH, INPUT_PATH, SYSTEM_PARAMS
from shaining.utils.param_keys.benchmark import MINERS
from shaining.utils.param_keys.coalition import FEATURE_NAMES, FEATURE_VALUES, CONTESTANTS
from shaining.utils.param_keys.generator import LOG

# TODO: Merge CONTESTANTS and MINERS in params
class SHAiningTask():
    #def __init__(self, feature_names: list, values: list, contestants: list, system_params: dict=None) -> None:
    def __init__(self, params: dict=None) -> None:
        print("=========================== SHAining ==========================")
        feature_names = params.get(FEATURE_NAMES) if params.get(FEATURE_NAMES) else None
        values = params.get(FEATURE_VALUES) if params.get(FEATURE_VALUES) else None
        contestants = params.get(CONTESTANTS) if params.get(CONTESTANTS) else None
        system_params = params.get(SYSTEM_PARAMS) if params.get(SYSTEM_PARAMS) else None
        if feature_names is None or values is None:
            raise ValueError("feature_names and values must not be None")
        if len(feature_names) != len(values):
            raise ValueError("feature_names and values must have the same length")
        if system_params is None:
            raise ValueError("system_params must not be None")
        self.contestants = contestants

        start = dt.now()
        lattice = Lattice(feature_names, values, system_params)
        potentials = lattice.describe_potential()
        benchmark_path = lattice.form(contestants)
        self.shapley_results = self.shapley_wrapper(benchmark_path, feature_names, contestants,
                                                    system_params.get(OUTPUT_PATH), potentials)
        print(f"SUCCESS: SHAining a light into {len(feature_names)} took {dt.now()-start} sec.")
              #+f"Generated {len(self.generated_features)} event log(s) and {len(shapley_experiments)}.")
        #print(f"         Saved generated logs in {self.output_path}")
        print("========================= ~ SHAining  ==========================")

    def shapley_wrapper(self, benchmark_path, feature_names, contestents, output_path, potentials):
        # TODO: Move calculation of potentials to shapley.py
        print("INFO: Reminders of potentials:", potentials)
        input_path = benchmark_path
        shapley = ShapleyTask({INPUT_PATH: input_path,
                               OUTPUT_PATH: output_path,
                               FEATURE_NAMES: feature_names,
                               MINERS: contestents})
        shapley_results = shapley.results
        return shapley_results
