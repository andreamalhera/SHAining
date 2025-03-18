import config
import pandas as pd
import warnings
import os
from datetime import datetime as dt
from shaining.benchmark import BenchmarkTest
from shaining.utils.param_keys import *
from shaining.generator import GenerateLogs
from shaining.shaining import SHAiningTask
from shaining.shapley import ShapleyTask
from shaining.coalition import Coalitions
from utils.default_argparse import ArgParser

warnings.filterwarnings("ignore")

def run(kwargs:dict, model_paramas_list: list, filename_list:list):
    """
    This function chooses the running option for the program.
    @param kwargs: dict
        contains the running parameters and the event-log file information
    @param model_params_list: list
        contains a list of model parameters, which are used to analyse this different models.
    @return:
    """
    params = kwargs[PARAMS]
    gen = pd.DataFrame(columns=['log'])

    for model_params in model_params_list:
        if model_params.get(PIPELINE_STEP) == 'event_logs_generation':
            gen = GenerateLogs(model_params)
            gen.run()
        elif model_params.get(PIPELINE_STEP) == 'benchmark_test':
            benchmark = BenchmarkTest(model_params, event_logs=gen['log'])
            # BenchmarkPlotter(benchmark.features, output_path="output/plots")
        elif model_params.get(PIPELINE_STEP) == 'shapley_computation':
            shapley = ShapleyTask(model_params)

        elif model_params.get(PIPELINE_STEP) == 'feature_extraction':
            ft = EventLogFeatures(**kwargs, logs=gen['log'], ft_params=model_params)
            FeaturesPlotter(ft.feat, model_params)
        elif model_params.get(PIPELINE_STEP) == 'coalition':
            coa = Coalitions(model_params)
        elif model_params.get(PIPELINE_STEP) == 'shaining_task':
            shaining_results = SHAiningTask(model_params)
        elif model_params.get(PIPELINE_STEP) == "evaluation_plotter":
            GenerationPlotter(gen, model_params, output_path=model_params['output_path'], input_path=model_params['input_path'])

if __name__=='__main__':
    start_tag = dt.now()
    print(f'INFO: SHAMPU starting {start_tag}')
    arg_parser = ArgParser()
    args = arg_parser.parse('SHAMPU main')

    model_params_list = config.get_model_params_list(args.alg_params_json)
    # print(model_params_list)
    run({'params':""}, model_params_list, [])

    print(f'SUCCESS: SHAMPU took {dt.now()-start_tag} sec.')
