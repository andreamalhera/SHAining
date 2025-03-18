import json
import multiprocessing
import os
import pandas as pd
import psutil
import subprocess
import random
import numpy as np

from datetime import datetime as dt
from datetime import timedelta
from functools import partial, partialmethod
from pathlib import Path
from pm4py import read_xes, convert_to_bpmn, read_bpmn, convert_to_petri_net, check_soundness
from pm4py import discover_petri_net_inductive, discover_petri_net_ilp, discover_petri_net_heuristics
from pm4py import fitness_alignments, fitness_token_based_replay
from pm4py import precision_alignments, precision_token_based_replay
from pm4py.algo.evaluation.generalization import algorithm as generalization_evaluator
from pm4py.algo.evaluation.simplicity import algorithm as simplicity_evaluator
from pm4py.objects.bpmn.obj import BPMN
from pm4py.objects.log.importer.xes import importer as xes_importer
from shaining.pruning import prune_ELnames_per_level
from shaining.utils.io_helpers import dump_features_json, _results_already_exists
from shaining.utils.merge_csvs import merge_csvs
from shaining.utils.param_keys import INPUT_PATH, OUTPUT_PATH, EVENTLOGS_SELECTION, SYSTEM_PARAMS
from shaining.utils.param_keys import SIMILARITY_THRESHOLD, FEAT_PATH, TARGET_SIMILARITY
from shaining.utils.param_keys.benchmark import MINERS, TIMEOUT, CURRENT_LEVEL, MAX_MEMORY, OVERALL_MAX_MEMORY
from shaining.utils.param_keys.generator import LOG
from tqdm import tqdm
from func_timeout import func_timeout, FunctionTimedOut
import psutil
import time
import threading

MEMORY_THRESHOLD = 18*1024**3#68719476736#64GB #137438953472 #128GB in bytes
OVERALL_RAM_LIMIT = 200*1024**3#200GB #137438953472 #128GB in bytes
TIME_THRESHOLD = 5*60#2 min *60# 2 hours 43200 #12 hours
RANDOM_SEED = 10
OVERALL_TIME_THRESHOLD = 3*24*60*60 #24 hours
random.seed(RANDOM_SEED)

#TODO: Instead of killing the whole process. Think of alternatives. Kill oldest subprocess when memory exceeds
def check_memory_limit(overall_ram_limit=OVERALL_RAM_LIMIT):
    """Check if memory usage exceeds the limit."""
    mem = psutil.virtual_memory()
    if mem.used >= overall_ram_limit:
        excess_memory = mem.used - overall_ram_limit
        print(f"    ERROR: Memory usage exceeded the specified limit by {excess_memory / (1024**3):.2f} GB,"
             +f"limit is {overall_ram_limit / (1024**3):.2f} GB")
        return True
    return False

class MemoryMonitor:
    def __init__(self, threshold=MEMORY_THRESHOLD):
        self.threshold = threshold
        self.process = psutil.Process(os.getpid())
        self.running = True
        self.exceeded = False

    def monitor(self):
        """Continuously checks memory usage in a separate thread."""
        while self.running:
            mem = self.process.memory_info().rss
            if mem > self.threshold:
                print(f"    FAILED: Memory exceeded: {mem} > {self.threshold}")
                self.exceeded = True
                self.running = False  # Stop monitoring
                break
            time.sleep(1)  # Check memory every second

    def start_monitoring(self):
        """Starts the monitoring thread."""
        self.monitor_thread = threading.Thread(target=self.monitor, daemon=True)
        self.monitor_thread.start()

    def stop_monitoring(self):
        """Stops the monitoring process."""
        self.running = False
        self.monitor_thread.join()

def get_dump_path(output_prefix, input_dir, miners=[]):
    dump_path= os.path.join(output_prefix,
                            os.path.join(*os.path.normpath(input_dir).split(os.path.sep)[1:]))
    if dump_path.endswith(".xes"):
        event_log = os.path.split(dump_path)[-1]
        dump_path = os.path.split(dump_path)[0]
    else:
        if len(miners) >= 1:
            dump_path = os.path.join(dump_path, "benchmark")
    if len(miners) == 1:
        dump_path = dump_path + f"_{miners[0]}"
    return dump_path

class BenchmarkTest:
    def __init__(self, params=None, event_logs=None):
        params[SYSTEM_PARAMS] = params.get(SYSTEM_PARAMS) if params.get(SYSTEM_PARAMS) else {}
        params[EVENTLOGS_SELECTION] = params.get(EVENTLOGS_SELECTION) if params.get(EVENTLOGS_SELECTION) else {}
        self.timeout = params.get(SYSTEM_PARAMS).get(TIMEOUT) if params.get(SYSTEM_PARAMS).get(TIMEOUT) else TIME_THRESHOLD #43200 # Default timeout 12 hours per mining one EL
        self.memory_threshold = params.get(SYSTEM_PARAMS).get(MAX_MEMORY) if params.get(SYSTEM_PARAMS).get(MAX_MEMORY) else MEMORY_THRESHOLD
        self.overall_memory_threshold = params.get(SYSTEM_PARAMS).get(OVERALL_MAX_MEMORY) if params.get(SYSTEM_PARAMS).get(OVERALL_MAX_MEMORY) else OVERALL_RAM_LIMIT

        if event_logs is None or len(event_logs) == 0:
            log_path = params[INPUT_PATH]
            if log_path.endswith(".xes"):
                event_logs = [""]
            else:
                files_in_path = os.listdir(log_path)
                event_logs = sorted([filename for filename in files_in_path if filename.endswith(".xes")])

        if params != None:
            self.params = params
        self.results = pd.DataFrame(columns=['log'])

        self.run(params, event_logs)

    def prune_eventlog_selection(self, files_in_path: list, dump_path: str, feat_path: str, similarity_threshold: float, current_level=1):
        all_feats = pd.read_csv(feat_path)
        filtered_feats = all_feats[all_feats[TARGET_SIMILARITY] >= float(similarity_threshold)]
        filtered_ELnames = prune_ELnames_per_level(filtered_feats[LOG].to_list(), current_level)
        filtered_feats = [log+'.xes' for log in filtered_ELnames]

        files_in_path = set(files_in_path)&set(filtered_feats)
        already_computed = [filename.replace(".json", ".xes")
                            for filename in os.listdir(os.path.join(dump_path,''))] if os.path.isdir(dump_path) else []
        pending_files = set(files_in_path)-set(already_computed)
        #TODO: Fix numbers in print.
        print(f"STARTINFO: {len(files_in_path)}/{len(filtered_feats)} ELs in disk, searching for stored benchmark results. "
                +f"{min(len(already_computed), len(filtered_feats))}/{max(len(files_in_path), len(filtered_feats))} ELs in {dump_path} already benchmarked: {already_computed[:5]}...{already_computed[-5:]}")
        if len(pending_files) > 0:
            pending_files = list(pending_files)
            print(f"    INFO: {len(pending_files)}/{len(files_in_path)} ELs will be benchmarked: {pending_files[:5]}...{pending_files[-5:]}")
        return pending_files

    def get_expected_results_path(self, input_path:str, output_path:str, miners:list):
        benchmark_suffix = "benchmark_"+ miners[0] if len(miners) == 1 else "benchmark"
        return os.path.join(output_path,
                            os.path.split(input_path)[-1].replace(".xes",""),
                            benchmark_suffix+'.csv')

    def _miner_complete_bechmark_in_disk(self, event_logs:list, expected_results_path:str, miner:str):
        event_logs = [filename.rsplit(".xes")[0] for filename in event_logs]
        if _results_already_exists(expected_results_path, event_logs):
                print(f"SUCCESS: All {len(event_logs)} ELs were already benchmarked for {miner} and stored in {expected_results_path}. Skipping.")
                return True
        return False

    def run_benchmark(self, event_logs, miner, benchpath_suffix):
        while True:
            try:
                start = dt.now()

                if not event_logs:
                    print("ERROR: No event logs to process. Exiting benchmark.")
                    return None

                num_cores = multiprocessing.cpu_count() if len(event_logs) >= multiprocessing.cpu_count() else len(event_logs)

                print(f"INFO: Benchmark starting with {miner} at {start.strftime('%H:%M:%S')} using {num_cores} cores for {len(event_logs)} files...")

                # Multiprocessing per MINER. Uncomment next line for debugging.
                #self.benchmark_wrapper(event_logs[0], miner=miner)#TESTING
                with multiprocessing.Pool(num_cores) as p:
                    random.seed(RANDOM_SEED)
                    results = p.map_async(partial(self.benchmark_wrapper, miner=miner), event_logs)
                    while not results.ready():
                        if check_memory_limit(self.overall_memory_threshold):
                            print(f"    ERROR: Memory usage exceeded {self.overall_memory_threshold/1024} KB. Terminating benchmark process.")
                            p.terminate()
                            p.join()
                            raise MemoryError("    WARNING: Benchmarking process stopped due to HIGH MEMORY USAGE.")
                        time.sleep(10)

                    completed_results = results.get(OVERALL_TIME_THRESHOLD)
                    break

            except multiprocessing.TimeoutError:
                print(f"ERROR: Benchmarking exceeded {OVERALL_TIME_THRESHOLD} hours and was terminated.")
                self.results = pd.DataFrame(columns=['log'])
                break

            except MemoryError as e:
                print(str(e))
                print(f"WARNING: OVERALL RAM EXCEEDED {OVERALL_RAM_LIMIT/1024} KB. Restarting benchmark process for {miner}.")

                _ , new_pending_ELs = self.get_pending_ELs_json(self.params, miner, event_logs)

                event_logs = new_pending_ELs if new_pending_ELs else event_logs
                if event_logs is None or not event_logs or len(event_logs) == 0:
                    print(f"DEBUG: No pending event logs found for miner {miner}. Exiting.")
                    break
            finally:
                # Ensure multiprocessing pool is always cleaned up
                if 'p' in locals():
                    p.close()
                    p.join()

        # Aggregates metafeatures in saved Jsons into dataframe
        path_to_json = os.path.join(self.params[OUTPUT_PATH],
                                    *Path(self.root_path).parts[1:],
                                    benchpath_suffix)
        os.makedirs(path_to_json, exist_ok=True)
        if path_to_json.endswith(".xes"):
            path_to_json = path_to_json.rsplit("/", 1)[0]

        return path_to_json

    def append_unique_rows(self, benchmark_results, file_path):
        os.makedirs(os.path.split(file_path)[0], exist_ok=True)
        benchmark_results.to_csv(file_path, mode='a',
                                 header=not os.path.exists(file_path),
                                 index=False)
        try:
            df = pd.read_csv(file_path)
            df = df.dropna()
            df = df.drop_duplicates()
            df.to_csv(file_path, index=None)
        except pd.errors.EmptyDataError as e:
            print(f"    ERROR: Benchmark could not be computed due to overall memory ({OVERALL_RAM_LIMIT}/1024) KB or timeout limit ({TIME_THRESHOLD} sec)", str(e))
            df=pd.DataFrame(columns=['log'])
            pass
        return file_path, df

    def get_pending_ELs_json(self, params, miner, files_in_path):
        dump_path = get_dump_path(params[OUTPUT_PATH], params[INPUT_PATH], miners=[miner])
        if params.get(EVENTLOGS_SELECTION) is not None and len(params.get(EVENTLOGS_SELECTION)) > 0:
            event_logs = self.prune_eventlog_selection(files_in_path, dump_path, params.get(EVENTLOGS_SELECTION).get(FEAT_PATH),
                                                        params.get(EVENTLOGS_SELECTION).get(SIMILARITY_THRESHOLD),
                                                        params.get(EVENTLOGS_SELECTION).get(CURRENT_LEVEL))
            event_logs =  event_logs if event_logs is not None else []
            return dump_path, event_logs
        return dump_path, []

    def get_pending_ELs_csv(self, params, miner, event_logs):
        expected_results_path = self.get_expected_results_path(params[INPUT_PATH], params[OUTPUT_PATH], [miner])
        if len(event_logs) > 0:
            miner_tasks = event_logs.copy()
            for filename in event_logs:
                if self._miner_complete_bechmark_in_disk([filename], expected_results_path, miner):
                    print(f"    INFO: Found {filename} for {miner} stored in {expected_results_path}. Skipping.")
                    miner_tasks.remove(filename)
            event_logs = miner_tasks
        return event_logs

    def run(self, params, files_in_path):
        self.current_level = self.params.get(EVENTLOGS_SELECTION).get(CURRENT_LEVEL)
        print(f"=========================== BenchmarkTest {self.current_level}  ==========================")
        print(f"INFO: Running BenchmarkTest with {params} and per-miner timeout {self.timeout} seconds, memory limit per single run {self.memory_threshold} bytes and overall memory limit of {self.overall_memory_threshold} bytes. To change open shaining/benckmark.py")
        start = dt.now()
        self.num_tasks = {}

        for miner in params[MINERS]:
            event_logs = [filename.rsplit(".xes")[0] for filename in files_in_path if filename.endswith(".xes")]

            if params.get(EVENTLOGS_SELECTION):# Only a selected subset is of interest
                dump_path, event_logs = self.get_pending_ELs_json(params, miner, files_in_path)

            expected_results_path = self.get_expected_results_path(params[INPUT_PATH], params[OUTPUT_PATH], [miner])

            if len(event_logs) > 0 and self._miner_complete_bechmark_in_disk(event_logs, expected_results_path, miner):
                self.results = pd.read_csv(expected_results_path)
                event_logs = []
                self.filepath = expected_results_path

            filtered_ELs = self.get_pending_ELs_csv(params, miner, event_logs)
            event_logs = filtered_ELs if filtered_ELs else event_logs
            self.num_tasks[miner] = len(event_logs)
            self.root_path = params[INPUT_PATH]
            benchpath_suffix = "benchmark_" + miner

            if self.num_tasks[miner] > 0:
                path_to_json = self.run_benchmark(event_logs, miner, benchpath_suffix)
            else:
                path_to_json = dump_path

            df = pd.DataFrame(columns=['log'])

            if os.path.exists(path_to_json):
                # Iterate over the files in the directory
                for filename in os.listdir(path_to_json):
                    if filename.endswith('.json'):
                        i_path = os.path.join(path_to_json, filename)
                        try:
                            with open(i_path) as f:
                                data = json.load(f)
                                #data = dict(sorted(data.items()))
                                temp_df = pd.DataFrame([data])
                                df = pd.concat([df, temp_df], ignore_index=True)
                        except pandas.errors.EmptyDataError as e:
                            print(f"WARNING: Problem with empty file: ", i_path, " ", e, "Removing file and continuing to recompute benchmark.")
                            os.remove(i_path)
                            continue
                benchmark_results = df

                self.filename = os.path.join(os.path.split(self.root_path)[-1].replace(".xes",""),
                                                benchpath_suffix) +'.csv'
                self.output_path = self.params[OUTPUT_PATH]
                self.filepath = os.path.join(self.output_path, self.filename)
                self.filepath, self.results= self.append_unique_rows(benchmark_results, self.filepath)

                print(f"    SUCCESS: Saved {len(self.results)} event-logs with {len(self.results.columns)} "
                +f"in {self.filepath}.")

        merged_csv_path = self.filepath
        if len(params[MINERS])>1:
            merged_csv_path = self.filepath.rsplit("_",1)[0]+".csv"
            self.filename = os.path.join(*Path(merged_csv_path).parts[1:])
            self.results = merge_csvs(self.filepath.rsplit("_",1)[0], merged_csv_path)
        self.filepath = merged_csv_path
        print(f"SUCCESS: BenchmarkTest took {dt.now()-start} sec for {len(params[MINERS])} miner(s)"+\
              f" {params[MINERS]} and {len(self.results)} event-logs. Saved benchmark to {merged_csv_path}.")
        print(f"========================= ~ BenchmarkTest {self.current_level} ==========================")

    def set_log_name(self, event_log, log_counter):
        # TODO: Low priority. Use iteratevely generated name for log name in dataframe for passed unnamed logs instead of whole log. E.g. genEL_1, genEL_2,...
        if isinstance(event_log, str):
            log_name = event_log.replace(".xes", "")
            results = {LOG: log_name}
        else:
            log_name = "genEL_"+str(log_counter)
            results = {"log": event_log}
            #results = {"log": log_name}
        return log_name,results

    def benchmark_wrapper(self, event_log="test", miner='inductive', log_counter=0):
        random.seed(RANDOM_SEED)
        dump_path = get_dump_path(self.params[OUTPUT_PATH], self.params[INPUT_PATH])
        benchmark_results = pd.DataFrame()
        log_name, results = self.set_log_name(event_log, log_counter)

        benchmark_type = "benchmark_"+miner
        miner_cols = [f"fitness_{miner}",
                        f"precision_{miner}",
                        f"fscore_{miner}",
                        f"size_{miner}",
                        f"cfc_{miner}",
                        f"pnsize_{miner}"]# f"generalization_{miner}",f"simplicity_{miner}"]
        start_miner = dt.now()

        try:
            # Applying timeout to the benchmark_discovery call
            benchmark_results = func_timeout(self.timeout, self.benchmark_discovery_with_memory, args=(results[LOG], miner, self.params))
        except FunctionTimedOut:
            print(f"    TIMEOUT: for miner {miner} on log {event_log} after {self.timeout} seconds.")
            benchmark_results = None
        except Exception as e:
            print(f"    ERROR: In miner {miner} on log {event_log}: {e}")
            benchmark_results = None

        if benchmark_results is None:
            # Set default values if timeout or error occurred
            results[f"fitness_{miner}"] = None
            results[f"precision_{miner}"] = None
            results[f"fscore_{miner}"] = None
            results[f"size_{miner}"] = None
            results[f"cfc_{miner}"] = None
            results[f"pnsize_{miner}"] = None
            results[f"exectime_{miner}"] = None
            results[f"benchtime_{miner}"] = None
        else:
            results[f"fitness_{miner}"] = benchmark_results[0]
            results[f"precision_{miner}"] = benchmark_results[1]
            if (benchmark_results[0] is None) or (benchmark_results[1] is None):
                f_score_temp = None
            elif (benchmark_results[0] + benchmark_results[1]) == 0:
                f_score_temp = 0
            else:
                f_score_temp = 2*(benchmark_results[0]*benchmark_results[1]/(benchmark_results[0]+ benchmark_results[1]))
            results[f"fscore_{miner}"] = f_score_temp
            results[f"size_{miner}"]=benchmark_results[2]
            results[f"cfc_{miner}"]=benchmark_results[3]
            results[f"pnsize_{miner}"]=benchmark_results[4]
            results[f"exectime_{miner}"]=benchmark_results[5]
            results[f"benchtime_{miner}"]=benchmark_results[6]

        dump_features_json(results, dump_path, log_name, content_type=benchmark_type)
        print(f"    SUCCESS: Miner {miner} for EL {log_name}... took {dt.now()-start_miner} sec.")
        results.clear()
        return

    def split_miner_wrapper(self, log_path="data/real_event_logs/BPI_Challenges/BPI_Challenge_2012.xes", version=1.0):
        random.seed(RANDOM_SEED)
        os.environ["DISPLAY"] = ":99" # For running on github CI
        filename = os.path.split(log_path)[-1].rsplit(".",1)[0]
        bpmn_path = os.path.join("output", "bpmns_split", filename)
        os.makedirs(os.path.split(bpmn_path)[0], exist_ok=True)
        if version==2.0:
            command = [
                    "java",
                    "-cp",
                    f"{os.getcwd()}/miners/split-miner-2.0/sm2.jar{os.pathsep}{os.getcwd()}/miners/split-miner-2.0/lib/*",
                    "au.edu.unimelb.services.ServiceProvider",
                    "SM2",
                    f"{os.getcwd()}/{log_path}",
                    f"{os.getcwd()}/{bpmn_path}",
                    "0.05"
                    ]
        elif version==1.0:
            command = [
                "java",
                "-cp",
                f"{os.getcwd()}/miners/splitminer/splitminer.jar{os.pathsep}{os.getcwd()}/miners/splitminer/lib/*",
                "au.edu.unimelb.services.ServiceProvider",
                "SMD",
                "0.1",
                "0.4",
                "false",
                f"{os.getcwd()}/{log_path}",
                f"{os.getcwd()}/{bpmn_path}",
            ]

        print("        COMMAND", " ".join(command))
        output = subprocess.run(
            command,
            capture_output=True,
            text=True,
        )
        try:
            if "\nERROR:" in output.stdout:
                print(f"        FAILED: SplitMiner v{version} could not create BPMN for", log_path)
                print("            SplitMiner:", output.stderr)
                return None
            return read_bpmn(bpmn_path+'.bpmn')
        except ValueError:
            print(output.stdout)


    def benchmark_discovery_with_memory(self, log, miner, params=None):
        total_memory = self.memory_threshold  # Replace this with an appropriate threshold
        monitor = MemoryMonitor(threshold=total_memory)
        monitor.start_monitoring()

        result = self.benchmark_discovery(log, miner, params)  # Execute the main function

        monitor.stop_monitoring()  # Stop monitoring once function completes

        if monitor.exceeded:
            return None, None, None, None, None, None, None  # If memory exceeded, return failure

        return result  # Return normal results if memory was within limits


    def benchmark_discovery(self, log, miner, params=None):
        """
        Runs discovery algorithms on a specific log and returns their performance.

        :param str/EventLog log: log from pipeline step before or string to .xes file.
        :param str miner: Specifies process discovery miner to be run on log.
        :param Dict params: Params from config file

        """
        #print("Running benchmark_discovery with", self, log, miner, params)
        random.seed(RANDOM_SEED)
        NOISE_THRESHOLD = 0.2
        miner_params=''
        tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)
        start_bench = dt.now()
        start_bench = start_bench-timedelta(days=0)

        if type(log) is str:
            if params[INPUT_PATH].endswith('.xes'):
                log_path = params[INPUT_PATH]
            else:
                log_path = os.path.join(params[INPUT_PATH], log+".xes")
            success_msg = f"    SUCCESS: Benchmarking event-log {log} with {miner} took "# {dt.now()-start_bench} sec."
            try:
                log = xes_importer.apply(f"{log_path}", parameters={"show_progress_bar": False})
            except FileNotFoundError:
                raise FileNotFoundError(f"        FAILED: Cannot find {log_path}" )
        else:
            log=log
            success_msg = f"    SUCCESS: Benchmarking one event-log with {miner} took "# {dt.now()-start_bench} sec."
        if miner == 'sm1':
            bpmn_graph = self.split_miner_wrapper(log_path, version=1.0)
            if bpmn_graph is None:
                return None
            '''TESTING
            from pm4py.visualization.bpmn.visualizer import apply as get_bpmn_fig
            from pm4py.visualization.bpmn.visualizer import matplotlib_view as view_bpmn_fig
            bpmn_fig = get_bpmn_fig(bpmn_graph)
            view_bpmn_fig(bpmn_fig)
            '''
            net, im, fm = convert_to_petri_net(bpmn_graph)
        elif miner == 'sm2':
            bpmn_graph = self.split_miner_wrapper(log_path, version=2.0)
            if bpmn_graph is None:
                return None
            net, im, fm = convert_to_petri_net(bpmn_graph)
        else:
            if miner == 'imf':
                miner = 'inductive'
                miner_params = f', noise_threshold={NOISE_THRESHOLD}'
            net, im, fm = eval(f"discover_petri_net_{miner}(log {miner_params})")
            bpmn_graph = convert_to_bpmn(net, im, fm)
        now = dt.now()
        time_m = round((now-start_bench).total_seconds(),2)
        try:
            fitness = fitness_alignments(log, net, im, fm)['log_fitness']
            precision = precision_alignments(log, net, im, fm)
        except Exception: 
            print(f"    ERROR: Alignment for {log_path}")
            fitness = -1
            precision = -1
        pn_size = len(net._PetriNet__places)
        size = len(bpmn_graph._BPMN__nodes)
        cfc = sum([isinstance(node, BPMN.ExclusiveGateway) for node in bpmn_graph._BPMN__nodes])
        #generalization = generalization_evaluator.apply(log, net, im, fm)
        #simplicity = simplicity_evaluator.apply(net)
        now = dt.now()
        metric_time = round((now-start_bench).total_seconds(),2)
        print(success_msg + f"{now-start_bench} sec. Miner time {time_m} sec. Complete computation time including metrics {metric_time} sec.")
        return fitness, precision, size, cfc, pn_size, time_m, metric_time  #, generalization, simplicity
