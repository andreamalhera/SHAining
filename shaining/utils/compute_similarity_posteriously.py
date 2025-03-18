import argparse
import glob
import json
import multiprocessing
import numpy as np
import pandas as pd
import os
import shutil
import sys

from functools import partial
from gedi.utils.io_helpers import compute_similarity, dump_features_json
from gedi.features import compute_features_from_event_data
from lxml import etree
from pathlib import Path
from pm4py.objects.log.obj import EventLog
from pm4py.objects.log.importer.xes import importer as xes_importer

#current_dir = os.path.dirname(os.path.abspath(__file__))
#parent_dir = os.path.dirname(current_dir)
#sys.path.insert(0, parent_dir)

from shaining.utils.io_helpers import get_output_key_value_location
from shaining.utils.param_keys.generator import GENERATOR_PARAMS, EXPERIMENT
from shaining.utils.merge_jsons import json_to_csv

"""
Run using
python compute_similarity_posteriously.py path_to_gen_ELs path_to_orig_config_file.json SIMILARITY_THRESHOLD

python -m shaining.utils.compute_similarity_posteriously path_to_gen_ELs path_to_orig_config_file.json SIMILARITY_THRESHOLD
E.g.
python -m shaining.utils.compute_similarity_posteriously data/generated/3_nt_rvpnot_tlm config_files/test/generation.json 0.7
"""
def get_targets(config_file: str):
    with open(config_file) as f:
        config = json.load(f)
    targets = config[0].get(GENERATOR_PARAMS).get(EXPERIMENT)
    return targets

def name_genEl_path(genEL_dir: str, target: dict, target_keys: list):
    for target_key in target_keys:
        target[target_key] = target[target_key] if target_key in target.keys() else np.nan
    genEL_path = get_output_key_value_location(target,
                                                os.path.split(genEL_dir)[0],
                                                identifier="genEL")+'.xes'
    if os.path.split(genEL_path)[0] == genEL_dir:
        return genEL_path
    else:
        raise ValueError(f"ERROR: Target keys don't match keys in generated ELs path: {target_keys}."+
                         f"Expected generated ELs directory: '{os.path.split(genEL_path)[0]}'. Got '{genEL_dir}' instead.")

def read_xes_safe(file_name):
    try:
        return xes_importer.apply(file_name)  # Load the XES file normally
    except etree.XMLSyntaxError as e:
        print(f"XML Syntax Error in {file_name}: {e}")
        return {"error": str(e)}  # Convert to string to prevent pickling issues
    except Exception as e:
        return {"error": str(e)}  # Catch all other exceptions

def get_target_similarity(file_name: str, target: dict):
    filtered_target = {k: v for k, v in target.items() if pd.notna(v)}
    target_keys = list(target.keys())
    pattern = os.path.join(os.path.split(file_name)[0], os.path.split(file_name)[-1].split("_",1)[0]+"*"+os.path.split(file_name)[-1].split("_",1)[1])
    matches = glob.glob(pattern)
    if len(matches)>0:
        file_name = matches[0]
        feat_path = os.path.join(Path(file_name).parts[0],"features",*Path(file_name).parts[1:]).replace(".xes", ".json")
        if os.path.exists(feat_path):
            print(f"Skipping {file_name}")
            return None
            with open(feat_path, 'r') as f:
                print(f"INFO: Feature file found. {feat_path}.")
                feats = json.load(f)
                if not set(list(filtered_target.keys())).issubset(set(list(feats.keys()))):
                    raise ValueError("ERROR: Missing feature names in feature file. Expected: {target_keys}. Please check if all are present in {feat_path}.")
        else:
            print(f"INFO: Feature file not found. {feat_path}. Computing features.")
            event_log = read_xes_safe(file_name)
            try:
                feats = compute_features_from_event_data(list(filtered_target.keys()), event_log)
            except (TypeError, Exception) as e:
                print(f"ERROR: in {file_name}. If Exception is mentioned, it is pyossibly due to malformation in the event log:  ",e)
                return None 
            del event_log
        if feats is None:
            raise ValueError(f"feats for {file_name} are None")
        try:
            feats['target_similarity'] = compute_similarity(feats, filtered_target)
            feats['log']= os.path.split(file_name)[1].split(".")[0]
            dump_features_json(feats, file_name)
        except Exception as e:
            print(f"ERROR: Exception in {file_name}. If Exception is mentioned, it is possibly due to malformation in the event log: ", e)
        return feats
    else:
        print(f"INFO: Cannot find ELs matching: {pattern}")
        return None

def similarity_from_target(target: dict, gen_els_dir: str, target_keys: list):
    if not os.path.isdir(gen_els_dir) or not os.listdir(gen_els_dir):
        raise ValueError(f"ERROR: {gen_els_dir} does not exist or is empty")
    file_name = name_genEl_path(gen_els_dir, target, target_keys)
    feats = get_target_similarity(file_name, target)
    return feats

def copy_qualifying_genEL(event_log: str, gen_els_dir: str):
    src = os.path.join(gen_els_dir, event_log+".xes")
    dst = os.path.join(Path(gen_els_dir).parts[0],"filtered_genELs",*Path(gen_els_dir).parts[1:],event_log+".xes")
    os.makedirs(os.path.split(dst)[0], exist_ok=True)
    shutil.copyfile(src, dst)
    #print(f"INFO: Copied {src} to {dst}.")

def compute_feats_posteriously(gen_els_dir: str, targets: list):
    target_keys = list({k: v for d in targets for k, v in d.items()}.keys())

    # Multiprocessing per target. Use line below for debugging.
    #similarity_from_target(targets[0], gen_els_dir, target_keys) #TESTING
    num_cores = multiprocessing.cpu_count() if len(targets) >= multiprocessing.cpu_count() else len(targets)
    with multiprocessing.Pool(processes=num_cores) as p:
        print(f"INFO: Using {num_cores} cores to compute {len(targets)} target similarities.")
        partial_wrapper = partial(similarity_from_target,
                                  gen_els_dir=gen_els_dir,
                                  target_keys=target_keys)
        feats = p.map(partial_wrapper, targets)

    feats_dir = os.path.join(Path(gen_els_dir).parts[0],"features",*Path(gen_els_dir).parts[1:])
    all_feats = json_to_csv(feats_dir, gen_els_dir+".csv")
    print(f"INFO: Got features for {len(all_feats)} genereated ELs of {len(targets)} targets.")
    return all_feats

def filter_genEls_posteriously(gen_els_dir: str, original_config_file: str, similarity_threshold: float):
    targets = get_targets(original_config_file)

    all_feats = compute_feats_posteriously(gen_els_dir, targets)
    filtered_feats = all_feats[all_feats['target_similarity'] >= float(similarity_threshold)]
    print(f"INFO: {len(filtered_feats)}/{len(all_feats)} ELs have a target similarity over {similarity_threshold}.")
    print(f"SUCCESS: Got features for {len(all_feats)} genereated ELs of {len(targets)} targets.")
    print(f"SUCCESS: Qualifying genELs: {len(filtered_feats)} out of {len(all_feats)}, with a threshold of {similarity_threshold}.")

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Compute target_similarites of generated ELs posteriously.')
    parser.add_argument('gen_els_dir', type=str, help='The directory containing the generated ELs')
    parser.add_argument('original_config_file', type=str, help='The path to the original config file for generation')
    parser.add_argument('similarity_threshold', type=float, help='The path to the original config file for generation')
    args = parser.parse_args()

    filter_genEls_posteriously(args.gen_els_dir, args.original_config_file, args.similarity_threshold)
