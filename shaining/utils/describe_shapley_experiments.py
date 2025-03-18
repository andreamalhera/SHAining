import argparse
import numpy as numpy   
import matplotlib.pyplot as plt
import pandas as pd
import re

from collections import defaultdict, Counter
from itertools import combinations

"""
Run using
python describe_shapley_experiments.py path_to_genEL_feats.csv "names" "targeted" "feature" THRESHOLD
E.g. python shaining/utils/describe_shapley_experiments.py data/8fts_1to3_target_similarities.csv 'aq1' 'ekbr3' 'nusa' 'rt5v' 'svo' 'saq1' 'tlkh' 'tlv' 0.7
"""

# Function to extract feature-value pairs
def extract_feature_values(data, feature_names):
    cleaned_data = []

    for row in data:
        # Split the row into individual values (each separated by '_')
        values = row.split('_')

        # Pair the feature names with their corresponding values
        feature_value_pairs = list(zip(feature_names, values))

        # Filter out any pairs where the value is 'nan'
        filtered_pairs = [(feature, value) for feature, value in feature_value_pairs if value != 'nan']

        if filtered_pairs:
            # Extract just the feature names and corresponding values
            features = [feature for feature, _ in filtered_pairs]
            values = [value for _, value in filtered_pairs]
            cleaned_data.append((features, values))

    return cleaned_data

# Format and print the result
def format_and_print(result):
    final_dict = []
    for features, values in result:
        feature_str = ', '.join(features)
        values_tuple = tuple(values)
        #print(f"{feature_str}: {values_tuple}")
        final_dict.append({feature_str: values_tuple})
    return final_dict

def counts_values_per_feature(data):
    unique_values_per_key = defaultdict(set)

    for entry in data:
        for key, value in entry:
            unique_values_per_key[key].add(value)

    # Convert set counts to number of unique values per key
    unique_value_counts = {key: len(values) for key, values in unique_values_per_key.items()}
    print(f"INFO: Number of different values per feature (of {len(unique_value_counts)} features):", unique_value_counts)
    #for key, count in unique_value_counts.items():
    #    print(f"{key}: {count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Counts for possible shapley experimets given generated ELs with target similarity over threshold.')
    parser.add_argument('genEL_feats', type=str, help='The path to csv containing names of generated ELs and their target_similarities')
    parser.add_argument('feature_names', nargs="+", help='List of targeted feature names. Values will be derived from genEL names.')
    parser.add_argument('similarity_threshold', type=float, help='The path to the original config file for generation')
    args = parser.parse_args()

def describe_shapley_experiments(data, feature_names, similarity_threshold=0.0):
    #feature_names = ['aq1', 'ekbr3', 'nusa', 'rt5v', 'svo', 'saq1', 'tlkh', 'tlv']
    data = data.dropna()
    if similarity_threshold == 0.0:
        data['target_similarity'] = data.get('target_similarity', 1.0)
    data = data[data['target_similarity'] >= float(similarity_threshold)]
    data = data['log']

    data = [s.split("_",1)[1] for s in data]
    data = list(set(data))
    print("INFO: Number of event-logs: ",len(data))

    # Extract the feature values
    cleaned_data = extract_feature_values(data, feature_names)

    # Sort by the length of the value tuple in descending order
    cleaned_data.sort(key=lambda x: len(x[1]), reverse=True)
    final_dict = format_and_print(cleaned_data)

    # Extract all feature-value combinations and organize by levels
    keys_by_level = {1: set(), 2: set(), 3: set()}

    for entry in final_dict:
        for key, values in entry.items():
            features = key.split(", ")
            feature_value_combinations = tuple(sorted(zip(features, values)))  # Pair features with their values
            level = len(feature_value_combinations)  # Determine the level
            if level in keys_by_level:
                keys_by_level[level].add(feature_value_combinations)
            else:
                keys_by_level[level] = {feature_value_combinations}

    counts = [{key: len(value) for key, value in d.items()} for d in final_dict]
    count_values = [list(d.values())[0] for d in counts]
    count_frequencies = dict(Counter(count_values))
    print("INFO: Level combination counts: ",count_frequencies)

    # Generate subsets for each level 3 feature-value combination and check their existence
    results = []
    for level_3_key in keys_by_level.get(3, []):
        level_3_subsets = {
            2: list(combinations(level_3_key, 2)),  # Generate level 2 subsets
            1: list(combinations(level_3_key, 1)),  # Generate level 1 subsets
        }

        subset_status = {"key": level_3_key, "present": {}, "missing": {}}

        for level, subsets in level_3_subsets.items():
            present = []
            missing = []
            for subset in subsets:
                subset_tuple = tuple(sorted(subset))  # Ensure consistent order
                if subset_tuple in keys_by_level[level]:
                    present.append(subset_tuple)
                else:
                    missing.append(subset_tuple)

            subset_status["present"][level] = present
            subset_status["missing"][level] = missing

        results.append(subset_status)

    # Output the results
    """
    for result in results:
        print(f"Level 3 Key: {result['key']}")
        for level in [2, 1]:
            print(f"  Level {level} Present: {result['present'][level]}")
            print(f"  Level {level} Missing: {result['missing'][level]}")
        print("-" * 40)
    """
    filtered_data = [item["missing"] for item in results if any(len(v) > 0 for v in item["missing"].values())]
    #print("f{len(filtered_data)} Missing combinations:", filtered_data)
    feasible_experiments = [item["key"] for item in results if all(len(v) == 0 for v in item["missing"].values())]
    print(f"INFO: {len(feasible_experiments)} feasible experiments.")#: {feasible_experiments}")
    key_combinations = Counter(frozenset(tuple(k for k, v in entry)) for entry in feasible_experiments)
    key_combinations = {tuple(sorted(list(k))): v for k, v in key_combinations.items()}

    print(f"INFO: Number of experiments per feature selection ({len(key_combinations)} different combinations):", key_combinations)
    counts_values_per_feature(feasible_experiments)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Counts for possible shapley experimets given generated ELs with target similarity over threshold.')
    parser.add_argument('genEL_feats', type=str, help='The path to csv containing names of generated ELs and their target_similarities')
    parser.add_argument('feature_names', nargs="+", help='List of targeted feature names. Values will be derived from genEL names.')
    parser.add_argument('similarity_threshold', type=float, help='The path to the original config file for generation')
    args = parser.parse_args()

    data = pd.read_csv(args.genEL_feats)
    describe_shapley_experiments(data, args.feature_names, args.similarity_threshold)
