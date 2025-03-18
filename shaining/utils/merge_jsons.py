import argparse
import csv
import json
import os
import pandas as pd

"""
Run using:
python merge_jsons.py path_to_your_json_directory output.csv

"""
def json_to_csv(json_dir, output_csv):
    os.makedirs(json_dir, exist_ok=True)
    json_files = [os.path.join(json_dir, file) for file in os.listdir(json_dir) if file.endswith('.json')]

    # Collect data from all JSON files
    all_data = []
    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)
            data = dict(sorted(data.items()))
            all_data.append(data)

    # Extract the headers from the first JSON object
    if all_data:
        headers = {elem for s in [set(i) for i in [d.keys() for d in all_data]] for elem in s}
    else:
        raise ValueError(f"No data found in JSON files in {json_dir}")

    df = pd.DataFrame(all_data)
    df = df[sorted(df.columns)]
    df = df.sort_values('log')
    df = df[['log'] + [col for col in df.columns if col != 'log']].reset_index(drop=True)

    # Write data to CSV
    df.to_csv(output_csv, index=False)
    print(f"SUCCESS: Saved dataframe with {df.shape} in {output_csv}")
    return df

# Example usage
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert JSON files in a directory to a CSV file.')
    parser.add_argument('json_dir', type=str, help='The directory containing JSON files')
    parser.add_argument('output_csv', type=str, help='The output CSV file path')
    args = parser.parse_args()

    json_to_csv(args.json_dir, args.output_csv)

