# change each row from csv to json
import csv
import json
import os
import pandas as pd

def convert_csv_to_jsonl(csv_file_path, jsonl_file_path):
    """
    Convert a CSV file to a JSONL file.
    
    Args:
        csv_file_path (str): Path to the input CSV file.
        jsonl_file_path (str): Path to the output JSONL file.
    """
    # Read the CSV file
    df = pd.read_csv(csv_file_path)
    
    # Convert each row to JSON and write to the JSONL file
    with open(jsonl_file_path, 'w') as jsonl_file:
        for _, row in df.iterrows():
            jsonl_file.write(json.dumps(row.to_dict()) + '\n')
    print(f"Converted {csv_file_path} to {jsonl_file_path}")

if __name__ == "__main__":
    csv_p = 'data_for_training/paas_plans/blocksworld/paas_plans.csv'
    jsonl_p = 'data_for_training/paas_plans/blocksworld/data.jsonl'
    convert_csv_to_jsonl(csv_p, jsonl_p)