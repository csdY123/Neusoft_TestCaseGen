#!/usr/bin/env python3
"""
Split JSONL file randomly into two files with 3:7 ratio (30% and 70%)
"""

import json
import random
import os

def split_jsonl(input_file: str, output_file_30: str, output_file_70: str, seed: int = 42):
    """
    Split JSONL file randomly into two files with 30% and 70% ratio
    
    Args:
        input_file: Path to input JSONL file
        output_file_30: Path to output file with 30% of data
        output_file_70: Path to output file with 70% of data
        seed: Random seed for reproducibility
    """
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Read all lines from input file
    data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    item = json.loads(line)
                    data.append(item)
                except json.JSONDecodeError as e:
                    print(f"Warning: Failed to parse line: {e}")
                    continue
    
    print(f"Total items loaded: {len(data)}")
    
    # Shuffle data randomly
    random.shuffle(data)
    
    # Calculate split point (30% and 70%)
    split_point = int(len(data) * 0.3)
    data_30 = data[:split_point]
    data_70 = data[split_point:]
    
    print(f"Split into: {len(data_30)} items (30%) and {len(data_70)} items (70%)")
    
    # Write 30% data to first file
    with open(output_file_30, 'w', encoding='utf-8') as f:
        for item in data_30:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"Written {len(data_30)} items to {output_file_30}")
    
    # Write 70% data to second file
    with open(output_file_70, 'w', encoding='utf-8') as f:
        for item in data_70:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"Written {len(data_70)} items to {output_file_70}")
    
    print("Split completed successfully!")


if __name__ == "__main__":
    # File paths
    input_file = "/media/a100/c5e1bf65-7974-432f-8aed-7a1345241efe/chensenda/codes/Neusoft/TestCaseGen/update_jsonl/ai4test_neusoft_eval_data.jsonl"
    output_file_30 = "/media/a100/c5e1bf65-7974-432f-8aed-7a1345241efe/chensenda/codes/Neusoft/TestCaseGen/update_jsonl/ai4test_neusoft_eval_data_30.jsonl"
    output_file_70 = "/media/a100/c5e1bf65-7974-432f-8aed-7a1345241efe/chensenda/codes/Neusoft/TestCaseGen/update_jsonl/ai4test_neusoft_eval_data_70.jsonl"
    
    split_jsonl(input_file, output_file_30, output_file_70)

