import argparse
import json

import pandas as pd

def make_index(subset, name):
    subset_with_index = []
    length = len(subset)
    length_str = len(str(length))
    for i in range(len(subset)):
        index = f'{name}_{str(i).rjust(length_str, "0")}'
        subset_with_index.append([index] + subset[i])
    return subset_with_index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='dataset_close_5.json')
    parser.add_argument('--output', type=str, default='dataset_close_5_index.json')
    args = parser.parse_args()

    with open(args.input) as f:
        dataset = json.load(f)

    dataset_with_index = {}

    for k, v in dataset.items():
        dataset_with_index[k] = make_index(v, k)

    with open(args.output, 'w') as f:
        json.dump(dataset_with_index, f)


if __name__ == '__main__':
    main()
