# predict generated molecules

import argparse
import json
import os
from os.path import dirname as up

from rdkit import Chem
from rdkit.Chem.Crippen import MolLogP
from rdkit.Contrib.SA_Score import sascorer

import sys
sys.path.append('..')
from ..bayesian.trainer import BayesianPredictor


def check_smiles_list(molecules):
    smiles_list = type(molecules[0]) is str
    return smiles_list


def remove_duplicates(original, new):
    is_smiles_list = check_smiles_list(new)

    original = [Chem.MolToSmiles(Chem.MolFromSmiles(s)) for s in original]

    removed = []
    for n in new:
        if is_smiles_list:
            n_new = n
        else:
            n_new = n['decoded_molecule']
        st_n = Chem.MolToSmiles((Chem.MolFromSmiles(n)))
        if st_n not in original:
            removed.append(n)

    return removed


def bayesian_to_generated(bayesian_list):
    return [n['decoded_molecule'] for n in bayesian_list]


def generated_to_bayesian(generate_list):
    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--clean', action='store_true')
    parser.add_argument('--to_smiles', action='store_true')
    parser.add_argument('--file', type=str)
    parser.add_argument('--save', type=str)
    parser.add_argument('--original', type=str, required=False, default=False)
    args = parser.parse_args()

    if args.original:
        with open(args.original) as f:
            original_data = json.load(f)
    with open(args.file) as f:
        files_data = json.load(f)

    if args.to_smiles:
        new = bayesian_to_generated(args.files)
    elif args.clean:
        new = remove_duplicates(original_data, files_data)

    # to be done
