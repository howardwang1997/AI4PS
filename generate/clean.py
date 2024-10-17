# predict generated molecules

import argparse
import json
import os
from os.path import dirname as up

from rdkit import Chem
from rdkit.Chem.Crippen import MolLogP
from rdkit.Contrib.SA_Score import sascorer

import sys
sys.path.append('../..')
from AI4PS.bayesian.trainer import BayesianPredictor
from AI4PS.bayesian.bayesian import _get_predictor


def check_smiles_list(molecules):
    smiles_list = type(molecules[0]) is str
    return smiles_list


def remove_duplicates(original, new):
    is_smiles_list = check_smiles_list(new)
    # print(original[0])

    original = [Chem.MolToSmiles(Chem.MolFromSmiles(s[1])) for s in original]

    removed = []
    for n in new:
        if is_smiles_list:
            n_new = n
        else:
            n_new = n['decoded_molecule']
        st_n = Chem.MolToSmiles((Chem.MolFromSmiles(n_new)))
        if st_n not in original:
            removed.append(n)

    print(f'Removed {len(original) - len(removed)} duplicates from {len(original)} to {len(removed)}.')
    return removed


def bayesian_to_generated(bayesian_list):
    return [n['decoded_molecule'] for n in bayesian_list]


def generated_to_bayesian(generate_list):
    is_smiles_list = check_smiles_list(generate_list)

    if is_smiles_list:
        checkpoints0 = [
            '/mlx_devbox/users/howard.wang/playground/molllm/ai4ps_logs/checkpoints/soqy_final_rg_ens_0_seed_42_fold_0_checkpoint.pt',
            '/mlx_devbox/users/howard.wang/playground/molllm/ai4ps_logs/checkpoints/soqy_final_rg_ens_1_seed_42_fold_1_checkpoint.pt',
            '/mlx_devbox/users/howard.wang/playground/molllm/ai4ps_logs/checkpoints/soqy_final_rg_ens_2_seed_42_fold_2_checkpoint.pt',
            '/mlx_devbox/users/howard.wang/playground/molllm/ai4ps_logs/checkpoints/soqy_final_rg_ens_3_seed_42_fold_3_checkpoint.pt',
            '/mlx_devbox/users/howard.wang/playground/molllm/ai4ps_logs/checkpoints/soqy_final_rg_ens_4_seed_42_fold_4_checkpoint.pt',
        ]
        checkpoints1 = [
            '/mlx_devbox/users/howard.wang/playground/molllm/ai4ps_logs/checkpoints/abs_final_rg_ens_0_seed_42_fold_0_checkpoint.pt',
            '/mlx_devbox/users/howard.wang/playground/molllm/ai4ps_logs/checkpoints/abs_final_rg_ens_1_seed_42_fold_1_checkpoint.pt',
            '/mlx_devbox/users/howard.wang/playground/molllm/ai4ps_logs/checkpoints/abs_final_rg_ens_2_seed_42_fold_2_checkpoint.pt',
            '/mlx_devbox/users/howard.wang/playground/molllm/ai4ps_logs/checkpoints/abs_final_rg_ens_3_seed_42_fold_3_checkpoint.pt',
            '/mlx_devbox/users/howard.wang/playground/molllm/ai4ps_logs/checkpoints/abs_final_rg_ens_4_seed_42_fold_4_checkpoint.pt',
        ]
        predictor = _get_predictor(checkpoints0, checkpoints1)

    predicted = []
    for data in generate_list:
        add_data = {}
        if is_smiles_list:
            m = data
            pred = predictor.predict([[data, 'O']])
            soqy, absorption = [pred['soqy_mean'].item(), pred['soqy_std'].item()], [[pred['abs_mean'].item(), pred['abs_std'].item()]]
            add_data['max_absorption'] = absorption
            add_data['phi_singlet_oxygen'] = soqy
        else:
            m = data['decoded_molecule']
            add_data['max_absorption'] = data['max_absorption']
            add_data['phi_singlet_oxygen'] = data['phi_singlet_oxygen']
        add_data['decoded_molecule'] = m
        m = Chem.MolFromSmiles(m)
        sas = sascorer.calculateScore(m)
        logp = MolLogP(m)
        add_data['log_p'] = [logp, 0]
        add_data['sas'] = [sas, 0]
        predicted.append(add_data)

    return predicted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--clean', action='store_true')
    parser.add_argument('--to_smiles', action='store_true')
    parser.add_argument('--predict', action='store_true')
    parser.add_argument('--file', type=str)
    parser.add_argument('--save', type=str)
    parser.add_argument('--original', type=str, required=False, default=False)
    args = parser.parse_args()

    if args.original:
        with open(args.original) as f:
            original_data = json.load(f)
        if type(original_data) is dict:
            if 'all_ps' in original_data.keys():
                original_data = original_data['all_ps']
            elif 'all_smiles' in original_data.keys():
                original_data = original_data['all_smiles']
            else:
                raise KeyError('original dict does not contain all_ps or all_smiles')
    with open(args.file) as f:
        files_data = json.load(f)

    new = []
    if args.to_smiles:
        new = bayesian_to_generated(files_data)
    if args.predict:
        new = generated_to_bayesian(files_data)
    elif args.clean:
        new = remove_duplicates(original_data, files_data)

    with open(args.save, 'w') as f:
        json.dump(new, f)


if __name__ == '__main__':
    main()
