import argparse
import json

from rdkit import Chem
import torch
from molecule_generation import load_model_from_directory


def encode(smiles, model_dir):
    with load_model_from_directory(model_dir) as model:
        embeddings = model.encode(smiles)

    return torch.tensor(embeddings)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_path', type=str)
    parser.add_argument('--save_path', type=str)
    parser.add_argument('--model_dir', type=str, default="/mlx_devbox/users/howard.wang/playground/molllm/moler_weights")
    args = parser.parse_args()

    with open(args.src_path) as f:
        # data = json.load(f)
        data = f.readlines()

    if data[-1] == '':
        data = data[:-1]

    new_data = []
    for d in data:
        a = Chem.MolFromSmiles(d)
        if 'DATIVE' in [b.GetBondType().name for b in a.GetBonds()]:
            continue
        else:
            new_data.append(d)

    embedding = encode(new_data, args.model_dir)

    torch.save(embedding, args.save_path)


if __name__ == '__main__':
    main()
