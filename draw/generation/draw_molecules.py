import json
import os

import rdkit
from rdkit import Chem
from rdkit.Chem import Draw
from tqdm import tqdm


def draw_molecules(smiles_list, output_path):
    mols = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]
    for i in tqdm(range(len(mols))):
        img = Draw.MolToImage(mols[i], size=(200, 200))
        img.save(os.path.join(output_path, f"{i}.png"))
    # img = Draw.MolsToGridImage(mols, molsPerRow=4, subImgSize=(200, 200))
    # img.save(output_path)


def draw_molecules_from_json(json_path, output_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    smiles_list = [item["decoded_molecule"] for item in data]
    draw_molecules(smiles_list, output_path)


if __name__ == "__main__":
    # Example usage
    json_path = "/mnt/bn/ai4s-hl/bamboo/hongyi/debug/moler/data/bayesian_generated_30_33_predicted_cleaned.json"
    output_path = "/mnt/bn/ai4s-hl/bamboo/hongyi/debug/moler/data/mol_img_30_33"
    draw_molecules_from_json(json_path, output_path)
