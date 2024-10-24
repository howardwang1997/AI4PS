import argparse
import json
from tqdm import tqdm
import time

import numpy as np
from rdkit import Chem
from molecule_generation import load_model_from_directory


def generate(scaffolds, rep=100, noise_std=0.5):
    model_dir = "/mlx_devbox/users/howard.wang/playground/molllm/moler_weights"
    encode_smi = scaffolds[0]
    example_smiles = [encode_smi]
    scaffolds = scaffolds * rep

    with load_model_from_directory(model_dir) as model:
        embeddings = model.encode(example_smiles)
        noise = np.random.normal(0, noise_std, (len(scaffolds), embeddings[0].shape[-1]))
        noise = noise.astype(embeddings[0].dtype)
        noise_embedding = embeddings[0] + noise
        print(noise_embedding.shape)

        # The i-th scaffold will be used when decoding the i-th latent vector.
        decoded_scaffolds = model.decode(noise_embedding, scaffolds=scaffolds)

    return decoded_scaffolds

def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--scaffolds", type=str, help="path to json file of scaffolds.")
    # parser.add_argument("--rep", type=int, default=100)
    # args = parser.parse_args()

    with open('/mlx_devbox/users/howard.wang/playground/molllm/AI4PS/data/scaffolds_v1.json') as f:
        scaffolds = json.load(f)

    st = time.time()
    # decoded_hypocrellin = generate(scaffolds[:-9], rep=100)
    # print('hypocrellin done! time: ', time.time() - st)
    # st = time.time()
    # decoded_6 = generate(scaffolds[-9:-7], rep=300)
    # print('6 done! time: ', time.time() - st)
    # st = time.time()
    # decoded_porphyrin = generate(scaffolds[-7:-1], rep=300)
    # print('porphyrin done! time: ', time.time() - st)
    # st = time.time()
    # decoded_bodipy = generate(scaffolds[-1:], rep=300)
    # print('bodipy done! time: ', time.time() - st)

    # decoded = list(set(decoded_hypocrellin + decoded_6 + decoded_porphyrin + decoded_bodipy))
    decoded = []
    for noise_std in [0.5,0.7,0.9]:
        for s in scaffolds:
            decoded = decoded + generate(scaffolds=[s], rep=100, noise_std=noise_std)
    decoded = list(set(decoded))
    print(f'done, {len(decoded)} decoded, time: {time.time() - st}')
    with open('/mlx_devbox/users/howard.wang/playground/molllm/datasets/decoded_09.json', 'w') as f:
        json.dump(decoded, f)

if __name__ == '__main__':
    main()
