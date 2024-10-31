import json
import os
from os.path import dirname as up
os.environ['CUDA_LAUNCH_BLOCKING']='1'

import numpy as np
import torch
from ax.plot.pareto_frontier import plot_pareto_frontier
from ax.plot.pareto_utils import compute_posterior_pareto_frontier
from ax.service.ax_client import AxClient
from ax.service.utils.instantiation import ObjectiveProperties
from rdkit import Chem
from molecule_generation import load_model_from_directory

from trainer import BayesianPredictor
from autoencoder import Autoencoder

CODE_PATH = up(up(os.path.abspath(__file__)))
DIMENSION = 50
MODEL_DIR = "/mlx_devbox/users/howard.wang/playground/molllm/moler_weights"
MODEL = load_model_from_directory(MODEL_DIR)
SOLVENT = 'O'
AUTOENCODER_PATH = '/mnt/bn/ai4s-hl/bamboo/hongyi/debug/checkpoints/autoencoder_03.pt'
AUTOENCODER = Autoencoder()
AUTOENCODER.load_state_dict(torch.load(AUTOENCODER_PATH, map_location=torch.device('cpu')))
AUTOENCODER.eval()

def _make_parameters(scaffolds, dimension=DIMENSION):
    parameters = {i: [0.0, 1.0] for i in range(dimension)}
    # parameters['scaffolds': scaffolds]

    parameter_list = [
        {
                "name": 'scaffold',
                "type": "choice",
                "values": scaffolds,
        }
    ]
    for k, v in parameters.items():
        parameter_list.append(
            {
                "name": str(k),
                "type": "range",
                "bounds": v,
            }
        )

    return parameter_list


def _make_scaffolds(path='/mlx_devbox/users/howard.wang/playground/molllm/AI4PS/data/scaffolds_v1.json'):
    with open(path, 'r') as f:
        scaffolds = json.load(f)
    return scaffolds


def _make_objectives():
    return {
        'phi_singlet_oxygen': ObjectiveProperties(minimize=False, threshold=torch.tensor(0.)),
        'max_absorption': ObjectiveProperties(minimize=False, threshold=torch.tensor(300.))
    }


def _get_photosensitizers(smiles_path=os.path.join(CODE_PATH, 'data', 'decoded_all.json')):
    with open(smiles_path, 'r') as f:
        smiles = json.load(f)
    return smiles


def _get_solvents(smiles_path=os.path.join(CODE_PATH, 'data', 'solvents_all.json')):
    return ["O"]


def _get_predictor(checkpoint0, checkpoint1):
    predictor = BayesianPredictor(checkpoint0, checkpoint1, device='cpu')
    return predictor


def parameters_to_embeddings(parameters, dimension=DIMENSION):
    embeddings = [parameters[str(i)] for i in range(dimension)]
    embeddings = np.array(embeddings)
    scaffold = parameters['scaffold']
    return embeddings, scaffold


def generate(parameters):
    embeddings_bias, scaffold = parameters_to_embeddings(parameters)
    embeddings_bias = torch.tensor(embeddings_bias, dtype=torch.float32)
    embeddings_bias = AUTOENCODER.decode(embeddings_bias).detach().numpy()
    # model_dir = "/mlx_devbox/users/howard.wang/playground/molllm/moler_weights"
    example_smiles = [scaffold]
    scaffolds = [scaffold]

    with load_model_from_directory(MODEL_DIR) as model:
        embeddings = model.encode(example_smiles)
        noise = np.random.normal(0, 0.2, (len(scaffolds), DIMENSION))
        noise = noise.astype(embeddings[0].dtype)
        embeddings_bias = embeddings_bias.astype(embeddings[0].dtype)
        noise_embedding = embeddings[0] + embeddings_bias # + noise
        noise_embedding = noise_embedding.reshape(1, -1)
        # print(noise_embedding.shape)

        # The i-th scaffold will be used when decoding the i-th latent vector.
        # print(noise_embedding.shape, scaffolds)
        decoded_scaffolds = model.decode(noise_embedding, scaffolds=scaffolds)
    return decoded_scaffolds


def evaluate(parameters, predictor):
    """
    parameters: check format
    """
    # print(parameters) # debug
    # parameters_conversion = [[parameters['photosensitizer'], parameters['solvent']]]
    decoded_scaffolds = generate(parameters)
    parameters_conversion = [[decoded_scaffolds[0], SOLVENT]]

    pred = predictor.predict(parameters_conversion)
    soqy, absorption = pred['soqy_mean'].item(), pred['abs_mean'].item()
    loss_soqy = pred['soqy_std'].item()
    loss_absorption = pred['abs_std'].item()
    results = {"decoded_molecule": decoded_scaffolds[0], "phi_singlet_oxygen": (soqy, loss_soqy), "max_absorption": (absorption, loss_absorption)}
    print('RESULTS:', results)
    return results


def plot_frontier(frontier):
    pass


def save_experiment_and_frontier(experiment, frontier, path='pareto_frontier.json', manual=False, components=None):
    raw_param_dicts = frontier.param_dicts
    if manual:
        def calculate_trial_index(components, parameters):
            return parameters['photosensitizer'] * len(components['solvent']) + parameters['solvent']

        param_dicts = []
        trial_dicts = []
        for i in range(len(raw_param_dicts)):
            param_data = {
                'photosensitizer': components['photosensitizer'][raw_param_dicts[i]['photosensitizer']],
                'solvent': components['solvent'][raw_param_dicts[i]['solvent']],
                }
            param_dicts.append(param_data)
            trial_data = experiment.fetch_trials_data_results(trial_indices=[calculate_trial_index(components, raw_param_dicts[i])],
                                                              metrics=experiment.metrics.values()).values()
            trial_data = list(trial_data)[0]

            trial_data = {
                'phi_singlet_oxygen': trial_data['phi_singlet_oxygen'].value.df['mean'][0], 
                'max_absorption': trial_data['max_absorption'].value.df['mean'][0]
            }
            trial_dicts.append(trial_data)
    else:
        param_dicts = raw_param_dicts
        for i in range(len(raw_param_dicts)):
            trial_data = experiment.fetch_trials_data_results(trial_indices=[calculate_trial_index(components, raw_param_dicts[i])],
                                                              metrics=experiment.metrics.values()).values()
            trial_data = list(trial_data)[0]

            trial_data = {
                'phi_singlet_oxygen': trial_data['phi_singlet_oxygen'].value.df['mean'][0], 
                'max_absorption': trial_data['max_absorption'].value.df['mean'][0]
            }
            trial_dicts.append(trial_data)
        raise NotImplementedError('trial data not implemented')

    # print(trial_dicts, param_dicts)
    with open(path, 'w') as f:
        json.dump({'param_dicts': param_dicts, 'trial_dicts': trial_dicts}, f)


def screen(parameter_list: list,
           objectives: dict,
           predictor: BayesianPredictor,
           iterations: int = 100,
           plot: bool = False,
           num_point: int = 20):
    ax_client = AxClient()
    ax_client.create_experiment(
        name="screen_photosensitizer_solution",
        parameters=parameter_list,
        objectives=objectives,
        overwrite_existing_experiment=True,
        is_test=True,
    )
    eval_results = []

    for _ in range(iterations):
        parameters, trial_index = ax_client.get_next_trial()
        # Local evaluation here can be replaced with deployment to external system.
        result = evaluate(parameters, predictor)
        eval_results.append(result)
        raw_data = {"phi_singlet_oxygen": result['phi_singlet_oxygen'], "max_absorption": result['max_absorption']}

        ax_client.complete_trial(trial_index=trial_index, raw_data=raw_data)

    ax_objectives = ax_client.experiment.optimization_config.objective.objectives
    frontier = compute_posterior_pareto_frontier(
        experiment=ax_client.experiment,
        data=ax_client.experiment.fetch_data(),
        primary_objective=ax_objectives[1].metric,
        secondary_objective=ax_objectives[0].metric,
        absolute_metrics=['phi_singlet_oxygen', 'max_absorption'],
        num_points=num_point,
    )

    if plot:
        plot_frontier(frontier)

    return ax_client, eval_results, frontier


def manual_screen(generations: list,
           objectives: dict,
        #    predictor: BayesianPredictor,
           plot: bool = False,
           num_point: int = 20):
    ax_client = AxClient()
    ax_client.create_experiment(
        name="screen_photosensitizer_solution",
        parameters=[
            {
                "name": 'photosensitizer',
                "type": "range",
                "bounds": [-1, len(generations)+1],
            }
        ],
        objectives=objectives,
        overwrite_existing_experiment=True,
        is_test=True,
    )
    eval_results = []

    trial_index = 0
    # manually screen photosensitizer and solvent
    for i in range(len(generations)):
        parameters_trial = [{
            'photosensitizer': i,
        }]
        print(parameters_trial)
        ax_client.experiment.attach_trial(parameters_trial)
        # Local evaluation here can be replaced with deployment to external system.
        results = {
            "phi_singlet_oxygen": tuple(generations[i]['phi_singlet_oxygen']), 
            "max_absorption": tuple(generations[i]['max_absorption']),
            }
        ax_client.complete_trial(trial_index=trial_index, raw_data=results)
        trial_index += 1

    ax_objectives = ax_client.experiment.optimization_config.objective.objectives
    frontier = compute_posterior_pareto_frontier(
        experiment=ax_client.experiment,
        data=ax_client.experiment.fetch_data(),
        primary_objective=ax_objectives[1].metric,
        secondary_objective=ax_objectives[0].metric,
        absolute_metrics=['phi_singlet_oxygen', 'max_absorption'],
        num_points=num_point,
    )

    if plot:
        save_experiment_and_frontier(ax_client.experiment, frontier, manual=True, generations=generations, path='/mlx_devbox/users/howard.wang/playground/molllm/datasets/pareto_frontier.json')

    return ax_client.experiment


def main():
    parameter_list = _make_parameters(_make_scaffolds())
    objectives = _make_objectives()
    checkpoints0 = [
        '/mnt/bn/ai4s-hl/bamboo/hongyi/debug/checkpoints/soqy_final_rg_ens_0_seed_52_fold_3_checkpoint.pt',
        '/mnt/bn/ai4s-hl/bamboo/hongyi/debug/checkpoints/soqy_final_rg_ens_1_seed_52_fold_3_checkpoint.pt',
        '/mnt/bn/ai4s-hl/bamboo/hongyi/debug/checkpoints/soqy_final_rg_ens_2_seed_52_fold_3_checkpoint.pt',
        '/mnt/bn/ai4s-hl/bamboo/hongyi/debug/checkpoints/soqy_final_rg_ens_3_seed_52_fold_3_checkpoint.pt',
        '/mnt/bn/ai4s-hl/bamboo/hongyi/debug/checkpoints/soqy_final_rg_ens_4_seed_52_fold_3_checkpoint.pt',
    ]
    checkpoints1 = [
        '/mnt/bn/ai4s-hl/bamboo/hongyi/debug/checkpoints/abs_final_rg_ens_0_seed_52_fold_3_checkpoint.pt',
        '/mnt/bn/ai4s-hl/bamboo/hongyi/debug/checkpoints/abs_final_rg_ens_1_seed_52_fold_3_checkpoint.pt',
        '/mnt/bn/ai4s-hl/bamboo/hongyi/debug/checkpoints/abs_final_rg_ens_2_seed_52_fold_3_checkpoint.pt',
        '/mnt/bn/ai4s-hl/bamboo/hongyi/debug/checkpoints/abs_final_rg_ens_3_seed_52_fold_3_checkpoint.pt',
        '/mnt/bn/ai4s-hl/bamboo/hongyi/debug/checkpoints/abs_final_rg_ens_4_seed_52_fold_3_checkpoint.pt',
    ]
    predictor = _get_predictor(checkpoints0, checkpoints1)
    experiment = screen(parameter_list=parameter_list,
                        objectives=objectives,
                        predictor=predictor,
                        iterations=600,
                        plot=True)
    client, results, frontier = experiment
    # save
    """
    NEED IMPLEMENTATION
    """
    with open('/mnt/bn/ai4s-hl/bamboo/hongyi/debug/moler/data/bayesian_generated_33.json', 'w') as f:
        json.dump(results, f)
    print(frontier)
    # with open('/mlx_devbox/users/howard.wang/playground/molllm/ai4ps_logs/data/bayesian_frontier_02.json', 'w') as f:
    #     json.dump(frontier, f)
    torch.save(frontier, '/mnt/bn/ai4s-hl/bamboo/hongyi/debug/moler/data/bayesian_frontier_33.pt')
    client.save_to_json_file(filepath='/mnt/bn/ai4s-hl/bamboo/hongyi/debug/moler/data/bayesian_client_33.json')


def debug():
    """
    FOR DEBUG ONLY. 
    """
    # done
    components = _make_parameters(_get_photosensitizers('/mlx_devbox/users/howard.wang/playground/molllm/datasets/decoded_all.json'), 
                                  _get_solvents('/mlx_devbox/users/howard.wang/playground/molllm/datasets/solvents_all.json'))
    objectives = _make_objectives()
    for k,v in components.items():
        print(k, len(v), type(v), v[0])


if __name__ == '__main__':
    main()
