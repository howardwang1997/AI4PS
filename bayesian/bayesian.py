import json
import os
from os.path import dirname as up

import torch
from ax.plot.pareto_frontier import plot_pareto_frontier
from ax.plot.pareto_utils import compute_posterior_pareto_frontier
from ax.service.ax_client import AxClient
from ax.service.utils.instantiation import ObjectiveProperties

from trainer import BayesianPredictor

CODE_PATH = up(up(os.path.abspath(__file__)))


def _make_parameters(photosensitizers, solvents):
    return {'photosensitizer': photosensitizers, 'solvent': solvents}


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
    with open(smiles_path, 'r') as f:
        smiles = json.load(f)
    return smiles


def _get_predictor(checkpoint0, checkpoint1):
    predictor = BayesianPredictor(checkpoint0, checkpoint1)
    return predictor


def evaluate(parameters, predictor):
    """
    parameters: check format
    """
    # print(parameters) # debug
    parameters_conversion = [[parameters['photosensitizer'], parameters['solvent']]]

    pred = predictor.predict(parameters_conversion)#, verbose=False)
    soqy, absorption = pred[0].item(), pred[1].item()
    loss_soqy = predictor.val_loss_soqy
    loss_absorption = predictor.val_loss_abs
    results = {"phi_singlet_oxygen": (soqy, loss_soqy), "max_absorption": (absorption, loss_absorption)}
    return results


def plot_frontier(frontier):
    pass


def screen(components: dict,
           objectives: dict,
           predictor: BayesianPredictor,
           iterations: int = 100,
           plot: bool = False,
           num_point: int = 20):
    ax_client = AxClient()
    ax_client.create_experiment(
        name="screen_photosensitizer_solution",
        parameters=[
            {
                "name": k,
                "type": "choice",
                "values": v,
            }
            for k, v in components.items()
        ],
        objectives=objectives,
        overwrite_existing_experiment=True,
        is_test=True,
    )

    for _ in range(iterations):
        parameters, trial_index = ax_client.get_next_trial()
        # Local evaluation here can be replaced with deployment to external system.
        ax_client.complete_trial(trial_index=trial_index, raw_data=evaluate(parameters, predictor))

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

    return ax_client.experiment


def main():
    components = _make_parameters(_get_photosensitizers('/mlx_devbox/users/howard.wang/playground/molllm/datasets/decoded_all.json'), 
                                  _get_solvents('/mlx_devbox/users/howard.wang/playground/molllm/datasets/solvents_all.json'))
    objectives = _make_objectives()
    predictor = _get_predictor('/mlx_devbox/users/howard.wang/playground/molllm/ai4ps_logs/checkpoints/soqy_rg_3_checkpoint.pt',
                               '/mlx_devbox/users/howard.wang/playground/molllm/ai4ps_logs/checkpoints/abs_rg_0_checkpoint.pt')
    experiment = screen(components=components,
                        objectives=objectives,
                        predictor=predictor,
                        iterations=50,
                        plot=True)
    # save
    """
    NEED IMPLEMENTATION
    """


def debug():
    """
    FOR DEBUG ONLY. 
    """
    # done
    components = _make_parameters(_get_photosensitizers('/mlx_devbox/users/howard.wang/playground/molllm/datasets/decoded_all.json')[:9], 
                                  _get_solvents('/mlx_devbox/users/howard.wang/playground/molllm/datasets/solvents_all.json')[:9])
    objectives = _make_objectives()
    ax_client = AxClient()
    # status = {
    #     'photosensitizer': components['photosensitizer'][0],
    #     'solvent': components['solvent'][2],
    # }
    data = [
        {
        'photosensitizer': components['photosensitizer'][0],
        'solvent': components['solvent'][2],
        # 'phi_singlet_oxygen': 0.5,
        # 'max_absorption': 600,
        },
        # {
        # 'photosensitizer': components['photosensitizer'][1],
        # 'solvent': components['solvent'][2],
        # # 'phi_singlet_oxygen': 0.6,
        # # 'max_absorption': 610,
        # }
    ]
    ax_client.create_experiment(
        name="screen_photosensitizer_solution",
        parameters=[
            {
                "name": k,
                "type": "choice",
                "values": v,
            }
            for k, v in components.items()
        ],
        objectives=objectives,
        overwrite_existing_experiment=True,
        is_test=True,
        # status_quo=status,
    )
    predictor = _get_predictor('/mlx_devbox/users/howard.wang/playground/molllm/ai4ps_logs/checkpoints/soqy_rg_3_checkpoint.pt',
                               '/mlx_devbox/users/howard.wang/playground/molllm/ai4ps_logs/checkpoints/abs_rg_0_checkpoint.pt')
    
    for _ in range(2):
        parameters, trial_index = ax_client.get_next_trial()
        # Local evaluation here can be replaced with deployment to external system.
        ax_client.complete_trial(trial_index=trial_index, raw_data=evaluate(parameters, predictor))
    for d in ax_client.experiment.fetch_data_results():
        print('EXPERIMENT_0\n', d)
    print('DATA_0\n', ax_client.experiment.fetch_data().df)
    ax_client.experiment.attach_trial(data)
    for d in ax_client.experiment.fetch_data_results():
        print('EXPERIMENT_1\n', d)
    print('DATA_1\n', ax_client.experiment.fetch_data().df)
    print('TRIAL_0\n', ax_client.experiment.get_trials_by_indices([0]))
    print('TRIAL_1\n', ax_client.experiment.get_trials_by_indices([1]))
    print('TRIAL_2\n', ax_client.experiment.get_trials_by_indices([2]))
    
    ax_client.complete_trial(trial_index=2, raw_data={"phi_singlet_oxygen": (600, 0.2), "max_absorption": (234, 0.1)})
    print('DATA_1\n', ax_client.experiment.fetch_data().df)
    print('TRIAL_2\n', ax_client.experiment.get_trials_by_indices([2]))


if __name__ == '__main__':
    debug()
