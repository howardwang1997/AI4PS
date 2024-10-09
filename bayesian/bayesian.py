import json
import os
from os.path import dirname as up

import torch
from ax.plot.pareto_frontier import plot_pareto_frontier
from ax.plot.pareto_utils import compute_posterior_pareto_frontier
from ax.service.ax_client import AxClient
from ax.service.utils.instantiation import ObjectiveProperties

import sys
sys.path.append('..')
from bayesian.trainer import BayesianPredictor

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

            # print(type(trial_data['phi_singlet_oxygen']), trial_data['phi_singlet_oxygen'].value.df)
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


def manual_screen(components: dict,
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
                "type": "range",
                "bounds": [-1, len(v)+1],
            }
            for k, v in components.items()
        ],
        objectives=objectives,
        overwrite_existing_experiment=True,
        is_test=True,
    )

    trial_index = 0
    # manually screen photosensitizer and solvent
    for i in range(len(components['photosensitizer'])):
        for j in range(len(components['solvent'])):
            parameters_trial = [{
                'photosensitizer': i,
                'solvent': j,
            }]
            parameters_eval = {
                'photosensitizer': components['photosensitizer'][i],
                'solvent': components['solvent'][j],
            }
            print(parameters_trial)
            ax_client.experiment.attach_trial(parameters_trial)
            # Local evaluation here can be replaced with deployment to external system.
            ax_client.complete_trial(trial_index=trial_index, raw_data=evaluate(parameters_eval, predictor))
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
        plot_frontier(frontier)
        save_experiment_and_frontier(ax_client.experiment, frontier, manual=True, components=components, path='/mlx_devbox/users/howard.wang/playground/molllm/datasets/pareto_frontier.json')

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
    components = _make_parameters(_get_photosensitizers('/mlx_devbox/users/howard.wang/playground/molllm/datasets/decoded_all.json')[:5], 
                                  _get_solvents('/mlx_devbox/users/howard.wang/playground/molllm/datasets/solvents_all.json')[:2])
    objectives = _make_objectives()
    predictor = _get_predictor('/mlx_devbox/users/howard.wang/playground/molllm/ai4ps_logs/checkpoints/soqy_rg_3_checkpoint.pt',
                               '/mlx_devbox/users/howard.wang/playground/molllm/ai4ps_logs/checkpoints/abs_rg_0_checkpoint.pt')
    experiment = manual_screen(components=components,
                        objectives=objectives,
                        predictor=predictor,
                        iterations=50,
                        plot=True,
                        num_point=10)


if __name__ == '__main__':
    main()
