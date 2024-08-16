import torch
from ax.plot.pareto_frontier import plot_pareto_frontier
from ax.plot.pareto_utils import compute_posterior_pareto_frontier
from ax.service.ax_client import AxClient
from ax.service.utils.instantiation import ObjectiveProperties

from trainer import BayesianPredictor


def _make_parameters(photosensitizers, solvents):
    return {'photosensitizer': photosensitizers, 'solvent': solvents}


def _make_objectives():
    return {
        'phi_singlet_oxygen': ObjectiveProperties(minimize=False, threshold=torch.tensor(0.)),
        'max_absorption': ObjectiveProperties(minimize=False, threshold=torch.tensor(300.))
    }


def _get_photosensitizers(smiles=['']):
    """
    NEED IMPLEMENTATION, read data
    """
    return smiles


def _get_solvents(smiles=['']):
    """
    NEED IMPLEMENTATION, read data
    """
    return smiles


def evaluate(parameters, predictor):
    soqy, absorption = predictor.predict(parameters)[0].item(), predictor.predict(parameters)[1].item()
    loss_soqy = predictor.val_loss_soqy
    loss_absorption = predictor.val_loss_abs
    results = {"phi_singlet_oxygen": (soqy, loss_soqy), "max_absorption": (absorption, loss_absorption)}
    return results


def plot_frontier(frontier):
    pass


def screen(components: dict,
           objectives: dict,
           predictor,
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
                "bounds": v,
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
    components = _make_parameters(_get_photosensitizers(), _get_solvents())
    objectives = _make_objectives()
    predictor = BayesianPredictor(checkpoint0='',
                                  checkpoint1='',
                                  device='cpu')
    experiment = screen(components=components,
                        objectives=objectives,
                        predictor=predictor,
                        iterations=50,
                        plot=True)
#     save


if __name__ == '__main__':
    main()
