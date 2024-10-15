from ax.service.ax_client import AxClient, ObjectiveProperties
from ax.utils.measurement.synthetic_functions import branin

from ax.modelbridge.dispatch_utils import choose_generation_strategy
from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from ax.modelbridge.modelbridge_utils import get_pending_observation_features
from ax.modelbridge.registry import ModelRegistryBase, Models

from ax.utils.testing.core_stubs import get_branin_experiment, get_branin_search_space

from botorch.models.approximate_gp import ApproximateGPyTorchModel, SingleTaskVariationalGP
from botorch.acquisition.multi_objective.monte_carlo import qNoisyExpectedHypervolumeImprovement as qNEHVI
from botorch.acquisition.multi_objective.logei import qLogNoisyExpectedHypervolumeImprovement as qLNEHVI
from ax.models.torch.botorch_modular.surrogate import Surrogate
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.fully_bayesian_multitask import SaasFullyBayesianMultiTaskGP
from botorch.models.multitask import MultiTaskGP


import gpytorch, torch, numpy as np, matplotlib.pyplot as plt, tqdm, botorch
from scipy.interpolate import griddata


from botorch.test_functions.multi_objective import BraninCurrin

import torch

branin_currin = BraninCurrin(negate=True).to(
    dtype=torch.double,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
)


# model_list = ModelListGP(ApproximateGPyTorchModel())
gs = GenerationStrategy(
    steps=[
        # 1. Initialization step (does not require pre-existing data and is well-suited for
        # initial sampling of the search space)
        GenerationStep(
            model=Models.SOBOL,
            num_trials=5,  # How many trials should be produced from this generation step
            min_trials_observed=3,  # How many trials need to be completed to move to next model
            max_parallelism=5,  # Max parallelism for this step
            model_kwargs={"seed": 999},  # Any kwargs you want passed into the model
            model_gen_kwargs={},  # Any kwargs you want passed to `modelbridge.gen`
        ),
        # 2. Bayesian optimization step (requires data obtained from previous phase and learns
        # from all data available at the time of each new candidate generation call)
        GenerationStep(
            model=Models.BOTORCH_MODULAR,
            num_trials=-1,  # No limitation on how many trials should be produced from this step
            max_parallelism=3,  # Parallelism limit for this step, often lower than for Sobol
            # model_kwargs={
            #     "surrogate": Surrogate(MultiTaskGP),
            #     "botorch_acqf_class": qLNEHVI,
            # }
        ),
    ]
)

def evaluate(parameters):
    evaluation = branin_currin(
        torch.tensor([parameters.get("x1"), parameters.get("x2")])
    )
    # In our case, standard error is 0, since we are computing a synthetic function.
    # Set standard error to None if the noise level is unknown.
    # return {"a": (evaluation[0].item()), "b": (evaluation[1].item())}
    return {"a": (evaluation[0].item(), 0.0), "b": (evaluation[1].item(), 0.0)}

# ax_client = AxClient(generation_strategy=gs)
ax_client = AxClient()
ax_client.create_experiment(
    name="branin_test_experiment",
    parameters=[
        {
            "name": "x1",
            "type": "range",
            "bounds": [-5.0, 10.0],
            "value_type": "float",
        },
        {
            "name": "x2",
            "type": "range",
            "bounds": [0.0, 10.0],
        },
    ],
    objectives={
        # `threshold` arguments are optional
        "a": ObjectiveProperties(minimize=False, threshold=branin_currin.ref_point[0]),
        "b": ObjectiveProperties(minimize=False, threshold=branin_currin.ref_point[1]),
    },
)

print(ax_client.generation_strategy, ax_client.generation_strategy._steps[1])
for _ in range(15):
    parameters, trial_index = ax_client.get_next_trial()
    ax_client.complete_trial(trial_index=trial_index, raw_data=evaluate(parameters))

best_parameters, metrics = ax_client.get_best_parameters()
