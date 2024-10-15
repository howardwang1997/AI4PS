from ax import *
from ax.metrics.branin import BraninMetric
from ax.utils.measurement.synthetic_functions import branin

from botorch.models.approximate_gp import ApproximateGPyTorchModel, SingleTaskVariationalGP
from botorch.acquisition.multi_objective.monte_carlo import qNoisyExpectedHypervolumeImprovement as qNEHVI
from ax.models.torch.botorch_modular.surrogate import Surrogate
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.gp_regression_fidelity import SingleTaskMultiFidelityGP
from botorch.models.fully_bayesian_multitask import SaasFullyBayesianMultiTaskGP
from botorch.models.multitask import MultiTaskGP
from botorch.test_functions.multi_objective import BraninCurrin
from ax.models.torch.randomforest import RandomForest
from ax.models.random.sobol import SobolGenerator
from botorch.utils.datasets import SupervisedDataset

import torch

branin_currin = BraninCurrin(negate=True).to(
    dtype=torch.double,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
)


class MockRunner(Runner):
    def run(self, trial):
        return {"name": str(trial.index)}


branin_search_space = SearchSpace(
    parameters=[
        RangeParameter(
            name="x1", parameter_type=ParameterType.FLOAT, lower=-5, upper=10
        ),
        RangeParameter(
            name="x2", parameter_type=ParameterType.FLOAT, lower=0, upper=15
        ),
    ]
)
exp = Experiment(
    name="test_branin",
    search_space=branin_search_space,
    optimization_config=OptimizationConfig(
        objective=Objective(
            metric=BraninMetric(name="branin", param_names=["x1", "x2"]),
            minimize=True,
        ),
    ),
    runner=MockRunner(),
)

all_trials = []
sobol = Models.SOBOL(exp.search_space)
for i in range(5):
    trial = exp.new_trial(generator_run=sobol.gen(1))
    print(trial)
    parameters = trial.arm.parameters
    print(parameters)
    all_trials.append([parameters["x1"], parameters["x2"]])
    trial.run()
    trial.update_trial_data(raw_data=branin(parameters["x1"], parameters["x2"]))
    print(branin(parameters["x1"], parameters["x2"]))
    trial.mark_completed()
    print(trial)
    print(i)
print(exp.fetch_data().df)
train_y = torch.tensor(exp.fetch_data().df['mean'].to_numpy()).reshape(-1,1)
print(train_y)
train_x = torch.tensor(all_trials)
print(train_x)

best_arm = None
for i in range(15):

    # gpei = Models.BOTORCH_MODULAR(
    #     experiment=exp, 
    #     data=exp.fetch_data(),
    #     surrogate=Surrogate(SobolGenerator),
    #     botorch_acqf_class=qNEHVI,
    #     )
    gpei = RandomForest()
    dataset = SupervisedDataset(train_x, train_y, feature_names=['x1', 'x2'], outcome_names=['branin'])
    gpei.fit(datasets=dataset)
    generator_run = gpei.gen(1)
    best_arm, _ = generator_run.best_arm_predictions
    trial = exp.new_trial(generator_run=generator_run)
    print(trial)
    # trial.run()
    # trial.mark_completed()
    print(i)

exp.fetch_data()
best_parameters = best_arm.parameters