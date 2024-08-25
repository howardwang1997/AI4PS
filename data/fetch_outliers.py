import json

from sklearn.metrics import r2_score
import torch
from torch.nn import L1Loss


def make_ensemble_predictions(predictions: list, targets: torch.Tensor):
    """
    Make ensemble predictions by taking the mean of the predictions.

    Args:
        predictions (list): A list of predictions, each prediction is a tensor with shape (n_samples, n_targets).
        targets (torch.Tensor): The targets tensor with shape (n_samples, n_targets).

    Returns:
        torch.Tensor: The ensemble predictions tensor with shape (n_samples, n_targets).
    """
    # Concatenate the predictions along the ensemble dimension
    predictions = [p.reshape(1, -1) for p in predictions]
    concatenated_predictions = torch.cat(predictions, dim=0)
    mean_predictions = torch.mean(concatenated_predictions, dim=0)
    std_predictions = torch.std(concatenated_predictions, dim=0)
    diff = torch.abs(mean_predictions - targets)

    return mean_predictions, std_predictions, diff


def calculate_r2(mean_predictions: torch.Tensor, targets: torch.Tensor):
    # Calculate the R2 score for the concatenated predictions
    r2 = r2_score(targets.reshape(-1).cpu().numpy(), mean_predictions.cpu().numpy())

    return r2


def calculate_l1(mean_predictions: torch.Tensor, targets: torch.Tensor):
    # Calculate the L1 loss for the concatenated predictions
    l1_loss = L1Loss()(mean_predictions, targets)

    return l1_loss


def get_outliers(predictions: list, targets: torch.Tensor, threshold: float = 1, names: list = None):
    """
    Identify outliers in the predictions by comparing them to the targets.

    Args:
        predictions (list): A list of predictions, each prediction is a tensor with shape (n_samples, n_targets).
        targets (torch.Tensor): The targets tensor with shape (n_samples, n_targets).
        threshold (float): The threshold for identifying outliers.

    Returns:
        list: A list of outlier indices.
    """
    # Concatenate the predictions along the ensemble dimension
    mean_predictions, std_predictions, diff = make_ensemble_predictions(predictions, targets)
    mae = calculate_l1(mean_predictions, targets)
    r2 = calculate_r2(mean_predictions, targets)
    outliers = []
    outliers_index = []

    for i in range(len(diff)):
        if diff[i] > threshold*mae:
            print(f'{i} {names[i]} {diff[i]} {mean_predictions[i]} {targets[i]} {mae} {r2}')
            outliers.append(names[i])
            outliers_index.append(i)

    return outliers, outliers_index


def main():
    path = '/mlx_devbox/users/howard.wang/playground/molllm/ai4ps_logs/predictions/rmo2_soqy_5f_rg__fold_0_predictions.pt'
    all_outliers = []
    thresholds = [1.5, 1.5, 1.5, 1.5, 1.5]
    for i in range(1,2):
        results = torch.load(path.replace('_fold_0_', f'_fold_{i}_'))
        all_predictions = []
        for k, v in results.items():
            all_predictions.append(v['predictions'])
            names = v['names']
            targets = v['targets']
        outliers, outliers_index = get_outliers(all_predictions, targets, threshold=thresholds[i], names=names)
        all_outliers.extend(outliers)
    print(len(all_outliers))
    path = '/mlx_devbox/users/howard.wang/playground/molllm/ai4ps_logs/predictions/rmo2_soqy_5f_rg__outliers.json'
    with open(path, 'w') as f:
        json.dump(all_outliers, f)


if __name__ == '__main__':
    main()
