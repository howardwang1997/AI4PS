import json


def remove_outliers(outliers, name, dataset):
    new_dataset = {}
    assert name in dataset.keys()

    for k, v in dataset.items():
        if k == name:
            subset = []
            for x in v:
                if x[0] not in outliers:
                    subset.append(x)
            new_dataset[k] = subset
            print(f'{len(outliers)} (outliers) +  {len(subset)} (subset) = {len(v)} (total)')
        else:
            new_dataset[k] = v
    
    return new_dataset


def main():
    dataset_path = '/mlx_devbox/users/howard.wang/playground/molllm/datasets/dataset_close_5_index_rmo2.json'
    name = 'soqy'
    outliers_path = '/mlx_devbox/users/howard.wang/playground/molllm/ai4ps_logs/predictions/rmo2_soqy_5f_rg__outliers.json'

    with open(outliers_path, 'r') as f:
        outliers = json.load(f)

    with open(dataset_path, 'r') as f:
        dataset = json.load(f)

    new_dataset = remove_outliers(outliers, name, dataset)

    with open('/mlx_devbox/users/howard.wang/playground/molllm/datasets/dataset_close_5_index_rmo3.json', 'w') as f:
        json.dump(new_dataset, f)


if __name__ == '__main__':
    main()
