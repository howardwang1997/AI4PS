import random


def split(dataset: list, val_ratio=0.2, seed=42, fold=0):
    l = len(dataset)
    random.seed(seed)
    border = (int(l * val_ratio * fold), int(l * val_ratio * (fold+1)))
    shuffled = list(range(l))
    random.shuffle(shuffled)

    train_idx = shuffled[0:border[0]] + shuffled[border[1]:]
    val_idx = shuffled[border[0]:border[1]]
    train_set = [dataset[i] for i in train_idx]
    val_set = [dataset[i] for  i in val_idx]

    return train_set, val_set


def dataset_converter(dataset):
    solution = len(dataset[0]) == 3

    if solution:
        mols, sols, outs = [], [], []
        for data in dataset:
            mols.append(data[0])
            sols.append(data[1])
            outs.append(data[2])
        return mols, sols, outs
    else:
        mols, outs = [], []
        for data in dataset:
            mols.append(data[0])
            outs.append(data[1])
        return mols, outs
