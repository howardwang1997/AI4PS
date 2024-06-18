import random


def split(dataset: list, val_ratio=0.1):
    l = len(dataset)
    boarder = int(l * val_ratio)
    shuffled = list(range(l))
    random.shuffle(shuffled)

    train_idx = shuffled[boarder:]
    val_idx = shuffled[:boarder]
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
