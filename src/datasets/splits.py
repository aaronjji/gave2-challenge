"""K-fold split generation over the 50 labeled GAVE2 training cases.

No camera-vendor metadata ships with the dataset, so folds are plain random
K-fold (not vendor-stratified, despite the plan's original intent) -- revisit
if vendor labeling is done manually later.
"""
import numpy as np


def kfold_case_ids(n_cases: int = 50, n_folds: int = 5, seed: int = 77) -> list[tuple[list[int], list[int]]]:
    """Returns a list of (train_ids, val_ids) 1-indexed case-id lists, one per fold."""
    ids = np.arange(1, n_cases + 1)
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    folds = np.array_split(ids, n_folds)

    splits = []
    for i in range(n_folds):
        val_ids = sorted(folds[i].tolist())
        train_ids = sorted(np.concatenate([folds[j] for j in range(n_folds) if j != i]).tolist())
        splits.append((train_ids, val_ids))
    return splits
