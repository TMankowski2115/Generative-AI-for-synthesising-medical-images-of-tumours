# sampler_utils.py

import torch
from torch.utils.data import WeightedRandomSampler

def make_oversampling_sampler(dataset, pos_factor=5):
    """
    dataset: LIDCDiffusionDataset
    pos_factor: ile razy mocniej ważyć pozytywy względem negatywów
    """
    has_nodule = dataset.has_nodule_flags  # lista booli
    weights = []
    for flag in has_nodule:
        if flag:
            weights.append(float(pos_factor))
        else:
            weights.append(1.0)

    weights = torch.tensor(weights, dtype=torch.double)
    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),  # tyle "sample" na epokę
        replacement=True
    )
    return sampler
