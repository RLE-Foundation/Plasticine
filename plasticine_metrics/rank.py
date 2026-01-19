import torch
import numpy as np

def compute_stable_rank(features: torch.Tensor) -> torch.Tensor:
    """
    Compute the stable rank of the features.

    Args:
        features (torch.Tensor): The feature tensor with the data shape [batch_size, num_features].

    Returns:
        torch.Tensor: The stable rank of the features.
    """
    assert features.ndim == 2, "features should be a 2D tensor"
    assert features.size(0) >= features.size(1), "batch_size should be greater than num_features"
    singular_values = torch.linalg.svdvals(features)
    cumsum_sorted_singular_values = torch.cumsum(singular_values, dim=-1) / torch.sum(singular_values)
    return torch.sum(cumsum_sorted_singular_values < 0.99) + 1


def compute_effective_rank(features: torch.Tensor) -> torch.Tensor:
    """
    Compute the effective rank of the features.

    Args:
        features (torch.Tensor): The feature tensor with the data shape [batch_size, num_features].

    Returns:
        torch.Tensor: The effective rank of the features.
    """
    assert features.ndim == 2, "features should be a 2D tensor"
    assert features.size(0) >= features.size(1), "batch_size should be greater than num_features"
    singular_values = torch.linalg.svdvals(features)
    probs = singular_values / torch.sum(torch.abs(singular_values))
    probs = probs[probs > 0]
    entropy = -torch.sum(probs * torch.log(probs))
    return torch.exp(entropy)
