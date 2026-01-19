from typing import Dict

import torch
import numpy as np

def compute_l2_norm_difference(current_model: torch.nn.Module, prev_model_state: Dict) -> torch.Tensor:
    """
    Compute the L2 norm difference between the current model and the previous model state.
    This is used to measure the change in model parameters.

    Args:
        current_model (torch.nn.Module): The current model.
        prev_model_state (Dict): The previous model state.

    Returns:
        torch.Tensor: The L2 norm difference between the current model and the previous model state.
    """
    l2_diff = 0.0
    num_params = 0

    for name, param in current_model.named_parameters():
        if name in prev_model_state:
            param_diff = param - prev_model_state[name]
            l2_diff += torch.sum(param_diff**2)
            num_params += param.numel()
        else:
            raise ValueError(
                f"Saved initial state does not contain parameters '{name}'."
            )

    return torch.sqrt(torch.Tensor(l2_diff) / num_params)

def compute_feature_norm(features: torch.Tensor) -> torch.Tensor:
    """
    Compute the feature norm of the features.

    Args:
        features (torch.Tensor): The feature tensor with the data shape [batch_size, num_features].

    Returns:
        torch.Tensor: The feature norm of the features.
    """
    assert features.ndim == 2, "features should be a 2D tensor"
    assert features.size(0) >= features.size(1), "batch_size should be greater than num_features"
    return torch.mean(torch.norm(features, dim=1))

