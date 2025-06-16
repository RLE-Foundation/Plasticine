# Based on: https://github.com/awjuliani/deep-rl-plasticity

import torch
import numpy as np

from typing import Dict

from utils import CReLU4Linear, CReLU4Conv2d, DFFLayer4Linear, DFFLayer4Conv2d

def compute_dormant_units(model: torch.nn.Module, data: torch.Tensor, activation: str, tau: float) -> torch.Tensor:
    """
    Compute the ratio of dormant units (RDU) in the model.

    Args:
        model (torch.nn.Module): The model to evaluate.
        data (torch.Tensor): The data for evaluating the model.
        activation (str): The activation function used in the model.
        tau (float): The threshold for dormant units.

    Returns:
        torch.Tensor: The ratio of dormant units.
    """

    with torch.no_grad():
        # Create a dictionary to store the s scores for each layer
        s_scores_dict = {}

        # Register a forward hook to capture the activations of each layer
        activations = {}
        hooks = []

        def hook(module, input, output):
            activations[module] = output.detach()

        act_set = (torch.nn.ReLU, 
                   torch.nn.Tanh, 
                   torch.nn.Sigmoid, 
                   CReLU4Linear, 
                   CReLU4Conv2d, 
                   DFFLayer4Linear, 
                   DFFLayer4Conv2d)

        for name, module in model.named_modules():
            if isinstance(module, act_set):
                handle = module.register_forward_hook(hook)
                hooks.append(handle)

        # Forward pass through the model
        model(data)

        # Calculate the s scores for each layer
        for name, module in model.named_modules():
            if isinstance(module, act_set):
                layer_activations = activations[module]
                if activation in ['tanh', 'dff']:
                    layer_activations = torch.abs(layer_activations)
                s_scores = layer_activations / (torch.mean(layer_activations, axis=1, keepdim=True) + 1e-6)
                s_scores = torch.mean(s_scores, axis=0)
                if len(s_scores.shape) > 1:
                    s_scores = torch.mean(s_scores, axis=tuple(range(1, len(s_scores.shape))))
                s_scores_dict[name] = s_scores

        # Remove the hooks to prevent memory leaks
        for handle in hooks:
            handle.remove()

        rdu = []
        # Calculate the ratio of dormant units
        for name, s_scores in s_scores_dict.items():
            reset_mask = s_scores <= tau
            rdu.append(torch.mean(reset_mask.float()))
        
    return torch.mean(torch.Tensor(rdu))


def compute_active_units(features: torch.Tensor, activation: str) -> torch.Tensor:
    """
    Compute the fraction of active units (FAU) in the features.

    Args:
        features (torch.Tensor): The feature tensor with the data shape [batch_size, num_features].
        activation (str): The activation function used in the model.

    Returns:
        torch.Tensor: The fraction of active units units.
    """
    assert features.ndim == 2, "features should be a 2D tensor"
    assert features.size(0) >= features.size(1), "batch_size should be greater than num_features"
    if activation in ["ln-relu", "gn-relu", "relu", "crelu"]:
        return 1. - torch.mean((features == 0).float())
    elif activation == "tanh":
        return 1. - torch.mean((torch.abs(features) > 0.99).float())
    elif activation == "sigmoid":
        return 1. - torch.mean(((features < 0.01) | (features > 0.99)).float())
    elif activation == "dff":
        return 1. - torch.mean((torch.abs(features) > (2 ** 0.5 - 0.01)).float())
    else:
        raise NotImplementedError
    

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


def compute_feature_variance(features: torch.Tensor) -> torch.Tensor:
    """
    Compute the feature variance of the features.

    Args:
        features (torch.Tensor): The feature tensor with the data shape [batch_size, num_features].

    Returns:
        torch.Tensor: The feature variance of the features.
    """
    assert features.ndim == 2, "features should be a 2D tensor"
    assert features.size(0) >= features.size(1), "batch_size should be greater than num_features"
    return torch.mean(torch.var(features, dim=1))


def compute_weight_magnitude(model: torch.nn.Module) -> torch.Tensor:
    """
    Compute the weight magnitude of the model.

    Args:
        model (torch.nn.Module): The model to compute the weight magnitude for.

    Returns:
        torch.Tensor: The weight magnitude of the model.
    """
    assert isinstance(model, torch.nn.Module), "model should be a torch.nn.Module"
    l2_norm_squared = 0.0
    num_params = 0
    for param in model.parameters():
        l2_norm_squared += torch.sum(param**2)
        num_params += param.numel()
    return torch.sqrt(torch.Tensor(l2_norm_squared) / num_params)


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


def save_model_state(model: torch.nn.Module) -> Dict:
    """
    Save the initial state of the model parameters.

    Args:
        model (torch.nn.Module): The model to save the initial state for.

    Returns:
        Dict: A dictionary containing the initial state of the model parameters.
    """
    initial_state = {
        name: param.clone().detach() for name, param in model.named_parameters()
    }
    return initial_state