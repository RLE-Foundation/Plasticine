import torch
import numpy as np

from typing import Dict

from plasticine.utils import CReLU4Linear, CReLU4Conv2d, DFF4Linear, DFF4Conv2d

def compute_dormant_units(model: torch.nn.Module, data: torch.Tensor, activation: str, tau: float) -> torch.Tensor:
    """
    Compute the ratio of dormant units (RDU) in the model.

    This implementation is based on the ReDo paper:
    https://github.com/google/dopamine/blob/ce36aab6528b26a699f5f1cefd330fdaf23a5d72/dopamine/labs/redo/weight_recyclers.py

    Args:
        model (torch.nn.Module): The model to evaluate.
        data (torch.Tensor): The data for evaluating the model.
        activation (str): The activation function used in the model.
        tau (float): The threshold for dormant units.

    Returns:
        torch.Tensor: The ratio of dormant units.
    """

    with torch.no_grad():
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
                   DFF4Linear, 
                   DFF4Conv2d)

        for name, module in model.named_modules():
            if isinstance(module, act_set):
                handle = module.register_forward_hook(hook)
                hooks.append(handle)

        # Forward pass through the model
        model(data)

        # Remove the hooks to prevent memory leaks
        for handle in hooks:
            handle.remove()

        rdu = []
        # Calculate the ratio of dormant units for each layer
        for name, module in model.named_modules():
            if isinstance(module, act_set):
                layer_activations = activations[module]
                
                # For tanh and dff activations, use absolute values
                if activation in ['tanh', 'dff']:
                    layer_activations = torch.abs(layer_activations)
                
                # Compute score based on activation tensor dimensions
                # Taking the mean conforms to the expectation under D in the ReDo paper's formula
                if layer_activations.ndim == 4:
                    # Conv layer: [batch, channels, height, width]
                    # Average over batch, height, width to get per-channel score
                    score = layer_activations.abs().mean(dim=(0, 2, 3))
                else:
                    # Linear layer: [batch, features]
                    # Average over batch to get per-feature score
                    score = layer_activations.abs().mean(dim=0)
                
                # Normalize by the mean of all neurons to make threshold independent of layer size
                # See: https://github.com/google/dopamine/blob/ce36aab6528b26a699f5f1cefd330fdaf23a5d72/dopamine/labs/redo/weight_recyclers.py#L314
                normalized_score = score / (score.mean() + 1e-9)
                
                # Create dormant mask
                if tau > 0.0:
                    dormant_mask = normalized_score <= tau
                else:
                    dormant_mask = torch.isclose(normalized_score, torch.zeros_like(normalized_score))
                
                rdu.append(torch.mean(dormant_mask.float()))
        
    return torch.mean(torch.stack(rdu)) if rdu else torch.tensor(0.0)


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
    