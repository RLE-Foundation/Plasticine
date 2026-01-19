from torch import nn
from torch.nn import functional as F
from typing import Dict

import numpy as np
import torch

class CReLU4Linear(nn.Module):
    """
    CReLU activation function for Linear layers.
    """

    def __init__(self):
        super(CReLU4Linear, self).__init__()

    def forward(self, x):
        return torch.cat((F.relu(x), F.relu(-x)), dim=-1)

class CReLU4Conv2d(nn.Module):
    """
    CReLU activation function for Conv2d layers.
    """

    def __init__(self):
        super(CReLU4Conv2d, self).__init__()

    def forward(self, x):
        return torch.cat((F.relu(x), F.relu(-x)), dim=1)
    

class DFF4Linear(nn.Module):
    """
    Deep Fourier Feature Layer for Linear layers.
    """

    def __init__(self):
        super(DFF4Linear, self).__init__()

    def forward(self, x):
        return torch.cat((torch.sin(x), torch.cos(-x)), dim=-1)
    

class DFF4Conv2d(nn.Module):
    """
    Deep Fourier Feature Layer for Conv2d layers.
    """

    def __init__(self):
        super(DFF4Conv2d, self).__init__()

    def forward(self, x):
        return torch.cat((torch.sin(x), torch.cos(-x)), dim=1)

class PRLinear(nn.Linear):
    """
    Linear layer with Parseval regularization.
    
    Args:
        Same as nn.Linear plus:
        lambda_reg (float): Regularization strength
        s (float): Target scale for singular values
    """
    def __init__(self, in_features, out_features, bias=True, lambda_reg=1e-3, s=1.0):
        super().__init__(in_features, out_features, bias)
        self.lambda_reg = lambda_reg
        self.s = s
        
    def pr_loss(self):
        """
        Computes the Parseval Regularization (PR) loss for this layer's weight matrix.
        """
        # Skip if not training or regularization is disabled
        if not self.training or self.lambda_reg <= 0:
            return torch.tensor(0.0, device=self.weight.device)
            
        # Compute WW^T
        wwt = torch.mm(self.weight, self.weight.t())
        
        # Target is s*I
        target = self.s * torch.eye(wwt.shape[0], device=self.weight.device)
        
        # Frobenius norm squared of difference
        loss = torch.square(torch.norm(wwt - target, p='fro'))
        
        return self.lambda_reg * loss

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def get_exp_name(args):
    if args.use_shrink_and_perturb:
        exp_name = "snp"
    elif args.use_normalize_and_project:
        exp_name = "nap"
    elif args.use_plasticity_injection:
        exp_name = "pi"
    elif args.use_layer_resetting:
        exp_name = "lrt"
    elif args.use_redo:
        exp_name = "redo"
    elif args.use_regenerative_regularization:
        exp_name = "rr"
    elif args.use_parseval_regularization:
        exp_name = "pr"
    elif args.use_crelu_activation:
        exp_name = "crelus"
    elif args.use_dff_activation:
        exp_name = "dff"
    elif args.use_layer_norm:
        exp_name = "ln"
    elif args.use_l2_norm:
        exp_name = "l2n"
    elif args.use_trac_optimizer:
        exp_name = "trac"
    elif args.use_kron_optimizer:
        exp_name = "kron"
    else:
        exp_name = "vanilla"
    return exp_name

def get_activation(af_name):
    if af_name == "crelu-linear":
        return CReLU4Linear()
    elif af_name == "crelu-conv":
        return CReLU4Conv2d()
    elif af_name == "dff-linear":
        return DFF4Linear()
    elif af_name == "dff-conv":
        return DFF4Conv2d()
    elif af_name == "relu":
        return nn.ReLU()
    elif af_name == "tanh":
        return nn.Tanh()
    else:
        raise NotImplementedError(f"Activation function {af_name} not implemented")

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