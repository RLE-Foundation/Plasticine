from torch import nn
from torch.nn import functional as F

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
    

class DFFLayer4Linear(nn.Module):
    """
    Deep Fourier Feature Layer for Linear layers.
    """

    def __init__(self):
        super(DFFLayer4Linear, self).__init__()

    def forward(self, x):
        return torch.cat((torch.sin(x), torch.cos(-x)), dim=-1)
    

class DFFLayer4Conv2d(nn.Module):
    """
    Deep Fourier Feature Layer for Conv2d layers.
    """

    def __init__(self):
        super(DFFLayer4Conv2d, self).__init__()

    def forward(self, x):
        return torch.cat((torch.sin(x), torch.cos(-x)), dim=1)