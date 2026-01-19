# Based on: https://github.com/awjuliani/deep-rl-plasticity


from .norm import compute_l2_norm_difference, compute_feature_norm
from .rank import compute_stable_rank, compute_effective_rank
from .units import compute_dormant_units, compute_active_units

# def compute_feature_variance(features: torch.Tensor) -> torch.Tensor:
#     """
#     Compute the feature variance of the features.

#     Args:
#         features (torch.Tensor): The feature tensor with the data shape [batch_size, num_features].

#     Returns:
#         torch.Tensor: The feature variance of the features.
#     """
#     assert features.ndim == 2, "features should be a 2D tensor"
#     assert features.size(0) >= features.size(1), "batch_size should be greater than num_features"
#     return torch.mean(torch.var(features, dim=1))


# def compute_weight_magnitude(model: torch.nn.Module) -> torch.Tensor:
#     """
#     Compute the weight magnitude of the model.

#     Args:
#         model (torch.nn.Module): The model to compute the weight magnitude for.

#     Returns:
#         torch.Tensor: The weight magnitude of the model.
#     """
#     assert isinstance(model, torch.nn.Module), "model should be a torch.nn.Module"
#     l2_norm_squared = 0.0
#     num_params = 0
#     for param in model.parameters():
#         l2_norm_squared += torch.sum(param**2)
#         num_params += param.numel()
#     return torch.sqrt(torch.Tensor(l2_norm_squared) / num_params)

