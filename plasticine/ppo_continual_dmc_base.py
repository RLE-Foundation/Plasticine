from torch.distributions.normal import Normal
from copy import deepcopy

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


from plasticine.utils import CReLU4Linear, DFF4Linear, PRLinear, layer_init

class PlasticineAgent(nn.Module):
    def __init__(self, envs, args):
        super().__init__()
        self.obs_shape = envs.single_observation_space.shape
        self.obs_dim = np.array(self.obs_shape).prod()
        self.action_dim = np.prod(envs.single_action_space.shape)
        self.hidden_dim = 256

        # generate the actor encoder, actor output layer, and critic networks
        self.args = args
        self.actor_encoder, self.actor, self.af_name = self.generate_actor_encoder_and_actor()
        self.actor_logstd = nn.Parameter(torch.zeros(1, self.action_dim))
        self.critic = self.generate_critic()
        if args.use_shrink_and_perturb or args.use_regenerative_regularization:
            # back up the initial weights of the actor and critic
            self.init_actor_encoder = deepcopy(self.actor_encoder)
            self.init_actor = deepcopy(self.actor)
            self.init_critic = deepcopy(self.critic)

            # save the initial weights of the actor and critic
            self.init_params = [deepcopy(p.data) for p in self.parameters()]

        if args.use_normalize_and_project:
            # save the initial norms of the actor and critic
            self.initial_norms = {}
            for name, param in self.named_parameters():
                if 'weight' in name and 'norm' not in name:  # Skip normalization layer params
                    self.initial_norms[name] = param.data.norm(2).item()
    
    def forward(self, x):
        """for computing the RDU"""
        return self.actor_encoder(x)

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        hidden = self.actor_encoder(x)
        action_mean = self.actor(hidden)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)

    # NOTE: functions for supporting Plasticine starts from here
    def generate_actor_encoder_and_actor(self):
        """
        Generate an actor encoder and actor output layer for the agent.

        This function can generate networks with different activation functions and regularization methods:
        - CReLU, Paper: Loss of Plasticity in Continual Deep Reinforcement Learning (https://arxiv.org/pdf/2303.07507)
        - DFF, Paper: Plastic Learning with Deep Fourier Features (https://arxiv.org/pdf/2410.20634)
        - Layer Normalization, Paper: Disentangling the Causes of Plasticity Loss in Neural Networks (https://arxiv.org/pdf/2402.18762)
        - Parseval Regularization, Paper: Parseval Regularization for Continual Reinforcement Learning (https://arxiv.org/pdf/2412.07224)
        """
        if self.args.use_crelu_activation:
            actor_encoder = nn.Sequential(
                layer_init(nn.Linear(self.obs_dim, self.hidden_dim)),
                CReLU4Linear(),  # CReLU4Linear will double the output size
                nn.Linear(self.hidden_dim * 2, self.hidden_dim),
                CReLU4Linear(),  # CReLU4Linear will double the output size
            )
            actor = layer_init(nn.Linear(self.hidden_dim * 2, self.action_dim), std=0.01)
            return actor_encoder, actor, 'crelu'
        elif self.args.use_dff_activation:
            actor_encoder = nn.Sequential(
                layer_init(nn.Linear(self.obs_dim, self.hidden_dim)),
                DFF4Linear(),  # DFFLayer4Linear will double the output channels
                nn.Linear(self.hidden_dim * 2, self.hidden_dim),
                DFF4Linear(),  # DFFLayer4Linear will double the output channels
            )
            actor = layer_init(nn.Linear(self.hidden_dim * 2, self.action_dim), std=0.01)
            return actor_encoder, actor, 'dff'
        elif self.args.use_layer_norm:
            actor_encoder = nn.Sequential(
                layer_init(nn.Linear(self.obs_dim, self.hidden_dim)),
                nn.ReLU(),
                layer_init(nn.Linear(self.hidden_dim, self.hidden_dim)),
                nn.ReLU(),
                nn.LayerNorm(self.hidden_dim, elementwise_affine=False),
            )
            actor = layer_init(nn.Linear(self.hidden_dim, self.action_dim), std=0.01)
            return actor_encoder, actor, 'relu'
        elif self.args.use_parseval_regularization:
            actor_encoder = nn.Sequential(
                layer_init(nn.Linear(self.obs_dim, self.hidden_dim)),
                nn.ReLU(),
                PRLinear(
                    in_features=self.hidden_dim,
                    out_features=self.hidden_dim,
                    lambda_reg=getattr(self.args, 'parseval_lambda', 1e-3),
                    s=getattr(self.args, 'parseval_s', 1.0)),
                nn.ReLU(),
            )
            actor = layer_init(nn.Linear(self.hidden_dim, self.action_dim), std=0.01)
            return actor_encoder, actor, 'relu'
        else:
            actor_encoder = nn.Sequential(
                layer_init(nn.Linear(self.obs_dim, self.hidden_dim)),
                nn.ReLU(),
                layer_init(nn.Linear(self.hidden_dim, self.hidden_dim)),
                nn.ReLU(),
            )
            actor = layer_init(nn.Linear(self.hidden_dim, self.action_dim), std=0.01)
            return actor_encoder, actor, 'relu'
    
    def generate_critic(self):
        """
        Generate a critic network for the agent.
        """
        return nn.Sequential(
            layer_init(nn.Linear(self.obs_dim, self.hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(self.hidden_dim, self.hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(self.hidden_dim, 1), std=1.0),
        )

    def plasticine_shrink_and_perturb(self, shrink_p=0.999999):
        """
        Implementation of the Shrink and Perturb (SNP) algorithm.
        Paper: On Warm-Starting Neural Network Training (https://arxiv.org/pdf/1910.08475)

        Args:
            shrink_p (float): The shrink factor.
        """
        perturb_p = 1.0 - shrink_p
        def sp_module(current_module, init_module, shrink_factor, epsilon):
            use_device = next(current_module.parameters()).device
            init_params = list(init_module.to(use_device).parameters())
            for idx, current_param in enumerate(current_module.parameters()):
                current_param.data *= shrink_factor
                current_param.data += epsilon * init_params[idx].data

        # shrink the actor encoder and actor
        sp_module(self.actor_encoder, self.init_actor_encoder, shrink_p, perturb_p)
        sp_module(self.actor, self.init_actor, shrink_p, perturb_p)
        # shrink the critic
        sp_module(self.critic, self.init_critic, shrink_p, perturb_p)
        # TODO: if reset actor_logstd is needed, uncomment the following line
        # self.actor_logstd.data.zero_()
    
    def plasticine_reset_layers(self, reset_actor=True, reset_critic=True):
        """
        Implementation of the layer resetting algorithm.
        In default, we only reset the final layers of the agent.
        Paper: The Primacy Bias in Deep Reinforcement Learning (https://arxiv.org/pdf/2205.07802)

        Args:
            reset_actor (bool): Whether to reset the actor.
            reset_critic (bool): Whether to reset the critic.
        """
        device = next(self.parameters()).device
        if reset_actor:
            self.actor_encoder, self.actor, self.af_name = self.generate_actor_encoder_and_actor()
            self.actor_encoder.to(device)
            self.actor.to(device)
            self.actor_logstd.data.zero_().to(device)
        if reset_critic:
            self.critic = self.generate_critic()
            self.critic.to(device)
            
    def plasticine_normalize_and_project(self):
        """
        Implementation of the Normalize and Project (NaP) algorithm.
        Paper: Normalization and Effective Learning Rates in Reinforcement Learning (https://arxiv.org/pdf/2407.01800)
        """
        with torch.no_grad():
            for name, param in self.named_parameters():
                if 'weight' in name and 'norm' not in name and name in self.initial_norms:
                    # Project weight matrices to initial norm
                    target_norm = self.initial_norms[name]
                    current_norm = param.data.norm(2)
                    if current_norm > 0:
                        param.data.mul_(target_norm / current_norm)

                # Handle scale and offset parameters if specified
                if ('scale' in name or 'bias' in name):
                    # For homogeneous activations (like ReLU), we can normalize scale and offset together
                    if 'scale' in name:
                        # Find corresponding offset/bias parameter
                        offset_name = name.replace('scale', 'bias')
                        if offset_name in dict(self.named_parameters()):
                            scale_param = param
                            offset_param = dict(self.named_parameters())[offset_name]

                            # Compute joint norm and project
                            joint_norm = torch.sqrt(scale_param.data.pow(2).sum() + offset_param.data.pow(2).sum())
                            target_norm = torch.sqrt(torch.tensor(2.0))  # Initial norm for scale=1, offset=0
                            if joint_norm > 0:
                                scale_param.data.mul_(target_norm / joint_norm)
                                offset_param.data.mul_(target_norm / joint_norm)
    
    def plasticine_plasticity_injection(self):
        """
        Implementation of the Plasticity Injection (PI) algorithm.
        Paper: Deep Reinforcement Learning with Plasticity Injection (https://arxiv.org/pdf/2305.15555)
        """
        class Injector(nn.Module):
            def __init__(self, original, in_size=64, out_size=10):
                super(Injector, self).__init__()
                self.original = original
                self.new_a = nn.Linear(in_size, out_size)
                self.new_b = deepcopy(self.new_a)

            def forward(self, x):
                return self.original(x) + self.new_a(x) - self.new_b(x).detach()
        

        self.actor = Injector(self.actor, self.hidden_dim, self.action_dim)
        # Critic receives raw observation, so use obs_dim instead of hidden_dim
        self.critic = Injector(self.critic, self.obs_dim, 1)
        self.actor.to(next(self.parameters()).device)
        self.critic.to(next(self.parameters()).device)
    
    def plasticine_redo(self, batch_obs, tau=0.025):
        """
        Implementation of the Recycling Dormant neurons (ReDo) algorithm.
        Paper: The Dormant Neuron Phenomenon in Deep Reinforcement Learning (https://arxiv.org/pdf/2302.12902)

        This implementation is based on:
        https://github.com/google/dopamine/blob/ce36aab6528b26a699f5f1cefd330fdaf23a5d72/dopamine/labs/redo/weight_recyclers.py

        Args:
            batch_obs (torch.Tensor): A batch of observations.
            tau (float): The threshold for the ReDo operation.
        """
        def _get_activation(name, activations):
            """Hook function to capture activations with ReLU applied."""
            def hook(layer, input, output):
                activations[name] = F.relu(output)
            return hook

        def _get_redo_masks(activations, tau):
            """
            Computes the ReDo mask for a given set of activations.
            The returned mask has True where neurons are dormant and False where they are active.
            """
            masks = []
            # Process all activations from encoder Linear layers (all have ReLU after them)
            activation_items = list(activations.items())
            
            for name, activation in activation_items:
                # Taking the mean here conforms to the expectation under D in the main paper's formula
                if activation.ndim == 4:
                    # Conv layer: average over batch, height, width
                    score = activation.abs().mean(dim=(0, 2, 3))
                else:
                    # Linear layer: average over batch
                    score = activation.abs().mean(dim=0)

                # Divide by activation mean to make the threshold independent of the layer size
                # see https://github.com/google/dopamine/blob/ce36aab6528b26a699f5f1cefd330fdaf23a5d72/dopamine/labs/redo/weight_recyclers.py#L314
                normalized_score = score / (score.mean() + 1e-9)

                layer_mask = torch.zeros_like(normalized_score, dtype=torch.bool)
                if tau > 0.0:
                    layer_mask[normalized_score <= tau] = True
                else:
                    layer_mask[torch.isclose(normalized_score, torch.zeros_like(normalized_score))] = True
                masks.append(layer_mask)
            return masks

        def _kaiming_uniform_reinit(layer, mask):
            """Partially re-initializes the weights and bias of a layer according to the Kaiming uniform scheme."""
            fan_in = nn.init._calculate_correct_fan(tensor=layer.weight, mode="fan_in")
            gain = nn.init.calculate_gain(nonlinearity="relu", param=np.sqrt(5))
            std = gain / np.sqrt(fan_in)
            bound = np.sqrt(3.0) * std
            layer.weight.data[mask, ...] = torch.empty_like(layer.weight.data[mask, ...]).uniform_(-bound, bound)

            if layer.bias is not None:
                if fan_in != 0:
                    bound = 1 / np.sqrt(fan_in)
                    layer.bias.data[mask, ...] = torch.empty_like(layer.bias.data[mask, ...]).uniform_(-bound, bound)

        def _reset_dormant_neurons(layers, redo_masks, output_layer):
            """Re-initializes the dormant neurons of a model.
            
            Args:
                layers: List of (name, layer) tuples for encoder Linear layers
                redo_masks: List of masks for each layer
                output_layer: The actor output layer (used as next_layer for the last encoder layer)
            """
            assert len(redo_masks) == len(layers), "Number of masks must match the number of layers"

            for i in range(len(layers)):
                mask = redo_masks[i]
                layer = layers[i][1]
                # Use output_layer as next_layer for the last encoder layer
                next_layer = layers[i + 1][1] if i < len(layers) - 1 else output_layer

                # Skip if there are no dormant neurons
                if torch.all(~mask):
                    continue

                # 1. Reset the ingoing weights using the initialization distribution
                _kaiming_uniform_reinit(layer, mask)

                # 2. Reset the outgoing weights to 0
                if isinstance(layer, nn.Conv2d) and isinstance(next_layer, nn.Linear):
                    # Special case: Transition from conv to linear layer
                    num_repetition = next_layer.weight.data.shape[1] // mask.shape[0]
                    linear_mask = torch.repeat_interleave(mask, num_repetition)
                    next_layer.weight.data[:, linear_mask] = 0.0
                else:
                    # Standard case: layer and next_layer are both conv or both linear
                    next_layer.weight.data[:, mask, ...] = 0.0

        with torch.no_grad():
            activations = {}
            handles = []

            # Get all Linear layers from actor_encoder (all have ReLU after them)
            layers = [(name, layer) for name, layer in list(self.actor_encoder.named_modules())
                      if isinstance(layer, nn.Linear)]

            # Register hooks for all Linear layers
            for name, module in layers:
                handle = module.register_forward_hook(_get_activation(name, activations))
                handles.append(handle)

            # Forward pass to calculate activations
            _ = self.actor_encoder(batch_obs)

            # Remove the hooks
            for handle in handles:
                handle.remove()

            # Calculate the masks for resetting (all encoder layers)
            masks = _get_redo_masks(activations, tau)

            # Re-initialize the dormant neurons (use self.actor as next_layer for last encoder layer)
            _reset_dormant_neurons(layers, masks, self.actor)
    
    def plasticine_regenerative_regularization(self, rr_weight=0.01):
        """
        Implementation of the Regenerative Regularization (RR) algorithm.
        Paper: Maintaining Plasticity in Continual Learning via Regenerative Regularization (https://arxiv.org/abs/2308.11958)

        Args:
            rr_weight (float): The weight of the regenerative regularization loss.
        """
        # Get the device of current parameters
        device = next(self.parameters()).device
        params = torch.cat([p.view(-1) for p in self.parameters()])
        # Move initial parameters to the same device as current parameters
        params_0 = torch.cat([p.view(-1).to(device) for p in self.init_params])
        rr_loss = torch.norm(params - params_0.detach(), 2)
        return rr_weight * rr_loss
    
    def plasticine_parseval_regularization(self):
        """
        Implementation of the Parseval Regularization (PR) algorithm.
        Paper: Parseval Regularization for Continual Reinforcement Learning (https://arxiv.org/pdf/2412.07224)
        """
        total_loss = torch.tensor(0.0, device=next(self.parameters()).device)
        
        # Check all modules in actor_encoder for PRLinear
        for module in self.actor_encoder:
            if isinstance(module, PRLinear):
                total_loss = total_loss + module.pr_loss()
                    
        return total_loss
