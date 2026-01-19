import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from copy import deepcopy

from plasticine.utils import CReLU4Conv2d, CReLU4Linear, DFF4Conv2d, DFF4Linear, PRLinear

class PlasticineQNetwork(nn.Module):
    def __init__(self, env, args, n_atoms=101, v_min=-100, v_max=100):
        super().__init__()
        self.env = env
        self.args = args
        self.n_atoms = n_atoms
        self.register_buffer("atoms", torch.linspace(v_min, v_max, steps=n_atoms))
        self.n = env.single_action_space.n
        self.obs_shape = (4, 84, 84)  # Atari observation shape after preprocessing
        self.hidden_dim = 512

        # generate the encoder
        self.encoder, self.af_name = self.generate_encoder()
        # generate the policy network
        self.policy = self.generate_policy()

        if args.use_shrink_and_perturb or args.use_regenerative_regularization:
            # back up the initial weights of the encoder and policy
            self.init_encoder = deepcopy(self.encoder)
            self.init_policy = deepcopy(self.policy)
            # save the initial weights of the encoder and policy
            self.init_params = [deepcopy(p.data) for p in self.parameters()]
    
        if args.use_normalize_and_project:
            # save the initial norms of the encoder and policy
            self.initial_norms = {}
            for name, param in self.named_parameters():
                if 'weight' in name and 'norm' not in name:  # Skip normalization layer params
                    self.initial_norms[name] = param.data.norm(2).item()
    
    def forward(self, x):
        """for computing the RDU"""
        hidden = self.encoder(x / 255.0)
        logits = self.policy(hidden)
        return hidden, logits

    def get_action(self, x, action=None):
        hidden = self.encoder(x / 255.0)
        logits = self.policy(hidden)
        # probability mass function for each action
        pmfs = torch.softmax(logits.view(len(x), self.n, self.n_atoms), dim=2)
        q_values = (pmfs * self.atoms).sum(2)
        if action is None:
            action = torch.argmax(q_values, 1)
        return action, pmfs[torch.arange(len(x)), action]

    # NOTE: functions for supporting Plasticine starts from here
    def generate_encoder(self):
        """
        Generate an encoder for the agent.

        This function can generate an encoder with different activation functions and regularization methods:
        - CReLU, Paper: Loss of Plasticity in Continual Deep Reinforcement Learning (https://arxiv.org/pdf/2303.07507)
        - DFF, Paper: Plastic Learning with Deep Fourier Features (https://arxiv.org/pdf/2410.20634)
        - Layer Normalization, Paper: Disentangling the Causes of Plasticity Loss in Neural Networks (https://arxiv.org/pdf/2402.18762)
        - Parseval Regularization, Paper: Parseval Regularization for Continual Reinforcement Learning (https://arxiv.org/pdf/2412.07224)
        """
        if self.args.use_crelu_activation:
            return nn.Sequential(
                nn.Conv2d(4, 32, 8, stride=4),
                CReLU4Conv2d(), # CReLU4Conv2d will double the output channels
                nn.Conv2d(32*2, 64, 4, stride=2),
                CReLU4Conv2d(), # CReLU4Conv2d will double the output channels
                nn.Conv2d(64*2, 64, 3, stride=1),
                CReLU4Conv2d(), # CReLU4Conv2d will double the output channels
                nn.Flatten(),
                nn.Linear(3136*2, 512),
                CReLU4Linear(),
            ), 'crelu'
        elif self.args.use_dff_activation:
            return nn.Sequential(
                nn.Conv2d(4, 32, 8, stride=4),
                DFF4Conv2d(), # DFF4Conv2d will double the output channels
                nn.Conv2d(32*2, 64, 4, stride=2),
                DFF4Conv2d(), # DFF4Conv2d will double the output channels
                nn.Conv2d(64*2, 64, 3, stride=1),
                DFF4Conv2d(), # DFF4Conv2d will double the output channels
                nn.Flatten(),
                nn.Linear(3136*2, 512),
                DFF4Linear(),
            ), 'dff'
        elif self.args.use_layer_norm:
            return nn.Sequential(
                nn.Conv2d(4, 32, 8, stride=4),
                nn.LayerNorm([32, 20, 20]), # add layer norm
                nn.ReLU(),
                nn.Conv2d(32, 64, 4, stride=2),
                nn.LayerNorm([64, 9, 9]), # add layer norm
                nn.ReLU(),
                nn.Conv2d(64, 64, 3, stride=1),
                nn.LayerNorm([64, 7, 7]), # add layer norm
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(3136, 512),
                nn.LayerNorm(512), # add layer norm
            ), 'relu'
        elif self.args.use_parseval_regularization:
            return nn.Sequential(
                nn.Conv2d(4, 32, 8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, 4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, 3, stride=1),
                nn.ReLU(),
                nn.Flatten(),
                PRLinear(3136, 512),
                nn.ReLU(),
            ), 'relu'
        else:
            return nn.Sequential(
                nn.Conv2d(4, 32, 8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, 4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, 3, stride=1),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(3136, 512),
                nn.ReLU(),
            ), 'relu'

    def generate_policy(self):
        """
        Generate a policy network for the agent, which aligns with the encoder.
        """
        if self.args.use_crelu_activation:
            return nn.Linear(512*2, self.n * self.n_atoms)
        elif self.args.use_dff_activation:
            return nn.Linear(512*2, self.n * self.n_atoms)
        return nn.Linear(512, self.n * self.n_atoms)

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

        # shrink the encoder and policy
        sp_module(self.encoder, self.init_encoder, shrink_p, perturb_p)
        sp_module(self.policy, self.init_policy, shrink_p, perturb_p)
    
    def plasticine_reset_layers(self, reset_encoder=False, reset_policy=True):
        """
        Implementation of the layer resetting algorithm.
        In default, we only reset the final layers of the agent.
        Paper: The Primacy Bias in Deep Reinforcement Learning (https://arxiv.org/pdf/2205.07802)

        Args:
            reset_encoder (bool): Whether to reset the encoder.
            reset_policy (bool): Whether to reset the policy.
        """
        device = next(self.parameters()).device
        if reset_encoder:
            self.encoder, _ = self.generate_encoder()
            self.encoder.to(device)
        if reset_policy:
            self.policy = self.generate_policy()
            self.policy.to(device)
    
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
            def __init__(self, original, in_size=512, out_size=10):
                super(Injector, self).__init__()
                self.original = original
                self.new_a = nn.Linear(in_size, out_size)
                self.new_b = deepcopy(self.new_a)

            def forward(self, x):
                return self.original(x) + self.new_a(x) - self.new_b(x).detach()
        
        # For policy, we need to get the input size of the last layer
        last_layer = None
        for module in reversed(list(self.policy.modules())):
            if isinstance(module, nn.Linear):
                last_layer = module
                break
        if last_layer is not None:
            last_layer_input_size = last_layer.in_features
        else:
            last_layer_input_size = self.hidden_dim
        
        self.policy = Injector(self.policy, last_layer_input_size, self.n * self.n_atoms)
        self.policy.to(next(self.parameters()).device)
    
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
            # Last activation are the output values, which are never reset
            activation_items = list(activations.items())[:-1]
            
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

        def _reset_dormant_neurons(layers, redo_masks):
            """Re-initializes the dormant neurons of a model."""
            assert len(redo_masks) == len(layers) - 1, "Number of masks must match the number of layers - 1"

            for i in range(len(layers) - 1):
                mask = redo_masks[i]
                layer = layers[i][1]
                next_layer = layers[i + 1][1]

                # Skip if there are no dormant neurons
                if torch.all(~mask):
                    continue

                # 1. Reset the ingoing weights using the initialization distribution
                _kaiming_uniform_reinit(layer, mask)

                # 2. Reset the outgoing weights to 0
                if isinstance(layer, nn.Conv2d) and isinstance(next_layer, nn.Linear):
                    # Special case: Transition from conv to linear layer
                    # Reset the outgoing weights to 0 with a mask created from the conv filters
                    num_repetition = next_layer.weight.data.shape[1] // mask.shape[0]
                    linear_mask = torch.repeat_interleave(mask, num_repetition)
                    next_layer.weight.data[:, linear_mask] = 0.0
                else:
                    # Standard case: layer and next_layer are both conv or both linear
                    next_layer.weight.data[:, mask, ...] = 0.0

        with torch.no_grad():
            activations = {}
            handles = []

            # Get all Conv2d and Linear layers (skip the Sequential container at index 0)
            layers = [(name, layer) for name, layer in list(self.encoder.named_modules())[1:]
                      if isinstance(layer, (nn.Conv2d, nn.Linear))]

            # Register hooks for all Conv2d and Linear layers
            for name, module in layers:
                handle = module.register_forward_hook(_get_activation(name, activations))
                handles.append(handle)

            # Forward pass to calculate activations
            _ = self.encoder(batch_obs / 255.0)

            # Remove the hooks
            for handle in handles:
                handle.remove()

            # Calculate the masks for resetting
            masks = _get_redo_masks(activations, tau)

            # Re-initialize the dormant neurons
            _reset_dormant_neurons(layers, masks)
    
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
        
        # Check all modules in encoder
        for module in self.encoder:
            if isinstance(module, PRLinear):
                total_loss = total_loss + module.pr_loss()
                    
        return total_loss
    
