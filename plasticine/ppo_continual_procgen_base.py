import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from torch.distributions.categorical import Categorical
from copy import deepcopy

from utils import PRLinear, layer_init, get_activation

# taken from https://github.com/AIcrowd/neurips2020-procgen-starter-kit/blob/142d09586d2272a17f44481a115c4bd817cf6a94/models/impala_cnn_torch.py
class ResidualBlock(nn.Module):
    def __init__(self, channels, activation='relu'):
        super().__init__()
        dilation = 2 if 'crelu' in activation or 'dff' in activation else 1
        self.block = nn.Sequential(
            get_activation(activation),
            nn.Conv2d(in_channels=channels * dilation, out_channels=channels, kernel_size=3, padding=1),
            get_activation(activation),
            nn.Conv2d(in_channels=channels * dilation, out_channels=channels, kernel_size=3, padding=1)
        )
        
    def forward(self, x):
        return self.block(x) + x

class ConvSequence(nn.Module):
    def __init__(self, input_shape, out_channels, activation='relu'):
        super().__init__()
        self._input_shape = input_shape
        self._out_channels = out_channels
        self.conv = nn.Conv2d(in_channels=self._input_shape[0], out_channels=self._out_channels, kernel_size=3, padding=1)
        self.res_block0 = ResidualBlock(self._out_channels, activation)
        self.res_block1 = ResidualBlock(self._out_channels, activation)

    def forward(self, x):
        x = self.conv(x)
        x = nn.functional.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x = self.res_block0(x)
        x = self.res_block1(x)
        assert x.shape[1:] == self.get_output_shape()
        return x

    def get_output_shape(self):
        _c, h, w = self._input_shape
        return (self._out_channels, (h + 1) // 2, (w + 1) // 2)

class PlasticineAgent(nn.Module):
    def __init__(self, envs, args):
        super().__init__()
        self.obs_shape = envs.single_observation_space.shape
        self.action_dim = envs.single_action_space.n

        # generate the encoder, policy network, and value network
        self.args = args
        self.encoder, self.af_name = self.generate_encoder()
        self.actor = self.generate_actor()
        self.critic = self.generate_critic()

        if args.use_shrink_and_perturb or args.use_regenerative_regularization:
            # back up the initial weights of the encoder, actor, and critic
            self.init_encoder = deepcopy(self.encoder)
            self.init_actor = deepcopy(self.actor)
            self.init_critic = deepcopy(self.critic)

            # save the initial weights of the encoder, actor, and critic
            # Convert parameters generator to list and deepcopy each parameter's data
            self.init_params = [deepcopy(p.data).to(args.device) for p in self.parameters()]

        if args.use_normalize_and_project:
            # save the initial norms of the encoder, actor, and critic
            self.initial_norms = {}
            for name, param in self.named_parameters():
                if 'weight' in name and 'norm' not in name:  # Skip normalization layer params
                    self.initial_norms[name] = param.data.norm(2).item()
    
    def forward(self, x):
        """for computing the RDU"""
        hidden = self.encoder(x.permute((0, 3, 1, 2)) / 255.0)  # "bhwc" -> "bchw"
        logits = self.actor(hidden)
        values = self.critic(hidden)

        return hidden, logits, values

    def get_value(self, x):
        return self.critic(self.encoder(x.permute((0, 3, 1, 2)) / 255.0))  # "bhwc" -> "bchw"

    def get_action_and_value(self, x, action=None):
        hidden = self.encoder(x.permute((0, 3, 1, 2)) / 255.0)  # "bhwc" -> "bchw"
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)

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
        h, w, c = self.obs_shape
        shape = (c, h, w)
        conv_seqs = []
        
        if self.args.use_crelu_activation:
            for out_channels in [16, 32, 32]:
                conv_seq = ConvSequence(shape, out_channels, activation='crelu-conv')
                shape = conv_seq.get_output_shape()
                conv_seqs.append(conv_seq)
            conv_seqs += [
                nn.Flatten(),
                get_activation('crelu-linear'), # CReLU4Linear will double the output size
                nn.Linear(in_features=shape[0] * shape[1] * shape[2] * 2, out_features=256),
                get_activation('crelu-linear'), # CReLU4Linear will double the output size
            ]
            return nn.Sequential(*conv_seqs), 'crelu'
        elif self.args.use_dff_activation:
            for out_channels in [16, 32, 32]:
                conv_seq = ConvSequence(shape, out_channels, activation='dff-conv')
                shape = conv_seq.get_output_shape()
                conv_seqs.append(conv_seq)
            conv_seqs += [
                nn.Flatten(),
                get_activation('dff-linear'), # DFFLayer4Linear will double the output channels
                nn.Linear(in_features=shape[0] * shape[1] * shape[2] * 2, out_features=256),
                get_activation('dff-linear'), # DFFLayer4Linear will double the output channels
            ]
            return nn.Sequential(*conv_seqs), 'dff'
        elif self.args.use_layer_norm:
            for out_channels in [16, 32, 32]:
                conv_seq = ConvSequence(shape, out_channels)
                shape = conv_seq.get_output_shape()
                conv_seqs.append(conv_seq)
            conv_seqs += [
                nn.Flatten(),
                get_activation('relu'),
                nn.Linear(in_features=shape[0] * shape[1] * shape[2], out_features=256),
                nn.LayerNorm(256), # add layer norm
                get_activation('relu'),
            ]
            return nn.Sequential(*conv_seqs), 'relu'
        elif self.args.use_parseval_regularization:
            for out_channels in [16, 32, 32]:
                conv_seq = ConvSequence(shape, out_channels)
                shape = conv_seq.get_output_shape()
                conv_seqs.append(conv_seq)
            conv_seqs += [
                nn.Flatten(),
                get_activation('relu'),
                PRLinear(
                    in_features=shape[0] * shape[1] * shape[2], 
                    out_features=256,
                    lambda_reg=getattr(self.args, 'parseval_lambda', 1e-3),
                    s=getattr(self.args, 'parseval_s', 1.0)),
                get_activation('relu')
            ]
            return nn.Sequential(*conv_seqs), 'relu'
        else:
            for out_channels in [16, 32, 32]:
                conv_seq = ConvSequence(shape, out_channels)
                shape = conv_seq.get_output_shape()
                conv_seqs.append(conv_seq)
            conv_seqs += [
                nn.Flatten(),
                get_activation('relu'),
                nn.Linear(in_features=shape[0] * shape[1] * shape[2], out_features=256),
                get_activation('relu'),
            ]
            return nn.Sequential(*conv_seqs), 'relu'

    def generate_actor(self):
        """
        Generate an actor network for the agent.
        """
        dilation = 2 if 'crelu' in self.af_name or 'dff' in self.af_name else 1
        return layer_init(nn.Linear(256 * dilation, self.action_dim), std=0.01)
    
    def generate_critic(self):
        """
        Generate a critic network for the agent.
        """
        dilation = 2 if 'crelu' in self.af_name or 'dff' in self.af_name else 1
        return layer_init(nn.Linear(256 * dilation, 1), std=1)

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

        # shrink the encoder, actor, and critic
        sp_module(self.encoder, self.init_encoder, shrink_p, perturb_p)
        sp_module(self.actor, self.init_actor, shrink_p, perturb_p)
        sp_module(self.critic, self.init_critic, shrink_p, perturb_p)
    
    def plasticine_reset_layers(self, reset_encoder=False, reset_critic=True, reset_actor=True):
        """
        Implementation of the layer resetting algorithm.
        In default, we only reset the final layers of the agent.
        Paper: The Primacy Bias in Deep Reinforcement Learning (https://arxiv.org/pdf/2205.07802)

        Args:
            reset_encoder (bool): Whether to reset the encoder.
            reset_critic (bool): Whether to reset the critic.
            reset_actor (bool): Whether to reset the actor.
        """
        device = next(self.parameters()).device
        if reset_encoder:
            self.encoder, self.af_name = self.generate_encoder()
            self.encoder.to(device)
        if reset_critic:
            self.critic = self.generate_critic()
            self.critic.to(device)
        if reset_actor:
            self.actor = self.generate_actor()
            self.actor.to(device)
    
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
            def __init__(self, original, in_size=256, out_size=10):
                super(Injector, self).__init__()
                self.original = original
                self.new_a = nn.Linear(in_size, out_size)
                self.new_b = deepcopy(self.new_a)

            def forward(self, x):
                return self.original(x) + self.new_a(x) - self.new_b(x).detach()
        
        self.actor = Injector(self.actor, 256, self.action_dim)
        self.critic = Injector(self.critic, 256, 1)
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
            """Hook function to capture ReLU activations."""
            def hook(layer, input, output):
                activations[name] = output.detach()
            return hook

        def _get_redo_masks(activations, tau):
            """
            Computes the ReDo mask for a given set of activations.
            The returned mask has True where neurons are dormant and False where they are active.
            """
            masks = {}
            for name, activation in activations.items():
                # Taking the mean conforms to the expectation under D in the main paper's formula
                if activation.ndim == 4:
                    # Conv layer: average over batch, height, width
                    score = activation.abs().mean(dim=(0, 2, 3))
                else:
                    # Linear layer: average over batch
                    score = activation.abs().mean(dim=0)

                # Divide by activation mean to make the threshold independent of the layer size
                normalized_score = score / (score.mean() + 1e-9)

                layer_mask = torch.zeros_like(normalized_score, dtype=torch.bool)
                if tau > 0.0:
                    layer_mask[normalized_score <= tau] = True
                else:
                    layer_mask[torch.isclose(normalized_score, torch.zeros_like(normalized_score))] = True
                masks[name] = layer_mask
            return masks

        def _kaiming_uniform_reinit(layer, mask):
            """Partially re-initializes weights and bias according to Kaiming uniform scheme."""
            fan_in = nn.init._calculate_correct_fan(tensor=layer.weight, mode="fan_in")
            gain = nn.init.calculate_gain(nonlinearity="relu", param=np.sqrt(5))
            std = gain / np.sqrt(fan_in)
            bound = np.sqrt(3.0) * std
            layer.weight.data[mask, ...] = torch.empty_like(layer.weight.data[mask, ...]).uniform_(-bound, bound)

            if layer.bias is not None:
                if fan_in != 0:
                    bound = 1 / np.sqrt(fan_in)
                    layer.bias.data[mask, ...] = torch.empty_like(layer.bias.data[mask, ...]).uniform_(-bound, bound)

        def _reset_dormant_neurons(modules, links, masks):
            """Re-initializes dormant neurons based on layer links.
            
            Args:
                modules: Dict of module name -> module
                links: List of (current_layer, relu_layer, next_layer) tuples
                masks: Dict of relu_layer name -> dormant mask
            """
            for current_name, relu_name, next_name in links:
                if relu_name not in masks:
                    continue
                mask = masks[relu_name]
                if torch.all(~mask):
                    continue

                current_layer = modules[current_name]
                next_layer = modules[next_name]

                # 1. Reset ingoing weights
                _kaiming_uniform_reinit(current_layer, mask)

                # 2. Reset outgoing weights to 0
                if isinstance(current_layer, nn.Conv2d) and isinstance(next_layer, nn.Linear):
                    # Conv2d -> Flatten -> Linear: need to expand mask
                    num_repetition = next_layer.weight.data.shape[1] // mask.shape[0]
                    linear_mask = torch.repeat_interleave(mask, num_repetition)
                    next_layer.weight.data[:, linear_mask] = 0.0
                else:
                    next_layer.weight.data[:, mask, ...] = 0.0

        with torch.no_grad():
            activations = {}
            handles = []
            modules = dict(self.named_modules())

            # Define links: (current_layer, relu_layer, next_layer)
            # Each link represents: current_layer's output -> relu_layer -> next_layer
            links = []
            for enc_sub in ['encoder.0', 'encoder.1', 'encoder.2']:
                links.extend([
                    (f'{enc_sub}.conv',               f'{enc_sub}.res_block0.block.0', f'{enc_sub}.res_block0.block.1'),
                    (f'{enc_sub}.res_block0.block.1', f'{enc_sub}.res_block0.block.2', f'{enc_sub}.res_block0.block.3'),
                    (f'{enc_sub}.res_block0.block.3', f'{enc_sub}.res_block1.block.0', f'{enc_sub}.res_block1.block.1'),
                    (f'{enc_sub}.res_block1.block.1', f'{enc_sub}.res_block1.block.2', f'{enc_sub}.res_block1.block.3'),
                ])
            # Add encoder.5 (Linear) -> encoder.6 (ReLU) -> actor
            links.append(('encoder.5', 'encoder.6', 'actor'))

            # Get unique ReLU layer names to hook
            relu_names = set(link[1] for link in links)

            # Register hooks for ReLU layers
            for name in relu_names:
                if name in modules:
                    handle = modules[name].register_forward_hook(_get_activation(name, activations))
                    handles.append(handle)

            # Forward pass to calculate activations
            _ = self.encoder(batch_obs.permute((0, 3, 1, 2)) / 255.0)

            # Remove hooks
            for handle in handles:
                handle.remove()

            # Calculate masks for all ReLU layers
            masks = _get_redo_masks(activations, tau)

            # Reset dormant neurons
            _reset_dormant_neurons(modules, links, masks)
    
    def plasticine_regenerative_regularization(self, rr_weight=0.01):
        """
        Implementation of the Regenerative Regularization (RR) algorithm.
        Paper: Maintaining Plasticity in Continual Learning via Regenerative Regularization (https://arxiv.org/abs/2308.11958)

        Args:
            rr_weight (float): The weight of the regenerative regularization loss.
        """
        params = torch.cat([p.view(-1) for p in self.parameters()])
        params_0 = torch.cat([p.view(-1) for p in self.init_params])
        rr_loss = torch.norm(params - params_0.detach(), 2)
        return rr_weight * rr_loss
    
    def plasticine_parseval_regularization(self):
        """
        Implementation of the Parseval Regularization (PR) algorithm.
        Paper: Parseval Regularization for Continual Reinforcement Learning (https://arxiv.org/pdf/2412.07224)
        """
        total_loss = torch.tensor(0.0, device=next(self.parameters()).device)
        
        # Check all modules in policy and value encoders
        for module in [self.encoder]:
            for layer in module:
                if layer.__class__.__name__ == "PRLinear":
                    total_loss = total_loss + layer.pr_loss()
                    
        return total_loss