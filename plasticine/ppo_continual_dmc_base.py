import torch
import torch.nn as nn
import numpy as np

from torch.distributions.normal import Normal
from copy import deepcopy

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

        Args:
            batch_obs (torch.Tensor): A batch of observations.
            tau (float): The threshold for the ReDo operation.
        """
        def _compute_neuron_scores(model, data):
            # Create a dictionary to store the s scores for each layer
            s_scores_dict = {}

            # Register a forward hook to capture the activations of each layer
            activations = {}
            hooks = []

            def hook(module, input, output):
                activations[module] = output.detach()

            for name, module in model.named_modules():
                if isinstance(module, torch.nn.ReLU):
                    handle = module.register_forward_hook(hook)
                    hooks.append(handle)

            # Forward pass through the model
            model(data)

            # Calculate the s scores for each layer
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.ReLU):
                    layer_activations = activations[module]
                    s_scores = layer_activations / (torch.mean(layer_activations, axis=0, keepdim=True) + 1e-6)
                    s_scores = torch.mean(s_scores, axis=0)
                    s_scores_dict[name] = s_scores

            # Remove the hooks to prevent memory leaks
            for handle in hooks:
                handle.remove()

            return s_scores_dict

        def _reinitialize_weights(module, reset_mask, next_module):
            """
            Reinitializes weights and biases of a module based on a reset mask.

            Args:
                module (torch.nn.Module): The module whose weights are to be reinitialized.
                reset_mask (torch.Tensor): A boolean tensor indicating which weights to reset.
                next_module (torch.nn.Module): The next module in the network.
            """
            # Reinitialize weights
            new_weights = torch.empty_like(module.weight.data)
            torch.nn.init.kaiming_uniform_(new_weights, a=np.sqrt(5))
            module.weight.data[reset_mask] = new_weights[reset_mask].to(module.weight.device)
            
            # Reinitialize bias if exists
            if module.bias is not None:
                new_bias = torch.empty_like(module.bias.data)
                fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(module.weight)
                bound = 1 / np.sqrt(fan_in)
                torch.nn.init.uniform_(new_bias, -bound, bound)
                module.bias.data[reset_mask] = new_bias[reset_mask].to(module.bias.device)

            # Set outgoing weights to zero for reset neurons
            if next_module is not None:
                if isinstance(module, nn.Conv2d) and isinstance(next_module, nn.Linear):
                    # Conv2d -> Flatten -> Linear: need to expand mask
                    C = reset_mask.shape[0]  # number of channels
                    spatial_size = next_module.in_features // C  # H * W
                    expanded_mask = reset_mask.unsqueeze(1).expand(-1, spatial_size).flatten()
                    next_module.weight.data[:, expanded_mask] = 0.0
                elif isinstance(next_module, (nn.Linear, nn.Conv2d)):
                    next_module.weight.data[:, reset_mask] = 0.0

        with torch.no_grad():
            s_scores_dict = _compute_neuron_scores(self.actor_encoder, batch_obs)
            modules = dict(self.actor_encoder.named_modules())

            # Process each layer in the actor_encoder Sequential
            layer_names = [name for name, _ in self.actor_encoder.named_modules() if isinstance(_, nn.Linear)]
            for i, layer_name in enumerate(layer_names):
                if i < len(layer_names) - 1:
                    # Find the ReLU activation after this layer
                    next_layer_name = layer_names[i + 1]
                    # Find the ReLU between these layers
                    for relu_name, relu_module in modules.items():
                        if isinstance(relu_module, nn.ReLU) and layer_name < relu_name < next_layer_name:
                            s_scores = s_scores_dict[relu_name]
                            reset_mask = s_scores <= tau
                            current_layer = modules[layer_name]
                            next_layer = modules[next_layer_name] if next_layer_name in modules else None
                            _reinitialize_weights(current_layer, reset_mask, next_layer)
                            break
    
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
