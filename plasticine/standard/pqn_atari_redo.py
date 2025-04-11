# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/pqn/#pqn_atari_envpoolpy
import os
import random
import time
from collections import deque
from dataclasses import dataclass

import envpool
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter

from plasticine.metrics import (compute_active_units,
                                compute_dormant_units, 
                                compute_stable_rank, 
                                compute_effective_rank, 
                                compute_feature_norm, 
                                compute_feature_variance,
                                compute_weight_magnitude, 
                                compute_l2_norm_difference, 
                                save_model_state
                                )

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "Breakout-v5"
    """the id of the environment"""
    total_timesteps: int = 10000000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 8
    """the number of parallel game environments"""
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    max_grad_norm: float = 10.0
    """the maximum norm for the gradient clipping"""
    start_e: float = 1
    """the starting epsilon for exploration"""
    end_e: float = 0.01
    """the ending epsilon for exploration"""
    exploration_fraction: float = 0.10
    """the fraction of `total_timesteps` it takes from start_e to end_e"""
    q_lambda: float = 0.65
    """the lambda for the Q-Learning algorithm"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""

    """------------------------Plasticine------------------------"""
    plasticity_eval_interval: int = 10
    """the interval of evaluating the plasticity metrics"""
    # Arguments for the ReDo operation
    redo_tau: float = 0.025
    """the weight of the ReDo loss"""
    redo_frequency: int = 10
    """the frequency of the ReDo operation"""
    """------------------------Plasticine------------------------"""


class RecordEpisodeStatistics(gym.Wrapper):
    def __init__(self, env, deque_size=100):
        super().__init__(env)
        self.num_envs = getattr(env, "num_envs", 1)
        self.episode_returns = None
        self.episode_lengths = None

    def reset(self, **kwargs):
        observations = super().reset(**kwargs)
        self.episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        self.lives = np.zeros(self.num_envs, dtype=np.int32)
        self.returned_episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.returned_episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        return observations

    def step(self, action):
        observations, rewards, dones, infos = super().step(action)
        self.episode_returns += infos["reward"]
        self.episode_lengths += 1
        self.returned_episode_returns[:] = self.episode_returns
        self.returned_episode_lengths[:] = self.episode_lengths
        self.episode_returns *= 1 - infos["terminated"]
        self.episode_lengths *= 1 - infos["terminated"]
        infos["r"] = self.returned_episode_returns
        infos["l"] = self.returned_episode_lengths
        return (
            observations,
            rewards,
            dones,
            infos,
        )


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.action_dim = env.single_action_space.n
        self.encoder = self.gen_encoder()
        self.value = self.gen_value()

    def gen_encoder(self):
        return nn.Sequential(
            layer_init(nn.Conv2d(4, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(3136, 512)),
            nn.ReLU(),
        )
    
    def get_features(self, x):
        return self.encoder(x / 255.0)

    def gen_value(self):
        return layer_init(nn.Linear(512, self.action_dim))

    def forward(self, x):
        return self.value(self.encoder(x / 255.0))


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


def redo_reset(model, batch_obs, tau):
    """
    Apply the ReDo operation to the model.

    Args:
        model (torch.nn.Module): The model to be modified.
        batch_obs (torch.Tensor): The batch of observations.
        tau (float): The threshold for the ReDo operation.

    Returns:
        None
    """
    with torch.no_grad():
        s_scores_dict = compute_neuron_scores(model, batch_obs)
        modules = [m for m in model.named_modules() if isinstance(m[1], torch.nn.Linear)]

        # check if there are any conv layers in the network
        has_conv = any(isinstance(m[1], torch.nn.Conv2d) for m in modules)

        for i, (name, module) in enumerate(modules):
            # Skip the first entry, which is the model itself in named_modules()
            if not isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
                continue  # Skip non-relevant modules
            if not has_conv:
                base_name_parts = name.split(".")[:-1] + ["1.0"]
                base_name = ".".join(base_name_parts)
            elif ("policy" not in name) and ("value" not in name):
                base_name_parts = name.split(".")
                base_name_parts[-1] = str(int(base_name_parts[-1]) + 1)
                base_name_parts.append("0")
                base_name = ".".join(base_name_parts)
            else:
                continue
            if base_name in s_scores_dict:
                s_scores = s_scores_dict[base_name]
                reset_mask = s_scores <= tau

                # Check if there is a next module in the list and get it
                next_module = modules[i + 1][1] if i + 1 < len(modules) else None
                # Assuming reinitialize_weights is modified to handle the next_module
                # You would need to adjust reinitialize_weights to apply the necessary changes
                # to both the current and next modules based on reset_mask.
                reinitialize_weights(module, reset_mask, next_module)


def compute_neuron_scores(model, data):
    # Create a dictionary to store the s scores for each layer
    s_scores_dict = {}

    # Register a forward hook to capture the activations of each layer
    activations = {}
    hooks = []

    def hook(module, input, output):
        activations[module] = output.detach()

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Tanh):
            handle = module.register_forward_hook(hook)
            hooks.append(handle)

    # Forward pass through the model
    model(data)

    # Calculate the s scores for each layer
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.ReLU):
            layer_activations = activations[module]
            s_scores = layer_activations / (torch.mean(layer_activations, axis=1, keepdim=True) + 1e-6)
            s_scores = torch.mean(s_scores, axis=0)
            if len(s_scores.shape) > 1:
                s_scores = torch.mean(s_scores, axis=tuple(range(1, len(s_scores.shape))))
            s_scores_dict[name] = s_scores

    # Remove the hooks to prevent memory leaks
    for handle in hooks:
        handle.remove()

    return s_scores_dict


def reinitialize_weights(module, reset_mask, next_module):
    """
    Reinitializes weights and biases of a module based on a reset mask.

    Args:
        module (torch.nn.Module): The module whose weights are to be reinitialized.
        reset_mask (torch.Tensor): A boolean tensor indicating which weights to reset.
    """
    # Reinitialize weights
    new_weights = torch.empty_like(module.weight.data)
    torch.nn.init.orthogonal_(new_weights, np.sqrt(2))
    module.weight.data[reset_mask] = new_weights[reset_mask].to(module.weight.device)

    # Set outgoing weights to zero for reset neurons
    if type(module) == type(next_module):
        next_module.weight.data[:, reset_mask] = 0.0


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    log_dir = 'std_pqn_atari_redo_runs'
    writer = SummaryWriter(f"{log_dir}/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = envpool.make(
        args.env_id,
        env_type="gym",
        num_envs=args.num_envs,
        episodic_life=True,
        reward_clip=True,
        seed=args.seed,
    )
    envs.num_envs = args.num_envs
    envs.single_action_space = envs.action_space
    envs.single_observation_space = envs.observation_space
    envs = RecordEpisodeStatistics(envs)
    assert isinstance(envs.action_space, gym.spaces.Discrete), "only discrete action space is supported"

    q_network = QNetwork(envs).to(device)
    # save the initial state of the model
    q_network_copy = save_model_state(q_network)
    optimizer = optim.RAdam(q_network.parameters(), lr=args.learning_rate)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    avg_returns = deque(maxlen=20)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs = torch.Tensor(envs.reset()).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    for iteration in range(1, args.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)

            random_actions = torch.randint(0, envs.single_action_space.n, (args.num_envs,)).to(device)
            with torch.no_grad():
                q_values = q_network(next_obs)
                max_actions = torch.argmax(q_values, dim=1)
                values[step] = q_values[torch.arange(args.num_envs), max_actions].flatten()

            explore = torch.rand((args.num_envs,)).to(device) < epsilon
            action = torch.where(explore, random_actions, max_actions)
            actions[step] = action

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, next_done, info = envs.step(action.cpu().numpy())
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            for idx, d in enumerate(next_done):
                if d and info["lives"][idx] == 0:
                    print(f"global_step={global_step}, episodic_return={info['r'][idx]}")
                    avg_returns.append(info["r"][idx])
                    writer.add_scalar("charts/avg_episodic_return", np.average(avg_returns), global_step)
                    writer.add_scalar("charts/episodic_return", info["r"][idx], global_step)
                    writer.add_scalar("charts/episodic_length", info["l"][idx], global_step)

        # Compute Q(lambda) targets
        with torch.no_grad():
            returns = torch.zeros_like(rewards).to(device)
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    next_value, _ = torch.max(q_network(next_obs), dim=-1)
                    nextnonterminal = 1.0 - next_done
                    returns[t] = rewards[t] + args.gamma * next_value * nextnonterminal
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    next_value = values[t + 1]
                    returns[t] = (
                        rewards[t]
                        + args.gamma * (args.q_lambda * returns[t + 1] + (1 - args.q_lambda) * next_value) * nextnonterminal
                    )

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_returns = returns.reshape(-1)

        total_grad_norm = []

        # Optimizing the Q-network
        b_inds = np.arange(args.batch_size)
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                old_val = q_network(b_obs[mb_inds]).gather(1, b_actions[mb_inds].unsqueeze(-1).long()).squeeze()
                loss = F.mse_loss(b_returns[mb_inds], old_val)

                # optimize the model
                optimizer.zero_grad()
                loss.backward()
                batch_grad_norm = nn.utils.clip_grad_norm_(q_network.parameters(), args.max_grad_norm)
                optimizer.step()

                total_grad_norm.append(batch_grad_norm.item())

        """------------------------Plasticine------------------------"""
        # ReDo operation
        if iteration % args.redo_frequency == 0 and iteration > 0:
            redo_reset(q_network, b_obs[mb_inds], args.redo_tau)
        """------------------------Plasticine------------------------"""

        # compute plasticity metrics
        if iteration % args.plasticity_eval_interval == 0:
            with torch.no_grad():
                features = q_network.get_features(b_obs)
                metrics = {
                    "effective_rank": compute_effective_rank(features).item(),
                    "stable_rank": compute_stable_rank(features),
                    "feature_variance": compute_feature_variance(features),
                    "feature_norm": compute_feature_norm(features),
                    "weight_magnitude": compute_weight_magnitude(q_network),
                    "l2_norm_difference": compute_l2_norm_difference(q_network, q_network_copy),
                    "active_units": compute_active_units(features, 'relu'),
                    "dormant_units": compute_dormant_units(q_network, b_obs, 'relu', 0.025),
                    "grad_norm": np.mean(total_grad_norm)
                }
                for metric_name, metric_value in metrics.items():
                    writer.add_scalar(f"plasticity/{metric_name}", metric_value, global_step)

        writer.add_scalar("losses/td_loss", loss, global_step)
        writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()

    # save model 
    torch.save(q_network.state_dict(), f"{log_dir}/{run_name}/agent.pt")