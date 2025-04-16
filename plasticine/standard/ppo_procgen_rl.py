# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_procgenpy
import os
import random
import time
from dataclasses import dataclass

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from procgen import ProcgenEnv
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

from plasticine.metrics import (compute_dormant_units, 
                                compute_active_units,
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
    wandb_project_name: str = "Plasticine"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "starpilot"
    """the id of the environment"""
    total_timesteps: int = int(25e6)
    """total timesteps of the experiments"""
    learning_rate: float = 5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 64
    """the number of parallel game environments"""
    num_steps: int = 256
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = False
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.999
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 8
    """the number of mini-batches"""
    update_epochs: int = 3
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""

    """------------------------Plasticine------------------------"""
    # Arguments for resetting the layers
    reset_type: str = 'final'
    """the type of resetting layer, can be 'final' or 'all'"""
    reset_frequency: int = 1000
    """the frequency of resetting layer"""
    """------------------------Plasticine------------------------"""


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


# taken from https://github.com/AIcrowd/neurips2020-procgen-starter-kit/blob/142d09586d2272a17f44481a115c4bd817cf6a94/models/impala_cnn_torch.py
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)
        )
        
    def forward(self, x):
        return self.block(x) + x


class ConvSequence(nn.Module):
    def __init__(self, input_shape, out_channels):
        super().__init__()
        self._input_shape = input_shape
        self._out_channels = out_channels
        self.conv = nn.Conv2d(in_channels=self._input_shape[0], out_channels=self._out_channels, kernel_size=3, padding=1)
        self.res_block0 = ResidualBlock(self._out_channels)
        self.res_block1 = ResidualBlock(self._out_channels)

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


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.obs_shape = envs.single_observation_space.shape
        self.action_dim = envs.single_action_space.n

        self.encoder = self.gen_encoder()
        self.policy = self.gen_policy()
        self.value = self.gen_value()

    def gen_encoder(self):
        h, w, c = self.obs_shape
        shape = (c, h, w)
        conv_seqs = []
        for out_channels in [16, 32, 32]:
            conv_seq = ConvSequence(shape, out_channels)
            shape = conv_seq.get_output_shape()
            conv_seqs.append(conv_seq)
        conv_seqs += [
            nn.Flatten(),
            nn.ReLU(),
            nn.Linear(in_features=shape[0] * shape[1] * shape[2], out_features=256),
            nn.ReLU(),
        ]
        return nn.Sequential(*conv_seqs)

    def gen_policy(self):
        return layer_init(nn.Linear(256, self.action_dim), std=0.01)
    
    def gen_value(self):
        return layer_init(nn.Linear(256, 1), std=1)

    def get_value(self, x):
        return self.value(self.encoder(x.permute((0, 3, 1, 2)) / 255.0))  # "bhwc" -> "bchw"

    def get_action_and_value(self, x, action=None, check=False):
        hidden = self.encoder(x.permute((0, 3, 1, 2)) / 255.0)  # "bhwc" -> "bchw"
        logits = self.policy(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()

        if check:
            with torch.no_grad():
                active_units = compute_active_units(hidden, 'relu')
                stable_rank = compute_stable_rank(hidden)
                effective_rank = compute_effective_rank(hidden)
                feature_norm = compute_feature_norm(hidden)
                feature_var = compute_feature_variance(hidden)
                plasticity_metrics = {
                    "active_units": active_units.item(),
                    "stable_rank": stable_rank.item(),
                    "effective_rank": effective_rank.item(),
                    "feature_norm": feature_norm.item(),
                    "feature_var": feature_var.item(),
                }
            return action, probs.log_prob(action), probs.entropy(), self.value(hidden), plasticity_metrics
        else:
            return action, probs.log_prob(action), probs.entropy(), self.value(hidden)

    def forward(self, x):
        """for computing the RDU"""
        return self.encoder(x.permute((0, 3, 1, 2)) / 255.0)  # "bhwc" -> "bchw"

    """------------------------Plasticine------------------------"""
    def shrink_perturb(self, reset_type):
        shrink_p, perturb_p = 0.0, 1.0
        if reset_type == 'final':
            shrink_encoder, shrink_value, shrink_policy = False, True, True
        elif reset_type == 'all':
            shrink_encoder, shrink_value, shrink_policy = True, True, True
        else:
            raise NotImplementedError(f"reset_type {reset_type} not implemented")

        if shrink_encoder:
            new_enc = self.gen_encoder()
            self.sp_module(self.encoder, new_enc, shrink_p, perturb_p)
        if shrink_value:
            new_value = self.gen_value()
            self.sp_module(self.value, new_value, shrink_p, perturb_p)
        if shrink_policy:
            new_policy = self.gen_policy()
            self.sp_module(self.policy, new_policy, shrink_p, perturb_p)

    def sp_module(self, current_module, init_module, shrink_factor, epsilon):
        use_device = next(current_module.parameters()).device
        init_params = list(init_module.to(use_device).parameters())
        for idx, current_param in enumerate(current_module.parameters()):
            current_param.data *= shrink_factor
            current_param.data += epsilon * init_params[idx].data
    """------------------------Plasticine------------------------"""


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
    log_dir = 'std_ppo_procgen_rl_runs'
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
    envs = ProcgenEnv(num_envs=args.num_envs, env_name=args.env_id, num_levels=0, start_level=0, distribution_mode="easy")
    envs = gym.wrappers.TransformObservation(envs, lambda obs: obs["rgb"])
    envs.single_action_space = envs.action_space
    envs.single_observation_space = envs.observation_space["rgb"]
    envs.is_vector_env = True
    envs = gym.wrappers.RecordEpisodeStatistics(envs)
    if args.capture_video:
        envs = gym.wrappers.RecordVideo(envs, f"videos/{run_name}")
    envs = gym.wrappers.NormalizeReward(envs, gamma=args.gamma)
    envs = gym.wrappers.TransformReward(envs, lambda reward: np.clip(reward, -10, 10))
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

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

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, next_done, info = envs.step(action.cpu().numpy())
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            for item in info:
                if "episode" in item.keys():
                    print(f"global_step={global_step}, episodic_return={item['episode']['r']}")
                    writer.add_scalar("charts/episodic_return", item["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", item["episode"]["l"], global_step)
                    break

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []

        # plasticity metrics
        total_active_units = []
        total_stable_rank = []
        total_effective_rank = []
        total_grad_norm = []
        total_policy_ent = []
        total_feature_norm = []
        total_feature_var = []
        # copy the agent
        agent_copy = save_model_state(agent)

        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue, plasticity_metrics = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds], check=True)
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                batch_grad_norm = nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

                # log plasticity metrics
                total_active_units.append(plasticity_metrics["active_units"])
                total_stable_rank.append(plasticity_metrics["stable_rank"])
                total_effective_rank.append(plasticity_metrics["effective_rank"])
                total_feature_norm.append(plasticity_metrics["feature_norm"])
                total_feature_var.append(plasticity_metrics["feature_var"])
                total_grad_norm.append(batch_grad_norm.item())
                total_policy_ent.append(entropy_loss.item())

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        """------------------------Plasticine------------------------"""
        # shrink and perturb the agent (episode-level)
        if iteration % args.reset_frequency == 0:
            agent.shrink_perturb(reset_type=args.reset_type)
        """------------------------Plasticine------------------------"""

        # compute the l2 norm difference
        diff_l2_norm = compute_l2_norm_difference(agent, agent_copy)
        # compute weight magnitude
        weight_magnitude = compute_weight_magnitude(agent)

        # compute dormant units
        if iteration % 10 == 0:
            dormant_units = compute_dormant_units(agent, b_obs[mb_inds], 'relu', tau=0.025)
            writer.add_scalar("plasticity/dormant_units", dormant_units, global_step)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        # add plasticity metrics
        writer.add_scalar("plasticity/active_units", np.mean(total_active_units), global_step)
        writer.add_scalar("plasticity/stable_rank", np.mean(total_stable_rank), global_step)
        writer.add_scalar("plasticity/effective_rank", np.mean(total_effective_rank), global_step)
        writer.add_scalar("plasticity/weight_magnitude", weight_magnitude.item(), global_step)
        writer.add_scalar("plasticity/l2_norm_difference", diff_l2_norm.item(), global_step)
        writer.add_scalar("plasticity/grad_norm", np.mean(total_grad_norm), global_step)
        writer.add_scalar("plasticity/policy_entropy", np.mean(total_policy_ent), global_step)
        writer.add_scalar("plasticity/feature_norm", np.mean(total_feature_norm), global_step)
        writer.add_scalar("plasticity/feature_variance", np.mean(total_feature_var), global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    writer.close()

    # save model 
    torch.save(agent.state_dict(), f"{log_dir}/{run_name}/agent.pt")