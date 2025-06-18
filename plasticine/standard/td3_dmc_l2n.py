import os
os.environ["MUJOCO_GL"] = "egl"
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from stable_baselines3.common.buffers import ReplayBuffer
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
from plasticine.dmc_wrappers import StandardDMC

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
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    env_id: str = "Hopper-v4"
    """the id of the environment"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    policy_noise: float = 0.2
    """the scale of policy noise"""
    exploration_noise: float = 0.1
    """the scale of exploration noise"""
    learning_starts: int = 25e3
    """timestep to start learning"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    noise_clip: float = 0.5
    """noise clip parameter of the Target Policy Smoothing Regularization"""

    """------------------------Plasticine------------------------"""
    plasticity_eval_interval: int = 1000
    """the interval of evaluating the plasticity metrics"""
    plasticity_eval_size: int = 1000
    """the size of the evaluation data for the plasticity metrics"""
    # l2 norm arguments
    weight_decay: float = 1e-3
    """the weight decay coefficient"""
    """------------------------Plasticine------------------------"""


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        domain, task = env_id.split("_")
        if capture_video and idx == 0:
            env = StandardDMC(domain, task, task_kwargs={"random": seed})
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = StandardDMC(domain, task, task_kwargs={"random": seed})
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk


# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.input_dim = np.array(env.single_observation_space.shape).prod() + \
            np.prod(env.single_action_space.shape)
        self.value_encoder = self.gen_encoder()
        self.value = self.gen_value()
    
    def gen_encoder(self):
        enc = nn.Sequential(
            nn.Linear(self.input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        return enc
    
    def gen_value(self):
        return nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = self.value_encoder(x)

        return self.value(x)

    def get_features(self, x, a):
        x = torch.cat([x, a], 1)
        x = self.value_encoder(x)
        return x


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.input_dim = np.array(env.single_observation_space.shape).prod()
        self.action_dim = np.prod(env.single_action_space.shape)
        self.policy_encoder = self.gen_encoder()
        self.policy = self.gen_policy()

        # action rescaling
        self.register_buffer(
            "action_scale",
            torch.tensor(
                (env.single_action_space.high - env.single_action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor(
                (env.single_action_space.high + env.single_action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )

    def gen_encoder(self):
        enc = nn.Sequential(
            nn.Linear(self.input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        return enc
    
    def gen_policy(self):
        return nn.Linear(256, self.action_dim)
    
    def forward(self, x):
        x = self.policy_encoder(x)
        x = torch.tanh(self.policy(x))
        return x * self.action_scale + self.action_bias
    
    def get_features(self, x):
        x = self.policy_encoder(x)
        return x


if __name__ == "__main__":
    import stable_baselines3 as sb3

    if sb3.__version__ < "2.0":
        raise ValueError(
            """Ongoing migration: run the following command to install the new dependencies:
poetry run pip install "stable_baselines3==2.0.0a1"
"""
        )

    args = tyro.cli(Args)
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
    log_dir = 'std_td3_dmc_l2n_runs'
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
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    actor = Actor(envs).to(device)
    qf1 = QNetwork(envs).to(device)
    qf2 = QNetwork(envs).to(device)
    qf1_target = QNetwork(envs).to(device)
    qf2_target = QNetwork(envs).to(device)
    target_actor = Actor(envs).to(device)
    target_actor.load_state_dict(actor.state_dict())
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    
    """------------------------Plasticine------------------------"""
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.learning_rate, weight_decay=args.weight_decay)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.learning_rate, weight_decay=args.weight_decay)
    """------------------------Plasticine------------------------"""

    # save the initial state of the model
    actor_copy = save_model_state(actor)
    qf1_copy = save_model_state(qf1)

    envs.single_observation_space.dtype = np.float32
    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        n_envs=args.num_envs,
        handle_timeout_termination=False,
    )
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            with torch.no_grad():
                actions = actor(torch.Tensor(obs).to(device))
                actions += torch.normal(0, actor.action_scale * args.exploration_noise)
                actions = actions.cpu().numpy().clip(envs.single_action_space.low, envs.single_action_space.high)

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info is not None:
                    print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                    writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                    break

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)
            with torch.no_grad():
                clipped_noise = (torch.randn_like(data.actions, device=device) * args.policy_noise).clamp(
                    -args.noise_clip, args.noise_clip
                ) * target_actor.action_scale

                next_state_actions = (target_actor(data.next_observations) + clipped_noise).clamp(
                    envs.single_action_space.low[0], envs.single_action_space.high[0]
                )
                qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                qf2_next_target = qf2_target(data.next_observations, next_state_actions)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target)
                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (min_qf_next_target).view(-1)

            qf1_a_values = qf1(data.observations, data.actions).view(-1)
            qf2_a_values = qf2(data.observations, data.actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            # optimize the model
            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            # get the qf gradient norm but don't clip it
            qf_grad_norm = torch.nn.utils.clip_grad_norm_(qf1.parameters(), 1e10)
            writer.add_scalar("plasticity/value_grad_norm", qf_grad_norm.item(), global_step)

            if global_step % args.policy_frequency == 0:
                actor_loss = -qf1(data.observations, actor(data.observations)).mean()
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                # get the actor gradient norm but don't clip it
                actor_grad_norm = torch.nn.utils.clip_grad_norm_(actor.parameters(), 1e10)
                writer.add_scalar("plasticity/policy_grad_norm", actor_grad_norm.item(), global_step)

                # update the target network
                for param, target_param in zip(actor.parameters(), target_actor.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            # evaluate the plasticity metrics
            if global_step % args.plasticity_eval_interval == 0:
                eval_data = rb.sample(args.plasticity_eval_size)

                policy_hidden = actor.get_features(eval_data.observations)
                value_hidden = qf1.get_features(eval_data.observations, eval_data.actions)

                policy_dormant_units = compute_dormant_units(actor.policy_encoder, eval_data.observations, 'relu', tau=0.025)
                value_dormant_units = compute_dormant_units(qf1.value_encoder, torch.cat([eval_data.observations, eval_data.actions], dim=1), 'relu', tau=0.025)
                policy_active_units, value_active_units = compute_active_units(policy_hidden, 'relu'), compute_active_units(policy_hidden, 'relu')
                policy_stable_rank, value_stable_rank = compute_stable_rank(policy_hidden), compute_stable_rank(policy_hidden)
                policy_effective_rank, value_effective_rank = compute_effective_rank(policy_hidden), compute_effective_rank(policy_hidden)
                policy_feature_norm, value_feature_norm = compute_feature_norm(policy_hidden), compute_feature_norm(policy_hidden)
                policy_feature_var, value_feature_var = compute_feature_variance(policy_hidden), compute_feature_variance(policy_hidden)

                # overall metrics
                weight_magnitude = compute_weight_magnitude(actor) + compute_weight_magnitude(qf1)
                diff_l2_norm = compute_l2_norm_difference(actor, actor_copy) + compute_l2_norm_difference(qf1, qf1_copy)

                # log the metrics
                writer.add_scalar("plasticity/policy_active_units", policy_active_units.item(), global_step)
                writer.add_scalar("plasticity/policy_dormant_units", policy_dormant_units.item(), global_step)
                writer.add_scalar("plasticity/policy_stable_rank", policy_stable_rank.item(), global_step)
                writer.add_scalar("plasticity/policy_effective_rank", policy_stable_rank.item(), global_step)
                writer.add_scalar("plasticity/policy_feature_norm", policy_feature_norm.item(), global_step)
                writer.add_scalar("plasticity/policy_feature_variance", policy_feature_var.item(), global_step)
                
                writer.add_scalar("plasticity/value_active_units", value_active_units.item(), global_step)
                writer.add_scalar("plasticity/value_dormant_units", value_dormant_units.item(), global_step)
                writer.add_scalar("plasticity/value_stable_rank", value_stable_rank.item(), global_step)
                writer.add_scalar("plasticity/value_effective_rank", value_effective_rank.item(), global_step)
                writer.add_scalar("plasticity/value_feature_norm", value_feature_norm.item(), global_step)
                writer.add_scalar("plasticity/value_feature_variance", value_feature_var.item(), global_step)

                writer.add_scalar("plasticity/weight_magnitude", weight_magnitude.item(), global_step)
                writer.add_scalar("plasticity/l2_norm_difference", diff_l2_norm.item(), global_step)

                # save the new model states
                actor_copy = save_model_state(actor)
                qf1_copy = save_model_state(qf1)


            if global_step % 100 == 0:
                writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar(
                    "charts/SPS",
                    int(global_step / (time.time() - start_time)),
                    global_step,
                )

    envs.close()
    writer.close()

    # save model
    torch.save(actor.state_dict(), f"{log_dir}/{run_name}/agent.pth")