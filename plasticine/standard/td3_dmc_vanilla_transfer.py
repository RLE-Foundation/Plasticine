# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/td3/#td3_continuous_actionpy
import os
import random
import time
from dataclasses import dataclass,field
import glob

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro

import dmc_wrappers

os.environ['MUJOCO_GL'] = 'egl'

from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter

from plasticine.metrics import (compute_dormant_units, 
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
    random_tasks: bool = True
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
    env_ids: list = field(default_factory=list)
    """the ids of the environments"""
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

    plasticity_eval_interval: int = 1000
    """the interval of evaluating the plasticity metrics"""
    plasticity_eval_size: int = 1000
    """the size of the evaluation data for the plasticity metrics"""
    
    """------------------------Plasticine------------------------"""
    num_steps_per_round: int = 1000
    """the number of episodes per round"""
    cont_mode: str = "task"
    """the mode of the continual learning task, `level` or `task`"""
    """------------------------Plasticine------------------------"""

# def make_env(env_ids, seed, idx, capture_video, run_name):
#     print(f"env_id: {env_ids[idx]}")
#     env = dmc_wrappers.DeepMindControl(
#         env_ids=env_ids,
#         mode=args.cont_mode
#     )
#     if capture_video and idx == 0:
#         env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
    
#     return env

def make_env(env_ids, seed, idx, capture_video, run_name):
    def thunk():
        env = dmc_wrappers.DeepMindControl(
            env_ids=env_ids,
            mode=args.cont_mode
        )
        if capture_video and idx == 0:
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        
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
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    
    actor = None
    qf1 = None
    qf2 = None
    target_actor = None
    qf1_target = None
    qf2_target = None
    q_optimizer = None
    actor_optimizer = None

    global_timestamp = int(time.time())
    
    # if args.random_tasks:
    #     import random
    #     random.shuffle(args.env_ids)
    
    for cycle in range(3):
        for per_round in range(len(args.env_ids)):  
            # TRY NOT TO MODIFY: seeding
            random.seed(args.seed)
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            torch.backends.cudnn.deterministic = args.torch_deterministic

            # env setup
            # envs = make_env(args.env_ids, args.seed, 1, args.capture_video, run_name) 
            # breakpoint()
            log_dir = 'std_td3_mujoco_vanilla_runs'
            parent_dir = f"{log_dir}/{args.exp_name}_{args.seed}_{global_timestamp}"
            run_name = parent_dir
            
            envs = gym.vector.SyncVectorEnv(
                [make_env(args.env_ids, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
            )
            assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"
            
            log_dir = 'std_td3_mujoco_vanilla_runs'
            parent_dir = f"{log_dir}/{args.exp_name}_{args.seed}_{global_timestamp}"
            run_name = parent_dir
            if args.track:
                import wandb
                wandb.tensorboard.unpatch()
                wandb.tensorboard.patch(root_logdir=parent_dir)
                wandb.init(
                    project=args.wandb_project_name,
                    entity=args.wandb_entity,
                    sync_tensorboard=True,
                    config=vars(args),
                    name=run_name,
                    monitor_gym=True,
                    save_code=True,
                )
            
            for env in envs.envs:
                env_id = env.env_id
                break
            
            env_subdir = f"{parent_dir}/{env_id}"
            # os.makedirs(env_subdir, exist_ok=True)
            writer = SummaryWriter(f"{env_subdir}")
            writer.add_text(
                "hyperparameters",
                "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
            )

            
            if actor is None:  
                actor = Actor(envs).to(device)
                qf1 = QNetwork(envs).to(device)
                qf2 = QNetwork(envs).to(device)
                qf1_target = QNetwork(envs).to(device)
                qf2_target = QNetwork(envs).to(device)
                target_actor = Actor(envs).to(device)
                target_actor.load_state_dict(actor.state_dict())
                qf1_target.load_state_dict(qf1.state_dict())
                qf2_target.load_state_dict(qf2.state_dict())
                q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.learning_rate)
                actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.learning_rate)
            else:  
                
                source_agent = f"{parent_dir}/{pre_env_id}/agent.pth"
                if os.path.exists(source_agent):
                    
                    print(f"Loading {pre_env_id} model from {parent_dir}/{pre_env_id}")
                    actor.load_state_dict(torch.load(f"{parent_dir}/{pre_env_id}/agent.pth"))
                    qf1.load_state_dict(torch.load(f"{parent_dir}/{pre_env_id}/qf1.pth"))
                    qf2.load_state_dict(torch.load(f"{parent_dir}/{pre_env_id}/qf2.pth"))
                    target_actor.load_state_dict(actor.state_dict())
                    qf1_target.load_state_dict(qf1.state_dict())
                    qf2_target.load_state_dict(qf2.state_dict())
                
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

                    if global_step % args.policy_frequency == 0:
                        actor_loss = -qf1(data.observations, actor(data.observations)).mean()
                        actor_optimizer.zero_grad()
                        actor_loss.backward()
                        actor_optimizer.step()

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
                        hidden = torch.hstack([policy_hidden, value_hidden])
                        dormant_units = compute_dormant_units(hidden, 'relu')
                        stable_rank = compute_stable_rank(hidden)
                        effective_rank = compute_effective_rank(hidden)
                        feature_norm = compute_feature_norm(hidden)
                        feature_var = compute_feature_variance(hidden)
                        weight_magnitude = compute_weight_magnitude(actor) + compute_weight_magnitude(qf1)
                        diff_l2_norm = compute_l2_norm_difference(actor, actor_copy) + compute_l2_norm_difference(qf1, qf1_copy)
                        feature_norm = compute_feature_norm(hidden)
                        feature_var = compute_feature_variance(hidden)
                        writer.add_scalar("plasticity/dormant_units", dormant_units.item(), global_step)
                        writer.add_scalar("plasticity/stable_rank", stable_rank.item(), global_step)
                        writer.add_scalar("plasticity/effective_rank", effective_rank.item(), global_step)
                        writer.add_scalar("plasticity/weight_magnitude", weight_magnitude.item(), global_step)
                        writer.add_scalar("plasticity/l2_norm_difference", diff_l2_norm.item(), global_step)
                        writer.add_scalar("plasticity/feature_norm", feature_norm.item(), global_step)
                        writer.add_scalar("plasticity/feature_variance", feature_var.item(), global_step)

                    if global_step % 100 == 0:
                        writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                        writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step)
                        writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                        writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                        writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                        writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                        print("SPS:", int(global_step / (time.time() - start_time)))
                        writer.add_scalar(
                            "charts/SPS", int(global_step / (time.time() - start_time)), global_step,
                        )

            envs.close()
            writer.close()

            # save model
            torch.save(actor.state_dict(), f"{env_subdir}/agent.pth")
            torch.save(qf1.state_dict(), f"{env_subdir}/qf1.pth")
            torch.save(qf2.state_dict(), f"{env_subdir}/qf2.pth")
            pre_env_id = env_id
            
            """------------------------Plasticine------------------------"""
            # shift the environment
            for env in envs.envs:
                env.shift()
            """------------------------Plasticine------------------------"""