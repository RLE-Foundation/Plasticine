# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/c51/#c51_ataripy
import os
import random
import time
from dataclasses import dataclass
from collections import deque

import gymnasium as gym
import ale_py
gym.register_envs(ale_py)
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter

from kron_torch import Kron

from plasticine_envs.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)
from plasticine.buffers import ReplayBuffer

from plasticine.c51_atari_base import PlasticineQNetwork
from plasticine.utils import get_exp_name, save_model_state
from plasticine.trac import start_trac
from plasticine_metrics.metrics import (compute_dormant_units, 
                                        compute_active_units,
                                        compute_stable_rank, 
                                        compute_effective_rank,
                                        compute_l2_norm_difference,
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
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    env_id: str = "BreakoutNoFrameskip-v4"
    """the id of the environment"""
    total_timesteps: int = 10000000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    n_atoms: int = 51
    """the number of atoms"""
    v_min: float = -10
    """the return lower bound"""
    v_max: float = 10
    """the return upper bound"""
    buffer_size: int = 1000000
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    target_network_frequency: int = 10000
    """the timesteps it takes to update the target network"""
    batch_size: int = 32
    """the batch size of sample from the reply memory"""
    start_e: float = 1
    """the starting epsilon for exploration"""
    end_e: float = 0.01
    """the ending epsilon for exploration"""
    exploration_fraction: float = 0.10
    """the fraction of `total-timesteps` it takes from start-e to go end-e"""
    learning_starts: int = 80000
    """timestep to start learning"""
    train_frequency: int = 4
    """the frequency of training"""

    # Plasticine specific arguments
    use_shrink_and_perturb: bool = False
    """whether to use the Shrink and Perturb (SNP) algorithm"""
    use_normalize_and_project: bool = False
    """whether to use the Normalize and Project (NaP) algorithm"""
    use_regenerative_regularization: bool = False
    """whether to use the Regenerative Regularization (RR) algorithm"""
    use_parseval_regularization: bool = False
    """whether to use the Parseval Regularization (PR) algorithm"""
    use_crelu_activation: bool = False
    """whether to use the Concatenated ReLUs (CReLUs) activation"""
    use_dff_activation: bool = False
    """whether to use the Deep Fourier Features (DFF) activation"""
    use_layer_resetting: bool = False
    """whether to use the Layer Resetting algorithm"""
    use_redo: bool = False
    """whether to use the Recycling Dormant Neurons (ReDo) algorithm"""
    use_plasticity_injection: bool = False
    """whether to use the Plasticity Injection algorithm"""
    use_l2_norm: bool = False
    """whether to use the L2 Normalization (L2N) algorithm"""
    use_layer_norm: bool = False
    """whether to use the Layer Normalization (LN) algorithm"""
    use_trac_optimizer: bool = False
    """whether to use the TRAC optimizer"""
    use_kron_optimizer: bool = False
    """whether to use the Kron optimizer"""
    plasticine_frequency: int = 10000
    """the frequency of the plasticity operations (in timesteps)"""


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)

        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayScaleObservation(env)
        env = gym.wrappers.FrameStack(env, 4)

        env.action_space.seed(seed)
        return env

    return thunk


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


if __name__ == "__main__":
    args = tyro.cli(Args)
    assert args.num_envs == 1, "vectorized envs are not supported at the moment"
    group_name = f"{args.env_id}__{get_exp_name(args)}"
    run_name = f"{group_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            group=group_name,
            # monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"c51_atari_runs/{run_name}")
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
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    q_network = PlasticineQNetwork(envs, args, n_atoms=args.n_atoms, v_min=args.v_min, v_max=args.v_max).to(device)
    print(q_network)
    """🎯============================== Plasticine Operations ==============================🎯"""
    if args.use_trac_optimizer:
        optimizer = start_trac(f'c51_atari_runs/{run_name}/trac.text', torch.optim.Adam)(
            q_network.parameters(), lr=args.learning_rate, eps=0.01 / args.batch_size)
    elif args.use_kron_optimizer:
        optimizer = Kron(q_network.parameters(), lr=args.learning_rate)
    elif args.use_l2_norm:
        optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate, eps=0.01 / args.batch_size, weight_decay=1e-3)
    else:
        optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate, eps=0.01 / args.batch_size)
    """🎯============================== Plasticine Operations ==============================🎯"""
    target_network = PlasticineQNetwork(envs, args, n_atoms=args.n_atoms, v_min=args.v_min, v_max=args.v_max).to(device)
    target_network.load_state_dict(q_network.state_dict())

    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        optimize_memory_usage=True,
        handle_timeout_termination=False,
    )
    start_time = time.time()
    
    # Track if plasticity operations have been applied at mid-training
    plasticity_injection_mid_applied = False
    layer_resetting_mid_applied = False
    total_grad_norm = deque(maxlen=10)
    q_network_copy = save_model_state(q_network)

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            actions, pmf = q_network.get_action(torch.Tensor(obs).to(device))
            actions = actions.cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                    writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

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
            if global_step % args.train_frequency == 0:
                data = rb.sample(args.batch_size)
                with torch.no_grad():
                    _, next_pmfs = target_network.get_action(data.next_observations)
                    next_atoms = data.rewards + args.gamma * target_network.atoms * (1 - data.dones)
                    # projection
                    delta_z = target_network.atoms[1] - target_network.atoms[0]
                    tz = next_atoms.clamp(args.v_min, args.v_max)

                    b = (tz - args.v_min) / delta_z
                    l = b.floor().clamp(0, args.n_atoms - 1)
                    u = b.ceil().clamp(0, args.n_atoms - 1)
                    # (l == u).float() handles the case where bj is exactly an integer
                    # example bj = 1, then the upper ceiling should be uj= 2, and lj= 1
                    d_m_l = (u + (l == u).float() - b) * next_pmfs
                    d_m_u = (b - l) * next_pmfs
                    target_pmfs = torch.zeros_like(next_pmfs)
                    for i in range(target_pmfs.size(0)):
                        target_pmfs[i].index_add_(0, l[i].long(), d_m_l[i])
                        target_pmfs[i].index_add_(0, u[i].long(), d_m_u[i])

                _, old_pmfs = q_network.get_action(data.observations, data.actions.flatten())
                loss = (-(target_pmfs * old_pmfs.clamp(min=1e-5, max=1 - 1e-5).log()).sum(-1)).mean()

                """🎯============================== Plasticine Operations ==============================🎯"""
                # NOTE: every training batch
                if args.use_regenerative_regularization:
                    rr_loss = q_network.plasticine_regenerative_regularization(rr_weight=0.01)
                    loss += rr_loss
                if args.use_parseval_regularization:
                    pr_loss = q_network.plasticine_parseval_regularization()
                    loss += pr_loss
                """🎯============================== Plasticine Operations ==============================🎯"""

                if global_step % 100 == 0:
                    writer.add_scalar("losses/loss", loss.item(), global_step)
                    old_val = (old_pmfs * q_network.atoms).sum(1)
                    writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)
                    print("SPS:", int(global_step / (time.time() - start_time)))
                    writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

                # optimize the model
                optimizer.zero_grad()
                loss.backward()
                # get the gradient norm but don't clip it
                batch_grad_norm = torch.nn.utils.clip_grad_norm_(q_network.parameters(), 1e10)
                total_grad_norm.append(batch_grad_norm.item())
                optimizer.step()

                """🎯============================== Plasticine Operations ==============================🎯"""
                # NOTE: every training batch
                if args.use_shrink_and_perturb:
                    q_network.plasticine_shrink_and_perturb(shrink_p=0.999999)
                """🎯============================== Plasticine Operations ==============================🎯"""

            # update target network
            if global_step % args.target_network_frequency == 0:
                target_network.load_state_dict(q_network.state_dict())

            """🎯============================== Plasticine Operations ==============================🎯"""
            # NOTE: based on environment steps (global_step)
            # if global_step % args.plasticine_frequency == 0 and global_step > args.learning_starts:
            if global_step > 4096:
                if args.use_normalize_and_project:
                    q_network.plasticine_normalize_and_project()
                elif args.use_redo:
                    # Sample a batch from replay buffer for ReDo
                    redo_data = rb.sample(2048)
                    q_network.plasticine_redo(redo_data.observations, tau=0.025)
                    print('did redo')
            # NOTE: plasticity injection at the middle of training
            if args.use_plasticity_injection and not plasticity_injection_mid_applied and global_step >= args.total_timesteps // 2:
                q_network.plasticine_plasticity_injection()
                plasticity_injection_mid_applied = True
            # NOTE: layer resetting at the middle of training
            if args.use_layer_resetting and not layer_resetting_mid_applied and global_step >= args.total_timesteps // 2:
                q_network.plasticine_reset_layers(reset_encoder=True, reset_policy=True)
                layer_resetting_mid_applied = True
            """🎯============================== Plasticine Operations ==============================🎯"""

            """🎯============================== Plasticine Operations ==============================🎯"""
            # NOTE: compute plasticity metrics
            if global_step % 10000 == 0 and global_step > args.learning_starts:
                # Sample a batch from replay buffer for metrics computation
                metrics_data = rb.sample(2048)
                with torch.no_grad():
                    hidden, _ = q_network(metrics_data.observations)
                dormant_units = compute_dormant_units(q_network.encoder, metrics_data.observations / 255.0, q_network.af_name, tau=0.025)
                active_units = compute_active_units(hidden, q_network.af_name)
                stable_rank = compute_stable_rank(hidden)
                effective_rank = compute_effective_rank(hidden)
                diff_l2_norm = compute_l2_norm_difference(q_network, q_network_copy)
                grad_norm = np.mean(total_grad_norm)

                writer.add_scalar("plasticity/dormant_units", dormant_units, global_step)
                writer.add_scalar("plasticity/active_units", active_units, global_step)
                writer.add_scalar("plasticity/stable_rank", stable_rank, global_step)
                writer.add_scalar("plasticity/effective_rank", effective_rank, global_step)
                writer.add_scalar("plasticity/gradient_norm", grad_norm, global_step)
                writer.add_scalar("plasticity/l2_norm_difference", diff_l2_norm.item(), global_step)

                q_network_copy = save_model_state(q_network)
            """🎯============================== Plasticine Operations ==============================🎯"""

    envs.close()
    writer.close()
