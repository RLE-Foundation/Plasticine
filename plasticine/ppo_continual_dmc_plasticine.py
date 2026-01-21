# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter

from kron_torch import Kron

from plasticine.trac import start_trac
from plasticine.ppo_continual_dmc_base import PlasticineAgent
from plasticine.utils import get_exp_name, save_model_state
from plasticine_envs.dmc_wrappers import ContinualDMC
from plasticine_metrics.metrics import (compute_dormant_units, 
                                        compute_active_units,
                                        compute_stable_rank, 
                                        compute_effective_rank,
                                        compute_l2_norm_difference,
                                        )

@dataclass
class Args:
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
    env_id: str = "quadruped"
    """the id of the environment (dog, walker, or quadruped)"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    num_steps: int = 2048
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = False
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 32
    """the number of mini-batches"""
    update_epochs: int = 10
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.0
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

    # Plasticine specific arguments
    number_of_individual_tasks: int = 0
    """the number of individual tasks (loaded in runtime)"""
    task_timesteps: int = int(1e6)
    """the timesteps for each task"""
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
    plasticine_frequency: int = 10
    """the frequency of the plasticity operations"""

if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.task_timesteps // args.batch_size
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
    writer = SummaryWriter(f"ppo_continual_dmc_runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = ContinualDMC(
        env_id=args.env_id,
        num_envs=args.num_envs,
        seed=args.seed,
    )
    args.number_of_individual_tasks = envs.num_tasks
    print(f"number of individual tasks: {args.number_of_individual_tasks}")
    print(f"task list: {envs.get_task_list()}")
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    agent = PlasticineAgent(envs, args).to(device)

    # NOTE: Plasticine operations
    if args.use_trac_optimizer:
        optimizer = start_trac(f'ppo_continual_dmc_runs/{run_name}/trac.text', torch.optim.Adam)(agent.parameters(), lr=args.learning_rate)
    elif args.use_l2_norm:
        optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5, weight_decay=1e-3)
    elif args.use_kron_optimizer:
        optimizer = Kron(agent.parameters(), lr=args.learning_rate)
    else:
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
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    # switch individual task
    for task_id in range(1, args.number_of_individual_tasks + 1):
        """🎯============================== Plasticine Operations ==============================🎯"""
        # NOTE: switch task
        if task_id > 1:
            envs.switch()
            next_obs, _ = envs.reset(seed=args.seed)
            next_obs = torch.Tensor(next_obs).to(device)
            next_done = torch.zeros(args.num_envs).to(device)
        """🎯============================== Plasticine Operations ==============================🎯"""
        # individual task training
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
                next_obs, reward, terminated, truncated, infos = envs.step(action.cpu().numpy())
                done = np.logical_or(terminated, truncated)
                rewards[step] = torch.tensor(reward).to(device).view(-1)
                next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)

                if "final_info" in infos:
                    for info in infos["final_info"]:
                        if info and "episode" in info:
                            print(f"current task={envs.get_current_task()}", f"iteration={iteration}", f"episodic_return={info['episode']['r']}", f"global_step={global_step}")
                            writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                            writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
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

            total_grad_norm = []
            agent_copy = save_model_state(agent)

            for epoch in range(args.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, args.batch_size, args.minibatch_size):
                    end = start + args.minibatch_size
                    mb_inds = b_inds[start:end]

                    _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
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

                    # NOTE: plasticity operations
                    """🎯============================== Plasticine Operations ==============================🎯"""
                    # NOTE: every mini-batch
                    if args.use_regenerative_regularization:
                        rr_loss = agent.plasticine_regenerative_regularization(rr_weight=0.01)
                        loss += rr_loss
                    if args.use_parseval_regularization:
                        pr_loss = agent.plasticine_parseval_regularization()
                        loss += pr_loss
                    """🎯============================== Plasticine Operations ==============================🎯"""

                    optimizer.zero_grad()
                    loss.backward()
                    batch_grad_norm = nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    total_grad_norm.append(batch_grad_norm.item())
                    optimizer.step()

                    """🎯============================== Plasticine Operations ==============================🎯"""
                    # NOTE: every mini-batch
                    if args.use_shrink_and_perturb:
                        agent.plasticine_shrink_and_perturb(shrink_p=0.999999)
                    """🎯============================== Plasticine Operations ==============================🎯"""

                if args.target_kl is not None and approx_kl > args.target_kl:
                    break

            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

            """🎯============================== Plasticine Operations ==============================🎯"""
            # NOTE: after each rollout
            if iteration % args.plasticine_frequency == 0 and iteration > 0:
                if args.use_normalize_and_project:
                    agent.plasticine_normalize_and_project()
                elif args.use_redo:
                    agent.plasticine_redo(b_obs, tau=0.025)
                elif args.use_plasticity_injection:
                    agent.plasticine_plasticity_injection()
            """🎯============================== Plasticine Operations ==============================🎯"""

            # TRY NOT TO MODIFY: record rewards for plotting purposes
            writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
            writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
            writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
            writer.add_scalar("losses/explained_variance", explained_var, global_step)
            print("SPS:", int(global_step / (time.time() - start_time)))
            writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

            # NOTE: compute plasticity metrics
            if iteration % 10 == 0:
                # use the whole rollout data to compute plasticity metrics
                hidden = agent.actor_encoder(b_obs)
                dormant_units = compute_dormant_units(agent.actor_encoder, b_obs, agent.af_name, tau=0.025)
                active_units = compute_active_units(hidden, agent.af_name)
                stable_rank = compute_stable_rank(hidden)
                effective_rank = compute_effective_rank(hidden)
                # NOTE: plasticity injection will change the model architecture
                try:    
                    diff_l2_norm = compute_l2_norm_difference(agent, agent_copy)
                except:
                    diff_l2_norm = torch.tensor(0.0).to(device)
                grad_norm = np.mean(total_grad_norm)

                writer.add_scalar("plasticity/dormant_units", dormant_units, global_step)
                writer.add_scalar("plasticity/active_units", active_units, global_step)
                writer.add_scalar("plasticity/stable_rank", stable_rank, global_step)
                writer.add_scalar("plasticity/effective_rank", effective_rank, global_step)
                writer.add_scalar("plasticity/gradient_norm", grad_norm, global_step)
                writer.add_scalar("plasticity/l2_norm_difference", diff_l2_norm.item(), global_step)

        """🎯============================== Plasticine Operations ==============================🎯"""
        # NOTE: after the whole task is done
        if args.use_plasticity_injection:
            agent.plasticine_plasticity_injection()
        elif args.use_layer_resetting:
            agent.plasticine_reset_layers(reset_actor=True, reset_critic=False)
        """🎯============================== Plasticine Operations ==============================🎯"""

    envs.close()
    writer.close()
