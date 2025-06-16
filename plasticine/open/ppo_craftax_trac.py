# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_ataripy
import os
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'  
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
import random
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
import jax

from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter


from craftax.craftax_env import make_craftax_env_from_name
from plasticine.craftax_wrappers import (LogWrapper, 
                                         OptimisticResetVecEnvWrapper
                                         )
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
from plasticine.trac import start_trac

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
    env_id: str = "Craftax-Symbolic-v1"
    """the id of the environment"""
    total_timesteps: int = 100000000
    """total timesteps of the experiments"""
    learning_rate: float = 2e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1024
    """the number of parallel game environments"""
    num_steps: int = 64
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.8
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 8
    """the number of mini-batches"""
    update_epochs: int = 4
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
    max_grad_norm: float = 1.0
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


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, obs_shape, action_dim):
        super().__init__()
        self.action_dim = action_dim
        self.obs_shape = obs_shape

        self.policy_encoder = self.gen_encoder()
        self.policy = self.gen_policy(action_dim)
        self.value_encoder = self.gen_encoder()
        self.value = self.gen_value()

    def gen_encoder(self):
        # generate the encoder
        return nn.Sequential(
            layer_init(nn.Linear(self.obs_shape[0], 512)),
            nn.Tanh(),
            layer_init(nn.Linear(512, 512)),
            nn.Tanh(),
            layer_init(nn.Linear(512, 512)),
            nn.Tanh(),
        )

    def gen_policy(self, action_dim):
        return layer_init(nn.Linear(512, action_dim), std=0.01)

    def gen_value(self):
        return layer_init(nn.Linear(512, 1), std=1.0)

    def get_value(self, x):
        return self.value(self.value_encoder(x))

    def get_action_and_value(self, x, action=None, check=False):
        policy_x = self.policy_encoder(x)
        value_x = self.value_encoder(x)

        logits = self.policy(policy_x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()

        if check:
            with torch.no_grad():
                policy_active_units, value_active_units = compute_active_units(policy_x, 'tanh'), compute_active_units(value_x, 'tanh')
                policy_stable_rank, value_stable_rank = compute_stable_rank(policy_x), compute_stable_rank(value_x)
                policy_effective_rank, value_effective_rank = compute_effective_rank(policy_x), compute_effective_rank(value_x)
                policy_feature_norm, value_feature_norm = compute_feature_norm(policy_x), compute_feature_norm(value_x)
                policy_feature_var, value_feature_var = compute_feature_variance(policy_x), compute_feature_variance(value_x)
                plasticity_metrics = {
                    "policy_active_units": policy_active_units.item(),
                    "policy_stable_rank": policy_stable_rank.item(),
                    "policy_effective_rank": policy_effective_rank.item(),
                    "policy_feature_norm": policy_feature_norm.item(),
                    "policy_feature_var": policy_feature_var.item(),
                    "value_active_units": value_active_units.item(),
                    "value_stable_rank": value_stable_rank.item(),
                    "value_effective_rank": value_effective_rank.item(),
                    "value_feature_norm": value_feature_norm.item(),
                    "value_feature_var": value_feature_var.item(),
                }
            return action, probs.log_prob(action), probs.entropy(), self.value(value_x), plasticity_metrics
        else:
            return action, probs.log_prob(action), probs.entropy(), self.value(value_x)

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
    log_dir = 'std_ppo_craftax_trac_runs'
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
    env = make_craftax_env_from_name(args.env_id, False)
    env_params = env.default_params
    env = LogWrapper(env)
    env = OptimisticResetVecEnvWrapper(env, num_envs=args.num_envs, reset_ratio=min(16, args.num_envs))
    obs_shape = env.observation_space(env_params).shape
    action_shape = env.action_space(env_params).shape
    action_dim = env.action_space(env_params).n

    # agent setup
    agent = Agent(obs_shape=obs_shape, action_dim=action_dim).to(device)
    """------------------------Plasticine------------------------"""
    # TRAC setup
    optimizer = start_trac(f'{log_dir}/{run_name}/trac.text', optim.Adam, eps=1e-5)(agent.parameters(), lr=args.learning_rate)
    """------------------------Plasticine------------------------"""

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + obs_shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + action_shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    rng = jax.random.PRNGKey(args.seed)
    rng, _rng = jax.random.split(rng)
    next_obs, env_state = env.reset(_rng, env_params)
    next_obs = torch.from_numpy(np.array(next_obs)).to(device)
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
            rng, _rng = jax.random.split(rng)
            next_obs, env_state, reward_e, done, infos = env.step(
                _rng, env_state, action.cpu().numpy(), env_params
            )
            next_done = done 
            rewards[step] = torch.from_numpy(np.array(reward_e)).to(device).view(-1)
            next_obs = torch.from_numpy(np.array(next_obs)).to(device)
            next_done = torch.from_numpy(np.array(next_done)).float().to(device)

            if True in done:
                # gather all the episode returns and lengths with done=True
                tag_idx = np.nonzero(done)[0][0]
                eps_return = np.array(infos['returned_episode_returns'][tag_idx])
                eps_length = np.array(infos['returned_episode_lengths'][tag_idx])
                print(f"global_step={global_step}, episodic_return={eps_return}")
                writer.add_scalar("charts/episodic_return", eps_return, global_step)
                writer.add_scalar("charts/episodic_length", eps_length, global_step)

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
        b_obs = obs.reshape((-1,) + obs_shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + action_shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []

        # plasticity metrics
        total_policy_active_units = []
        total_policy_stable_rank = []
        total_policy_effective_rank = []
        total_policy_feature_norm = []
        total_policy_feature_var = []
        total_value_active_units = []
        total_value_stable_rank = []
        total_value_effective_rank = []
        total_value_feature_norm = []
        total_value_feature_var = []

        total_grad_norm = []
        total_policy_ent = []

        # copy the agent
        agent_copy = save_model_state(agent)

        # update the agent
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]
                # set check as True to get the plasticity metrics
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
                total_policy_active_units.append(plasticity_metrics["policy_active_units"])
                total_policy_stable_rank.append(plasticity_metrics["policy_stable_rank"])
                total_policy_effective_rank.append(plasticity_metrics["policy_effective_rank"])
                total_policy_feature_norm.append(plasticity_metrics["policy_feature_norm"])
                total_policy_feature_var.append(plasticity_metrics["policy_feature_var"])
                total_value_active_units.append(plasticity_metrics["value_active_units"])
                total_value_stable_rank.append(plasticity_metrics["value_stable_rank"])
                total_value_effective_rank.append(plasticity_metrics["value_effective_rank"])
                total_value_feature_norm.append(plasticity_metrics["value_feature_norm"])
                total_value_feature_var.append(plasticity_metrics["value_feature_var"])
                total_grad_norm.append(batch_grad_norm.item())
                total_policy_ent.append(entropy_loss.item())

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # compute the l2 norm difference
        diff_l2_norm = compute_l2_norm_difference(agent, agent_copy)
        # compute weight magnitude
        weight_magnitude = compute_weight_magnitude(agent)
        # compute dormant units
        if iteration % 10 == 0:
            policy_dormant_units = compute_dormant_units(agent.policy_encoder, b_obs[mb_inds], 'tanh', tau=0.05)
            value_dormant_units = compute_dormant_units(agent.value_encoder, b_obs[mb_inds], 'tanh', tau=0.05)
            writer.add_scalar("plasticity/policy_dormant_units", policy_dormant_units, global_step)
            writer.add_scalar("plasticity/value_dormant_units", value_dormant_units, global_step)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        # individual plasticity metrics
        writer.add_scalar("plasticity/policy_active_units", np.mean(total_policy_active_units), global_step)
        writer.add_scalar("plasticity/policy_stable_rank", np.mean(total_policy_stable_rank), global_step)
        writer.add_scalar("plasticity/policy_effective_rank", np.mean(total_policy_effective_rank), global_step)
        writer.add_scalar("plasticity/policy_feature_norm", np.mean(total_policy_feature_norm), global_step)
        writer.add_scalar("plasticity/policy_feature_var", np.mean(total_policy_feature_var), global_step)
        writer.add_scalar("plasticity/value_active_units", np.mean(total_value_active_units), global_step)
        writer.add_scalar("plasticity/value_stable_rank", np.mean(total_value_stable_rank), global_step)
        writer.add_scalar("plasticity/value_effective_rank", np.mean(total_value_effective_rank), global_step)
        writer.add_scalar("plasticity/value_feature_norm", np.mean(total_value_feature_norm), global_step)
        writer.add_scalar("plasticity/value_feature_var", np.mean(total_value_feature_var), global_step)
        # overall plasticity metrics
        writer.add_scalar("plasticity/weight_magnitude", weight_magnitude.item(), global_step)
        writer.add_scalar("plasticity/l2_norm_difference", diff_l2_norm.item(), global_step)
        writer.add_scalar("plasticity/grad_norm", np.mean(total_grad_norm), global_step)
        writer.add_scalar("plasticity/policy_entropy", np.mean(total_policy_ent), global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    writer.close()

    # save model 
    torch.save(agent.state_dict(), f"{log_dir}/{run_name}/agent.pt")