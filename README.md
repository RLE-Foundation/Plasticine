<div align=center>
<p align="center"><img align="center" width="500px" src="assets/logo.png"></p>

## Plasticine: Clean Plasticity Optimization in Deep Reinforcement Learning
<img src="https://img.shields.io/badge/License-MIT-%230677b8"> <img src="https://img.shields.io/badge/Base-PyTorch-EF4B28"> <img src="https://img.shields.io/badge/Code%20style-Black-000000"> <img src="https://img.shields.io/badge/Python-%3E%3D3.9-%2335709F"> <a href="https://discord.gg/swMV6kgV">
  <img src="https://img.shields.io/badge/Discussion-Discord-5562EA" alt="Discussion Discord">

  ⚠️⚠️⚠️This project is undergoing fast development and iteration!!!⚠️⚠️⚠️
</a> 

</div>

**Plasticine** is a library that provides high-quality and single-file implementations of plasticity optimization algorithms in deep reinforcement learning. We highlight the features of **Plasticine** as follows:
- 📜 Single-file implementation;
- 🏞️ Support **Standard** and **Continual** RL Scenarios;
- 📊 Benchmarked Implementation (13+ algorithms and 10+ plasticity metrics);
- 🧱 Easy combination of different strategies;
- ⚙️ Local Reproducibility via Seeding;
- 🧫 Experiment Management with [Weights and Biases]().

> **Plasticine** is built on the top of [CleanRL](https://github.com/vwxyzjn/cleanrl). Thanks for the excellent project!

> Join the [Discord](https://discord.gg/swMV6kgV) channel for discussion!

## Quick Start
- Create an environment and install the dependencies:
``` sh
conda create -n plasticine python=3.9
pip install -r requirements/requirements-craftax.txt
```

- Clone the repository and run the training script:
``` sh
git clone https://github.com/RLE-Foundation/Plasticine
cd Plasticine
sh scripts/std_ppo_craftax.sh
```


## Implemented Algorithms and Metrics
### Vanilla
| **Algorithm** | **Standard** | **Continual** | **Open-ended**|
|:--|:--|:--|:--|
| Vanilla                     | [PPO+Procgen](./plasticine/standard/ppo_procgen_vanilla.py),[PQN+Atari](./plasticine/standard/pqn_atari_vanilla.py),[TD3+DMC](./plasticine/standard/td3_dmc_vanilla.py) | [PPO+Procgen](./plasticine/continual/ppo_procgen_vanilla.py),[TD3+DMC](./plasticine/continual/td3_dmc_vanilla.py) | [PPO+Craftax](./plasticine/open/ppo_craftax_vanilla.py) |
### Reset-based Interventions
| **Algorithm** | **Standard** | **Continual** | **Open-ended**|
|:--|:--|:--|:--|
| [Shrink and Perturb](https://arxiv.org/pdf/1910.08475)          | [PPO+Procgen](./plasticine/standard/ppo_procgen_snp.py),[PQN+Atari](./plasticine/standard/pqn_atari_snp.py),[TD3+DMC](./plasticine/standard/td3_dmc_snp.py)          | [PPO+Procgen](./plasticine/continual/ppo_procgen_snp.py),[TD3+DMC](./plasticine/continual/td3_dmc_snp.py)         | [PPO+Craftax](./plasticine/open/ppo_craftax_snp.py)     |
| [Plasticity Injection](https://arxiv.org/pdf/2305.15555)        | [PPO+Procgen](./plasticine/standard/ppo_procgen_pi.py),[PQN+Atari](./plasticine/standard/pqn_atari_pi.py),[TD3+DMC](./plasticine/standard/td3_dmc_pi.py)             | [PPO+Procgen](./plasticine/continual/ppo_procgen_pi.py),[TD3+DMC](./plasticine/continual/td3_dmc_pi.py)           | [PPO+Craftax](./plasticine/open/ppo_craftax_pi.py)      |
| [ReDo](https://arxiv.org/pdf/2302.12902)                        | [PPO+Procgen](./plasticine/standard/ppo_procgen_redo.py),[PQN+Atari](./plasticine/standard/pqn_atari_redo.py),[TD3+DMC](./plasticine/standard/td3_dmc_redo.py)       | [PPO+Procgen](./plasticine/continual/ppo_procgen_redo.py),[TD3+DMC](./plasticine/continual/td3_dmc_redo.py)       | [PPO+Craftax](./plasticine/open/ppo_craftax_redo.py)    |
| [Resetting Layer](https://arxiv.org/pdf/2205.07802)             | [PPO+Procgen](./plasticine/standard/ppo_procgen_rl.py),[PQN+Atari](./plasticine/standard/pqn_atari_rl.py),[TD3+DMC](./plasticine/standard/td3_dmc_rl.py)             | [PPO+Procgen](./plasticine/continual/ppo_procgen_rl.py),[TD3+DMC](./plasticine/continual/td3_dmc_rl.py)           | [PPO+Craftax](./plasticine/open/ppo_craftax_rl.py)      |


### Normlization

| **Algorithm** | **Standard** | **Continual** | **Open-ended**|
|:--|:--|:--|:--|
| [Layer Normalization](https://arxiv.org/pdf/2402.18762v1)         | [PPO+Procgen](./plasticine/standard/ppo_procgen_ln.py),[PQN+Atari](./plasticine/standard/pqn_atari_ln.py),[TD3+DMC](./plasticine/standard/td3_dmc_ln.py)             | [PPO+Procgen](./plasticine/continual/ppo_procgen_ln.py),[TD3+DMC](./plasticine/continual/td3_dmc_ln.py)           | [PPO+Craftax](./plasticine/open/ppo_craftax_ln.py)      |
| [Normalize-and-Project](https://arxiv.org/pdf/2407.01800)       | [PPO+Procgen](./plasticine/standard/ppo_procgen_nap.py),[PQN+Atari](./plasticine/standard/pqn_atari_nap.py),[TD3+DMC](./plasticine/standard/td3_dmc_nap.py)          | [PPO+Procgen](./plasticine/continual/ppo_procgen_nap.py),[TD3+DMC](./plasticine/continual/td3_dmc_nap.py)         | [PPO+Craftax](./plasticine/open/ppo_craftax_nap.py)     |


### Regularization

| **Algorithm** | **Standard** | **Continual** | **Open-ended**|
|:--|:--|:--|:--|
| [L2 Normalization](https://arxiv.org/pdf/2402.18762)            | [PPO+Procgen](./plasticine/standard/ppo_procgen_l2n.py),[PQN+Atari](./plasticine/standard/pqn_atari_l2n.py),[TD3+DMC](./plasticine/standard/td3_dmc_l2n.py)    | [PPO+Procgen](./plasticine/continual/ppo_procgen_l2n.py),[TD3+DMC](./plasticine/continual/td3_dmc_l2n.py)   | [PPO+Craftax](./plasticine/open/ppo_craftax_l2n.py)  |
| [Regenerative Regularization](https://arxiv.org/pdf/2308.11958) | [PPO+Procgen](./plasticine/standard/ppo_procgen_rr.py),[PQN+Atari](./plasticine/standard/pqn_atari_rr.py),[TD3+DMC](./plasticine/standard/td3_dmc_rr.py)       | [PPO+Procgen](./plasticine/continual/ppo_procgen_rr.py),[TD3+DMC](./plasticine/continual/td3_dmc_rr.py)     | [PPO+Craftax](./plasticine/open/ppo_craftax_rr.py)   |
| [Parseval Regularization](https://arxiv.org/pdf/2412.07224)     | [PPO+Procgen](./plasticine/standard/ppo_procgen_pr.py),[PQN+Atari](./plasticine/standard/pqn_atari_pr.py),[TD3+DMC](./plasticine/standard/td3_dmc_pr.py)       | [PPO+Procgen](./plasticine/continual/ppo_procgen_pr.py),[TD3+DMC](./plasticine/continual/td3_dmc_pr.py)     | [PPO+Craftax](./plasticine/open/ppo_craftax_pr.py)   |


### Activation

| **Algorithm** | **Standard** | **Continual** | **Open-ended**|
|:--|:--|:--|:--|
| [CReLU Activation](https://arxiv.org/pdf/2303.07507)      | [PPO+Procgen](./plasticine/standard/ppo_procgen_ca.py),[PQN+Atari](./plasticine/standard/pqn_atari_ca.py),[TD3+DMC](./plasticine/standard/td3_dmc_ca.py)       | [PPO+Procgen](./plasticine/continual/ppo_procgen_ca.py),[TD3+DMC](./plasticine/continual/td3_dmc_ca.py)     | [PPO+Craftax](./plasticine/open/ppo_craftax_ca.py)   |
| [Deep Fourier Features](https://arxiv.org/pdf/2410.20634) | [PPO+Procgen](./plasticine/standard/ppo_procgen_dff.py),[PQN+Atari](./plasticine/standard/pqn_atari_dff.py),[TD3+DMC](./plasticine/standard/td3_dmc_dff.py)    | [PPO+Procgen](./plasticine/continual/ppo_procgen_dff.py),[TD3+DMC](./plasticine/continual/td3_dmc_dff.py)   | [PPO+Craftax](./plasticine/open/ppo_craftax_dff.py)  |


### Optimizer


| **Algorithm** | **Standard** | **Continual** | **Open-ended**|
|:--|:--|:--|:--|
| [TRAC](https://arxiv.org/pdf/2405.16642)          | [PPO+Procgen](./plasticine/standard/ppo_procgen_trac.py),[PQN+Atari](./plasticine/standard/pqn_atari_trac.py),[TD3+DMC](./plasticine/standard/td3_dmc_trac.py) | [PPO+Procgen](./plasticine/continual/ppo_procgen_trac.py),[TD3+DMC](./plasticine/continual/td3_dmc_trac.py) | [PPO+Craftax](./plasticine/open/ppo_craftax_trac.py) |

### Evalution Metrics
- **Ratio of Dormant Units** $s_i^{\ell}=\frac{\mathbb{E}_{x \in D}\left|h_i^{\ell}(x)\right|}{\frac{1}{H^{\ell}} \sum_{k \in h} \mathbb{E}_{x \in D}\left|h_k^{\ell}(x)\right|}$
- **Fraction of Inactive Units** $\Phi_{\mathcal{M}}=\frac{\sum_{n \in \mathcal{M}} \mathbf{1}\left(a_n(x)>0\right)}{N}$
- **Stable Rank**
- **Effective Rank** $\rho(A)=-\sum_{i=1}^{\min (n, m)} \bar{\sigma}_i \log \left(\bar{\sigma}_i\right)$
- **Weight Magnitude**
- **Weight Difference**
- **Gradient Norm**
- **Feature Norm**
- **Feature Variance**
- **Policy Entropy**


<!-- |         **Metric**         | **Remark** |
|:--------------------------:|:----------:|
|   Ratio of Dormant Units   |    |
|         Stable Rank        | |
|       Effective Rank       |            |
|      Weight Magnitude      |            |
|      Weight Difference     |            |
|        Gradient Norm       |            |
|        Feature Norm        |            |
|      Feature Variance      |            |
|       Policy Entropy       |            | -->


## Dataset
## Discussion and Contribution

- For discussion and questions:
  + [GitHub issues](https://github.com/RLE-Foundation/Plasticine/issues)
  + [Discord channel](https://discord.gg/swMV6kgV)

- For contribution:
  - Read the `CONTRIBUTING.md` before contributing to the project!

## Cite Us
If you use Plasticine in your work, please cite our paper:
``` bib
@misc{yuan2025@plasticine,
    author = {Mingqi Yuan and Qi Wang and Guozheng Ma and Bo Li and Xin Jin and Wenjun Zeng},
    title = {Plasticine: Clean Plasticity Optimization in Deep Reinforcement Learning},
    year = {2025},
    publisher = {GitHub},
    journal = {GitHub Repository},
    howpublished = {\url{https://github.com/RLE-Foundation/Plasticine}}
}
```

## Acknowledgement
<!-- This project is supported by [The Hong Kong Polytechnic University](http://www.polyu.edu.hk/), [Eastern Institute for Advanced Study](http://www.eias.ac.cn/), and [FLW-Foundation](FLW-Foundation). [EIAS HPC](https://hpc.eias.ac.cn/) provides a GPU computing platform, and [HUAWEI Ascend Community](https://www.hiascend.com/) provides an NPU computing platform for our testing. Some code of this project is borrowed or inspired by several excellent projects, and we highly appreciate them. See [ACKNOWLEDGMENT.md](https://github.com/RLE-Foundation/rllte/blob/main/ACKNOWLEDGMENT.md). -->
