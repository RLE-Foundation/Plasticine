<div align=center>
<p align="center"><img align="center" width="500px" src="assets/logo.png"></p>

## Plasticine: Accelerating Research in Plasticity-Motivated Deep Reinforcement Learning
<img src="https://img.shields.io/badge/License-MIT-%230677b8"> <img src="https://img.shields.io/badge/Base-PyTorch-EF4B28"> <img src="https://img.shields.io/badge/Code%20style-Black-000000"> <img src="https://img.shields.io/badge/Python-%3E%3D3.9-%2335709F"> <a href="https://discord.gg/swMV6kgV">
  <img src="https://img.shields.io/badge/Discussion-Discord-5562EA" alt="Discussion Discord">

  ⚠️⚠️⚠️This project is undergoing fast development and iteration!!!⚠️⚠️⚠️
</a> 

</div>

**Plasticine** is a library that provides high-quality and single-file implementations of plasticity loss mitigation methods in deep reinforcement learning. We highlight the features of **Plasticine** as follows:
- 📜 Single-file implementation;
- 🏞️ Support **standard**, **continual**, and **open-ended** RL Scenarios;
- 📊 Benchmarked implementation (13+ algorithms and 10+ plasticity metrics);
- 🧱 Easy combination of different strategies;
- ⚙️ Local reproducibility via seeding;
- 🧫 Experiment management with [Weights and Biases]().

> **Plasticine** is built on the top of [CleanRL](https://github.com/vwxyzjn/cleanrl). Thanks for the excellent project!

> Join the [Discord](https://discord.gg/YGApGaXAHW) channel for discussion!

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

In each Python file, the core algorimic logic is marked like
``` py
"""------------------------Plasticine------------------------"""
...
nn.Linear(512, 512),
CReLU4Linear(), # CRELU4Linear() doubles the output size
nn.Linear(512*2, 512), 
...
"""------------------------Plasticine------------------------"""
```

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

### Evaluation Metrics
|                          |                  |     **Metric**    |               |                  |
|:------------------------:|:----------------:|:-----------------:|:-------------:|:----------------:|
|  Ratio of Dormant Units  |    Stable Rank   |   Effective Rank  |  Feature Norm | Feature Variance |
| Fraction of Active Units | Weight Magnitude | Weight Difference | Gradient Norm |  Policy Entropy  |

The detailed formulation of these metrics can be found in the [Paper]().

### Learning Scenarios
#### Standard
| <img src="assets/demon_attack.gif" style="height:150px;width: auto;"> | <img src="assets/bossfight.png" style="height:150px;width: auto;"> |<img src="assets/dog_run.png" style="height:150px;width: auto;"> | 
|:-----------------------|:-----------------------|:-----------------------|
|ALE|Procgen|DMC|

#### Continual
- Continual Procgen

| ![cont_procgen_ls](assets/cont_procgen_ls.png) | ![cont_procgen_ts](assets/cont_procgen_ts.png) |
|:-----------------------|:-----------------------|
|**Level-Shift**: The same task with a sequentially-incremented `start_level` parameter for each round. |**Task-Shift**: Different tasks with a same `start_level` parameter for each round.|

- Continual DMC

| ![cont_dmc_ds](assets/cont_dmc_ds.png) | ![cont_dmc_ts](assets/cont_dmc_ts.png) |
|:--|:--|
|**Dynamic-Shift**: The same task with a sequentially and randomly-sampled `coefficient_of_friction` parameter for each round.|**Task-Shift**: Different tasks with a same `coefficient_of_friction` parameter for each round.|

#### Open-ended
| ![craftax_farming](assets/farming.gif) | ![craftax_mining](assets/mining.gif) |![craftax_archery](assets/archery.gif) | ![craftax_magic](assets/magic.gif) |
|:-----------------------|:-----------------------|:-----------------------|:-----------------------|
|Farming|Mining|Archery|Magic|


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
    author = {Mingqi Yuan and Qi Wang and Guozheng Ma and Bo Li and Xin Jin and Yunbo Wang and Xiaokang Yang and Wenjun Zeng and Dacheng Tao},
    title = {Plasticine: Accelerating Research in Plasticity-Motivated Deep Reinforcement Learning},
    year = {2025},
    publisher = {GitHub},
    journal = {GitHub Repository},
    howpublished = {\url{https://github.com/RLE-Foundation/Plasticine}}
}
```

## Acknowledgement

This project is supported by [The Hong Kong Polytechnic University](http://www.polyu.edu.hk/), [Ningbo Institute of Digital Twin, Eastern Institute of Technology, Ningbo](https://idt.eitech.edu.cn/), [Shanghai Jiao Tong University](https://en.sjtu.edu.cn/), [Nanyang Technological University](https://www.ntu.edu.sg/), and [LimX Dynamics](https://limxdynamics.com/). We thank the high-performance computing center at Eastern Institute of Technology and Ningbo Institute of Digital Twin for providing the computing resources. Some code of this project is borrowed or inspired by several excellent projects, and we highly appreciate them.
