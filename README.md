<div align=center>
<p align="center"><img align="center" width="500px" src="assets/logo.png"></p>

## Plasticine: Clean Plasticity Optimization in Deep Reinforcement Learning
<img src="https://img.shields.io/badge/License-MIT-%230677b8"> <img src="https://img.shields.io/badge/Base-PyTorch-EF4B28"> <img src="https://img.shields.io/badge/Code%20style-Black-000000"> <img src="https://img.shields.io/badge/Python-%3E%3D3.9-%2335709F"> 

</div>

**Plasticine** is a library that provides high-quality and single-file implementations of plasticity optimization algorithms in deep reinforcement learning. We highlight the features of **Plasticine** as follows:
- 📜 Single-file implementation;
- 🏞️ Support **Standard** and **Continual** RL Scenarios;
- 📊 Benchmarked Implementation (10+ algorithms and 7+ plasticity metrics);
- 🧱 Easy combination of different strategies;
- ⚙️ Local Reproducibility via Seeding;
- 🧫 Experiment Management with [Weights and Biases]().

> **Plasticine** is built on the top of [CleanRL](https://github.com/vwxyzjn/cleanrl). Thanks for the excellent project!

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


## Implemented Algorithms
### Standard RL

|        **Algorithm**        |                     **Benchmark**                    | **Backbone** |                                                                                                         **Code**                                                                                                         |  **Remark**  |
|:---------------------------:|:----------------------------------------------------:|:------------:|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:------------:|
|           Vanilla           | Craftax (State)<br>Procgen (Pixel)<br>Mujoco (State) |   PPO, TD3   | [ppo_craftax_vanilla.py](./plasticine/standard/ppo_craftax_vanilla.py)<br>[ppo_procgen_vanilla.py](./plasticine/standard/ppo_procgen_vanilla.py)<br>[td3_mujoco_vanilla.py](./plasticine/standard/td3_mujoco_vanilla.py) |      N/A     |
|        Shrink+Perturb       |          Craftax (State)<br>Procgen (Pixel)          |      PPO     |                                               [ppo_craftax_sp.py](./plasticine/standard/ppo_craftax_sp.py)<br>[ppo_procgen_sp.py](./plasticine/standard/ppo_procgen_sp.py)                                               | Intermittent |
|     Plasticity Injection    | Craftax (State)<br>Procgen (Pixel)<br>Mujoco (State) |   PPO, TD3   |                [ppo_craftax_pi.py](./plasticine/standard/ppo_craftax_pi.py)<br>[ppo_procgen_pi.py](./plasticine/standard/ppo_procgen_pi.py)<br>[td3_mujoco_pi.py](./plasticine/standard/td3_mujoco_pi.py)                | Intermittent |
|             ReDo            | Craftax (State)<br>Procgen (Pixel)<br>Mujoco (State) |   PPO, TD3   |          [ppo_craftax_redo.py](./plasticine/standard/ppo_craftax_redo.py)<br>[ppo_procgen_redo.py](./plasticine/standard/ppo_procgen_redo.py)<br>[td3_mujoco_redo.py](./plasticine/standard/td3_mujoco_redo.py)          | Intermittent |
|       L2 Normalization      | Craftax (State)<br>Procgen (Pixel)<br>Mujoco (State) |   PPO, TD3   |             [ppo_craftax_l2n.py](./plasticine/standard/ppo_craftax_l2n.py)<br>[ppo_procgen_l2n.py](./plasticine/standard/ppo_procgen_l2n.py)<br>[td3_mujoco_l2n.py](./plasticine/standard/td3_mujoco_l2n.py)             |  Continuous  |
|     Layer Normalization     | Craftax (State)<br>Procgen (Pixel)<br>Mujoco (State) |   PPO, TD3   |                [ppo_craftax_ln.py](./plasticine/standard/ppo_craftax_ln.py)<br>[ppo_procgen_ln.py](./plasticine/standard/ppo_procgen_ln.py)<br>[td3_mujoco_ln.py](./plasticine/standard/td3_mujoco_ln.py)                |  Continuous  |
| Regenerative Regularization | Craftax (State)<br>Procgen (Pixel)<br>Mujoco (State) |   PPO, TD3   |                [ppo_craftax_rr.py](./plasticine/standard/ppo_craftax_rr.py)<br>[ppo_procgen_rr.py](./plasticine/standard/ppo_procgen_rr.py)<br>[td3_mujoco_rr.py](./plasticine/standard/td3_mujoco_rr.py)                |  Continuous  |
|     Soft Shrink+Perturb     | Craftax (State)<br>Procgen (Pixel)<br>Mujoco (State) |   PPO, TD3   |             [ppo_craftax_ssp.py](./plasticine/standard/ppo_craftax_ssp.py)<br>[ppo_procgen_ssp.py](./plasticine/standard/ppo_procgen_ssp.py)<br>[td3_mujoco_ssp.py](./plasticine/standard/td3_mujoco_ssp.py)             |  Continuous  |
|       CReLU Activation      |           Procgen (Pixel)<br>Mujoco (State)          |   PPO, TD3   |                 [ppo_procgen_ca.py](./plasticine/standard/ppo_procgen_ca.py)<br>[td3_mujoco_ca.py](./plasticine/standard/td3_mujoco_ca.py)<br>[td3_mujoco_ca.py](./plasticine/standard/td3_mujoco_ca.py)                 |  Continuous  |

- `Intermittent`: The method is applied only at specific points during training.
- `Continuous`: The method is applied at every step of optimization.


### Continual RL

|        **Algorithm**        |  **Benchmark**  | **Backbone** |                                 **Code**                                |  **Remark**  |
|:---------------------------:|:---------------:|:------------:|:-----------------------------------------------------------------------:|:------------:|
|           Vanilla           | Procgen (Pixel) |      PPO     | [ppo_procgen_vanilla.py](./plasticine/continual/ppo_procgen_vanilla.py) |      N/A     |
|        Shrink+Perturb       | Procgen (Pixel) |      PPO     |      [ppo_procgen_sp.py](./plasticine/continual/ppo_procgen_sp.py)      | Intermittent |
|     Plasticity Injection    | Procgen (Pixel) |      PPO     |      [ppo_procgen_pi.py](./plasticine/continual/ppo_procgen_pi.py)      | Intermittent |
|             ReDo            | Procgen (Pixel) |      PPO     |    [ppo_procgen_redo.py](./plasticine/continual/ppo_procgen_redo.py)    | Intermittent |
|    Resetting Final Layer    | Procgen (Pixel) |      PPO     |     [ppo_procgen_rfl.py](./plasticine/continual/ppo_procgen_rfl.py)     | Intermittent |
|     Resetting All Layer     | Procgen (Pixel) |      PPO     |     [ppo_procgen_ral.py](./plasticine/continual/ppo_procgen_ral.py)     | Intermittent |
|             TRAC            | Procgen (Pixel) |      PPO     |      [ppo_procgen_trac.py](./plasticine/continual/ppo_procgen_trac.py)      |  Continuous  |
|       L2 Normalization      | Procgen (Pixel) |      PPO     |     [ppo_procgen_l2n.py](./plasticine/continual/ppo_procgen_l2n.py)     |  Continuous  |
|     Layer Normalization     | Procgen (Pixel) |      PPO     |      [ppo_procgen_ln.py](./plasticine/continual/ppo_procgen_ln.py)      |  Continuous  |
| Regenerative Regularization | Procgen (Pixel) |      PPO     |      [ppo_procgen_rr.py](./plasticine/continual/ppo_procgen_rr.py)      |  Continuous  |
|     Soft Shrink+Perturb     | Procgen (Pixel) |      PPO     |     [ppo_procgen_ssp.py](./plasticine/continual/ppo_procgen_ssp.py)     |  Continuous  |
|       CReLU Activation      | Procgen (Pixel) |      PPO     |      [ppo_procgen_ca.py](./plasticine/continual/ppo_procgen_ca.py)      |  Continuous  |

- `Intermittent`: The method is applied only at specific points during training.
- `Continuous`: The method is applied at every step of optimization.


## Dataset

## Discussion and Contribution

## Cite Us
If you use Plasticine in your work, please cite our paper:
``` bib
@misc{yuan2025@plasticine,
    author = {Mingqi Yuan and Bo Li and Xin Jin and Wenjun Zeng},
    title = {Plasticine: Clean Plasticity Optimization in Deep Reinforcement Learning},
    year = {2025},
    publisher = {GitHub},
    journal = {GitHub Repository},
    howpublished = {\url{https://github.com/RLE-Foundation/Plasticine}}
}
```

## Acknowledgement
<!-- This project is supported by [The Hong Kong Polytechnic University](http://www.polyu.edu.hk/), [Eastern Institute for Advanced Study](http://www.eias.ac.cn/), and [FLW-Foundation](FLW-Foundation). [EIAS HPC](https://hpc.eias.ac.cn/) provides a GPU computing platform, and [HUAWEI Ascend Community](https://www.hiascend.com/) provides an NPU computing platform for our testing. Some code of this project is borrowed or inspired by several excellent projects, and we highly appreciate them. See [ACKNOWLEDGMENT.md](https://github.com/RLE-Foundation/rllte/blob/main/ACKNOWLEDGMENT.md). -->
