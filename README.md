<div align=center>
<p align="center"><img align="center" width="500px" src="assets/logo.png"></p>

## Plasticine: Clean Plasticity Optimization in Deep Reinforcement Learning
<img src="https://img.shields.io/badge/License-MIT-%230677b8"> <img src="https://img.shields.io/badge/Base-PyTorch-EF4B28"> <img src="https://img.shields.io/badge/Code%20style-Black-000000"> <img src="https://img.shields.io/badge/Python-%3E%3D3.9-%2335709F"> 

</div>

**Plasticine** is a library that provides high-quality and single-file implementations of plasticity optimization algorithms in deep reinforcement learning. We highlight the features of **Plasticine** as follows:
- 📜 Single-file implementation;
- 🏞️ Support **Standard** and **Continual** RL Scenarios;
- 📊 Benchmarked Implementation (9+ algorithms and 7+ plasticity metrics);
- 📈 Tensorboard Logging;
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

|        **Algorithm**        |           **Benchmark**          | **Backbone** |                                                                    **Code**                                                                    |  **Remark**  |
|:---------------------------:|:--------------------------------:|:------------:|:----------------------------------------------------------------------------------------------------------------------------------------------:|:------------:|
|           Vanilla           | Craftax (State)<br>Procgen (Pixel) |      PPO     | [ppo_craftax_vanilla.py](./plasticine/standard/ppo_craftax_vanilla.py)<br> [ppo_procgen_vanilla.py](./plasticine/standard/ppo_procgen_vanilla.py) |      N/A     |
|        Shrink+Perturb       | Craftax (State)<br>Procgen (Pixel) |      PPO     |            [ppo_craftax_sp.py](./plasticine/standard/ppo_craftax_sp.py)<br>[ppo_procgen_sp.py](./plasticine/standard/ppo_procgen_sp.py)           | Intermittent |
|    Resetting final layer    | Craftax (State)<br>Procgen (Pixel) |      PPO     |            [ppo_craftax_rf.py](./plasticine/standard/ppo_craftax_rf.py)<br>[ppo_procgen_rf.py](./plasticine/standard/ppo_procgen_rf.py)           | Intermittent |
|     Plasticity Injection    | Craftax (State)<br>Procgen (Pixel) |      PPO     |            [ppo_craftax_pi.py](./plasticine/standard/ppo_craftax_pi.py)<br>[ppo_procgen_pi.py](./plasticine/standard/ppo_procgen_pi.py)           | Intermittent |
|             ReDo            | Craftax (State)<br>Procgen (Pixel) |      PPO     |        [ppo_craftax_redo.py](./plasticine/standard/ppo_craftax_redo.py)<br>[ppo_procgen_redo.py](./plasticine/standard/ppo_procgen_redo.py)       | Intermittent |
|       L2 Normalization      | Craftax (State)<br>Procgen (Pixel) |      PPO     |          [ppo_craftax_l2n.py](./plasticine/standard/ppo_craftax_l2n.py)<br>[ppo_procgen_l2n.py](./plasticine/standard/ppo_procgen_l2n.py)         |  Continuous  |
|     Layer Normalization     | Craftax (State)<br>Procgen (Pixel) |      PPO     |            [ppo_craftax_ln.py](./plasticine/standard/ppo_craftax_ln.py)<br>[ppo_procgen_ln.py](./plasticine/standard/ppo_procgen_ln.py)           |  Continuous  |
| Regenerative Regularization | Craftax (State)<br>Procgen (Pixel) |      PPO     |            [ppo_craftax_rr.py](./plasticine/standard/ppo_craftax_rr.py)<br>[ppo_procgen_rr.py](./plasticine/standard/ppo_procgen_rr.py)           |  Continuous  |
|     Soft Shrink+Perturb     | Craftax (State)<br>Procgen (Pixel) |      PPO     |          [ppo_craftax_ssp.py](./plasticine/standard/ppo_craftax_ssp.py)<br>[ppo_procgen_ssp.py](./plasticine/standard/ppo_procgen_ssp.py)         |  Continuous  |
|       CReLU Activation      |          Procgen (Pixel)         |      PPO     |                                          [ppo_procgen_ca.py](./plasticine/standard/ppo_procgen_ca.py)                                          |  Continuous  |

- `Intermittent`: The method is applied only at specific points during training.
- `Continuous`: The method is applied at every step of optimization.


### Continual RL

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
