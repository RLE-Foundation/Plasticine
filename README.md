<div align=center>
<p align="center"><img align="center" width="500px" src="assets/logo.png"></p>

## Plasticine: Clean Plasticity Optimization in Deep Reinforcement Learning

</div>

**Plasticine** is a library that provides high-quality and single-file implementations of plasticity optimization algorithms in deep reinforcement learning. 


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
|           Vanilla           | Craftax (State), Procgen (Pixel) |      PPO     | [ppo_craftax_vanilla.py](./plasticine/standard/ppo_craftax_vanilla.py), [ppo_procgen_vanilla.py](./plasticine/standard/ppo_procgen_vanilla.py) |      N/A     |
|        Shrink+Perturb       | Craftax (State), Procgen (Pixel) |      PPO     |            [ppo_craftax_sp.py](./plasticine/standard/ppo_craftax_sp.py),[ppo_procgen_sp.py](./plasticine/standard/ppo_procgen_sp.py)           | Intermittent |
|    Resetting final layer    | Craftax (State), Procgen (Pixel) |      PPO     |            [ppo_craftax_rf.py](./plasticine/standard/ppo_craftax_rf.py),[ppo_procgen_rf.py](./plasticine/standard/ppo_procgen_rf.py)           | Intermittent |
|     Plasticity Injection    | Craftax (State), Procgen (Pixel) |      PPO     |            [ppo_craftax_pi.py](./plasticine/standard/ppo_craftax_pi.py),[ppo_procgen_pi.py](./plasticine/standard/ppo_procgen_pi.py)           | Intermittent |
|             ReDo            | Craftax (State), Procgen (Pixel) |      PPO     |        [ppo_craftax_redo.py](./plasticine/standard/ppo_craftax_redo.py),[ppo_procgen_redo.py](./plasticine/standard/ppo_procgen_redo.py)       | Intermittent |
|  Continual Backpropagation  | Craftax (State), Procgen (Pixel) |      PPO     |                                         [ppo_craftax_cbp.py](./plasticine/standard/ppo_craftax_cbp.py),                                        |  Continuous  |
|       L2 Normalization      | Craftax (State), Procgen (Pixel) |      PPO     |          [ppo_craftax_l2n.py](./plasticine/standard/ppo_craftax_l2n.py),[ppo_procgen_l2n.py](./plasticine/standard/ppo_procgen_l2n.py)         |  Continuous  |
|     Layer Normalization     | Craftax (State), Procgen (Pixel) |      PPO     |            [ppo_craftax_ln.py](./plasticine/standard/ppo_craftax_ln.py),[ppo_procgen_ln.py](./plasticine/standard/ppo_procgen_ln.py)           |  Continuous  |
| Regenerative Regularization | Craftax (State), Procgen (Pixel) |      PPO     |            [ppo_craftax_rr.py](./plasticine/standard/ppo_craftax_rr.py),[ppo_procgen_rr.py](./plasticine/standard/ppo_procgen_rr.py)           |  Continuous  |
|     Soft Shrink+Perturb     | Craftax (State), Procgen (Pixel) |      PPO     |          [ppo_craftax_ssp.py](./plasticine/standard/ppo_craftax_ssp.py),[ppo_procgen_ssp.py](./plasticine/standard/ppo_procgen_ssp.py)         |  Continuous  |
|       CReLU Activation      |          Procgen (Pixel)         |      PPO     |                                          [ppo_procgen_ca.py](./plasticine/standard/ppo_procgen_ca.py)                                          |  Continuous  |

- `Intermittent`: The method is applied only at specific points during training.
- `Continuous`: The method is applied at every step of optimization.


### Continual RL

## Dataset

## Discussion and Contribution

## Cite Us
If you use Plasticine in your work, please cite our paper:
``` bib
@article{yuan2025@plasticine,
author = {Mingqi Yuan and Bo Li and Xin Jin and Wenjun Zeng},
title = {Plasticine: Clean Plasticity Optimization in Deep Reinforcement Learning},
year = {2025}
}
```

## Acknowledgement
