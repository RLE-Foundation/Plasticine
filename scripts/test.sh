# export PYTHONPATH="${PYTHONPATH}:./plasticine"

# CUDA_VISIBLE_DEVICES=1 && python plasticine/standard/ppo_craftax_redo.py

export PYTHONPATH="${PYTHONPATH}:./plasticine"

CUDA_VISIBLE_DEVICES=1 && python plasticine/standard/ppo_procgen_vanilla.py