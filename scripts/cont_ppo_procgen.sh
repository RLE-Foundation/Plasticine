export PYTHONPATH="${PYTHONPATH}:./plasticine"

CUDA_VISIBLE_DEVICES=1 && python plasticine/continual/ppo_procgen_vanilla.py