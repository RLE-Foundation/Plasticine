export PYTHONPATH="${PYTHONPATH}:./plasticine"

# CUDA_VISIBLE_DEVICES=1 && python plasticine/standard/ppo_procgen_vanilla.py --track
CUDA_VISIBLE_DEVICES=0 && python plasticine/standard/ppo_procgen_vanilla.py --track --seed 2