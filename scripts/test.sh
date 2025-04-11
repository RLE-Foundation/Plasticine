export PYTHONPATH="${PYTHONPATH}:./plasticine"

# CUDA_VISIBLE_DEVICES=1 && python plasticine/open/ppo_craftax_vanilla.py

# export PYTHONPATH="${PYTHONPATH}:./plasticine"

# CUDA_VISIBLE_DEVICES=1 && python plasticine/standard/ppo_procgen_vanilla.py

# export PYTHONPATH="${PYTHONPATH}:./plasticine"

CUDA_VISIBLE_DEVICES=1 && python plasticine/standard/td3_dmc_vanilla.py

# CUDA_VISIBLE_DEVICES=0 && python plasticine/standard/pqn_atari_vanilla.py
