export PYTHONPATH="${PYTHONPATH}:./plasticine"


# CUDA_VISIBLE_DEVICES=0 && python plasticine/standard/ppo_procgen_ca.py
CUDA_VISIBLE_DEVICES=0 && python plasticine/continual/td3_dmc_vanilla.py