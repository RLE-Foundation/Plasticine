# export PYTHONPATH="${PYTHONPATH}:./plasticine"

# CUDA_VISIBLE_DEVICES=1 && python plasticine/standard/ppo_craftax_redo.py

# export PYTHONPATH="${PYTHONPATH}:./plasticine"

# CUDA_VISIBLE_DEVICES=1 && python plasticine/standard/ppo_procgen_redo.py

export PYTHONPATH="${PYTHONPATH}:./plasticine"

CUDA_VISIBLE_DEVICES=6 && python plasticine/standard/td3_dmc_vanilla_multi.py \
--track \
--total_timesteps 100000 \
--env_ids "humanoid_stand" "humanoid_walk" "humanoid_run" 
