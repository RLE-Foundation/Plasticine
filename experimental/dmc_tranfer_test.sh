# export PYTHONPATH="${PYTHONPATH}:./plasticine"

# CUDA_VISIBLE_DEVICES=1 && python plasticine/standard/ppo_craftax_redo.py

# export PYTHONPATH="${PYTHONPATH}:./plasticine"

# CUDA_VISIBLE_DEVICES=1 && python plasticine/standard/ppo_procgen_redo.py

# --env_ids "humanoid_stand" "humanoid_walk" "humanoid_run" 

export PYTHONPATH="${PYTHONPATH}:./plasticine"

CUDA_VISIBLE_DEVICES=6 && python plasticine/standard/td3_dmc_vanilla_transfer.py \
--track \
--total_timesteps 10000 \
--env_ids "dog_stand" "dog_walk" "dog_trot" "dog_run" "dog_fetch" 

