export PYTHONPATH="${PYTHONPATH}:./plasticine"

for env in bigfish starpilot;
do 
    for seed in 1 2 3;
        do
            CUDA_VISIBLE_DEVICES=0 python plasticine/standard/ppo_procgen_vanilla.py --env_id ${env} --seed ${seed}  > logs/cuda0.log 2>&1 &
            CUDA_VISIBLE_DEVICES=1 python plasticine/standard/ppo_procgen_dff.py     --env_id ${env} --seed ${seed}  > logs/cuda1.log 2>&1 &
            CUDA_VISIBLE_DEVICES=2 python plasticine/standard/ppo_procgen_l2n.py     --env_id ${env} --seed ${seed}  > logs/cuda2.log 2>&1 &
            CUDA_VISIBLE_DEVICES=3 python plasticine/standard/ppo_procgen_ln.py      --env_id ${env} --seed ${seed}  > logs/cuda3.log 2>&1 &
            CUDA_VISIBLE_DEVICES=4 python plasticine/standard/ppo_procgen_nap.py     --env_id ${env} --seed ${seed}  > logs/cuda4.log 2>&1 &
            CUDA_VISIBLE_DEVICES=5 python plasticine/standard/ppo_procgen_snp.py     --env_id ${env} --seed ${seed}  > logs/cuda5.log 2>&1 &
            CUDA_VISIBLE_DEVICES=6 python plasticine/standard/ppo_procgen_pr.py      --env_id ${env} --seed ${seed}  > logs/cuda6.log 2>&1 &
            CUDA_VISIBLE_DEVICES=7 python plasticine/standard/ppo_procgen_redo.py    --env_id ${env} --seed ${seed}  > logs/cuda7.log 2>&1 &
            wait

            CUDA_VISIBLE_DEVICES=0 python plasticine/standard/ppo_procgen_rl.py      --env_id ${env} --seed ${seed}  > logs/cuda0.log 2>&1 &
            CUDA_VISIBLE_DEVICES=1 python plasticine/standard/ppo_procgen_ca.py      --env_id ${env} --seed ${seed}  > logs/cuda1.log 2>&1 &
            CUDA_VISIBLE_DEVICES=2 python plasticine/standard/ppo_procgen_rr.py      --env_id ${env} --seed ${seed}  > logs/cuda2.log 2>&1 &
            CUDA_VISIBLE_DEVICES=3 python plasticine/standard/ppo_procgen_pi.py      --env_id ${env} --seed ${seed}  > logs/cuda3.log 2>&1 &
            CUDA_VISIBLE_DEVICES=4 python plasticine/standard/ppo_procgen_trac.py    --env_id ${env} --seed ${seed}  > logs/cuda4.log 2>&1 &
            wait
    done
done