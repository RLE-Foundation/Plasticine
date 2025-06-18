export PYTHONPATH="${PYTHONPATH}:./plasticine"

for seed in 1 2 3;
do
    CUDA_VISIBLE_DEVICES=0 python plasticine/open/ppo_craftax_vanilla.py --seed ${seed}  > logs/cuda0.log 2>&1 &
    CUDA_VISIBLE_DEVICES=1 python plasticine/open/ppo_craftax_dff.py     --seed ${seed}  > logs/cuda1.log 2>&1 &
    CUDA_VISIBLE_DEVICES=2 python plasticine/open/ppo_craftax_l2n.py     --seed ${seed}  > logs/cuda2.log 2>&1 &
    CUDA_VISIBLE_DEVICES=3 python plasticine/open/ppo_craftax_ln.py      --seed ${seed}  > logs/cuda3.log 2>&1 &
    CUDA_VISIBLE_DEVICES=4 python plasticine/open/ppo_craftax_nap.py     --seed ${seed}  > logs/cuda4.log 2>&1 &
    CUDA_VISIBLE_DEVICES=5 python plasticine/open/ppo_craftax_snp.py     --seed ${seed}  > logs/cuda5.log 2>&1 &
    CUDA_VISIBLE_DEVICES=6 python plasticine/open/ppo_craftax_pr.py      --seed ${seed}  > logs/cuda6.log 2>&1 &
    CUDA_VISIBLE_DEVICES=7 python plasticine/open/ppo_craftax_redo.py    --seed ${seed}  > logs/cuda7.log 2>&1 &
    wait

    CUDA_VISIBLE_DEVICES=0 python plasticine/open/ppo_craftax_rl.py      --seed ${seed}  > logs/cuda0.log 2>&1 &
    CUDA_VISIBLE_DEVICES=1 python plasticine/open/ppo_craftax_ca.py      --seed ${seed}  > logs/cuda1.log 2>&1 &
    CUDA_VISIBLE_DEVICES=2 python plasticine/open/ppo_craftax_rr.py      --seed ${seed}  > logs/cuda2.log 2>&1 &
    CUDA_VISIBLE_DEVICES=3 python plasticine/open/ppo_craftax_pi.py      --seed ${seed}  > logs/cuda3.log 2>&1 &
    CUDA_VISIBLE_DEVICES=4 python plasticine/open/ppo_craftax_trac.py    --seed ${seed}  > logs/cuda4.log 2>&1 &
    wait
done