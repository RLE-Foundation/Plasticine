export PYTHONPATH="${PYTHONPATH}:./plasticine"

## task mode

CUDA_VISIBLE_DEVICES=0 python plasticine/continual/ppo_procgen_vanilla.py --cont_mode task > logs/cuda0.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python plasticine/continual/ppo_procgen_dff.py     --cont_mode task > logs/cuda1.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python plasticine/continual/ppo_procgen_l2n.py     --cont_mode task > logs/cuda2.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 python plasticine/continual/ppo_procgen_ln.py      --cont_mode task > logs/cuda3.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 python plasticine/continual/ppo_procgen_nap.py     --cont_mode task > logs/cuda4.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 python plasticine/continual/ppo_procgen_pi.py      --cont_mode task > logs/cuda5.log 2>&1 &
CUDA_VISIBLE_DEVICES=6 python plasticine/continual/ppo_procgen_pr.py      --cont_mode task > logs/cuda6.log 2>&1 &
CUDA_VISIBLE_DEVICES=7 python plasticine/continual/ppo_procgen_redo.py    --cont_mode task > logs/cuda7.log 2>&1 &
wait

CUDA_VISIBLE_DEVICES=0 python plasticine/continual/ppo_procgen_rl.py      --cont_mode task > logs/cuda0.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python plasticine/continual/ppo_procgen_ca.py      --cont_mode task > logs/cuda1.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python plasticine/continual/ppo_procgen_rr.py      --cont_mode task > logs/cuda2.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 python plasticine/continual/ppo_procgen_snp.py     --cont_mode task > logs/cuda3.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 python plasticine/continual/ppo_procgen_trac.py    --cont_mode task > logs/cuda4.log 2>&1 &
wait

## level mode

CUDA_VISIBLE_DEVICES=0 python plasticine/continual/ppo_procgen_vanilla.py --cont_mode level > logs/cuda0.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python plasticine/continual/ppo_procgen_dff.py     --cont_mode level > logs/cuda1.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python plasticine/continual/ppo_procgen_l2n.py     --cont_mode level > logs/cuda2.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 python plasticine/continual/ppo_procgen_ln.py      --cont_mode level > logs/cuda3.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 python plasticine/continual/ppo_procgen_nap.py     --cont_mode level > logs/cuda4.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 python plasticine/continual/ppo_procgen_pi.py      --cont_mode level > logs/cuda5.log 2>&1 &
CUDA_VISIBLE_DEVICES=6 python plasticine/continual/ppo_procgen_pr.py      --cont_mode level > logs/cuda6.log 2>&1 &
CUDA_VISIBLE_DEVICES=7 python plasticine/continual/ppo_procgen_redo.py    --cont_mode level > logs/cuda7.log 2>&1 &
wait

CUDA_VISIBLE_DEVICES=0 python plasticine/continual/ppo_procgen_rl.py      --cont_mode level > logs/cuda0.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python plasticine/continual/ppo_procgen_ca.py      --cont_mode level > logs/cuda1.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python plasticine/continual/ppo_procgen_rr.py      --cont_mode level > logs/cuda2.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 python plasticine/continual/ppo_procgen_snp.py     --cont_mode level > logs/cuda3.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 python plasticine/continual/ppo_procgen_trac.py    --cont_mode level > logs/cuda4.log 2>&1 &
wait