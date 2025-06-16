export PYTHONPATH="${PYTHONPATH}:./plasticine"

CUDA_VISIBLE_DEVICES=0 python plasticine/standard/ppo_procgen_vanilla.py > logs/cuda0.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python plasticine/standard/ppo_procgen_dff.py > logs/cuda1.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python plasticine/standard/ppo_procgen_l2n.py > logs/cuda2.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 python plasticine/standard/ppo_procgen_ln.py > logs/cuda3.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 python plasticine/standard/ppo_procgen_nap.py > logs/cuda4.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 python plasticine/standard/ppo_procgen_pi.py > logs/cuda5.log 2>&1 &
CUDA_VISIBLE_DEVICES=6 python plasticine/standard/ppo_procgen_pr.py > logs/cuda6.log 2>&1 &
CUDA_VISIBLE_DEVICES=7 python plasticine/standard/ppo_procgen_redo.py > logs/cuda7.log 2>&1 &
wait

CUDA_VISIBLE_DEVICES=0 python plasticine/standard/ppo_procgen_rl.py > logs/cuda0.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python plasticine/standard/ppo_procgen_ca.py > logs/cuda1.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python plasticine/standard/ppo_procgen_rr.py > logs/cuda2.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 python plasticine/standard/ppo_procgen_snp.py > logs/cuda3.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 python plasticine/standard/ppo_procgen_trac.py > logs/cuda4.log 2>&1 &
wait