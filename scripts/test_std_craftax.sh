export PYTHONPATH="${PYTHONPATH}:./plasticine"

CUDA_VISIBLE_DEVICES=0 python plasticine/open/ppo_craftax_vanilla.py > logs/cuda0.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python plasticine/open/ppo_craftax_dff.py > logs/cuda1.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python plasticine/open/ppo_craftax_l2n.py > logs/cuda2.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 python plasticine/open/ppo_craftax_ln.py > logs/cuda3.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 python plasticine/open/ppo_craftax_nap.py > logs/cuda4.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 python plasticine/open/ppo_craftax_pi.py > logs/cuda5.log 2>&1 &
CUDA_VISIBLE_DEVICES=6 python plasticine/open/ppo_craftax_pr.py > logs/cuda6.log 2>&1 &
CUDA_VISIBLE_DEVICES=7 python plasticine/open/ppo_craftax_redo.py > logs/cuda7.log 2>&1 &
wait

CUDA_VISIBLE_DEVICES=0 python plasticine/open/ppo_craftax_rl.py > logs/cuda0.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python plasticine/open/ppo_craftax_ca.py > logs/cuda1.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python plasticine/open/ppo_craftax_rr.py > logs/cuda2.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 python plasticine/open/ppo_craftax_snp.py > logs/cuda3.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 python plasticine/open/ppo_craftax_trac.py > logs/cuda4.log 2>&1 &
wait