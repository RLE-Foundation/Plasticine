export PYTHONPATH="${PYTHONPATH}:./plasticine"

CUDA_VISIBLE_DEVICES=0 python plasticine/open/pqn_atari_vanilla.py > logs/cuda0.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python plasticine/open/pqn_atari_dff.py > logs/cuda1.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python plasticine/open/pqn_atari_l2n.py > logs/cuda2.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 python plasticine/open/pqn_atari_ln.py > logs/cuda3.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 python plasticine/open/pqn_atari_nap.py > logs/cuda4.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 python plasticine/open/pqn_atari_pi.py > logs/cuda5.log 2>&1 &
CUDA_VISIBLE_DEVICES=6 python plasticine/open/pqn_atari_pr.py > logs/cuda6.log 2>&1 &
CUDA_VISIBLE_DEVICES=7 python plasticine/open/pqn_atari_redo.py > logs/cuda7.log 2>&1 &
wait

CUDA_VISIBLE_DEVICES=0 python plasticine/open/pqn_atari_rl.py > logs/cuda0.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python plasticine/open/pqn_atari_ca.py > logs/cuda1.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python plasticine/open/pqn_atari_rr.py > logs/cuda2.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 python plasticine/open/pqn_atari_snp.py > logs/cuda3.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 python plasticine/open/pqn_atari_trac.py > logs/cuda4.log 2>&1 &
wait