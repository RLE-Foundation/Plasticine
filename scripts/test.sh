export PYTHONPATH="${PYTHONPATH}:./plasticine"

CUDA_VISIBLE_DEVICES=1 && python plasticine/standard/ppo_craftax_redo.py