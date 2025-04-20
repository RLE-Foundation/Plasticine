export PYTHONPATH="${PYTHONPATH}:./plasticine"
CUDA_VISIBLE_DEVICES=1 && python plasticine/standard/td3_dmc_vanilla.py