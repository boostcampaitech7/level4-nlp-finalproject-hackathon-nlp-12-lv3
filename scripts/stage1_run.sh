export ACCELERATE_USE_FSDP=1 
export FSDP_CPU_RAM_EFFICIENT_LOADING=1 
export NCCL_IB_DISABLE=1 

python nltk_download.py

CUDA_VISIBLE_DEVICES=0,1 accelerate launch train.py --cfg-path configs/train_stage1.yaml