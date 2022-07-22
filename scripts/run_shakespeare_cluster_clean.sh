#!/bin/bash

mkdir -p log
now=$(date +"%Y%m%d_%H%M%S")

data_dir=/mnt/lustre/$(whoami)/projects/easyfl/easyfl/datasets/data
root_dir=/mnt/lustre/$(whoami)/projects/leaf

export PYTHONPATH=$PYTHONPATH:${root_dir}

export PATH="/mnt/lustre/share/cuda-10.1/bin:$PATH"
export LD_LIBRARY_PATH="/mnt/lustre/share/cuda-10.1/lib64:$LD_LIBRARY_PATH"
export CUDA_VISIBLE_DEVICES=0

srun -u --partition=innova --job-name=leaf-shakespeare \
    -n8 --gres=gpu:8 --ntasks-per-node=8 -w SG-IDC2-10-51-3-50 \
  python -u ${root_dir}/models/main.py --dataset shakespeare --model stacked_lstm --eval-every 1 \
  --data_dir ${data_dir}/shakespeare/shakespeare_iid_10_10_1_0.2_0.2_sample_0.9/ \
  --clients-per-round 10 --num-rounds 20 --batch-size 64 --num-epochs 10 --lr 0.8 --test-all | tee log/leaf-shakespear-clean-${now}.log &
