#!/bin/bash

mkdir -p log
now=$(date +"%Y%m%d_%H%M%S")

data_dir=/mnt/lustre/$(whoami)/projects/easyfl/easyfl/datasets/data
root_dir=/mnt/lustre/$(whoami)/projects/leaf

export PYTHONPATH=$PYTHONPATH:${root_dir}

#export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PATH="/mnt/lustre/share/cuda-10.1/bin:$PATH"
export LD_LIBRARY_PATH="/mnt/lustre/share/cuda-10.1/lib64:$LD_LIBRARY_PATH"

python -u ${root_dir}/models/main.py --dataset femnist --model cnn --eval-every 1 \
      --data_dir ${data_dir}/femnist/femnist_iid_10_10_1_0.05_0.05_sample_0.9/ \
      --clients-per-round 10 --num-rounds 150 --batch-size 64 --num-epochs 10 --lr 0.01 --test-all | tee log/leaf-femnist-${now}.log &

