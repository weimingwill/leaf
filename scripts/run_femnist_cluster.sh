#!/bin/bash

mkdir -p log
now=$(date +"%Y%m%d_%H%M%S")

data_dir=/mnt/lustre/$(whoami)/projects/easyfl/easyfl/datasets/data
root_dir=/mnt/lustre/$(whoami)/projects/leaf

export PYTHONPATH=$PYTHONPATH:${root_dir}

srun -u --partition=innova --job-name=leaf-femnist \
    -n1 --gres=gpu:1 --ntasks-per-node=1 \
    python ${root_dir}/models/main.py --dataset femnist --model cnn --eval-every 1 \
    --data_dir ${data_dir}/femnist/femnist_iid_100_10_1_0.05_0.1_sample_0.9/ \
    --clients-per-round 10 --num-rounds 150 --batch-size 64 --num-epochs 5 --lr 0.01 | tee log/leaf-femnist-${now}.log &
