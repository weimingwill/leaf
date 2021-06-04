#!/bin/bash

mkdir -p log
now=$(date +"%Y%m%d_%H%M%S")

data_dir=/mnt/lustre/$(whoami)/projects/easyfl/easyfl/datasets/data
root_dir=/mnt/lustre/$(whoami)/projects/leaf

export PYTHONPATH=$PYTHONPATH:${root_dir}

srun -u --partition=innova --job-name=leaf-shakespeare \
    -n1 --gres=gpu:1 --ntasks-per-node=1 \
    python -u ${root_dir}/models/main.py --dataset shakespeare --model stacked_lstm --eval-every 1 \
      --data_dir ${data_dir}/shakespeare/shakespeare_iid_10_10_1_0.2_0.2_sample_0.9/ \
      --clients-per-round 10 --num-rounds 100 --batch-size 64 --num-epochs 10 --lr 0.8 --test-all | tee log/leaf-shakespear-small-${now}.log &
