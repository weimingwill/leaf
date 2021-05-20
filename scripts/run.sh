#!/bin/bash
python ~/personal-projects/leaf/models/main.py --dataset femnist --model cnn --eval-every 1 \
    --data_dir ~/personal-projects/easyfl/easyfl/datasets/data/femnist/femnist_iid_100_10_1_0.05_0.1_sample_0.9/ \
    --clients-per-round 2 --num-rounds 1 --batch-size 64 --num-epochs 1 --lr 0.01
