#!/bin/bash

model_path=$1
arch="resnet50"

eps="0.25"      # epsilon constraint for PGD
constraint="2"  # for L2-PGD, use "inf" for L_inf-PGD
stepsize="0.1"  # PGD learning rate 
iters="100"     # PGD iterations

python3 -m robustness.main --dataset cifar --data ./datasets --eval-only 1 \
                --out-dir ./eval_dir/ --arch ${arch} --adv-eval 1 \
                --resume "${model_path}" --eps ${eps} --constraint ${constraint} \
                --attack-steps ${iters} --attack-lr ${stepsize}