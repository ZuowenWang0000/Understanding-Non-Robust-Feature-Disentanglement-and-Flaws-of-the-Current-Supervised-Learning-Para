#!/bin/bash

model_path=$1
arch="resnet50"

constraint="spatial_grid"
maxrot_maxtrans="(30, 0.09375)"       # (max rotation, max translation as percentage of image size)
grid_granularity_trans_rot="(5, 31)"  # (values per translation direction, values for rotation)
                                      # the default settings of the previous two lines correspond to grid775 in the report

python3 -m robustness.main --dataset cifar --data ./datasets --eval-only 1 \
                --out-dir ./eval_dir/ --arch ${arch} --adv-eval 1 \
                --resume "${model_path}" --constraint ${constraint} \
                --eps "${maxrot_maxtrans}" --attack-lr "${grid_granularity_trans_rot}" \
                --batch-size 1 --attack-steps 10