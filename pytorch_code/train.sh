#!/bin/bash

export ARCHI_NAME="resnet50"  # one of {resnet18, resnet50, gresnet18, gresnet34, stnresnet18, stnresnet34, ptnresnet18, ptnresnet34}
export BATCH_RATIO="1.0"      # $\alpha$ from the report (proportion of robust images per training batch)
export USE_DA="0"             # 0 = std data augmentation; 1 = std* data augmentation
export START_LR="0.1"         # initial learning rate

python3 train.py