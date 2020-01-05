#!/bin/bash -l
python3 ../eval.py \
--config=$CONFIG_NAME \
--save_root_path=$repo_dir \
--exp_id_list=$EXP_LIST \
--eval_on_train=$EVAL_ON_TRAIN \
--save_filename=$SAVE_FNAME \
--linf_attack=$LINF_ATTACK
