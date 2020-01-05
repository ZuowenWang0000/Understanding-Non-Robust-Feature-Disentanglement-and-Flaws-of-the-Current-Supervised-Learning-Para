#!/bin/bash -l

python3 ../train_asr.py \
--config=$config_name \
--save-root-path=$repo_dir \
--local_json_dir_name=$local_json_folder_name \
--worstofk=$WORSTOFK \
--lambda-reg=$LAMBDA_REG \
--use_reg=$USE_REG \
--seed=$SEED \
--num-ids=$NUM_IDS \
--fo_epsilon=$FO_EPSILON \
--fo_num_steps=$FO_NUM_STEPS
