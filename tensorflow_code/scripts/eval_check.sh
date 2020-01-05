#!/bin/bash -l

export CONFIG_NAME='../eval_configs/eval_check.json' #the evaluation config file
export repo_dir='../logdir/check' #where the checkpoint is stored
export EVAL_ON_TRAIN=0  #if 1, evaluating on training set!
export SAVE_FNAME='result_check.json'
export LINF_ATTACK=0

echo "Using configuration: ${config_name}"

for VAR_EXP_LIST in 'eGjcGuW3zk_' #you can pass in multiple folders but this can't evaluate checkpoints under the same folder
do
    export EXP_LIST=$VAR_EXP_LIST
    sh submit-eval-single-cp.sh
done