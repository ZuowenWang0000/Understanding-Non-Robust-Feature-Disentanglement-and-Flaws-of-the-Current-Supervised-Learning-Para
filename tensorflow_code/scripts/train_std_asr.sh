#!/bin/sh

export config_name='../train_configs/std_asr.json'
export local_json_folder_name='train_std_asr'
export repo_dir='../logdir/std_asr'

echo "Using configuration: ${config_name}"

for VAR_WORSTOFK in 1
do
    export WORSTOFK=$VAR_WORSTOFK
    for VAR_LAMBDA_REG in 0
    do
        export LAMBDA_REG=$VAR_LAMBDA_REG
        for VAR_USE_REG in 0
        do
            export USE_REG=$VAR_USE_REG
            for VAR_SEED in 1
            do
                export SEED=$VAR_SEED
                for VAR_NUM_IDS in 64
                do
                    export NUM_IDS=$VAR_NUM_IDS
                    for VAR_FO_EPSILON in 16
                    do
                        export FO_EPSILON=$VAR_FO_EPSILON
                        for VAR_FO_NUM_STEPS in 5
                        do
                            export FO_NUM_STEPS=$VAR_FO_NUM_STEPS
                            sh submit-train-asr-local.sh
                        done
                    done
                done
            done
        done
    done
done
