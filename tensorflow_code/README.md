# Training and Attack Success Rate Evaluation

This repository contains code to train and evaluate models against
several adversarial attacks including spatial attack and linf-PGD attack. It can be used to reproduce the
main experiments of:

The code is based on https://github.com/MadryLab/adversarial_spatial. 

The main scipts to run are `train_asr.py` and `eval.py`, which will train and
evaluate a model respectively.


**Supported datasets:**
CIFAR-10, CIFAR-100, SVHN, imageNet

**Supported architectures:**

Resnet-8, 18, 34 (in file resnet.py) & Resnet-50 (in file resnet50.py)



Different training options are included in the folder `train_configs`
A template is annotated below. Please keep the default values if not mentioned in the comment.
Some the parameters are irrelevant for our experiments in this work.

```
{
  "model": {
      "output_dir": "output/cifar10",
      "pad_mode": "constant",
      "model_family": "resnet",
      "resnet_depth_n": 5,
      "filters": [16, 16, 32, 64],
      "pad_size": 32,
      "n_classes": 10,
      "use_reg": false   

  },

  "training": {
      "tf_random_seed": 1,
      "np_random_seed": 1,
      "max_num_training_steps": 78125,
      "num_output_steps": 3125,
      "num_summary_steps": 3125,
      "num_easyeval_steps": 3125,
      "num_eval_steps": 78125,
      "num_checkpoint_steps": 3125,
      "num_ids": 64,
      "batch_size": 64,
      "lr" : 0.1,
      "step_size_schedule": [[0, 0.1], [40000, 0.01], [60000, 0.001]],
      "momentum": 0.9,
      "weight_decay": 0.0002,
      "eval_during_training": true,
      "adversarial_training": true,
      "adversarial_ce": false,
      "nat_ce": false,
      "data_augmentation": true, 
      "data_augmentation_reg": false,
      "group_size": 2,
      "lambda_": 1
  }, 

  "eval": {
      "num_eval_examples": 10000,
      "batch_size": 128,
      "adversarial_eval": true  # when false, only evaluated with original test images
  },

#defense mechanism
  "defense": {
      "reg_type": "kl",
      "cce_adv_exp_wrt": "cce", 
      "reg_adv_exp_wrt": "kl",

      "use_linf": false,
      "use_spatial": true,
      "only_rotation": false,
      "only_translation": false,

      "loss_function": "xent",
      "epsilon": 8.0,
      "num_steps": 5,
      "step_size": 2.0,
      "random_start": false,

      "spatial_method": "random",
      "spatial_limits": [4, 4, 0],  #this if for std, for std* use [3, 3, 30]'
      "random_tries": 1,
      "grid_granularity": [5, 5, 31]
  },

#attack policy (for evaluation)
  "attack": {
      "use_l2": false,
      "use_linf": false,  #true for use linf-PGD evaluation
      "use_spatial": true, #true for spatial evaluation
      "only_rotation": false, 
      "only_translation": false,

      #linf-PGD parameters
      "loss_function": "xent",
      "epsilon": 8.0,
      "num_steps": 5,
      "step_size": 2.0,
      "random_start": false,

    ` # spatial evaluation parameters
      "spatial_method": "grid",
      "spatial_limits": [3, 3, 30],  
      "random_tries": 10,
      "grid_granularity": [5, 5, 31]
  },

  "data": {
    "dataset_name": "cifar-10",
    "data_path": "./datasets/cifar10"
  }
}

```

### Training 
For training models and saving multiple checkpoints for evaluation, you can run the train_std_asr.sh
or train_std_star_asr.sh scripts in ./scripts folder. Or run the train_asr.py script with parameters passed 
 
```
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
```

### Evaluation
If you want to solely evaluate a trained model, you can use the eval.py script.

```
python3 ../eval.py \
--config=$CONFIG_NAME \
--save_root_path=$repo_dir \
--exp_id_list=$EXP_LIST \
--eval_on_train=$EVAL_ON_TRAIN \
--save_filename=$SAVE_FNAME \
--linf_attack=$LINF_ATTACK
```


# References
[1] Harini Kannan, Alexey Kurakin, and Ian Goodfellow. Adversarial Logit Pairing. arXiv preprint
arXiv:1803.06373, 2018.


