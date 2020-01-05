# Training and Attack Success Rate Evaluation

This repository contains code to train and evaluate models against
adversarially chosen rotations and translations. It can be used to reproduce the
main experiments of:

**Invariance-inducing regularization using worst-case transformations suffices to boost accuracy and spatial robustness**<br>
*Fanny Yang, Zuowen Wang and Christina Heinze-Deml*<br>

The code is based on https://github.com/MadryLab/adversarial_spatial. 

The main scipts to run are `train.py` and `eval.py`, which will train and
evaluate a model respectively.

**Note:** `train.py` only supports groups of size 2 for now.

**Supported datasets:**
CIFAR-10, CIFAR-100, SVHN, imageNet

**Supported architectures:**

Resnet-8, 18, 34 (in file resnet.py) & Resnet-50 (in file resnet50.py)



Different training options are included in the folder `configs`
A template is annotated below. 

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
      "use_reg": true   #true if we use regularization

      # if using imagent, specify "dataset": "imagenet" here
  },

  "training": {
      "tf_random_seed": 1,
      "np_random_seed": 1,
      "max_num_training_steps": 80000,
      "num_output_steps": 5000,
      "num_summary_steps": 5000,
      "num_easyeval_steps": 5000,
      "num_eval_steps": 80000,
      "num_checkpoint_steps": 5000,
      "num_ids": 64,       #annotated as b
      "batch_size": 128,   #can be b, 2b or 3b
      "lr" : 0.1,
      "step_size_schedule": [[0, 0.1], [40000, 0.01], [60000, 0.001]],
      "momentum": 0.9,
      "weight_decay": 0.0002,
      "eval_during_training": true, #if true, will automatically enter evaluation after training
      "adversarial_training": true,
      "adversarial_ce": false,  # set to true if only adversarial examples are used for cross-entropy
      "nat_ce": false,          # set to true if only original examples are used for cross-entropy
      "data_augmentation": true,
      "data_augmentation_reg": false,
      "group_size": 2,
      "lambda_": 1          #the coefficient of the regularizer 
  }, 

  "eval": {
      "num_eval_examples": 10000,
      "batch_size": 128,
      "adversarial_eval": true
  },

#defense mechanism
  "defense": {
      "reg_type": "kl",
      "cce_adv_exp_wrt": "cce", # adversarial examples used for cross-entropy are generated w.r.t 
      "reg_adv_exp_wrt": "kl", # adversarial examples used for regularizer are generated w.r.t 

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
      "spatial_limits": [3, 3, 30],  # [24, 24, 30] for imageNet
      "random_tries": 10,
      "grid_granularity": [5, 5, 31]
  },

#attack policy (for evaluation)
  "attack": {
      "use_linf": false,
      "use_spatial": true,
      "only_rotation": false,
      "only_translation": false,

      "loss_function": "xent",
      "epsilon": 8.0,
      "num_steps": 5,
      "step_size": 2.0,
      "random_start": false,

      "spatial_limits": [3, 3, 30],  # [24, 24, 30] for imageNet
      "random_tries": 10,
      "grid_granularity": [5, 5, 31]
  },

  "data": {
    "dataset_name": "cifar-10",
    "data_path": "./datasets/cifar10"
  }
}

```

Run with a particular config file
```
python train.py --config PATH/TO/CONFIG_FILE
```

## Standard CIFAR data augmentation
By default data augmentation only includes random left-right flips. Standard CIFAR10
augmentation (+-4 pixel crops) can be achieved by setting
`adversarial_training: true`, `spatial_method: random`, `random_tries: 1`,
`spatial_limits: [4, 4, 0]`.

run 
```
python train.py --config ./configs/std.json
```
for standard training (std) with only translation as data augmentation

run 
```
python train.py --config ./configs/std_star.json
```
for training (std*) with translation and rotation as data augmentation




## Run with various settings for adversarial training
### Run with unregularized adversarial training
Set ```use_reg = false``` in the configuration file. 
See  ```configs/at_rob_wo_10.json``` for an example.
 

### Run with different batch types 

We can use solely original images for the cross-entropy part of the loss function.
To achieve that, set ```nat_ce = true, adversarial_ce = false``` in the configuration file.

Accordingly, ```nat_ce = false, adversarial_ce = true``` and ```nat_ce = false, adversarial_ce = false``` correspond to "rob" and "mix" in the 
paper respectively. For "nat" we only use the original examples for the cross-entropy and for "mix" we use both original 
and adversarial examples. 


### Generate adversarial examples w.r.t different functions
Regardless loss function, we can generate adversarial examples, which can be used independently for either
cross-entropy or regularizer, with respect to different functions.

To achieve this, we need to configure ```cce_adv_exp_wrt``` and ```reg_adv_exp_wrt```

For instance, to conduct training in the same way as adversarial logit pairing (ALP) [1],
we set both  ```cce_adv_exp_wrt``` and ```reg_adv_exp_wrt``` to ```cce```. Then the adversarial 
examples which entering the regularizer will be generated w.r.t. cross-entropy.

**Note:** if ```cce_adv_exp_wrt != reg_adv_exp_wrt``` and using a mixed batch for cross-entropy, we need to set 
```batch_size ==  3 * num_ids```, since we need two sets of different adversarial examples for 
cross-entropy and regularizer respectively.

Please refer to ```configs/l2_mix_wo_10.json```

### Evaluation
If you want to solely evaluate a trained model, you can use the eval.py script.

```
python3 eval.py \
--config=$config_path \
--save_root_path=$repo_dir \
--exp_id_list  '' \
--eval_on_train=$EVAL_ON_TRAIN \   # set to 0, otherwise evaluating on training set.
--save_filename=$SAVE_FNAME \
--linf_attack=0
```


# Citation
For citing this work please use the following bibtex:
```
@article{DBLP:journals/corr/abs-1906-11235,
  author    = {Fanny Yang and
               Zuowen Wang and
               Christina Heinze{-}Deml},
  title     = {Invariance-inducing regularization using worst-case transformations
               suffices to boost accuracy and spatial robustness},
  journal   = {CoRR},
  volume    = {abs/1906.11235},
  year      = {2019},
  url       = {http://arxiv.org/abs/1906.11235},
  archivePrefix = {arXiv},
  eprint    = {1906.11235},
  timestamp = {Thu, 27 Jun 2019 18:54:51 +0200},
  biburl    = {https://dblp.org/rec/bib/journals/corr/abs-1906-11235},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
# References
[1] Harini Kannan, Alexey Kurakin, and Ian Goodfellow. Adversarial Logit Pairing. arXiv preprint
arXiv:1803.06373, 2018.


