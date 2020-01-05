# Code for Questions 2 and 3

## Preparing the datasets

Download the datasets from [this link](https://github.com/MadryLab/constructed-datasets).
Put the folder `d_robust_CIFAR` into the `./datasets` folder.

## Training a model

Change the file `train.sh` to contain your desired training parameters and then run it. By default, the script stores model checkpoints in `./out_store` under a random directory name which is printed at the start of the output.

## Evaluating a model

### Under PGD attacks

1. Change the file `eval_pgd.sh` to contain your desired attack parameters (or leave them as-is).
2. Run `./eval_pgd.sh <model-path>`, where `<model-path>` is the path to a model checkpoint.

### Under spatial attacks

1. Change the file `eval_spatial.sh` to contain your desired attack parameters (or leave them as-is).
2. Run `./eval_spatial.sh <model-path>`, where `<model-path>` is the path to a model checkpoint.
