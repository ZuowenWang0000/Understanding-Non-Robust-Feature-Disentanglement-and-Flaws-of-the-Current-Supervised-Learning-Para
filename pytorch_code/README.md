# Code for Questions 2 and 3

The code for Questions 2 and 3 is heavily based on the PyTorch [robustness library](https://github.com/MadryLab/robustness) and the PyTorch [codebase for the ICML 2019 paper "Exploring the Landscape of Spatial Robustness"](https://github.com/MadryLab/spatial-pytorch).

Running the code will require installing several Python modules, most of which can be installed through `pip`. There is one module which must be installed manually (the version from `pip` does not have PyTorch support): [GrouPy](https://github.com/adambielski/GrouPy) must be downloaded and then installed via `python3 setup.py install`.

The other modules which can be installed from `pip` are: `torch`, `torchvision`, `pickle`, `dill`, `cox`, and `tensorboardX` (this list is not exhaustive). If you receive a runtime error about missing modules, simply install the offending module from `pip`.

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
