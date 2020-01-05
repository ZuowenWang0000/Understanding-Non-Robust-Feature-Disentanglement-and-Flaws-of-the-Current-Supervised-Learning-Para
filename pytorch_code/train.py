import sys
sys.path.append("..")
import os
from robustness import model_utils, datasets, train, defaults
from robustness.datasets import CIFAR
from robustness.datasets import robust_CIFAR, da_robust_CIFAR

# We use cox (http://github.com/MadryLab/cox) to log, store and analyze
# results. Read more at https//cox.readthedocs.io.
from cox.utils import Parameters
import cox.store

archi_name = str(os.environ['ARCHI_NAME'])
print("Using architecture: {}".format(archi_name))
if_da = int(os.environ['USE_DA'])
print("Using [3, 3, 30] data augmentation: {}".format(if_da))
start_lr = float(os.environ['START_LR'])
print("Using start learning rate: %f" % start_lr)
batch_ratio = float(os.environ['BATCH_RATIO'])
print("Using batch ratio: %f" % batch_ratio)

# Hard-coded dataset, architecture, batch size, workers
if if_da:
    ds = da_robust_CIFAR('./datasets')
else:
    ds = robust_CIFAR('./datasets')

m, _ = model_utils.make_and_restore_model(arch=archi_name, dataset=ds)
train_loader, val_loader = ds.make_loaders(batch_size=128, workers=8, batch_ratio=batch_ratio)

# Create a cox store for logging
out_store = cox.store.Store('./out_store/')

# Hard-coded base parameters
train_kwargs = {
    'lr': start_lr,
    'out_dir': "train_out",
    'adv_train': 0,
    'constraint': '2',
    'eps': 0.5,
    'attack_lr': 1.5,
    'attack_steps': 20
}
train_args = Parameters(train_kwargs)

# Fill whatever parameters are missing from the defaults
train_args = defaults.check_and_fill_args(train_args,
                        defaults.TRAINING_ARGS, robust_CIFAR)
train_args = defaults.check_and_fill_args(train_args,
                        defaults.PGD_ARGS, robust_CIFAR)

# Train a model
train.train_model(train_args, m, (train_loader, val_loader), store=out_store)