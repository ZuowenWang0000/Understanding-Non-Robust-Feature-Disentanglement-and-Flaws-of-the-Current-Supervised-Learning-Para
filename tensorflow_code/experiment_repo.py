from datetime import datetime, timezone
import getpass
import io
import json
import pathlib
import uuid
import os
import pickle
import hashlib
import os

import numpy as np


def gen_short_uuid(num_chars=None):
    num = uuid.uuid4().int
    alphabet = '23456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz'
    res = []
    while num > 0:
        num, digit = divmod(num, len(alphabet))
        res.append(alphabet[digit])
    res2 = ''.join(reversed(res))
    if num_chars is None:
        return res2
    else:
        return res2[:num_chars]


def get_experiment_key(experiment_id):
    return 'experiments/{}_experiment.pkl'.format(experiment_id)

def get_checkpoint_key(checkpoint_id):
    return 'checkpoints/{}_checkpoint.pkl'.format(checkpoint_id)


def get_training_metadata_key(experiment_id):
    return 'training_metadata/{}_training_metadata.pkl'.format(experiment_id)


def get_training_info_key(experiment_id):
    return 'training_info/{}_training_info.pkl'.format(experiment_id)


def get_logdir_key(experiment_id):
    return 'logdir/{}'.format(experiment_id)


class ExperimentRepo:
    def __init__(self, use_s3=False, save_in_local_json=False,
                 json_filename='experiments_lambda_new.json',
                 local_dir_name='local_json_files',
                 root_dir='/cluster/work/math/fanyang-broglil/CoreRepo'):
        self.save_in_local_json = save_in_local_json
        self.json_filename = json_filename
        self.local_dir_name = local_dir_name
        # Have to think about long term solution if training on different clusters
        self.metadata_filepath = (pathlib.Path(__file__).parent /  root_dir).resolve()
        self.experiment_metadata_filepath = self.metadata_filepath / json_filename
        if self.save_in_local_json:
            self.experiments = {}
            self.experiment_id_by_name = {}
        else:
            self.load_experiment_metadata()
        if use_s3:
            import s3_utils
            cache_root_path = (pathlib.Path(__file__).parent / '../data/s3_cache').resolve()
            self.wrapper = s3_utils.S3Wrapper(cache_root_path=cache_root_path)
        else:
            import leo_utils
            self.wrapper = leo_utils.LeoWrapper(save_root_path=root_dir)
        self.uuid_length = 10

    def gen_short_uuid(self):
        new_id = gen_short_uuid(self.uuid_length)
        assert new_id not in self.experiments
        assert new_id not in self.experiment_id_by_name
        return new_id

    def gen_checkpoint_uuid(self):
        return gen_short_uuid(num_chars=None)

    def store_experiment_metadata(self):
        with open(self.experiment_metadata_filepath, 'w') as f:
            json.dump(self.experiments, f, indent=2, sort_keys=True)

    def store_experiment_metadata_local(self):
        with open(self.experiment_metadata_filepath_local, 'w') as f:
            json.dump(self.experiments, f, indent=2, sort_keys=True)

    def load_experiment_metadata(self):
        # Create empty file with {} if doesn't exist yet
        if not os.path.isfile(self.experiment_metadata_filepath):
            with open(self.experiment_metadata_filepath, 'w') as f:
                json.dump({}, f, indent=2, sort_keys=True)

        with open(self.experiment_metadata_filepath, 'r') as f:
            self.experiments = json.load(f)
        self.experiment_id_by_name = {}
        for experiment in self.experiments.values():
            cur_name = experiment['name']
            if cur_name != '':
                assert cur_name not in self.experiment_id_by_name
                assert cur_name not in self.experiments
                self.experiment_id_by_name[cur_name] = experiment['id']

    def create_new_experiment(self, dataset, model_family, worstofk,
                              attack_style, attack_limits, penaltyweight,
                              num_grouped_ids, group_size, hyperparameters,
                              name='', description='', verbose=True):
        allowed_datasets = ['cifar-10', 'cifar-100', 'mnist', 'svhn', 'imagenet']
        allowed_model_families = ['vgg', 'resnet', 'wide_resnet', 'shakeshake', 'densenet', 'inception', 'resnet50']
        # Allow a combination (or just one) of attacks - array of transformations to do
        allowed_attack_style = ['rotate', 'translate', 'flip', 'linf']

        assert dataset in allowed_datasets
        assert model_family in allowed_model_families
        if isinstance(attack_style, list):
            assert all([style in allowed_attack_style for style in attack_style])
        else:
            assert attack_style in allowed_attack_style

        new_id = self.gen_short_uuid()
        try:
            lsf_job_id = os.environ["LSB_JOBID"]
        except:
            lsf_job_id = ""
        new_id = new_id + "_" + lsf_job_id
        creation_time = datetime.now(timezone.utc).strftime('%Y-%m-%d_%H-%M-%S_%Z')
        username = getpass.getuser()
        # name enables extra name for experiment ID, experiment_id_by_name maps name to ID
        if name != '':
            assert name not in self.experiment_id_by_name
            assert name not in self.experiments
        new_experiment = {'id': new_id,
                          'hyperparameters': hyperparameters,
                          'username': username,
                          'model_family': model_family,
                          'dataset': dataset,
                          'worstofk': worstofk,
                          'attack_style': attack_style,
                          'attack_limits': attack_limits,
                          'num_grouped_ids': num_grouped_ids,
                          'lambda_reg': penaltyweight,
                          'group_size': group_size,
                          'name': name,
                          'description': description,
                          'creation_time': creation_time,
                          'completed': False,
                          'train_nat_accuracy': -1.0,
                          'train_adv_accuracy': -1.0,
                          'train_inp_accuracy': -1.0,
                          'test_nat_accuracy': -1.0,
                          # Currently adv here means worstof1 attack
                          'test_adv_accuracy': -1.0,
                          'test_grid_accuracy': -1.0,
                          'runtime': 0,
                          'runtime_wo_eval': 0,
                          'runtime_adv_ex_creation': 0,
                          'runtime_train_step': 0}
        self.experiments[new_id] = new_experiment
        if name != '':
            self.experiment_id_by_name[name] = new_id

        # put experiment metadata to experiments.json
        if self.save_in_local_json:
            local_exp_json_folder = os.path.join(self.metadata_filepath,
                                                 self.local_dir_name)
            if not os.path.isdir(local_exp_json_folder):
                os.makedirs(local_exp_json_folder)
            json_filename_local = new_id + "_" + self.json_filename
            self.experiment_metadata_filepath_local = os.path.join(
                local_exp_json_folder, json_filename_local)
            self.store_experiment_metadata_local()
        else:
            self.store_experiment_metadata()
        # create new expID_training_metadata.pkl
        new_training_metadata = {'checkpoints': {}, 'stored_training_info': False, 'logdir_files': []}
        self.put_training_metadata(new_id, new_training_metadata, verbose=verbose)
        if verbose:
            print(f'Created a new experiment with id "{new_id}"')
        return new_id

    def delete_experiment(self, experiment_id, verbose=True):
        assert experiment_id in self.experiments
        cur_exp = self.experiments[experiment_id]
        assert not cur_exp['completed']
        cur_metadata = self.get_training_metadata(experiment_id, verbose=verbose)
        assert len(cur_metadata['logdir_files']) == 0
        assert not cur_metadata['stored_training_info']
        assert len(cur_metadata['checkpoints']) == 0
        cur_name = cur_exp['name']
        if cur_name != '':
            del self.experiment_id_by_name[cur_name]
        del self.experiments[experiment_id]
        if self.save_in_local_json:
            self.store_experiment_metadata_local()
        else:
            self.store_experiment_metadata()
        if verbose:
            print(f'Deleted experiment with id "{experiment_id}"')

    def mark_experiment_as_completed(self, experiment_id, *,
                                     train_acc_nat=-1.0,
                                     train_acc_adv=-1.0,
                                     train_acc_inp=-1.0,
                                     test_acc_adv=-1.0, test_acc_nat=-1.0,
                                     test_acc_fo=-1.0, test_acc_grid=-1.0, 
                                     runtime=0,
                                     runtime_wo_eval=0,
                                     runtime_train_step=0,
                                     runtime_adv_ex_creation=0):
        assert experiment_id in self.experiments
        cur_exp = self.experiments[experiment_id]
        cur_exp['train_nat_accuracy'] = float(train_acc_nat)
        cur_exp['train_adv_accuracy'] = float(train_acc_adv)
        cur_exp['train_inp_accuracy'] = float(train_acc_inp)
        cur_exp['test_nat_accuracy'] = float(test_acc_nat)
        cur_exp['test_adv_accuracy'] = float(test_acc_adv)
        cur_exp['test_fo_accuracy'] = float(test_acc_fo)
        cur_exp['test_grid_accuracy'] = float(test_acc_grid)
        cur_exp['runtime'] = runtime
        cur_exp['runtime_wo_eval'] = runtime_wo_eval
        cur_exp['runtime_train_step'] = runtime_train_step
        cur_exp['runtime_adv_ex_creation'] = runtime_adv_ex_creation
        cur_exp['completed'] = True
        if self.save_in_local_json:
            self.store_experiment_metadata_local()
        else:
            self.store_experiment_metadata()

    def get_training_metadata(self, experiment_id, verbose=True):
        assert experiment_id in self.experiments
        key = get_training_metadata_key(experiment_id)
        metadata_bytes = self.wrapper.get(key, verbose=verbose)
        return pickle.loads(metadata_bytes)

    def put_training_metadata(self, experiment_id, metadata, verbose=True):
        assert experiment_id in self.experiments
        key = get_training_metadata_key(experiment_id)
        bytes_to_store = pickle.dumps(metadata)
        self.wrapper.put(bytes_to_store, key, verbose=verbose)


    # Checkpoint_data should include the weights of the model at checkpoint as an object
    def create_training_checkpoint(self, experiment_id, *, training_step, epoch,
                                   checkpoint_data=None,
                                   train_acc_nat=-1.0, train_acc_adv=-1.0,
                                   train_acc_inp=-1.0,
                                   test_acc_adv=-1.0, test_acc_nat=-1.0,
                                   test_acc_fo=-1.0, test_acc_grid=-1.0,
                                   test_loss_adv=-1.0, test_loss_nat=-1.0,
                                   test_loss_fo=-1.0, test_loss_grid=-1.0, 
                                   verbose=True):
        assert experiment_id in self.experiments
        creation_time = datetime.now(timezone.utc).strftime('%Y-%m-%d_%H-%M-%S_%Z')
        new_id = self.gen_checkpoint_uuid()
        new_checkpoint = {}
        new_checkpoint['id'] = new_id
        new_checkpoint['creation_time'] = creation_time
        new_checkpoint['training_step'] = training_step
        new_checkpoint['epoch'] = epoch
        new_checkpoint['train_nat_accuracy'] = train_acc_nat
        new_checkpoint['train_adv_accuracy'] = train_acc_adv
        new_checkpoint['train_inp_accuracy'] = train_acc_inp
        new_checkpoint['test_nat_accuracy'] = test_acc_nat
        new_checkpoint['test_adv_accuracy'] = test_acc_adv
        new_checkpoint['test_fo_accuracy'] = test_acc_fo
        new_checkpoint['test_grid_accuracy'] = test_acc_grid
        new_checkpoint['test_nat_loss'] = test_loss_nat
        new_checkpoint['test_adv_loss'] = test_loss_adv
        new_checkpoint['test_fo_loss'] = test_loss_fo
        new_checkpoint['test_grid_loss'] = test_loss_grid

        # Save chkpt data if there is - currently unused and data saved via tf_saver
        if checkpoint_data != None:
            bytes_to_store = pickle.dumps(checkpoint_data)
            # Put checkpoint_data to checkpoint.pkl
            key = get_checkpoint_key(new_id)
            self.wrapper.put(bytes_to_store, key, verbose=verbose)

        # Save checkpoint metadata
        cur_metadata = self.get_training_metadata(experiment_id, verbose=verbose)
        assert new_id not in cur_metadata['checkpoints']
        cur_metadata['checkpoints'][new_id] = new_checkpoint
        self.put_training_metadata(experiment_id, cur_metadata, verbose=verbose)
        if verbose:
            print(f'Created a new checkpoint with id "{new_id}" at iteration "{training_step}"')
        return new_id

    def get_training_checkpoint_data(self, checkpoint_id, verbose=True):
        key = get_checkpoint_key(checkpoint_id)
        checkpoint = self.wrapper.get(key, verbose=verbose)
        return pickle.loads(checkpoint)

    def store_training_info(self, experiment_id, training_info, verbose=True):
        assert experiment_id in self.experiments
        bytes_to_store = pickle.dumps(training_info)
        key = get_training_info_key(experiment_id)
        self.wrapper.put(bytes_to_store, key, verbose=verbose)
        cur_metadata = self.get_training_metadata(experiment_id, verbose=verbose)
        cur_metadata['stored_training_info'] = True
        self.put_training_metadata(experiment_id, cur_metadata, verbose=verbose)

    def get_training_info(self, experiment_id, verbose=True):
        assert experiment_id in self.experiments
        cur_metadata = self.get_training_metadata(experiment_id, verbose=verbose)
        assert cur_metadata['stored_training_info']
        key = get_training_info_key(experiment_id)
        training_info = self.wrapper.get(key, verbose=verbose)
        return pickle.loads(training_info)

    # Currently unused
    def store_logdir(self, experiment_id, logdir, verbose=True):
        assert experiment_id in self.experiments
        logdir_path = pathlib.Path(logdir).resolve()
        assert logdir_path.is_dir()
        tmp_filepaths = [x for x in logdir_path.glob('**/*') if x.is_file()]
        all_data = {}
        base_key = get_logdir_key(experiment_id) + '/'
        cur_logdir_files = []
        for cur_filepath in tmp_filepaths:
            with open(cur_filepath, 'rb') as f:
                cur_data = f.read()
            cur_relative_path = str(cur_filepath.relative_to(logdir_path))
            cur_logdir_files.append(cur_relative_path)
            cur_key = base_key + cur_relative_path
            all_data[cur_key] = cur_data
        self.wrapper.put_multiple(all_data, verbose=verbose)
        cur_metadata = self.get_training_metadata(experiment_id, verbose=verbose)
        cur_metadata['logdir_files'] = cur_logdir_files
        self.put_training_metadata(experiment_id, cur_metadata, verbose=verbose)
