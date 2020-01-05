"""
Evaluation of a given checkpoint in the standard and adversarial sense.  Can be
called as an infinite loop going through the checkpoints in the model directory
as they appear and evaluating them. Accuracy and average loss are printed and
added as tensorboard summaries.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from datetime import datetime
import json
import math
import os
import sys
import time
import copy

import numpy as np
import tensorflow as tf
from tqdm import trange

import cifar10_input
import cifar100_input
import svhn_input
# import imagenet_input
import resnet
import vgg
from spatial_attack_evaluation import SpatialAttack
from pgd_attack import LinfPGDAttack, L2PGDAttack
import utilities

# A function for evaluating a single checkpoint
def evaluate(model, attack, sess, config, attack_type, data_path,
             summary_writer=None, eval_on_train=False):
    num_eval_examples = config.eval.num_eval_examples
    eval_batch_size = config.eval.batch_size

    if config.data.dataset_name == "cifar-10":
        data_iterator = cifar10_input.CIFAR10Data(data_path)
    elif config.data.dataset_name == "cifar-100":
        data_iterator = cifar100_input.CIFAR100Data(data_path)
    elif config.data.dataset_name == "svhn":
        data_iterator = svhn_input.SVHNData(data_path)
    # elif config.data.dataset_name == "imagenet":
    #     data_iterator = imagenet_input.IMAGENETData(data_path, eval_batch_size, config.model.pad_size, sess)
    else:
        raise ValueError("Unknown dataset name.")

    global_step = tf.train.get_or_create_global_step()
    # Iterate over the samples batch-by-batch
    num_batches = int(math.ceil(num_eval_examples / eval_batch_size))
    total_xent_nat = 0.
    total_xent_adv = 0.
    total_corr_nat = 0
    total_corr_adv = 0
    total_succ_attack = 0

    for ibatch in trange(num_batches):
      if config.data.dataset_name == "imagenet":
          if eval_on_train:
              x_batch, y_batch, epoch_done = data_iterator.train_data.get_next_batch(eval_batch_size)
          else:
              x_batch, y_batch, epoch_done = data_iterator.eval_data.get_next_batch(eval_batch_size)

          noop_trans = np.zeros([len(x_batch), 3])
          if config.eval.adversarial_eval:
              x_batch_adv, adv_trans = attack.perturb(x_batch, y_batch, sess)
          else:
              x_batch_adv, adv_trans = x_batch, noop_trans
      else:
          bstart = ibatch * eval_batch_size
          bend = min(bstart + eval_batch_size, num_eval_examples)

          if eval_on_train:
            x_batch = data_iterator.train_data.xs[bstart:bend, :]
            y_batch = data_iterator.train_data.ys[bstart:bend]
          else:
            x_batch = data_iterator.eval_data.xs[bstart:bend, :]
            y_batch = data_iterator.eval_data.ys[bstart:bend]

          noop_trans = np.zeros([len(x_batch), 3])
          if config.eval.adversarial_eval:
              if config.attack.use_linf==True and config.attack.use_spatial==False: #PGD attack
                  x_batch_adv = attack.perturb(x_batch, y_batch, sess)
                  adv_trans = noop_trans
              else: # all the rest spatial methods
                  x_batch_adv, adv_trans = attack.perturb(x_batch, y_batch, sess)
          else:
              x_batch_adv, adv_trans = x_batch, noop_trans

      dict_nat = {model.x_input: x_batch,
                  model.y_input: y_batch,
                  model.transform: noop_trans,
                  model.is_training: False}

      dict_adv = {model.x_input: x_batch_adv,
                  model.y_input: y_batch,
                  model.transform: adv_trans,
                  model.is_training: False}

      cur_corr_nat, cur_xent_nat = sess.run([model.num_correct, model.xent],
                                            feed_dict = dict_nat)
      cur_corr_adv, cur_xent_adv = sess.run([model.num_correct, model.xent],
                                            feed_dict = dict_adv)

      # calculate attack success rate
      curr_batch_size = x_batch.shape[0]
      corr_pred_nat = sess.run(model.correct_prediction, feed_dict = dict_nat) # a boolean list about correctness on the nat. batch prediction
      cur_pred_adv = sess.run(model.predictions, feed_dict = dict_adv) #on this adv.batch, return the prediction list
      corr_pred_adv = cur_pred_adv == y_batch

      empty_index = np.arange(curr_batch_size)
      curr_corr_indices_nat = empty_index[corr_pred_nat]

      temp = np.take(corr_pred_adv, indices = curr_corr_indices_nat)

      cur_succ_attack = cur_corr_nat - np.sum(temp)

      total_xent_nat += cur_xent_nat
      total_xent_adv += cur_xent_adv
      total_corr_nat += cur_corr_nat
      total_corr_adv += cur_corr_adv

      total_succ_attack += cur_succ_attack
      # print("total successful attack")
      # print(total_succ_attack)
      # print("total correct nat predcition")
      # print(total_corr_nat)

    avg_xent_nat = total_xent_nat / num_eval_examples
    avg_xent_adv = total_xent_adv / num_eval_examples
    acc_nat = total_corr_nat / num_eval_examples
    acc_adv = total_corr_adv / num_eval_examples

    attack_success_rate = total_succ_attack / total_corr_nat

    if summary_writer:
        summary = tf.Summary(value=[
              tf.Summary.Value(tag='xent_adv_eval', simple_value= avg_xent_adv),
              tf.Summary.Value(tag='xent_nat_eval', simple_value= avg_xent_nat),
              # tf.Summary.Value(tag='xent_adv', simple_value= avg_xent_adv),
              # tf.Summary.Value(tag='xent_nat', simple_value= avg_xent_nat),
              tf.Summary.Value(tag='accuracy_adv_eval', simple_value= acc_adv),
              tf.Summary.Value(tag='accuracy_nat_eval', simple_value= acc_nat),
              # tf.Summary.Value(tag='accuracy_adv', simple_value= acc_adv),
              # tf.Summary.Value(tag='accuracy_nat', simple_value= acc_nat)
              tf.Summary.Value(tag='attack_success_rate', simple_value=attack_success_rate),
            ])
        summary_writer.add_summary(summary, global_step.eval(sess))

        # adv_image_eval, adv_image_after_stn_eval= sess.run([model.x_image,model.x_image_after_stn], feed_dict= dict_adv)
        # tf.summary.image('adv_x_eval', adv_image_eval, collections=['image'])
        # tf.summary.image('adv_x_after_stn_eval', adv_image_after_stn_eval, collections=['image'])
        # image_summaries = tf.summary.merge_all('image')
        # summary_image = sess.run(image_summaries, feed_dict=dict_adv)
        # summary_writer.add_summary(summary_image, global_step.eval(sess))

    step = global_step.eval(sess)
    print('Eval at step: {}'.format(step))
    print('  Adversary: ', attack_type)
    print('  natural: {:.2f}%'.format(100 * acc_nat))
    print('  adversarial: {:.2f}%'.format(100 * acc_adv))
    print('  avg nat xent: {:.4f}'.format(avg_xent_nat))
    print('  avg adv xent: {:.4f}'.format(avg_xent_adv))
    print('  attack success rate: {:.2f}%'.format(100 * attack_success_rate))

    return [100 * acc_nat, 100 * acc_adv, avg_xent_nat, avg_xent_adv, 100 * attack_success_rate]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                        description='Eval script options',
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-c', '--config', type=str,
                        help='path to config file',
                        default="configs/christinaconfig_cifar10_spatial_eval.json", required=False)
    parser.add_argument('--save_root_path', type=str,
                        help='path to repo dir',
                        default='/Users/heinzec/projects/core-da/repo_dir_7jan', required=False)
    parser.add_argument('--exp_id_list', type=str, nargs='+',
                        default=['3e3p7xPG98_1058376','4hbWroJkyE_1058258'])
    parser.add_argument('--eval_on_train', type=int,
                        help='flag whether to use training or test images',
                        default=0, required=False)
    parser.add_argument('-s', '--save_filename', type=str,
                        help='path to plots folder',
                        default='test.json', required=False)
    parser.add_argument('--linf_attack', type=int,
                        help='path to plots folder',
                        default=0, required=False)

    args = parser.parse_args()
    config_dict = utilities.get_config(args.config)
    dataset = config_dict['data']['dataset_name']
    # setting up save folders
    split = 'train' if args.eval_on_train else 'test'
    print(args.exp_id_list)

    # args.exp_id_list = args.exp_id_list[0].split("_")

    # if len(args.exp_id_list) > 1:
    #     exp_range = args.exp_id_list[0]+'_'+args.exp_id_list[-1]
    # else:
    #     exp_range = args.exp_id_list[0]
    # print(exp_range)
    save_folder = os.path.join(args.save_root_path,
        'additional_evals_{}'.format(dataset))
    import ntpath
    _, config_file_name = ntpath.split(args.config)
    config_file_name = config_file_name.replace('.json','')

    os.makedirs(save_folder, exist_ok=True)
    save_filename = os.path.join(save_folder,
        '{}_{}_{}_{}'.format(dataset, split, config_file_name, args.save_filename))

    if args.eval_on_train:
        if dataset == 'cifar-10' or dataset == 'cifar-100':
            config_dict['eval']['num_eval_examples'] = 50000
        elif dataset == 'svhn':
            config_dict['eval']['num_eval_examples'] = 73257
        else:
            raise NotImplementedError

    config_dict_copy = copy.deepcopy(config_dict)
    out_dict = {}
    out_dict['hyperparameters'] = config_dict_copy
    config = utilities.config_to_namedtuple(config_dict)

    # num_ids in model does not matter for eval
    num_ids = 64
    model_family = config.model.model_family
    if model_family == "resnet":
        if config.attack.use_spatial and config.attack.spatial_method == 'fo':
            diffable = True
        else:
            diffable = False
        model = resnet.Model(config.model, num_ids, diffable)
    elif model_family == "vgg":
        if config.attack.use_spatial and config.attack.spatial_method == 'fo':
            diffable = True
        else:
            diffable = False
        model = vgg.Model(config.model, num_ids, diffable)

    global_step = tf.train.get_or_create_global_step()
    if config.attack.use_l2:
        assert args.linf_attack==0 and config.attack.use_linf==0 and config.attack.use_spatial==0
        print("l2 pgd attack!")
        attack_name = 'l2_pgd'
        attack_eval = L2PGDAttack(model, config.attack,
                epsilon=config.attack.epsilon,
                step_size=config.attack.step_size,
                num_steps=config.attack.num_steps)

    elif config.attack.use_linf:
        if config.attack.use_spatial:
            assert args.linf_attack == 0
            print("spgd attack!")
            attack_name = 'spgd'
            attack_eval = SpatialAttack(model, config.attack, 'fo', 1,
                config.attack.spatial_limits,
                config.attack.epsilon,
                config.attack.step_size,
                config.attack.num_steps)
        else:  # set use_linf=tt + use_spatial=ff => PGD attack
            # TODO
            #  this design is not very good. need to leave all controls over attack methods in config files.
            # model, config, epsilon, step_size, num_steps
            print("pgd attack! ")
            attack_name = 'linf_pgd'
            attack_eval = LinfPGDAttack(model, config.attack,
                epsilon=config.attack.epsilon,
                step_size=config.attack.step_size,
                num_steps=config.attack.num_steps)
    else:
        if config.attack.spatial_method == 'random_eval':  # using random_eval instead of random is to be backwards compatible.
                                                # in the old configs, the random spatial method in attack still gives grid evaluation
            print("rand attack!")
            attack_name = 'rand_spatial'
            attack_eval = SpatialAttack(model, config.attack, 'random')
        else:
            print("grid attack!")
            attack_name = 'grid'
            attack_eval = SpatialAttack(model, config.attack, 'grid')

    saver = tf.train.Saver()

    for id in args.exp_id_list:
        out_dict[id] = {}
        model_dir = '%s/logdir/%s' % (args.save_root_path, id)
        ckpt = tf.train.get_checkpoint_state(model_dir)

        print("model dir!")
        print(model_dir)
        if ckpt is None:
            print('No checkpoint found.')
        else:
            with tf.Session() as sess:
                # Restore the checkpoint
                saver.restore(sess,
                    os.path.join(model_dir,
                                 ckpt.model_checkpoint_path.split("/")[-1]))
                [acc_nat, acc_grid, _, _,asr] = evaluate(
                    model, attack_eval, sess, config, attack_name,
                    config.data.data_path, eval_on_train=args.eval_on_train)
                out_dict[id]['{}_grid_accuracy'.format(split)] = acc_grid
                out_dict[id]['{}_nat_accuracy'.format(split)] = acc_nat
                out_dict[id]['{}_attack success rate'.format(split)] = asr
                # save results
                with open(save_filename, 'w') as result_file:
                    json.dump(out_dict, result_file, sort_keys=True, indent=4)

    grid_accuracy = []
    nat_accuracy = []
    asr = []
    for key in out_dict:
        if key != 'hyperparameters':
            grid_accuracy.append(out_dict[key]['{}_grid_accuracy'.format(split)])
            nat_accuracy.append(out_dict[key]['{}_nat_accuracy'.format(split)])
            asr.append(out_dict[key]['{}_attack success rate'.format(split)])

    out_dict['{}_grid_accuracy_summary'.format(split)] = (np.mean(grid_accuracy),
        np.std(grid_accuracy))
    out_dict['{}_nat_accuracy_summary'.format(split)] = (np.mean(nat_accuracy),
        np.std(nat_accuracy))
    out_dict['{}_asr_summary'.format(split)] = (np.mean(asr),
        np.std(asr))

    with open(save_filename, 'w') as result_file:
        json.dump(out_dict, result_file, sort_keys=True, indent=4)
