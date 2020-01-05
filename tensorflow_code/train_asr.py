"""Trains a model, saving checkpoints and tensorboard summaries along
   the way."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from datetime import datetime
import json
import math
import os
import shutil
import click
import sys
import copy
import time
import getpass
from timeit import default_timer as timer
import ipdb

import numpy as np
import tensorflow as tf

import cifar10_input
import cifar100_input
import svhn_input
# import imagenet_input
from eval import evaluate
import experiment_repo as exprepo
import resnet
import resnet50
import vgg
#for defense
from spatial_attack_defense import SpatialAttack
#for attack (evaluation)
from spatial_attack_evaluation import SpatialAttack as SpatialAttackEval
import utilities

import sys


def train(config='configs/cifar10_nat_res.json',
          save_root_path='./noAdvResnet',
          experiment_json_fname='experiments.json',
          local_json_dir_name='local_json_files',
          worstofk=None,
          attack_style=None,
          attack_limits=None,
          fo_epsilon=None,
          fo_step_size=None,
          fo_num_steps=None,
          lambda_reg=None,
          num_ids = None,
          group_size=None,
          use_reg=None,
          seed=None,
          save_in_local_json=True,
          this_repo=None):
    #check tensorflow version
    print(tf.__version__)


    # reset default graph (needed for running locally with run_jobs_ray.py)
    tf.reset_default_graph()

    # get configs
    config_dict = utilities.get_config(config)
    config_dict_copy = copy.deepcopy(config_dict)
    config = utilities.config_to_namedtuple(config_dict)

    # seeding randomness
    if seed == None:
        seed = config.training.tf_random_seed
    else:
        config_dict_copy['training']['tf_random_seed'] = seed
    tf.set_random_seed(seed)
    np.random.seed(seed)

    # Setting up training parameters
    max_num_training_steps = config.training.max_num_training_steps
    step_size_schedule = config.training.step_size_schedule
    weight_decay = config.training.weight_decay
    momentum = config.training.momentum

    if group_size == None:
        group_size = config.training.group_size
    else:
        config_dict_copy['training']['group_size'] = int(group_size)
    if num_ids == None:
        num_ids = config.training.num_ids
    else:
        config_dict_copy['training']['num_ids'] = int(num_ids)
    if lambda_reg == None:
        lambda_reg = config.training.lambda_
    else:
        config_dict_copy['training']['lambda_'] = float(lambda_reg)
    if use_reg == None:
        use_reg = config.model.use_reg
    else:
        config_dict_copy['model']['use_reg'] = use_reg

    batch_size = config.training.batch_size
    # number of groups with group size > 1
    # num_grouped_ids = batch_size - num_ids
    # # number of unique ids needs to be larger than half the desired batch size
    # # so that full batch can be filled up
    # assert num_ids >= batch_size/2
    # # currently, code is designed for groups of size 2
    # assert config.training.group_size == 2

    if use_reg & (config.defense.cce_adv_exp_wrt != config.defense.reg_adv_exp_wrt):
        print("three folds training batch!")
        num_grouped_ids = int(batch_size/3)
    else:
        num_grouped_ids = batch_size - num_ids

    adversarial_training = config.training.adversarial_training
    eval_during_training = config.training.eval_during_training
    if eval_during_training:
        num_eval_steps = config.training.num_eval_steps

    # Setting up output parameters
    num_output_steps = config.training.num_output_steps
    num_summary_steps = config.training.num_summary_steps
    num_checkpoint_steps = config.training.num_checkpoint_steps
    num_easyeval_steps = config.training.num_easyeval_steps

    # Setting up the data and the model
    data_path = config.data.data_path

    if config.data.dataset_name == "cifar-10":
        raw_iterator = cifar10_input.CIFAR10Data(data_path)
    elif config.data.dataset_name == "cifar-100":
        raw_iterator = cifar100_input.CIFAR100Data(data_path)
    elif config.data.dataset_name == "svhn":
        raw_iterator = svhn_input.SVHNData(data_path)
    elif config.data.dataset_name == "imagenet":
        print("using imagenet")
    else:
        raise ValueError("Unknown dataset name.")

    global_step = tf.train.get_or_create_global_step()

    model_family = config.model.model_family
    if model_family == "resnet":
        if config.defense.use_spatial and config.defense.spatial_method == 'fo':
            diffable = True
        else:
            diffable = False
        model = resnet.Model(config.model, num_ids, diffable,
            config.training.adversarial_ce, config.training.nat_ce, config.defense.reg_type,
                             config.defense.cce_adv_exp_wrt, config.defense.reg_adv_exp_wrt)
    elif model_family == "resnet50":
        if config.defense.use_spatial and config.defense.spatial_method == 'fo':
            diffable = True
        else:
            diffable = False
        model = resnet50.Model(config.model, num_ids, diffable,
            config.training.adversarial_ce, config.training.nat_ce, config.defense.reg_type,
                             config.defense.cce_adv_exp_wrt, config.defense.reg_adv_exp_wrt)
    elif model_family == "vgg":
        if config.defense.use_spatial and config.defense.spatial_method == 'fo':
            diffable = True
        else:
            diffable = False
        if config.training.adversarial_ce:
            raise NotImplementedError
        model = vgg.Model(config.model, num_ids, diffable, config.training.adversarial_ce,
                          config.training.nat_ce, config.defense.reg_type,
                          config.defense.cce_adv_exp_wrt, config.defense.reg_adv_exp_wrt)

    # uncomment to get a list of trainable variables
    # model_vars = tf.trainable_variables()

    # Setting up the optimizer
    boundaries = [int(sss[0]) for sss in step_size_schedule]
    boundaries = boundaries[1:]
    values = [sss[1] for sss in step_size_schedule]
    learning_rate = tf.train.piecewise_constant(
        tf.cast(global_step, tf.int32),
        boundaries,
        values)

    if use_reg and lambda_reg > 0:
        total_loss = (model.mean_xent + weight_decay * model.weight_decay_loss +
                      lambda_reg * model.reg_loss)
    else:
        total_loss = model.mean_xent + weight_decay * model.weight_decay_loss

    optimizer = tf.train.MomentumOptimizer(learning_rate, momentum)
    train_step = optimizer.minimize(total_loss, global_step=global_step)

    # Set up defense
    if worstofk == None:
        worstofk = config.defense.random_tries
    else:
        config_dict_copy['attack']['random_tries'] = worstofk
    if fo_epsilon == None:
        fo_epsilon = config.defense.epsilon
    else:
        config_dict_copy['attack']['epsilon'] = fo_epsilon
    if fo_step_size == None:
        fo_step_size = config.defense.step_size
    else:
        config_dict_copy['attack']['step_size'] = fo_step_size
    if fo_num_steps == None:
        fo_num_steps = config.defense.num_steps
    else:
        config_dict_copy['attack']['num_steps'] = fo_num_steps
    # @ Luzius: incorporate being able to choose multiple transformations
    if attack_style == None:
        attack_style = 'rotate'

    # Training attack (defense)
    # L-inf attack if use_spatial is False and use_linf is True
    # spatial attack if use_spatial is True and use_linf is False
    # spatial random attack if spatial_method is 'random'
    # spatial PGD attack if spatial_method is 'fo'
    attack = SpatialAttack(model, config.defense, config.defense.spatial_method,
                           worstofk, attack_limits, fo_epsilon,
                           fo_step_size, fo_num_steps)

    fo_epsilon_eval = config.attack.epsilon
    fo_step_size_eval = config.attack.step_size
    fo_num_steps_eval = config.attack.num_steps

    # Different eval attacks
    # Random attack
    # L-inf attack if use_spatial is False and use_linf is True
    # random (worst-of-1) spatial attack if use_spatial is True
    # and use_linf is False
    attack_eval_random = SpatialAttackEval(model, config.attack, 'random', 1,
                                           attack_limits, fo_epsilon_eval,
                                           fo_step_size_eval, fo_num_steps_eval)
    # First order attack
    # L-inf attack if use_spatial is False and use_linf is True
    # first-order spatial attack if use_spatial is True and use_linf is False
    attack_eval_fo = SpatialAttackEval(model, config.attack, 'fo', 1,
                                       attack_limits, fo_epsilon_eval,
                                       fo_step_size_eval, fo_num_steps_eval)

    # Grid attack
    # spatial attack if use_spatial is True and use_linf is False
    # not executed for L-inf attacks
    attack_eval_grid = SpatialAttackEval(model, config.attack, 'grid', None,
                                         attack_limits)

    # TODO(christina): add L-inf attack with random restarts

    # ------------------START EXPERIMENT -------------------------
    # Initialize the Repo
    print("==> Creating repo..")
    # Create repo object if it wasn't passed, comment out if repo has issues
    if this_repo == None:
        this_repo = exprepo.ExperimentRepo(
            save_in_local_json=save_in_local_json,
            json_filename=experiment_json_fname,
            local_dir_name=local_json_dir_name,
            root_dir=save_root_path)

    # Create new experiment
    if this_repo != None:
        exp_id = this_repo.create_new_experiment(config.data.dataset_name,
                                                 model_family,
                                                 worstofk,
                                                 attack_style,
                                                 attack_limits,
                                                 lambda_reg,
                                                 num_grouped_ids,
                                                 group_size,
                                                 config_dict_copy)

    # Setting up the Tensorboard and checkpoint outputs
    model_dir = '%s/logdir/%s' % (save_root_path, exp_id)

    # We add accuracy and xent twice so we can easily make three types of
    # comparisons in Tensorboard:
    # - train vs eval (for a single run)
    # - train of different runs
    # - eval of different runs

    saver = tf.train.Saver(max_to_keep=25)

    tf.summary.scalar('accuracy_nat_train', model.accuracy, collections=['nat'])
    # tf.summary.scalar('accuracy_nat', model.accuracy, collections=['nat'])
    tf.summary.scalar('xent_nat_train', model.xent / (num_ids * 2), collections=['nat'])
    # tf.summary.scalar('xent_nat', model.xent / (num_ids * 2), collections=['nat'])
    # tf.summary.image('images_post_preprocessing', model.x_image_pre, collections=['image'])
    # tf.summary.image('images_post_padding', model.x_image_after_pad, collections=['image'])
    tf.summary.image('images_nat_train', model.x_image, collections=['nat'])
    tf.summary.scalar('learning_rate', learning_rate, collections=['nat'])
    nat_summaries = tf.summary.merge_all('nat')
    # image_summaries = tf.summary.merge_all('image')
    # data augmentation used if config.training.data_augmentation_reg is True
    x_input_placeholder = tf.placeholder(tf.float32, shape=[None, None, None, 3])
    flipped = tf.map_fn(lambda img: tf.image.random_flip_left_right(img),
                        x_input_placeholder)

    with tf.Session() as sess:
        # initialize standard data augmentation
        if config.training.data_augmentation:
            if config.data.dataset_name == "cifar-10":
                data_iterator = cifar10_input.AugmentedCIFAR10Data(raw_iterator, sess)
            elif config.data.dataset_name == "cifar-100":
                data_iterator = cifar100_input.AugmentedCIFAR100Data(raw_iterator, sess)
            elif config.data.dataset_name == "svhn":
                data_iterator = svhn_input.AugmentedSVHNData(raw_iterator, sess)
            # elif config.data.dataset_name == "imagenet":
            #     data_iterator = imagenet_input.IMAGENETData(data_path, num_ids, config.model.pad_size, sess)
            else:
                raise ValueError("Unknown dataset name.")
        else:
            data_iterator = raw_iterator

        # eval_dict = {model.x_input: data_iterator.eval_data.xs,
        #              model.y_input: data_iterator.eval_data.ys,
        #              model.group:  np.arange(0, batch_size, 1, dtype="int32"),
        #              model.transform: np.zeros([data_iterator.eval_data.n, 3]),
        #              model.is_training: False}


        # Initialize the summary writer, global variables, and our time counter.
        summary_writer = tf.summary.FileWriter(model_dir, sess.graph)
        # if eval_during_training:
        eval_dir = os.path.join(model_dir, 'eval')
        os.makedirs(eval_dir, exist_ok=True)
        # eval_summary_writer = tf.summary.FileWriter(eval_dir)
        eval_summary_writer = tf.summary.FileWriter(eval_dir, sess.graph)

        sess.run(tf.global_variables_initializer())
        training_time = 0.0
        run_time_without_eval = 0.0
        run_time_adv_ex_creation = 0.0
        run_time_train_step = 0.0
        ####################################
        # Main training loop
        ####################################
        start_time = time.time()
        no_epochs_done = 0  # the same as epoch_count, need to merge
        start_epoch = timer()
        it_count = 0
        epoch_count = 0
        acc_sum = 0

        for ii in range(max_num_training_steps+1):
            # print("Iteration : {}".format(ii))
            # original batch
            x_batch, y_batch, epoch_done = data_iterator.train_data.get_next_batch(
                                num_ids, multiple_passes=True)

            no_epochs_done += epoch_done
            # noop trans
            noop_trans = np.zeros([len(x_batch), 3])
            # id_batch starts with IDs of original examples
            id_batch = np.arange(0, num_ids, 1, dtype="int32")

            if use_reg:
                start = timer()
                # IDs for grouped examples
                id_batch_adv = np.arange(0, num_grouped_ids, 1, dtype="int32")

                for _ in range(config.training.group_size-1):
                    if config.training.data_augmentation_reg:
                        x_batch_reg = sess.run(flipped,
                                               feed_dict={x_input_placeholder:
                                               x_batch[0:num_grouped_ids, :, :, :]})
                    else:
                        x_batch_reg = x_batch[0:num_grouped_ids, :, :, :]

                    # create rotated examples w.r.t regularizer
                    x_batch_adv, trans_adv = attack.perturb(
                        x_batch_reg, y_batch[0:num_grouped_ids], config.defense.reg_adv_exp_wrt, sess)
                    y_batch_adv = y_batch[0:num_grouped_ids]

                    # construct new batches including rotated examples
                    x_batch_inp = np.concatenate(
                        (x_batch, x_batch_adv), axis=0)

                    y_batch_inp = np.concatenate((y_batch, y_batch_adv), axis=0)

                    trans_inp = np.concatenate((noop_trans, trans_adv), axis=0)
                    id_batch_inp = np.concatenate(
                        (id_batch, id_batch_adv), axis=0)

                    if config.defense.cce_adv_exp_wrt != config.defense.reg_adv_exp_wrt:  # 3 fold batch
                        # create adv rotated examples w.r.t config.defense.cce_adv_exp_wrt
                        x_batch_adv_cce, trans_adv_cce = attack.perturb(x_batch, y_batch, config.defense.cce_adv_exp_wrt, sess)
                        # construct new batches including rotated examples
                        x_batch_inp = np.concatenate(
                            (x_batch_inp, x_batch_adv_cce), axis=0)
                        y_batch_inp = np.concatenate((y_batch_inp, y_batch), axis=0)
                        trans_inp = np.concatenate((trans_inp, trans_adv_cce), axis=0)
                        id_batch_inp = np.concatenate((id_batch_inp, id_batch), axis=0)

                end = timer()
                training_time += end - start

                run_time_without_eval += end - start
                run_time_adv_ex_creation += end - start
            else:
                if adversarial_training:
                    start = timer()
                    x_batch_inp, trans_inp = attack.perturb(x_batch, y_batch, "cce",
                                                            sess)
                    end = timer()
                    training_time += end - start
                    run_time_without_eval += end - start
                    run_time_adv_ex_creation += end - start
                else:
                    x_batch_inp, trans_inp = x_batch, noop_trans
                # for adversarial training and plain training, the following
                # variables coincide
                y_batch_inp = y_batch
                y_batch_adv = y_batch
                trans_adv = trans_inp
                x_batch_adv = x_batch_inp
                id_batch_inp = id_batch
                id_batch_adv = id_batch

            # feed_dict for training step
            inp_dict = {model.x_input: x_batch_inp,
                        model.y_input: y_batch_inp,
                        model.group: id_batch_inp,
                        model.transform: trans_inp,
                        model.is_training: False}

            # separate natural and adversarially transformed examples for eval
            nat_dict = {model.x_input: x_batch,
                        model.y_input: y_batch,
                        model.group: id_batch,
                        model.transform: noop_trans,
                        model.is_training: False}

            adv_dict = {model.x_input: x_batch_adv,
                        model.y_input: y_batch_adv,
                        model.group: id_batch_adv,
                        model.transform: trans_adv,
                        model.is_training: False}

            ########### Outputting/saving weights and evaluations ###########
            acc_grid_te = -1.0
            avg_xent_grid_te = -1.0
            acc_fo_te = -1.0
            avg_xent_fo_te = -1.0
            saved_weights = 0


            # Compute training accuracy on this minibatch
            nat_acc_tr = 100 * sess.run(model.accuracy, feed_dict=nat_dict)
            # print("nat_acc on current training batch: {}".format(nat_acc_tr))

            # prediction = sess.run(model.predictions, feed_dict=nat_dict)
            # true_label = sess.run(model.y_input, feed_dict=nat_dict)
            # pre_softmax = sess.run(model.pre_softmax, feed_dict=nat_dict)
            # y_xent_for_loss = sess.run(model.y_xent_for_loss, feed_dict=nat_dict)
            # print("prediction:  {}".format(prediction[:5]))
            # print("true_label:  {}".format(true_label[:5]))
            # print("pre_softmax:  {}".format(pre_softmax[:5]))
            # print("y_xent_for_loss:   {}".format(y_xent_for_loss[:5]))
            #
            # sys.stdout.flush()

            # Output to stdout
            if epoch_done:
                epoch_time = timer() - start_epoch
                # Average
                av_acc = acc_sum/it_count

                # ToDo: Log this to file as well

                # Training accuracy over epoch
                print('Epoch {}:    ({})'.format(epoch_count, datetime.now()))
                print('    training natural accuracy {:.4}%'.format(av_acc))
                print('    {:.4} seconds per epoch'.format(epoch_time))

                # Accuracy on entire test set
                # nat_acc_te = 100 * sess.run(model.accuracy, feed_dict=eval_dict)

                # print('    test set natural accuracy {:.4}%'.format(nat_acc_te))

                # Set loss sum, it count back to zero
                acc_sum = nat_acc_tr
                epoch_done = 0
                epoch_count += 1
                start_epoch = timer()
                it_count = 1

            else:
                it_count += 1
                acc_sum += nat_acc_tr


            # Output to stdout
            if ii % num_output_steps == 0:
                nat_acc_tr = 100 * sess.run(model.accuracy, feed_dict=nat_dict)
                adv_acc_tr = 100 * sess.run(model.accuracy, feed_dict=adv_dict)
                inp_acc_tr = 100 * sess.run(model.accuracy, feed_dict=inp_dict)

                print('Step {}:    ({})'.format(ii, datetime.now()))
                print('    training nat accuracy {:.4}%'.format(nat_acc_tr))
                print('    training adv accuracy {:.4}%'.format(adv_acc_tr))
                print('    training inp accuracy {:.4}%'.format(inp_acc_tr))
                # print('    training inp trades loss {:.4}'.format(reg_loss))

            # Tensorboard summaries and heavy checkpoints
            if ii % num_summary_steps == 0:
                summary = sess.run(nat_summaries, feed_dict=nat_dict)
                summary_writer.add_summary(summary, global_step.eval(sess))

                # summary_image = sess.run(image_summaries, feed_dict=nat_dict)
                # summary_writer.add_summary(summary_image, global_step.eval(sess))

            # Write a checkpoint and eval if it's time
            if ii % num_checkpoint_steps == 0 or ii == max_num_training_steps:
                # Save checkpoint data (weights)

                saver.save(sess,
                                   os.path.join(model_dir, 'checkpoint_iter_'+str(ii)),
                                   global_step=global_step)
                saved_weights = 1

            # Write evaluation meta data for checkpoint
            if ii % num_easyeval_steps == 0 or ii == max_num_training_steps:
                # Get training accuracies
                nat_acc_tr = 100 * sess.run(model.accuracy,
                                            feed_dict=nat_dict)
                # adv_acc_tr = 100 * sess.run(model.accuracy,
                #                             feed_dict=adv_dict)
                inp_acc_tr = 100 * sess.run(model.accuracy,
                                            feed_dict=inp_dict)

                # Evaluation on random and natural
                [acc_nat_te, acc_rand_adv_te, avg_xent_nat_te,
                avg_xent_adv_te] = evaluate(
                    model, attack_eval_random, sess, config, 'random',
                    data_path, None)

                print('Step {}:    ({})'.format(ii, datetime.now()))
                print('    test nat accuracy {:.4}%'.format(acc_nat_te))
                print('    test rand adv accuracy {:.4}%'.format(acc_rand_adv_te))
                # print('    test inp accuracy {:.4}%'.format(inp_acc_tr))

                # Evaluation on grid (only for spatial attacks)
                if ((eval_during_training and ii % num_eval_steps == 0
                    and ii > 0 and config.attack.use_spatial) or
                    (eval_during_training and ii == max_num_training_steps and
                    config.attack.use_spatial)):


                    if config.defense.use_spatial and config.defense.spatial_method == 'fo':
                        # Evaluation on first-order PDG attack (too expensive to
                        # evaluate more frequently on whole dataset)
                        [_, acc_fo_te, _, avg_xent_fo_te] = evaluate(
                            model, attack_eval_fo, sess, config, 'fo',
                            data_path, None)

                    # Evaluation on grid
                    print("Start Grid-Evaluation!")
                    sys.stdout.flush()
                    start_grid = timer()

                    [_, acc_grid_te, _, avg_xent_grid_te] = evaluate(model,
                        attack_eval_grid, sess, config, "grid", data_path,
                        eval_summary_writer)

                    end_grid = timer()
                    grid_eval_time = end_grid-start_grid
                    print("grid_evaluation time: {}".format(grid_eval_time))

                # if ((eval_during_training and ii % num_eval_steps == 0
                #     and ii > 0 and config.attack.use_linf) or
                #     (eval_during_training and ii == max_num_training_steps and
                #     config.attack.use_linf)):
                #     print("not implemented yet..")
                # #     TODO
                # #     linf attack

                chkpt_id = this_repo.create_training_checkpoint(exp_id,
                    training_step=ii, epoch=no_epochs_done,
                    train_acc_nat=nat_acc_tr, train_acc_adv=adv_acc_tr,
                    train_acc_inp=inp_acc_tr,
                    test_acc_nat=acc_nat_te,
                    test_acc_adv=acc_rand_adv_te,
                    test_acc_fo=acc_fo_te,
                    test_acc_grid=acc_grid_te,
                    test_loss_nat=avg_xent_nat_te,
                    test_loss_adv=avg_xent_adv_te,
                    test_loss_fo=avg_xent_fo_te,
                    test_loss_grid=avg_xent_grid_te)

                if saved_weights == 0:
                    # Save checkpoint data (weights)
                    saver.save(sess,
                                       os.path.join(model_dir,
                                                    '{}_checkpoint_{}'.format(chkpt_id)))

            # Actual training step
            start = timer()
            inp_dict[model.is_training] = True
            sess.run(train_step, feed_dict=inp_dict)
            end = timer()
            training_time += end - start
            run_time_without_eval += end - start
            run_time_train_step += end - start
            # print("run_time_train_step: {}".format(end-start))
            # sys.stdout.flush()

        runtime = time.time() - start_time
        this_repo.mark_experiment_as_completed(
            exp_id, train_acc_nat=nat_acc_tr, train_acc_adv=adv_acc_tr,
            train_acc_inp=inp_acc_tr,
            test_acc_nat=acc_nat_te,
            test_acc_adv=acc_rand_adv_te,
            test_acc_fo=acc_fo_te,
            test_acc_grid=acc_grid_te,
            runtime=runtime, runtime_wo_eval=run_time_without_eval,
            runtime_train_step=run_time_train_step,
            runtime_adv_ex_creation=run_time_adv_ex_creation)


    return 0


@click.command()
@click.option('--config', default='configs/std.json', type=str)
@click.option('--save-root-path',
              default='/cluster/work/math/fanyang-broglil/regRepo',
              help='path to project root dir')
@click.option('--experiment_json_fname',
              default='experiments.json',
              help='filename for json saving experimental results')
@click.option('--local_json_dir_name',
              default='local_json_files',
              help='foldername for local json files')
@click.option('--save_in_local_json', default=1, type=int)
@click.option('--worstofk', default=None, type=int)
@click.option('--attack-style', default=None, type=str,
              help='Size multipler for original CIFAR dataset')
# ToDo: should be an Array, currently unused
@click.option('--attack-limits', default=None)
@click.option('--lambda-reg', default=None, type=float)
@click.option('--fo_epsilon', default=None, type=float)
@click.option('--fo_step_size', default=None, type=float)
@click.option('--fo_num_steps', default=None, type=int)
@click.option('--num-ids', default=None, type=int)
@click.option('--group-size', default=None, type=int)
@click.option('--use_reg', default=None, type=bool)
@click.option('--seed', default=None, type=int)
def train_cli(config, save_root_path, experiment_json_fname, local_json_dir_name,
              worstofk, attack_style, attack_limits, fo_epsilon,
              fo_step_size, fo_num_steps,
              lambda_reg, num_ids, group_size, use_reg, seed,
              save_in_local_json):

    train(config, save_root_path, experiment_json_fname, local_json_dir_name,
          worstofk, attack_style, attack_limits, fo_epsilon,
          fo_step_size, fo_num_steps,
          lambda_reg, num_ids, group_size, use_reg, seed, save_in_local_json)

if __name__ == '__main__':
    train_cli()
