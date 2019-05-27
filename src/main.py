#!/usr/bin/env python
# -*- coding: utf-8 -*-

################################################################################
# Copyright (c) 2017. Vincenzo Lomonaco, Karan Desai, Eugenio Culurciello,     #
# Davide Maltoni. All rights reserved.                                         #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 27-05-2019                                                             #
# Authors: Vincenzo Lomonaco, Karan Desai, Eugenio Culurciello, Davide Maltoni #
# E-mail: vincenzo.lomonaco@unibo.it                                           #
# Website: vincenzolomonaco.com                                                #
################################################################################

import os.path
import numpy as np
import random
from model_utils import get_model
from doom_env import init_doom_env
import pprint
import vizdoom
import logging

from collections import OrderedDict

# Sacred dependecies
from sacred import Experiment
from sacred.observers import MongoObserver

# Creating the experiment
ex = Experiment('CL4RL')

# Setting custom logger
logger = logging.getLogger('mylogger')
logger.handlers = []
ch = logging.StreamHandler()
formatter = logging.Formatter('[%(levelname).1s] > %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)
ex.logger = logger

# We add the observer (if you don't have a configured DB
# then simply comment the line below line).
ex.observers.append(MongoObserver.create(db_name='experiments_db'))

@ex.config
def cfg():
    """ Default configuration parameters. Overwritten by specific exps
        configurations. """

    # train or test
    mode = None

    # number of experiments runs
    num_runs = 1

    # list of learning rates for each map
    learning_rate = None

    # number of steps in an episode
    episode_size = None

    # number of game instances running in parallel
    batch_size = None

    # number of episodes for training
    episode_num = None

    # number of full-lenght episodes for testing
    num_test_ep = None

    # number of steps per epoch
    epoch_game_steps = None

    # save best model based on epoch test results
    save_best = None

    # discount factor
    episode_discount = None

    # learning rate multiplier at each epoch
    lr_step = None

    # lambda (memory strength co-efficient) for EWC, zero lambda means no EWC
    ewc_lambda = 0

    # clip value for the fisher matrix
    clip_value = 0.00000001

    # number of game samples to be drawn for fisher matrix calculation in EWC
    fisher_sample_size = 100

    # number of episodes between two computations of the fisher matrix
    fisher_ep_freq = None

    # reward threshold for computing fisher
    fisher_threshold = None

    # long moving average time window
    long_tw_size = 50

    # short moving average time window
    short_tw_size = 6

    # random generator seed
    seed = None

    # Model type among 'aac' or 'aac_map'
    model = None

    # path to base model file and action set (to remove?)
    base_model = None
    action_set = None

    # path to model file (string in train, list in test)
    load = None

    # vizdoom config path
    vizdoom_config = None

    # list of maps to use in the wad file
    vizdoom_maps = None

    # vizdoom window visibility for the test
    vizdoom_test_visible = None

    # path to vizdoom
    vizdoom_path = os.path.dirname(vizdoom.__file__)

    # vizdoom basic utilities wad
    wad_path = 'cfgs/freedoom2.wad'

    # skiprate
    skiprate = None

    # number of frames per input
    frame_num = None

    # checkpoint file name
    checkpoint_file = None

    # after how many epoch to save the model
    checkpoint_rate = None

    # command to launch a bot (to remove?)
    bot_cmd = None

    # pkl file containing results to plot
    results_file = None

    # log level
    log_level = 'DEBUG'

    # backend: 'GPU' or 'CPU'
    backend = 'GPU'

# ex.add_config('cfgs/light/naive/train.json')


@ex.automain
def main(_config):
    """ Main script which for running a non-stationary doom environments
        composed of multiple maps. """

    # Setting logger
    log = ex.logger
    log.setLevel(_config['log_level'].upper())

    # Printing the conf for visual check
    log.debug(
        'Exp. configurations:\n'
        '----------------------------------------------------------------\n'
        + pprint.pformat(_config) + '\n' +
        '----------------------------------------------------------------\n'
    )

    init_doom_env(_config)
    random.seed(_config['seed'])
    np.random.seed(_config['seed'])
    model = get_model(_config)

    if _config['mode'] == 'train':
        test_crew = [[] for j in range(_config['num_runs'])]
        test_std = [[] for j in range(_config['num_runs'])]

        runs_avg_crew = [[[] for i in range(len(_config['vizdoom_maps']))]
                         for j in range(len(_config['vizdoom_maps']))]

        # train with multiple runs
        for run_id in range(_config['num_runs']):
            ex.info[str(run_id)] = {'test_crew': [], 'test_std': []}
            print("\n------------------ RUN {0} -------------------"
                  .format(run_id))
            for task_id in range(len(_config['vizdoom_maps'])):
                crew, std, movavg_stats = model.run_train(task_id=task_id)
                test_crew[run_id].append(crew[-1])
                test_std[run_id].append(std[-1])
                # save results into mongodb
                ex.info[str(run_id)]['test_crew'].append(crew[-1])
                ex.info[str(run_id)]['test_std'].append(std[-1])
                ex.info[str(run_id)]['mov_avg_stats'] = movavg_stats
                ex.info[str(run_id)]['tot_crew'] = crew
                ex.info[str(run_id)]['tot_std'] = std

            # prepare for next run
            del model
            model = get_model(_config)

        # printing results
        print(
            '----------------------------------------------------------------\n'
            '                          RESULTS\n' +
            '----------------------------------------------------------------\n'
        )
        for run_id in range(_config['num_runs']):
            print("\n------------------ RUN {0} -------------------"
                  .format(run_id))
            for task_id in range(len(_config['vizdoom_maps'])):
                avg_crew = 0
                avg_std = 0
                for map_id in range(len(_config['vizdoom_maps'])):
                    # print(test_crew[run_id][task_id])
                    # print(test_std[run_id][task_id])
                    print(
                        "[run %d], [task: %d], [map: %d], "
                        "[avg. cumulated reward: %f.2f], "
                        "[dev.std: %.2f]" % (run_id, task_id, map_id,
                                             test_crew[run_id][task_id][map_id],
                                             test_std[run_id][task_id][map_id])
                    )
                    avg_crew += test_crew[run_id][task_id][map_id]
                    avg_std += test_std[run_id][task_id][map_id]
                    runs_avg_crew[task_id][map_id].append(
                        test_crew[run_id][task_id][map_id]
                    )
                avg_crew /= len(_config['vizdoom_maps'])
                avg_std /= len(_config['vizdoom_maps'])
                print("[run %d] [avg_maps_crew %.3f], [avg_maps_std %.3f]\n" %
                      (run_id, avg_crew, avg_std))

        print(
            '----------------------------------------------------------------\n'
            '                       AVG RUNS RESULTS\n' +
            '----------------------------------------------------------------\n'
        )

        for task_id in range(len(_config['vizdoom_maps'])):
            tot_avg_crew = []
            for map_id in range(len(_config['vizdoom_maps'])):
                print("[task %d], [map: %d], [runs avg_crew %.3f],"
                      " [runs std %.3f]" %
                      (task_id, map_id,
                       float(np.mean(runs_avg_crew[task_id][map_id])),
                       float(np.std(runs_avg_crew[task_id][map_id]))))

                tot_avg_crew.append(np.mean(runs_avg_crew[task_id][map_id]))

            print("[avg_runs_maps_crew %.3f], [std %.3f]\n" %
                  (float(np.mean(tot_avg_crew)), float(np.std(tot_avg_crew))))

    else:
        # we assume to have a list in load for test
        res = {}
        for model_path in _config['load']:
            res[model_path] = []
            for task_id in range(len(_config['vizdoom_maps'])):
                crew, std = model.run_test(
                    num_test_ep=_config['num_test_ep'], load=model_path,
                    seed=_config['seed'], task_id=task_id)
                res[model_path].append((crew, std))

        res = OrderedDict(sorted(res.items(), key=lambda t: t[0]))
        for model_name, vec in res.items():
            avg_crew = 0
            avg_std = 0
            for i, (crew, std) in enumerate(vec):
                print(
                    "[name %s], [map: %d], [avg. cumulated reward: %f.2f],"
                    "[dev.std: %.2f]" % (model_name, i, crew, std)
                )
                avg_crew += crew
                avg_std += std
            avg_crew /= len(res[model_name])
            avg_std /= len(res[model_name])
            print("[avg_crew %.3f], [avg_std %.3f]\n" % (avg_crew, avg_std))
