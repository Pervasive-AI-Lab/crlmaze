#!/usr/bin/env python
# -*- coding: utf-8 -*-

################################################################################
# Copyright (c) 2019. Vincenzo Lomonaco, Karan Desai, Eugenio Culurciello,     #
# Davide Maltoni. All rights reserved.                                         #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 27-05-2019                                                             #
# Authors: Vincenzo Lomonaco, Karan Desai, Eugenio Culurciello, Davide Maltoni #
# E-mail: vincenzo.lomonaco@unibo.it                                           #
# Website: vincenzolomonaco.com                                                #
################################################################################

import os
import pickle as pkl
import time
from multiprocessing.pool import ThreadPool

import numpy as np
import torch
from torch import nn, optim

import cuda
from doom_instance import NormalizedState, prepare_doom_batch
from ewc_regularizer import EWCRegularizer

from builtins import *


class AACBase(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.optimizer = None
        self.ewc_reg = EWCRegularizer(self.cfg, self.cfg['ewc_lambda'])

        # stats fields
        self.cum_episode_id = 0
        self.long_rw_tw = []
        self.short_rw_tw = []
        self.long_tw_size = self.cfg['long_tw_size']
        self.short_tw_size = self.cfg['short_tw_size']
        self.fisher_id = 0
        self.mov_avg_stats = []
        self.test_crew = []
        self.test_crew_std = []

        if self.cfg['backend'] == 'CPU':
            cuda.USE_CUDA = False

    def on_task_update(self, task_id=0, consolidate=True):
        """This method should be called when tasks are switched, currently it
        is specifically written for EWC Regularization strategy.

        A list of game episodes are sampled according to task, and updates are
        taken to accumulate parameter gradients. These gradients will be used
        in fisher matrix calculation.

        Arguments
        =========
        task_id : int or str
            ID of recently completed task.
        consolidate: boolean
            if the F matrix should be used for consolidation.
        """
        self.train()
        fs_size = self.cfg['fisher_sample_size']
        games = prepare_doom_batch(fs_size, self.cfg, task_id)

        self.zero_grad()
        state = NormalizedState(
            screen=torch.Tensor(fs_size, *self.cfg['screen_size']),
            variables=torch.Tensor(fs_size, self.cfg['variable_num']),
            depth=None, labels=None
        )
        reward = torch.Tensor(fs_size, 1)
        terminal = torch.Tensor(fs_size, 1)
        episode_return = torch.zeros(fs_size)

        pool = ThreadPool()

        def get_state(game):
            """Get state of the game."""
            id = game.get_id()
            nstate = game.get_state_normalized()
            state.screen[id, :] = torch.from_numpy(nstate.screen)
            state.variables[id, :] = torch.from_numpy(nstate.variables)

        # fill up the initial state of games
        pool.map(get_state, games)

        for step in range(self.cfg['episode_size']):
            action = self.get_action(state)

            def step_game(game):
                """Step game (perform action) and get next state."""
                id = game.get_id()
                nstate, step_reward, finished = game.step_normalized(action[id][0])
                state.screen[id, :] = torch.from_numpy(nstate.screen)
                state.variables[id, :] = torch.from_numpy(nstate.variables)
                reward[id, 0] = step_reward
                if finished:
                    episode_return[id] = float(game.get_episode_return())
                    # cut rewards from future actions
                    terminal[id] = 0
                else:
                    terminal[id] = 1

            pool.map(step_game, games)
            self.set_reward(reward)
            self.set_terminal(terminal)

        # this will accumulate gradient of parameters
        self.backward()
        # terminate games
        pool.map(lambda game: game.release(), games)
        self.ewc_reg.update_fisher_optpar(
            self.fisher_id, self.named_parameters(),
            consolidate=consolidate
        )

    def run_train(self, task_id=0):
        print("Training for task {0}: {1}..."
              .format(task_id, self.cfg["vizdoom_maps"][task_id]))
        self.train()

        best_crew = -30000

        # set initial learning rate for the task
        self.optimizer = optim.Adam(self.parameters(),
                                    lr=self.cfg['learning_rate'][task_id])

        if self.cfg['load'] or task_id != 0:
            if self.cfg['load']:
                state = self.cfg['load']
            else:
                state = self.cfg['checkpoint_file'][:-4] + str(task_id - 1) + '.pth'

            if cuda.USE_CUDA:
                state_dict = torch.load(state)
            else:
                state_dict = torch.load(
                    state, map_location=lambda storage, loc: storage
                )
            self.optimizer.load_state_dict(state_dict.pop('optimizer'))
            self.ewc_reg.load_state_dict(state_dict.pop('ewc_reg'))
            self.load_state_dict(state_dict)

        self.optimizer.zero_grad()

        state = NormalizedState(screen=None, depth=None, labels=None, variables=None)
        state.screen = torch.Tensor(self.cfg['batch_size'], *self.cfg['screen_size'])
        state.variables = torch.Tensor(self.cfg['batch_size'], self.cfg['variable_num'])
        reward = torch.Tensor(self.cfg['batch_size'], 1)
        terminal = torch.Tensor(self.cfg['batch_size'], 1)
        episode_return = torch.zeros(self.cfg['batch_size'])

        games = prepare_doom_batch(self.cfg['batch_size'], self.cfg, task_id)
        pool = ThreadPool()

        def get_state(game):
            id = game.get_id()
            normalized_state = game.get_state_normalized()
            state.screen[id, :] = torch.from_numpy(normalized_state.screen)
            state.variables[id, :] = torch.from_numpy(normalized_state.variables)

        pool.map(get_state, games)
        # start training
        start_time = time.time()
        cur_lr = self.cfg['learning_rate'][task_id]
        for episode in range(1, self.cfg['episode_num'] + 1):
            self.cum_episode_id += 1

            batch_time = time.time()
            for step in range(self.cfg['episode_size']):
                # get action
                action = self.get_action(state)

                # step and get new state
                def step_game(game):
                    id = game.get_id()
                    normalized_state, step_reward, finished = \
                        game.step_normalized(action[id][0])
                    state.screen[id, :] = \
                        torch.from_numpy(normalized_state.screen)
                    state.variables[id, :] = \
                        torch.from_numpy(normalized_state.variables)
                    reward[id, 0] = step_reward
                    if finished:
                        episode_return[id] = float(game.get_episode_return())
                        # cut rewards from future actions
                        terminal[id] = 0
                    else:
                        terminal[id] = 1
                pool.map(step_game, games)
                self.set_reward(reward)
                self.set_terminal(terminal)

            # update model
            avg_rw, loss, ewc = self.backward()
            grads = []
            weights = []
            for p in self.parameters():
                if p.grad is not None:
                    grads.append(p.grad.data.view(-1))
                    weights.append(p.data.view(-1))
            grads = torch.cat(grads, 0)
            weights = torch.cat(weights, 0)
            grads_norm = grads.norm()
            weights_norm = weights.norm()

            self.optimizer.step()
            self.optimizer.zero_grad()

            if episode % 1 == 0:

                # managing the moving averages
                avg_rw_tot_ep = episode_return.mean()
                if len(self.long_rw_tw) >= self.long_tw_size:
                    del self.long_rw_tw[0]
                self.long_rw_tw.append(avg_rw)

                if len(self.short_rw_tw) >= self.short_tw_size:
                    del self.short_rw_tw[0]
                self.short_rw_tw.append(avg_rw)

                long_time_avg_rew = np.mean(self.long_rw_tw)
                short_time_avg_rew = np.mean(self.short_rw_tw)
                rw_diff = short_time_avg_rew - long_time_avg_rew
                # logging
                self.mov_avg_stats.append(
                    (long_time_avg_rew, short_time_avg_rew, rw_diff)
                )
                print("{} [avg_rw {:+.1f}]"
                      "[avg_rw_tot_ep {:.1f}]"
                      "[mavg {:.1f}]"
                      "[short_mavg {:.1f}]"
                      "[rw_diff {:+.1f}]"
                      "[tot_loss {:+.2f}]"
                      "[ewc_loss {:+.2f}]"
                      "[lr {:.1e}]"
                      "[t(m) {:.1f}]"
                      .format(episode, avg_rw, avg_rw_tot_ep, long_time_avg_rew,
                              short_time_avg_rew, rw_diff, loss, ewc,
                              cur_lr, (time.time() - start_time) / 60.0))

            # Compute Fisher matrix if needed
            if self.cfg['fisher_ep_freq'] is not None and \
                episode % self.cfg['fisher_ep_freq'] == 0:
                    print(
                        "computing fisher matrix " + str(self.fisher_id) + "..."
                    )

                    # here we decide if we consolidate or not depending on the
                    # strategy.
                    if self.cfg['fisher_threshold'] is None:
                        consolid = True
                    else:
                        consolid = False

                    self.on_task_update(task_id=task_id, consolidate=consolid)
                    print("Done.")
                    self.fisher_id += 1

            # Consolidate if needed
            if self.cfg['fisher_threshold'] and self.mov_avg_stats and \
                    self.mov_avg_stats[-1][-1] < self.cfg['fisher_threshold']:
                # if we detect the threshold we consolidate
                # previously acquired fisher
                self.ewc_reg.add_task_to_consolidate(self.fisher_id-1)

            # Drop learning rate if step reached
            if episode != 0 and episode % self.cfg['lr_step'] == 0:
                for param_group in self.optimizer.param_groups:
                    cur_lr = cur_lr - self.cfg['learning_rate'][task_id] / 100
                    param_group['lr'] = cur_lr

            # if the epoch is finished
            if episode != 0 and episode % self.cfg['epoch_game_steps'] == 0:

                torch.save(self.state_dict(), self.cfg['checkpoint_file'])

                # now we test on the three maps
                crew_list = []
                std_list = []
                for i in range(len(self.cfg['vizdoom_maps'])):
                    print("testing on task {0}: {1}..."
                          .format(i, self.cfg['vizdoom_maps'][i]))
                    crew, std = self.run_test(
                        self.cfg['num_test_ep'], seed=episode, task_id=i
                    )
                    crew_list.append(crew)
                    std_list.append(std)

                self.test_crew.append(crew_list)
                self.test_crew_std.append(std_list)
                self.reset()
                self.train()

                if self.cfg['save_best'] and crew > best_crew:
                    torch.save(self.state_dict(), 'artifacts/best_model.pth')
                    best_crew = crew
                    print('Saving best model so far...')

                # overrides it
                results = ((self.test_crew, self.test_crew_std),
                           self.mov_avg_stats)
                with open(self.cfg['results_file'] + '_' +
                          str(task_id) + '.pkl', 'wb') as f:
                    pkl.dump(results, f)

        # terminate games
        pool.map(lambda game: game.release(), games)

        # save model, optimizer and regularizer in single .pth file
        state_dict = self.state_dict()
        state_dict['optimizer'] = self.optimizer.state_dict()
        state_dict['ewc_reg'] = self.ewc_reg.state_dict()
        name = self.cfg['checkpoint_file'][:-4] + str(task_id) + '.pth'
        print("Saving model in: ", name)
        torch.save(state_dict, name)

        return self.test_crew, self.test_crew_std, self.mov_avg_stats

    def run_test(self, num_test_ep=10, seed=None, load=None, task_id=0):

        if load:
            print("Loading pre-trained model from {}...".format(load))
            if cuda.USE_CUDA:
                state_dict = torch.load(load)
            else:
                state_dict = torch.load(
                    load, map_location=lambda storage, loc: storage
                )
            state_dict.pop('optimizer')
            self.ewc_reg.load_state_dict(state_dict.pop('ewc_reg'))
            self.load_state_dict(state_dict)

        self.eval()

        state = NormalizedState(screen=None, depth=None, labels=None, variables=None)
        state.screen = torch.Tensor(self.cfg['batch_size'], *self.cfg['screen_size'])
        state.variables = torch.Tensor(self.cfg['batch_size'], self.cfg['variable_num'])

        games = prepare_doom_batch(self.cfg['batch_size'], self.cfg, task_id)
        pool = ThreadPool()

        def get_state(game):
            id = game.get_id()
            normalized_state = game.get_state_normalized()
            state.screen[id, :] = torch.from_numpy(normalized_state.screen)
            state.variables[id, :] = torch.from_numpy(normalized_state.variables)

        pool.map(get_state, games)

        cum_rewards = []
        for episode in range(1, num_test_ep // self.cfg['batch_size'] + 1):

            for step in range(1000 // self.cfg['skiprate']):
                # get action
                action = self.get_action(state)

                # step and get new state
                def step_game(game):
                    id = game.get_id()
                    normalized_state, step_reward, finished = \
                        game.step_normalized(action[id][0])
                    state.screen[id, :] = \
                        torch.from_numpy(normalized_state.screen)
                    state.variables[id, :] = \
                        torch.from_numpy(normalized_state.variables)
                    if finished:
                        cum_rewards.append(game.get_episode_return())

                pool.map(step_game, games)

            if cum_rewards:
                avg_cum_reward = np.average(cum_rewards)
                std_dev = np.std(cum_rewards)
            print("Finished: {}, Avg cumulative reward: {}, std: {}"
                  .format(len(cum_rewards), avg_cum_reward, std_dev))

        return avg_cum_reward, std_dev
