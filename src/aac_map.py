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

import random
from collections import namedtuple

import torch
import torch.nn.functional as F
from torch import nn

from aac_base import AACBase
import cuda

from builtins import *


class BaseModel(AACBase):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.screen_feature_num = 128
        self.conv1 = nn.Conv2d(in_channels=cfg['screen_size'][0] * cfg['frame_num'],
                               out_channels=32, kernel_size=(3, 3), stride=(2, 2))
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(2, 2))
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(1, 1))

        self.screen_features1 = nn.Linear(32 * 27 * 37, self.screen_feature_num)

        self.batch_norm = nn.BatchNorm1d(self.screen_feature_num)

        layer1_size = 128
        self.action1 = nn.Linear(self.screen_feature_num, layer1_size)
        self.action2 = nn.Linear(layer1_size + cfg['variable_num'], cfg['button_num'])
        self.batch_norm_action = nn.BatchNorm1d(layer1_size + cfg['variable_num'])

        self.value1 = nn.Linear(self.screen_feature_num, layer1_size)
        self.value2 = nn.Linear(layer1_size + cfg['variable_num'], 1)
        self.batch_norm_value = nn.BatchNorm1d(layer1_size + cfg['variable_num'])

        self.screens = None
        self.frame_num = cfg['frame_num']

    def forward(self, screen, variables):
        # cnn
        screen_features = F.max_pool2d(screen, kernel_size=(2, 2), stride=(2, 2))
        screen_features = F.selu(self.conv1(screen_features))
        screen_features = F.selu(self.conv2(screen_features))
        screen_features = F.selu(self.conv3(screen_features))
        screen_features = screen_features.view(screen_features.size(0), -1)

        # features
        input = self.screen_features1(screen_features)
        input = self.batch_norm(input)
        input = F.selu(input)

        # action
        action = F.selu(self.action1(input))
        action = torch.cat([action, variables], 1)
        action = self.batch_norm_action(action)
        action = self.action2(action)

        return action, input

    def transform_input(self, screen, variables):
        screen_batch = []
        if self.frame_num > 1:
            if self.screens is None:
                self.screens = [[]] * len(screen)
            for idx, screens in enumerate(self.screens):
                if len(screens) >= self.frame_num:
                    screens.pop(0)
                screens.append(screen[idx])
                if len(screens) == 1:
                    for i in range(self.frame_num - 1):
                        screens.append(screen[idx])
                screen_batch.append(torch.cat(screens, 0))
            screen = torch.stack(screen_batch)

        screen = cuda.Variable(screen, volatile=not self.training)
        variables = cuda.Variable(variables / 100, volatile=not self.training)
        return screen, variables

    def set_terminal(self, terminal):
        if self.screens is not None:
            indexes = torch.nonzero(terminal == 0).squeeze()
            for idx in range(len(indexes)):
                self.screens[indexes[idx]] = []


ModelOutput = namedtuple('ModelOutput', ['log_action', 'value'])


class AdvantageActorCriticMap(BaseModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        if cfg['base_model'] is not None:
            # load weights from the base model
            base_model = torch.load(cfg['base_model'])
            self.load_state_dict(base_model.state_dict())
            del base_model

        if cuda.USE_CUDA:
            super().cuda()

        self.discount = cfg['episode_discount']
        self.outputs = []
        self.rewards = []
        self.discounts = []

    def reset(self):
        self.outputs = []
        self.rewards = []
        self.discounts = []

    def forward(self, screen, variables):
        action_prob, input = super().forward(screen, variables)
        if not self.training:
            _, action = action_prob.max(1, keepdim=True)
            return action, None

        # greedy actions
        if random.random() < 0.1:
            action = torch.LongTensor(action_prob.size(0), 1).random_(0, action_prob.size(1))
            action = cuda.Variable(action)
            if cuda.USE_CUDA:
                action = action.cuda()
        else:
           _, action = action_prob.max(1, keepdim=True)

        # value prediction - critic
        value = F.selu(self.value1(input))
        value = torch.cat([value, variables], 1)
        value = self.batch_norm_value(value)
        value = self.value2(value)

        # save output for backpro
        action_prob = F.log_softmax(action_prob, dim=1)
        self.outputs.append(ModelOutput(action_prob.gather(-1, action), value))
        return action, value

    def get_action(self, state):
        action, _ = self.forward(*self.transform_input(state.screen, state.variables))
        return action.data

    def set_reward(self, reward):
        self.rewards.append(reward * 0.01)  # no clone() b/c of * 0.01

    def set_terminal(self, terminal):
        super().set_terminal(terminal)
        self.discounts.append(self.discount * terminal)

    def backward(self):

        # calculate step returns in reverse order
        rewards = self.rewards
        rew = torch.stack(self.rewards, dim=0)

        returns = torch.Tensor(len(rewards) - 1, *self.outputs[-1].value.data.size())
        step_return = self.outputs[-1].value.data.cpu()
        for i in range(len(rewards) - 2, -1, -1):
            step_return.mul_(self.discounts[i]).add_(rewards[i])
            returns[i] = step_return

        if cuda.USE_CUDA:
            returns = returns.cuda()

        # calculate losses
        policy_loss = 0
        value_loss = 0
        steps = len(self.outputs) - 1
        for i in range(steps):
            advantage = cuda.Variable(returns[i] - self.outputs[i].value.data)
            policy_loss += -self.outputs[i].log_action * advantage
            value_loss += F.smooth_l1_loss(self.outputs[i].value, cuda.Variable(returns[i]))

        weights_l2 = 0
        for param in self.parameters():
            weights_l2 += param.norm(2)

        loss = policy_loss.mean() / steps + value_loss / steps + 0.00001 * weights_l2
        ewc = self.ewc_reg.regularize(self.named_parameters())
        if cuda.USE_CUDA:
            ewc = ewc.cuda()
        loss = loss + ewc
        loss.backward()

        # reset state
        self.reset()

        # episode average reward, rew size: [40x20x1]
        ep_rew = torch.mean(torch.sum(rew, dim=0)) * 100

        return ep_rew, loss.data[0], ewc.data[0]
