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

import torch
from torch.autograd import Variable
from torch import nn
from cuda import Variable
import numpy as np


class EWCRegularizer(nn.Module):
    """Module for Elastic Weight Consolidation Regularization. It persists the
    fisher information matrix and optimal parameters for every task encountered
    by underlying model, and calculates component in overall loss criterion.

    Arguments
    =========
    ewc_lambda : float
        Co-efficient for EWC component in loss, zero means no regularization.

    Attributes
    ==========
    tasks_encountered : list
        List of sequential task ids as encountered.
    fisher : dict
        Dict with task ids as keys and values being dicts of named parameters.
    optpar : dict
        Similar structure as fisher dict, to store optimal parameters per task.
    """

    def __init__(self, cfg, ewc_lambda):
        super().__init__()
        self.ewc_lambda = ewc_lambda
        self.cfg = cfg
        self.tasks_encountered = []

        # both these dicts have a similar hierarchical structure
        self.fisher = {}
        self.optpar = {}

    def update_fisher_optpar(self, task_id, optimal_named_parameters,
                             consolidate=True):
        """Save optimal parameters and terms of fisher matrix. Additionally it
        needs few samples of cooresponding task, to perform backward through
        model and accumulate gradients, used to calculate fisher matrix.

        Arguments
        =========
        task_id : integer, string
            ID of task whose optimal parameters and fisher matrix to be saved.
        optimal_named_parameters : generator
            Optimal named parameters of model corresponding to task_id.
        consolidate: boolean
            if the F matrix should be used for consolidation.
        """
        if consolidate:
            self.tasks_encountered.append(task_id)
        self.fisher[task_id] = {}
        self.optpar[task_id] = {}

        # gradients would have be accumulated, can be used to calculate fisher
        for name, param in optimal_named_parameters:
            self.optpar[task_id][name] = param.data.clone()
            self.fisher[task_id][name] = param.grad.data.clone().pow(2)
            # clip the values for stability
            self.fisher[task_id][name] = torch.clamp(
                     self.fisher[task_id][name], 0, self.cfg['clip_value']
            )

        # saving fishers for stats
        f_values = []
        for i, (k, fish) in enumerate(self.fisher.items()):
            for k, values in fish.items():
                f_values = np.concatenate(
                    (f_values, values.cpu().numpy().flatten()))
            f_values.tofile(self.cfg['checkpoint_file'][:-4] +
                            '_fisher'+str(i)+'.bin')

    def forward(self, named_params):
        net_loss = Variable(torch.Tensor([0]))
        if not self.ewc_lambda:
            return net_loss
        for task_id in self.tasks_encountered:
            for name, param in named_params:
                fisher = Variable(self.fisher[task_id][name])
                optpar = Variable(self.optpar[task_id][name])
                net_loss += (fisher * (optpar - param).pow(2)).sum() * self.ewc_lambda
        return net_loss

    def regularize(self, named_params):
        """Calculate the EWC regularization component in the overall loss.
        For all the tasks encountered in past, L2-norm loss is calculated
        between current model parameters and optimal parameters of previous
        tasks, weighted by terms from fisher matrix.

        Arguments
        =========
        named_params : generator
            Named parameters of model to be regularized.
        """
        return self.forward(named_params)

    def add_task_to_consolidate(self, task_id):
        """ We use this function to hack the tasks_encountered and being able
            to add some F that has been calculated before but not used for
            consolidation.

        Arguments
        =========
        task_id : task/fisher_id
            id of the task/fisher to use for consolidation in future.
        """

        if task_id not in self.tasks_encountered and \
                task_id in self.fisher.keys():
            self.tasks_encountered.append(task_id)
            print("Consolidating Fishers: ", self.tasks_encountered)
