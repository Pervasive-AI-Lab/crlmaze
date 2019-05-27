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

import torch
from cuda import USE_CUDA
from aac_map import AdvantageActorCriticMap


def get_model(cfg):
    torch.set_default_tensor_type('torch.FloatTensor')
    torch.manual_seed(cfg['seed'])
    if USE_CUDA:
        torch.cuda.manual_seed_all(cfg['seed'])

    model_class = {
        'aac_map': AdvantageActorCriticMap
    }
    model = model_class[cfg['model']](cfg)
    return model
