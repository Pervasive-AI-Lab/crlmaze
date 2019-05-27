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

import numpy as np
from doom_instance import DoomInstance


def init_doom_env(conf):
    if conf['action_set'] is not None:
        conf['action_set'] = np.load(conf['action_set']).tolist()

    doom = DoomInstance(
        conf['vizdoom_config'],
        wad=conf['wad_path'],
        skiprate=conf['skiprate'],
        id=None,
        visible=False,
        actions=conf['action_set'])
    state = doom.get_state_normalized()

    conf['button_num'] = doom.get_button_num()
    conf['screen_size'] = state.screen.shape
    conf['variable_num'] = len(state.variables)
    if state.variables is not None:
        conf['variables_size'] = state.variables.shape
