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

import glob
import time
import argparse
from vizdoom import *


def replay(config, skiprate, path):

    game = DoomGame()
    #game.set_doom_game_path(wad)
    game.load_config(config)

    game.set_screen_resolution(ScreenResolution.RES_800X600)
    game.set_window_visible(True)
    game.set_render_hud(True)

    game.set_mode(Mode.ASYNC_PLAYER)
    game.init()

    for episode_file in glob.glob(path + '/*.lmp'):
        time.sleep(5)
        print('replay episode:', episode_file)
        game.replay_episode(episode_file)
        while not game.is_episode_finished():
            state = game.get_state()
            game.advance_action(skiprate)
            reward = game.get_last_reward()
            print('State #{}: reward = {}'.format(state.number, reward))

        print('total reward:', game.get_total_reward())

    game.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Doom Recorder')
    parser.add_argument('--vizdoom_config', default='world.cfg', help='vizdoom config path')
    parser.add_argument('--skiprate', type=int, default=1, help='number of skipped frames')
    parser.add_argument('--path', default='.', help='.lmp files path')

    args = parser.parse_args()

    replay(args.vizdoom_config, args.skiprate, args.path)
