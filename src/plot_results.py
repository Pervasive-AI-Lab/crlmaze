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

""" This simple script can be used to plot the cumulated reward results """

import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import sys

test_rw = False
mov_avg = True

if len(sys.argv) < 2:
    print("usage: plot_results.py file.pkl")

with open(sys.argv[1], 'rb') as f:
    results = pkl.load(f)

y = []
std = []

if test_rw:
    for avg_reward, err in zip(results[0][0], results[0][1]):
        y.append(avg_reward)
        std.append(err)
    x = list(range(len(y)))

    print(x)
    print(y)
    x, y, std = map(np.asarray, [x, y, std])
    plt.fill_between(x, y-std, y+std, facecolor='#d62728', alpha=0.2)
    # red dashes, blue squares and green triangles
    plt.plot(x, y, 'r--',)
    plt.show()
    print('Done.')

if mov_avg:

    x = []
    y1 = []
    y2 = []
    y3 = []
    print(results)
    for long_time_avg_rew, short_time_avg_rew, rw_diff in results[1]:

        y1.append(long_time_avg_rew)
        y2.append(short_time_avg_rew)
        y3.append(rw_diff)

    x = list(range(1, len(y2)+1))

    x, y1, y2, y3 = map(np.asarray, [x, y1, y2, y3])
    print(x)
    print(np.where(y3 < -250))
    # red dashes, blue squares and green triangles
    plt.plot(x, y1, x, y2, x, y3)
    plt.axvline(x=168, linewidth=1, ls='--')
    plt.axvline(x=334, linewidth=1, ls='--')
    plt.legend(['long_mov_avg', 'short_mov_avg', 'diff'])
    plt.show()
    print('Done.')
