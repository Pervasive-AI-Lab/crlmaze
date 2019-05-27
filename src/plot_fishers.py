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


""" This simple script can be used to plot the fisher matrices. """

import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import sys

if len(sys.argv) < 3:
    print("usage: plot_results.py file.bin")

with open(sys.argv[1], 'rb') as f:
    fisher = np.fromfile(f)
    print("loaded.")

n_bins = 100

print(fisher)
plt.hist(fisher, bins=n_bins, range=(0, np.max(fisher) * 1.5))

print("Tot params: ", fisher.shape[0])
over = np.where(fisher >= (np.max(fisher) - np.max(fisher)/10))[0].shape[0]
print("Values greater than {}: {}".format(np.max(fisher) - np.max(fisher)/10, over))
print("Percentage (%): ", (over / fisher.shape[0]) * 100)
print("max:", np.max(fisher))

plt.yscale('log', nonposy='clip')

plt.show()
print('Showing Fisher 0...')
