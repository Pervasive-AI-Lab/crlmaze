#!/usr/bin/env bash
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

echo "--------------------- Starting LIGTH exps ------------------------"

echo "--------------------- Starting LIGTH - NAIVE ------------------------"
python src/main.py with cfgs/light/naive/train.json -c "CRL v1.1 - LIGHT - Naive" > light_naive.out

echo "--------------------- Starting LIGTH - STATIC ------------------------"
python src/main.py with cfgs/light/ewc_static/train.json -c "CRL v1.1 - LIGHT - Static" > light_static.out

echo "--------------------- Starting LIGTH - SUP ------------------------"
python src/main.py with cfgs/light/ewc_sup/train.json -c "CRL v1.1 - LIGHT - Sup" > light_sup.out

echo "--------------------- Starting LIGTH - UNSUP ------------------------"
python src/main.py with cfgs/light/ewc_unsup/train.json -c "CRL v1.1 - LIGHT - Unsup" > light_unsup.out

echo "--------------------- Starting LIGTH - MULTIENV ------------------------"
python src/main.py with cfgs/light/multienv/train.json -c "CRL v1.1 - LIGHT - Multienv" > light_multienv.out

echo "--------------------- Starting OBJECT exps ------------------------"

echo "--------------------- Starting OBJECT - NAIVE ------------------------"
python src/main.py with cfgs/object/naive/train.json -c "CRL v1.1 - OBJECT - Naive" > object_naive.out

echo "--------------------- Starting OBJECT - STATIC ------------------------"
python src/main.py with cfgs/object/ewc_static/train.json -c "CRL v1.1 - OBJECT - Static" > object_static.out

echo "--------------------- Starting OBJECT - SUP ------------------------"
python src/main.py with cfgs/object/ewc_sup/train.json -c "CRL v1.1 - OBJECT - Sup" > object_sup.out

echo "--------------------- Starting OBJECT - UNSUP ------------------------"
python src/main.py with cfgs/object/ewc_unsup/train.json -c "CRL v1.1 - OBJECT - Unsup" > object_unsup.out

echo "--------------------- Starting OBJECT - MULTIENV ------------------------"
python src/main.py with cfgs/object/multienv/train.json -c "CRL v1.1 - OBJECT - Multienv" > object_multienv.out

echo "--------------------- Starting TEXTURE exps ------------------------"

echo "--------------------- Starting TEXTURE - NAIVE ------------------------"
python src/main.py with cfgs/texture/naive/train.json -c "CRL v1.1 - TEXTURE - Naive" > texture_naive.out

echo "--------------------- Starting TEXTURE - STATIC ------------------------"
python src/main.py with cfgs/texture/ewc_static/train.json -c "CRL v1.1 - TEXTURE - Static" > texture_static.out

echo "--------------------- Starting TEXTURE - SUP ------------------------"
python src/main.py with cfgs/texture/ewc_sup/train.json -c "CRL v1.1 - TEXTURE - Sup" > texture_sup.out

echo "--------------------- Starting TEXTURE - UNSUP ------------------------"
python src/main.py with cfgs/texture/ewc_unsup/train.json -c "CRL v1.1 - TEXTURE - Unsup" > texture_unsup.out

echo "--------------------- Starting TEXTURE - MULTIENV ------------------------"
python src/main.py with cfgs/texture/multienv/train.json -c "CRL v1.1 - TEXTURE - Multienv" > texture_multienv.out

echo "--------------------- Starting ALL exps ------------------------"

echo "--------------------- Starting ALL - NAIVE ------------------------"
python src/main.py with cfgs/all/naive/train.json -c "CRL v1.1 - ALL - Naive" > all_naive.out

echo "--------------------- Starting ALL - STATIC ------------------------"
python src/main.py with cfgs/all/ewc_static/train.json -c "CRL v1.1 - ALL - Static" > all_static.out

echo "--------------------- Starting ALL - SUP ------------------------"
python src/main.py with cfgs/all/ewc_sup/train.json -c "CRL v1.1 - ALL - Sup" > all_sup.out

echo "--------------------- Starting ALL - UNSUP ------------------------"
python src/main.py with cfgs/all/ewc_unsup/train.json -c "CRL v1.1 - ALL - Unsup" > all_unsup.out

echo "--------------------- Starting ALL - MULTIENV ------------------------"
python src/main.py with cfgs/all/multienv/train.json -c "CRL v1.1 - ALL - Multienv" > all_multienv.out