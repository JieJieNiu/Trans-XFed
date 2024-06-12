
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 10:08:49 2023

@author: jolie
"""

import argparse
import torch
import os

DEFAULT_DIR=os.getcwd()

"""Raw data directory and model directory"""

DATA_DIR = DEFAULT_DIR + os.path.sep + 'data' + os.path.sep
MODEL_SAVE = DEFAULT_DIR + os.path.sep + 'ckpt' + os.path.sep
MODEL_NAME="TransXFed"
LOG_PATH=DEFAULT_DIR + os.path.sep + 'log'
FIG_PATH = DEFAULT_DIR + os.path.sep + 'figure'
clients = ['data' + str(i) for i in range(0, 4)]



def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--M', type=str, default='TransFed', help='TransFed or FedProx or FedAvg')
    parser.add_argument('--E', type=int, default=200, help='number of rounds of training')
    parser.add_argument('--poc', type=bool, default=True, help='True power of choice, False random choice')
    parser.add_argument('--wa', type=bool, default=True, help='Weighted aggregation')
    parser.add_argument('--r', type=int, default=50, help='number of communication rounds')
    parser.add_argument('--K', type=int, default=4, help='number of total clients')
    parser.add_argument('--input_dim', type=int, default=21, help='input dimension')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--C', type=float, default=0.5, help='sampling rate')
    parser.add_argument('--B', type=int, default=64, help='local batch size')
    parser.add_argument('--w', type=int, default=0.25, help='majority class weight')
    parser.add_argument('--fixmu', type=bool, default=True, help='fix proximal term constant')
    parser.add_argument('--mu', type=float, default=0.01, help='proximal term constant')
    parser.add_argument('--Trans_D', type=float, default=32, help='transformer depth')
    parser.add_argument('--Trans_H', type=float, default=3, help='transformer heads')
    parser.add_argument('--loss', type=str, default='NLL', help='WCE weighted cross entropy or Focal or NLL')    
    parser.add_argument('--optimizer', type=str, default='SGD', help='type of optimizer')
    parser.add_argument('--file_name', type=dict, default=['data0','data1', 'data2','data3'],help='filename of different client')
    parser.add_argument('--samplesize', type=dict, default=[18368,19904,18816,13440],help='filename of different client')
    parser.add_argument('--device', default=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--step_size', type=float, default=5, help='lr scheduler step size')
    parser.add_argument('--gamma', type=float, default=0.1, help='lr scheduler gamma')
    clients = ['data' + str(i) for i in range(0, 4)]
    parser.add_argument('--clients', default=clients)

    args = parser.parse_args()

    return args


