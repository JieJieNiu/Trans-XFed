#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 14:33:34 2023

@author: jolie
"""

import sys
import numpy as np
import pandas as pd
import torch
import args
from args import args_parser

sys.path.append('../')
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import TensorDataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def load_data(file_name):
    df= pd.read_csv(args.DATA_DIR + file_name + '.csv')
    df=df.iloc[:,1:]
    return df


def load_data_test(file_name):
    df= pd.read_csv(args.DATA_DIR + file_name + '.csv')
    df=df.iloc[:,1:]
    return df


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)
    


def nn_seq_wind(file_name):
    print('data processing...')
    dataset = load_data(file_name)
    # split
    train = dataset[:int(len(dataset))]
    val = dataset[int(len(dataset) * 0.1):len(dataset)]

    def process(data):
        train_seq=np.array(data.iloc[:,1:-1], dtype=np.float32)
        train_label=np.array(data.iloc[:,-1], dtype=np.float32)
        train_seq = torch.tensor(train_seq)
        train_label = torch.tensor(train_label)
        dataset=TensorDataset(train_seq,train_label)

        seq=DataLoader(dataset=dataset,batch_size=64,shuffle=True)
        
        for i, dataset in enumerate(seq):
            data,label=dataset


        return seq

    Dtr = process(train)
    Val = process(val)

    return  Dtr, Val



def nn_seq_test(file_name):
    data = load_data_test(file_name)

    def process(data):
        test_seq=np.array(data.iloc[:,1:-1], dtype=np.float32)
        test_label=np.array(data.iloc[:,-1], dtype=np.float32)
        test_seq = torch.tensor(test_seq)
        test_label = torch.tensor(test_label)
        dataset=TensorDataset(test_seq,test_label)

        seq=DataLoader(dataset=dataset,batch_size=64,shuffle=True)
        
        for i, dataset in enumerate(seq):
            data,label=dataset


        return seq

    Dte = process(data)

    return  Dte



# def get_mape(x, y):
#     """
#     :param x:true
#     :param y:pred
#     :return:MAPE
#     """
#     return np.mean(np.abs((x - y) / x))



