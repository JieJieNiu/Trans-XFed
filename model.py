#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 13:03:57 2023

@author: jolie
"""

from torch import nn
import args


class TransFed(nn.Module):
    def __init__(self, args, name):
        super(TransFed, self).__init__()
        self.name = TransFed
        self.len = 0
        self.loss=0
        self.transformer = nn.Sequential(
            nn.TransformerEncoderLayer(d_model=21, nhead=args.Trans_H, dim_feedforward=args.Trans_D,batch_first=True),
        )
        self.classifer=nn.Sequential(
            nn.Linear(21, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2),
            )

    def forward(self, data):
        x=self.transformer(data)
        x=self.classifer(x)

    
        return x


