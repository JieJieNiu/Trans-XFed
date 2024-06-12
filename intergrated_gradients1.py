#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 13:52:19 2024

@author: jie
"""

import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch import nn
from torch.autograd import Variable
from torch import nn
import os
from captum.attr import (
    GradientShap,
    DeepLift,
    DeepLiftShap,
    IntegratedGradients,
    LayerConductance,
    NeuronConductance,
    NoiseTunnel,
)
import pandas as pd
import plotly.express as px
from IPython.display import HTML
import plotly.offline as py
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset, DataLoader
import seaborn as sns



class TransFed(nn.Module):
    def __init__(self):
        super(TransFed, self).__init__()
        self.len = 0
        self.loss=0
        self.transformer = nn.Sequential(
            nn.TransformerEncoderLayer(d_model=21, nhead=3, dim_feedforward=32,batch_first=True),
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
    
check_point = torch.load('best.pkl')
model=TransFed()
model.load_state_dict(check_point['state_dict'])
model.eval()    

df= pd.read_csv('./data/data0.csv')
inputs=torch.tensor(np.array(df.iloc[:,2:-1])).to(dtype=torch.float32)
outputs=torch.tensor(np.array(df.iloc[:,-1])).to(dtype=torch.float32)
dataset=TensorDataset(inputs,outputs)
data_loader=DataLoader(dataset=dataset,batch_size=64,shuffle=False)

for i, dataset in enumerate(data_loader):
    inputs, outputs=dataset
    
all_probs = []
all_labels = []

with torch.no_grad():
    for inputs, outputs in data_loader:
        predicts = model(inputs)
        probs = torch.softmax(predicts, dim=1)  
        all_probs.append(probs)

        
all_probs = torch.cat(all_probs)
all_probs_flat = pd.DataFrame(all_probs.tolist())


        

def ig_results(method):
    
    df= pd.read_csv('/Users/jolie/Desktop/project_Federated1.26/data/data0.csv')
    check_point = torch.load('/Users/jolie/Desktop/project_Federated1.26/FedAvg_29_last.pkl')
    model=TransFed()
    model.load_state_dict(check_point['state_dict'])
    model.eval()    
    a=df.iloc[:,2:]
    # if target_class == 0:
    #     a = a[a['Default_label']==0]
    # else:
    #     a =a[a['Default_label']==1]
    inputs=torch.tensor(a.iloc[:,0:-1].values, dtype=torch.float32)
    baseline=torch.zeros_like(inputs, dtype=torch.float32)
    target=a.iloc[:,-1]
    target=torch.tensor(target.values)
    ig= method(model)
    attributions, delta=ig.attribute(inputs, baseline, target=target, return_convergence_delta=True)
    attributions =attributions.detach().numpy()
    return attributions, delta

attributions, delta=ig_results(IntegratedGradients)


a=pd.concat([pd.DataFrame(attributions), all_probs_flat,df.iloc[:,-1]],axis=1)
a.to_csv('lg_results.csv')



weight=check_point['state_dict']['transformer.0.self_attn.in_proj_weight']
bias=check_point['state_dict']['transformer.0.self_attn.in_proj_bias']



wq=weight[0:21]
wk=weight[21:42]
wv=weight[42:63]
bq=bias[0:21]
bk=bias[21:42]
bv=bias[42:63]

df1=pd.read_csv('./data/data0.csv')
df2=pd.read_csv('./data/data1.csv')
df3=pd.read_csv('./data/data2.csv')
df4=pd.read_csv('./data/data3.csv')

defaulting=pd.concat([df1[df1['Default_label']==1],df2[df2['Default_label']==1],df3[df3['Default_label']==1],df4[df4['Default_label']==1]])
defaulting=np.array(defaulting.iloc[:,2:-1])
nondefaulting=pd.concat([df1[df1['Default_label']==0],df2[df2['Default_label']==0],df3[df3['Default_label']==0],df4[df4['Default_label']==0]])
nondefaulting=np.array(nondefaulting.iloc[:,2:-1])
columns=df1.columns[2:-1]

def plotattentionscore(data):
    inputx=torch.tensor(data).to(dtype=torch.float32).unsqueeze(1)
    q=inputx*wq+bq
    k=inputx*wk+bk
    att=torch.matmul(q,k.transpose(1,2))/np.sqrt(21)
    att = torch.softmax(att, -1)
    score = torch.zeros(21,21)
    for i in range(len(att)):
        score+=att[i]
    score=np.array(score)
    sns.set(font_scale=0.8)
    sns.heatmap(score, cmap="YlGnBu",robust=False,xticklabels=columns,yticklabels=columns)
    return score

score=plotattentionscore(nondefaulting)
