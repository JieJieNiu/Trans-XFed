#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 16:15:12 2024

@author: jolie
"""

import matplotlib
from matplotlib import ticker
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import make_interp_spline



df=pd.read_csv('non_default_results.csv')


def parallel_plot(df,cols,title,cmap='Spectral',spread=None,curved=0.05,curvedextend=0.1):
    '''Produce a parallel coordinates plot from pandas dataframe with line colour with respect to a column.
    Required Arguments:
        df: dataframe
        cols: columns to use for axes
        rank_attr: attribute to use for ranking
    Options:
        cmap: Colour palette to use for ranking of lines
        spread: Spread to use to separate lines at categorical values
        curved: Spline interpolation along lines
        curvedextend: Fraction extension in y axis, adjust to contain curvature
    Returns:
        x coordinates for axes, y coordinates of all lines'''
    colmap = matplotlib.cm.get_cmap(cmap)
    cols = cols 

    fig, axes = plt.subplots(1, len(cols)-1, sharey=False, figsize=(1.5*len(cols),5))
    valmat = np.ndarray(shape=(len(cols),len(df)))
    x = np.arange(0,len(cols),1)
    ax_info = {}
    for i,col in enumerate(cols):
        vals = df[col]
        if (vals.dtype == float) & (len(np.unique(vals)) > 20):
            minval = np.min(vals)
            maxval = np.max(vals)
            rangeval = maxval - minval
            vals = np.true_divide(vals - minval, maxval-minval)
            nticks = 5
            tick_labels = [round(minval + i*(rangeval/nticks),4) for i in range(nticks+1)]
            ticks = [0 + i*(1.0/nticks) for i in range(nticks+1)]
            valmat[i] = vals
            ax_info[col] = [tick_labels,ticks]
        else:
            vals = vals.astype('category')
            cats = vals.cat.categories
            c_vals = vals.cat.codes
            minval = 0
            maxval = len(cats)-1
            if maxval == 0:
                c_vals = 0.5
            else:
                c_vals = np.true_divide(c_vals - minval, maxval-minval)
            tick_labels = cats
            ticks = np.unique(c_vals)
            ax_info[col] = [tick_labels,ticks]
            if spread is not None:
                offset = np.arange(-1,1,2./(len(c_vals)))*2e-2
                np.random.shuffle(offset)
                c_vals = c_vals + offset
            valmat[i] = c_vals
            
    extendfrac = curvedextend if curved else 0.05  
    for i,ax in enumerate(axes):
        for idx in range(valmat.shape[-1]):
            if curved:
                x_new = np.linspace(0, len(x), len(x)*20)
                a_BSpline = make_interp_spline(x, valmat[:,idx],k=3,bc_type='clamped')
                y_new = a_BSpline(x_new)
                ax.plot(x_new,y_new,color=colmap(valmat[-1,idx]),alpha=0.3)
            else:
                ax.plot(x,valmat[:,idx],color=colmap(valmat[-1,idx]),alpha=0.3)
        ax.set_ylim(0-extendfrac,1+extendfrac)
        ax.set_xlim(i,i+1)
    
    for dim, (ax,col) in enumerate(zip(axes,cols)):
        ax.xaxis.set_major_locator(ticker.FixedLocator([dim]))
        ax.yaxis.set_major_locator(ticker.FixedLocator(ax_info[col][1]))
        ax.set_yticklabels(ax_info[col][0])
        ax.set_xticklabels([cols[dim]],fontsize=14,rotation=90)
    
    
    plt.subplots_adjust(wspace=0)
    norm = matplotlib.colors.Normalize(0,1)#*axes[-1].get_ylim())
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = plt.colorbar(sm,pad=0,ticks=None,extend='both',extendrect=True,extendfrac=extendfrac)
    if curved:
        cbar.ax.set_ylim(0-curvedextend,1+curvedextend)
    cbar.ax.set_yticklabels(ax_info[cols][0])
    cbar.ax.set_xlabel(cols)
    if title =='defaulting':
        plt.title('integrated gradients results of defaulting samples')
    else:
        plt.title('integrated gradients results of non-defaulting samples')
    plt.show()
            
    return x,valmat


cols=df.columns[:]


parallel_plot(df.iloc[:2000,:],cols,'non-defaulting')






###attention
import torch
import torch.nn as nn
import pandas as pd
import numpy as np


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





