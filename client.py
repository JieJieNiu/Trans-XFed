#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 14:00:39 2023

@author: jolie
"""

import copy
import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR
from torch import nn
from tqdm import tqdm
from get_data import nn_seq_wind, nn_seq_test
import get_data
from torch.autograd import Variable 
from focal import FocalLoss, WeightedFocalLoss
import torch.nn.functional as F
from torch.nn import init
import args
import math
from args import args_parser
from datetime import datetime



def get_val_loss(args, model, Val):
    model.eval()
    val_loss = []
    if args.loss == 'WCE':
        class_weight=torch.tensor([args.w, (1-args.w)])
        loss_function =nn.CrossEntropyLoss(weight=class_weight)
    elif args.loss== 'Focal':
        loss_function = FocalLoss(args, alpha=True, gamma=2)
    elif args.loss== 'NLL':
        class_weight=torch.tensor([args.w, (1-args.w)])
        loss_function = nn.NLLLoss(weight=class_weight)
    for (seq, label) in Val:
        with torch.no_grad():
            seq=seq.to(args.device)
            label=label.to(args.device)
            y_pred = model(seq)
            loss = loss_function(y_pred, label.to(dtype=torch.long))
            val_loss.append(loss.item())

    return np.mean(val_loss)

def initialize(self):
    for m in self.modules():
        if isinstance(m, nn.Linear):
            init.xavier_uniform_(m.weight.data) 


"""Starting training"""

def train(args,model,global_model):
    Dtr, Val = get_data.nn_seq_wind(model.name)
    model.len = len(Dtr)
    global_model.apply(initialize)
    global_model = copy.deepcopy(global_model)
    LEARNING_RATE = args.lr
    """Initializing settings"""
    print('[{}] Initializing optimizer...'.format(datetime.now()))
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(),lr=LEARNING_RATE, momentum=0.9,weight_decay=args.weight_decay)
    else:
        
        optimizer = torch.optim.SGD(model.parameters(),lr=LEARNING_RATE,momentum=0.9, weight_decay=args.weight_decay)
        
    #scheduler=StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    print('[{}] Initializing LOSS...'.format(datetime.now()))
    if args.loss == 'WCE':
        class_weight=torch.tensor([args.w, (1-args.w)])
        loss_function =nn.CrossEntropyLoss(weight=class_weight)
    elif args.loss== 'Focal':
        loss_function = FocalLoss(args, alpha=True, gamma=2)
    elif args.loss== 'NLL':
        class_weight=torch.tensor([args.w, (1-args.w)])
        loss_function = nn.NLLLoss(weight=class_weight,reduction='mean',reduce=None)
    # training
    correct_source = 0
    FN=0
    TN=0
    FP=0
    TP=0
    
    
    print('[{}] Start training...'.format(datetime.now()))
    

    for epoch in tqdm(range(args.E)):
        train_loss=0.0
        
        for (seq, label) in Dtr:
            model.train()
            seq = seq.to(args.device)
            label = label.to(args.device)
            y_pred = model(seq)
            preds_source = y_pred.data.max(1, keepdim=True)[1]
            zes=Variable(torch.zeros(args.B).type(torch.LongTensor))#全0变量
            ons=Variable(torch.ones(args.B).type(torch.LongTensor))#全1变量
            train_correct01 = ((preds_source.squeeze(1)==zes)&(label==ons)).sum() #原标签为1，预测为 0 的总数
            train_correct10 = ((preds_source.squeeze(1)==ons)&(label==zes)).sum() #原标签为0，预测为1 的总数
            train_correct11 = ((preds_source.squeeze(1)==ons)&(label==ons)).sum() #原标签为1，预测为 1 的总数
            train_correct00 = ((preds_source.squeeze(1)==zes)&(label==zes)).sum() #原标签为0，预测为 0 的总数
            FN += train_correct01
            FP += train_correct10
            TP += train_correct11
            TN += train_correct00
            correct_source += preds_source.eq(label.data.view_as(preds_source)).sum()
            

            optimizer.zero_grad()
            
            # compute proximal_term
                           
            proximal_term = 0.0
            for w, w_t in zip(model.parameters(), global_model.parameters()):
                proximal_term += (w - w_t).norm(2)
                    
            if args.M == "TransFed":
                if args.loss=="NLL":
                    m=nn.LogSoftmax(dim=1)
                    loss = loss_function(m(y_pred), label.to(dtype=torch.long)) + (args.mu / 2) * proximal_term
                else:
                    loss = loss_function(y_pred, label.to(dtype=torch.long)) + (args.mu / 2) * proximal_term
            elif args.M =="FedAvg":
                loss = loss_function(y_pred, label.to(dtype=torch.long))
            #train_loss.append(loss.item())
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()

        train_loss /= len(Dtr)
        #acc_source = float(correct_source) *100. / (len(Dtr)* args.E * args.B)
        recall=TP/(TP+FN)
        percision=TP/(TP+FP)
        F1=2*(recall*percision)/(recall+percision)
        

        # validation
        val_loss = get_val_loss(args, model, Val)
        #if epoch + 1 >= min_epochs and val_loss < min_val_loss:
            #min_val_loss = val_loss
            #best_model = copy.deepcopy(model)
        print("lr in %d: %.7f" % (epoch+1, optimizer.state_dict()['param_groups'][0]['lr']))
        print('model name {} epoch {:03d} train_loss {:.8f} val_loss {:.8f},Recall:{}, F1:{},Precision:{}'.format(model.name, epoch, train_loss, val_loss,recall,F1,percision))
        

    return model

def test(args, TransFed, file_name):
    TransFed.eval()
    Dte = get_data.nn_seq_test(file_name)
    if args.loss == 'WCE':
        class_weight=torch.tensor([args.w, (1-args.w)])
        loss_function =nn.CrossEntropyLoss(weight=class_weight)
    elif args.loss== 'Focal':
        loss_function = FocalLoss(args, alpha=True, gamma=2)
    elif args.loss== 'NLL':
        class_weight=torch.tensor([args.w, (1-args.w)])
        loss_function = nn.NLLLoss(weight=class_weight,reduction='mean',reduce=None)
    test_loss=0
    correct= 0
    FN_test=0
    TN_test=0
    FP_test=0
    TP_test=0
    for (seq, target) in Dte:
        with torch.no_grad():
            seq = seq.to(args.device)
            y_pred = TransFed(seq)
            if args.loss=="NLL":
                m=nn.LogSoftmax(dim=1)
                loss = loss_function(m(y_pred), target.to(dtype=torch.long))
            else:
                loss=loss_function(y_pred, target.to(dtype=torch.long))
            test_loss += loss # sum up batch loss
            test_pred = y_pred.data.max(1)[1]
            correct += test_pred.eq(target.data.view_as(test_pred)).cpu().sum()
            zes=Variable(torch.zeros(args.B).type(torch.LongTensor))#全0变量
            ons=Variable(torch.ones(args.B).type(torch.LongTensor))#全1变量
            test_correct01 = ((test_pred==zes)&(target==ons)).sum() #原标签为1，预测为 0 的总数
            test_correct10 = ((test_pred==ons)&(target==zes)).sum() #原标签为0，预测为1 的总数
            test_correct11 = ((test_pred==ons)&(target==ons)).sum()#原标签为1，预测为 1 的总数
            test_correct00 = ((test_pred==zes)&(target==zes)).sum()#原标签为0，预测为 0 的总数
            FN_test += test_correct01.item()
            FP_test += test_correct10.item()
            TP_test += test_correct11.item()
            TN_test += test_correct00.item()
            
            
            
            
        test_loss /= len(Dte)
        recall_test=TP_test/(TP_test+FN_test+1)
        percision_test=TP_test/(TP_test+FP_test+1)
        F1_test=2*(recall_test*percision_test)/(recall_test+percision_test)
    print('Target_test_set:Average classification loss: {:.4f}/tRecall:{}/tPrecision:{}/tF1:{}'.format(test_loss,recall_test,percision_test,F1_test))
    return test_loss

    