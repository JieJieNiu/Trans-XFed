#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 08:39:46 2024

@author: jie
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import args
 
class FocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in 
        Focal Loss for Dense Object Detection.
            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])
        The losses are averaged across observations for each minibatch.
        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5), 
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.
    """
    def __init__(self,args, alpha, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(2, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = torch.tensor([args.w, (1-args.w)])
            else:
                self.alpha = torch.tensor([args.w, (1-args.w)])
        self.gamma = gamma
        self.size_average = size_average
 
    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs,dim=1)
 
        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
 

        alpha = self.alpha[ids.data.view(-1)]
 
        probs = (P*class_mask).sum(1).view(-1,1)
 
        log_p = probs.log()

 
        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p 
 
 
        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss
    

class WeightedFocalLoss(nn.Module):
    "Non weighted version of Focal Loss"
    def __init__(self,args, alpha, gamma=2):
        super(WeightedFocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(2, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = torch.tensor([args.w, (1-args.w)])
            else:
                self.alpha = torch.tensor([args.w, (1-args.w)])
        self.alpha = torch.tensor([args.w, (1-args.w)])
        self.gamma = gamma

    def forward(self, inputs, targets):
        targets=F.one_hot(targets, 2).float()
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        p=torch.sigmoid(inputs)
        pt=p*targets+ (1-p)*(1-targets)
        
        #targets = targets.type(torch.long)
        #at = self.alpha.gather(0, targets.data.view(-1))
        #pt = torch.exp(-BCE_loss)
        F_loss = (1-pt)**self.gamma * BCE_loss
        a_t=self.alpha*targets +(1-self.alpha)*(1-targets)
        F_loss=a_t*F_loss
        return F_loss.mean()
    
