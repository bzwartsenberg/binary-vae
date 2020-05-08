#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  4 17:41:22 2020

@author: berend
"""

import torch.nn as nn
import torch.nn.functional as F
import torch


class VAE_discrete(nn.Module):

    def __init__(self, input_size=100, hidden_size=400, num_bernoullis=10):
        super(VAE_discrete, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_bernoullis = num_bernoullis

        self.FC1 = nn.Linear(self.input_size,self.hidden_size) #size is ..?
        self.FC2 = nn.Linear(self.hidden_size,self.num_bernoullis)

        self.FC3 = nn.Linear(self.num_bernoullis, self.hidden_size)
        self.FC4 = nn.Linear(self.hidden_size, self.input_size)

        #note: can make this be a trainable parameter as well
        self.logit_q_prior = torch.zeros((1,self.num_bernoullis))
        
        if torch.cuda.is_available():
            self.logit_q_prior = self.logit_q_prior.cuda()
            self.cuda()
        
        self.q_prior = torch.sigmoid(self.logit_q_prior)

    def encode(self, x, return_logits=True):
        #q(b|x)
        y = F.relu(self.FC1(x))
        if return_logits:
            return self.FC2(y)
        else:
            return torch.sigmoid(self.FC2(y))
        
    def decode(self, b, return_logits=True):
        #p(x|b)
        y = F.relu(self.FC3(b))
        if return_logits:
            return self.FC4(y)
        else:
            return torch.sigmoid(self.FC4(y))

    def sample_epsilon(self, b_size):
        eps = torch.randn(b_size, self.num_bernoullis)
        return eps

    def forward(self, x, sample=False, return_logits=False):
        q = self.encode(x)
        if sample:
            eps = torch.rand_like(q)
            b = (eps < q).float()
        else:
            b = q


        return self.decode(b,return_logits=return_logits)