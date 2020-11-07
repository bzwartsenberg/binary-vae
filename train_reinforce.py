#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 11:04:15 2020

@author: berend
"""

## test run with REINFORCE:


import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
import itertools
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from multiprocessing import freeze_support
#from trainer import Model_Trainer, Parameters
from scheduler import train_on_table
#
from model import VAE_discrete
from integrated_loss import expectation_loss, loss_arg

import torchvision
import torchvision.transforms as transforms

from torch.distributions import Bernoulli



transform = transforms.Compose(
    [transforms.ToTensor(),
    #   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                            shuffle=True, num_workers=0)

testset = torchvision.datasets.MNIST(root='./data', train=False,
                                        download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                            shuffle=False, num_workers=0)

import matplotlib.pyplot as plt
import numpy as np

# functions to show an image

def plot_in_and_out(images, rec_images):

    fig,ax = plt.subplots(images.shape[0],2)
    for i in range(images.shape[0]):
        ax[i][0].imshow(images[i].reshape(28,28), cmap='Greys')
        ax[i][1].imshow(rec_images[i].detach().numpy().reshape(28,28), cmap='Greys')


if __name__ == '__main__':
    freeze_support()        
    vae = VAE_discrete(input_size=28*28, hidden_size=400, num_bernoullis=10)
    optimizer = optim.Adam(vae.parameters(), lr=1e-4)        
    
    kl_rec_ratio = vae.num_bernoullis/(28*28)

    
    epochs = 2

    train_kl_losses = []
    train_rec_losses = []
    losses = []
    reinforce_losses = []
    total_losses = []
    print('Start of loop')
    for i in range(epochs):
        print('In first loop')
        for j,data in enumerate(trainloader):

            
            images,labels = data
            batch_X = images.reshape((-1,28*28))
            
            if torch.cuda.is_available():
                batch_X = batch_X.cuda()
            
            
            optimizer.zero_grad()
        
            logit_q = vae.encode(batch_X, return_logits=True)
            q = torch.sigmoid(logit_q)
            
            n_samples = 5
            batch_size = batch_X.shape[0]
            z_samples = Bernoulli(q).sample((n_samples,)).transpose(2,0).transpose(0,1)
            
            ## for reconstruct(z_samples) need:
            logit_p_x = vae.decode(z_samples.transpose(1,2).reshape(-1,vae.num_bernoullis)).reshape((batch_size, n_samples, -1)).transpose(1,2)
            
            logit_q = logit_q.unsqueeze(2)
            logit_q_prior = vae.logit_q_prior.unsqueeze(2)
            batch_X = batch_X.unsqueeze(2)
        
            
            #expanding for KL loss:, because crossentropy function does not broadcast
            logit_q = logit_q.expand(-1,-1,z_samples.shape[2])
            logit_q_prior = logit_q_prior.expand(*logit_q.shape)
            batch_X = batch_X.expand(*logit_p_x.shape)
            
            rec_loss, kl_loss = loss_arg(batch_X, z_samples, logit_q, logit_q_prior, logit_p_x)
            
            loss = -(rec_loss + kl_rec_ratio*kl_loss)
            
            
            ## now I want to gather grads, and calc:
            # grad(loss) + nograd(loss)*grad(log(q))
            
            #write as grad(loss + loss.clone().detach()*q_log_prob
            log_prob_q = -F.binary_cross_entropy_with_logits(logit_q, z_samples, reduction='none').sum(dim=1, keepdims=False)
            
            reinforce_loss = loss.clone().detach()*log_prob_q
        
            total_loss = loss + reinforce_loss
            
            total_loss.mean().backward()
            optimizer.step()

            train_kl_losses.append(kl_loss.mean().item())
            train_rec_losses.append(rec_loss.mean().item())
            losses.append(loss.mean().item())
            reinforce_losses.append(reinforce_loss.mean().item())
            total_losses.append(total_loss.mean().item())
            if j % 100 == 0:
                print(i, j)                   
                print(total_loss.mean())



