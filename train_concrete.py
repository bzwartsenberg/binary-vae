#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 16:18:36 2020

@author: berend
"""


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

from torch.distributions import Bernoulli, Uniform


bsize=4

transform = transforms.Compose(
    [transforms.ToTensor(),
    #   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=bsize,
                                            shuffle=True, num_workers=0)

testset = torchvision.datasets.MNIST(root='./data', train=False,
                                        download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=bsize,
                                            shuffle=False, num_workers=0)

# functions to show an image

def plot_in_and_out(images, rec_images):

    fig,ax = plt.subplots(images.shape[0],2)
    for i in range(min(images.shape[0],4)):
        ax[i][0].imshow(images[i].cpu().reshape(28,28), cmap='Greys')
        ax[i][1].imshow(rec_images[i].cpu().detach().numpy().reshape(28,28), cmap='Greys')



def log_density_binary_concrete(y, log_alpha, lam):
    log_alpha = log_alpha.unsqueeze(2)
    ## if x > 0:  x + log(1+exp(-x))
    ## if x < 0: 0 + log(1 + exp(x))
    arg = -y*lam + log_alpha
    log_one_plus_exp = torch.clamp(arg, min=0.0) + torch.log(1+torch.exp(-torch.abs(arg)))
    #stable: 
    #TODO: check correspondence between below and above, some seem to be offset by ~1e-7
    dens = np.log(lam) - lam*y + log_alpha - 2*log_one_plus_exp
    #old version:
#    dens = np.log(lam) - lam*y + log_alpha - 2*torch.log(1 + torch.exp(arg))
    return dens

def sample_binary_concrete(log_alpha, lam, n_samples):
    u = torch.rand(log_alpha.shape + (n_samples,))
    if torch.cuda.is_available():
        u = u.cuda()
    return (log_alpha.unsqueeze(2) + torch.log(u) - torch.log(1-u))/lam


if __name__ == '__main__':
    freeze_support()        
    vae = VAE_discrete(input_size=28*28, hidden_size=400, num_bernoullis=10)
    optimizer = optim.Adam(vae.parameters(), lr=1e-4)        
    
    lam = 0.8
    lam_decay = 0.8
    lam2 = 0.8
    lam2_decay = 0.8
    
    kl_rec_ratio = vae.num_bernoullis/(28*28)

    
    epochs = 5
    
    train_kl_losses = []
    train_rec_losses = []
    losses = []
    print('Start of loop')
    for i in range(epochs):
        print('In first loop')
    
        lam *= lam_decay
        lam2 *= lam2_decay
    
        for j,data in enumerate(trainloader):
    
            
            images,labels = data
            batch_X = images.reshape((-1,28*28))
            
            if torch.cuda.is_available():
                batch_X = batch_X.cuda()
    
            optimizer.zero_grad()
            
            logit_q = vae.encode(batch_X, return_logits=True)
            q = torch.sigmoid(logit_q)
            
            #sample y (n samples)
            
            n_samples = 1 
            
            y = sample_binary_concrete(logit_q, lam, n_samples)
            
            #reconstruct x -> x_rec from vae.decode(torch.sigmoid(y))    
            rec_X = vae.decode(torch.sigmoid(y).transpose(1,2).reshape((-1,vae.num_bernoullis))).reshape((bsize, n_samples,-1)).transpose(1,2)    
    
            # calculate likelihood from batch_X and rec_X
            log_likelihood = -F.binary_cross_entropy_with_logits(rec_X, batch_X.unsqueeze(2).expand(rec_X.shape), reduction='none').mean(axis=1)
            
            # calculate log_density from y, logit_q and lam  --> log_dens_posterior
            log_posterior = log_density_binary_concrete(y, logit_q, lam)
            
            # calcualte log_density from y, logit_prior_q and lam2 --> log_dens_prior
            log_prior = log_density_binary_concrete(y, vae.logit_q_prior, lam2)
    
            # neg_kl = sum(log_dens_prior) - sum(log_dens_posterior)
            neg_kl = log_prior.mean(axis=1) - log_posterior.mean(axis=1)
            
            elbo = (log_likelihood + kl_rec_ratio*neg_kl).mean()
            
            loss = -elbo
            
            loss.backward()
            
            optimizer.step()
    
            train_kl_losses.append(neg_kl.mean().item())
            train_rec_losses.append(log_likelihood.mean().item())
            losses.append(loss.mean().item())
    
            if j % 100 == 0:
                print(i, j)                   
                print(loss.mean())
            if j%1000 == 0:
                rec = vae.decode(torch.sigmoid(y[:,:,0]), return_logits=False)
                plot_in_and_out(batch_X[:,:], rec[:,:])
                plt.show()
