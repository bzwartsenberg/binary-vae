#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  5 08:24:52 2020

@author: berend
"""
import torch
import numpy as np
from plotting import save_and_print_images
import os
from integrated_loss import expectation_loss
import torch.optim as optim
import time

class Model_Trainer():
    
    def __init__(self, model, train_data, test_data, parameters):
        
        
        self.model = model
        self.train_data = train_data
        
        self.test_data = test_data
        
        
        self.p = parameters
        
        #these come as part of parameters so they can be saved
        self.loss = self.p.loss
        self.optimizer = self.p.optimizer(model.parameters(), lr=self.p.lr)
        self.scheduler = self.p.scheduler(self.optimizer, self.p.epochs_per_step, gamma=self.p.reduction_per_epoch)
        
        self.train_rec_history = []
        self.train_kl_history = []
        
        self.test_rec_histories = {}
        self.test_kl_histories = {}






    def train(self, verbose=True, save_image_samples_interval=None):
        #train
        if verbose:
            print('Starting training')
            
        #change optimizer for every epoch? --> use scheduler, see: https://pytorch.org/docs/master/optim.html
        for i in range(self.p.epochs):           
            print('Starting epoch {}'.format(i))
            
            for j,data in enumerate(self.train_data):
                images,labels = data
                batch_X = images.reshape((-1,28*28))
                self.optimizer.zero_grad()

                logit_q = self.model.encode(batch_X, return_logits=True)
                q = torch.sigmoid(logit_q)
        
                #for under/over flow issues
                if torch.any(torch.isinf(q)):
                    print('Infinity found')
                    break
                if torch.any(torch.isnan(q)):
                    print('Nan found')
                    break
        
                total, rec, kl = self.loss(batch_X,
                                           logit_q,
                                           self.model.logit_q_prior,
                                           self.model.decode,
                                           self.p.kl_rec_ratio)
        
                self.train_rec_history.append(rec.item())
                self.train_kl_history.append(kl.item())
        
                #maximizing the total, minimizing the -total
                loss = -total
                loss.backward()
                self.optimizer.step()

                if j%100 == 0:
                    print(j)
                    if verbose:    
                        print('total loss: ', total.item())
                        print('rec loss: ', self.train_rec_history[-1])
                        print('KL loss: ', self.train_kl_history[-1])
        

                if save_image_samples_interval is not None:        
                    if j%save_image_samples_interval == 0:
                        x_rec = self.model.forward(batch_X, sample=True)
                        savename = self.p.image_save_path + '{}_epoch_{}_batch_{}.png'.format(self.p.id, i, j)
                        save_and_print_images(batch_X, x_rec, savename, show = True)
            #stepping learn rate scheduler
            self.scheduler.step()
            test_rec_history, test_kl_history = self.test_on_set(self.test_data)
            print('Epoch {} test rec loss mean (std): {} ({})'.format(i, test_rec_history.mean(), test_rec_history.std()))
            print('Epoch {} test kl loss mean (std): {} ({})'.format(i, test_kl_history.mean(), test_kl_history.std()))
            
            self.test_rec_histories[i] = test_rec_history
            self.test_rec_histories[i] = test_rec_history
        
    def test_on_set(self, test_set):
        ### change!
        test_rec_history = []
        test_kl_history = []
        with torch.no_grad():
            for j,data in enumerate(test_set):
                images,labels = data
                batch_X = images.reshape((-1,28*28))
    
                logit_q = self.model.encode(batch_X, return_logits=True)
                q = torch.sigmoid(logit_q)
        
                #for under/over flow issues
                if torch.any(torch.isinf(q)):
                    print('Infinity found')
                    break
                if torch.any(torch.isnan(q)):
                    print('Nan found')
                    break
        
                total, rec, kl = self.loss(batch_X,
                                           logit_q,
                                           self.model.logit_q_prior,
                                           self.model.decode,
                                           self.p.kl_rec_ratio)
        
                test_rec_history.append(rec.item())
                test_kl_history.append(kl.item())
                if j % 100 == 0:
                    print(j)
        
        return np.array(test_rec_history), np.array(test_kl_history)
        
    
    def save(self):
        #1 save model
        torch.save(self.model.state_dict(), self.p.save_path + 'model')
        #2 save history
        np.savetxt(self.p.save_path + 'train_rec_losses', self.train_rec_history)
        np.savetxt(self.p.save_path + 'train_kl_losses', self.train_kl_history)
        for i in self.test_rec_histories.keys():
            np.savetxt(self.p.save_path + 'epoch_{}_test_rec_losses'.format(i), self.test_rec_histories[i])
            np.savetxt(self.p.save_path + 'epoch_{}_test_kl_losses'.format(i), self.test_kl_histories[i])
            
        #3 save parameters
        self.p.save()
        
        
        
# to do:
        #- use lr scheduler
        # make a save_info function
        #fill in rest of parameters
        # make a train_scheduler --> basically create a table of hyperparameters, 
        #then go through that table, generating an id, training, and when training is complete add the id to the table
        #
        
class Parameters():
    
    
    def __init__(self, path='./', lr=1e-4, optimizer=optim.Adam, loss=expectation_loss, 
                 epochs=2, kl_rec_ratio=0.12, reduction_per_epoch=0.1, epochs_per_step=2):
        
        self.id = 'asdf' ##
        self.path = path
        if not os.path.exists(self.path):
            os.mkdir(self.path)        
        self.save_path = self.path + self.id + '/'
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)
        self.image_save_path = self.save_path + 'imgs/'
        if not os.path.exists(self.image_save_path):
            os.mkdir(self.image_save_path)
        self.lr = lr
        self.optimizer = optimizer
        self.loss = loss
        self.epochs = epochs
        self.kl_rec_ratio = kl_rec_ratio
        #fix this for now:
        self.scheduler = optim.lr_scheduler.StepLR
        
        self.reduction_per_epoch = reduction_per_epoch
        self.epochs_per_step = epochs_per_step
        
    
    def save(self, filename):
        #print an overview of all parameters
        pass
    
    
    
        
    
        
        
    
    