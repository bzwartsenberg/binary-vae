#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  6 09:01:13 2020

@author: berend
"""

import pandas as pd

from integrated_loss import expectation_loss
from trainer import Model_Trainer, Parameters
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch
from model import VAE_discrete


def create_table(path):
    ## create the initial table:
    num_bernoullis = [9,10]
    
    lrs = [5e-4, 1e-4, 5e-5]
    
    kl_rec_ratios = [1/780., 10/780., 100/780., 1., 10.]
    
    batch_sizes = [4,8]
    
    p_combs = []
    for num_b in num_bernoullis:
        for lr in lrs:
            for kl_rec_ratio in kl_rec_ratios:
                for batch_size in batch_sizes:
                    p_dict = {
                            'num_bernoullis' : num_b,
                            'hidden_size' : 400,
                            'lr' : lr,
                            'kl_rec_ratio' : kl_rec_ratio,
                            'batch_size' : batch_size,
                            
                            'optimizer' : 'adam',
                            'loss' : 'expectation_loss',
                            'epochs' : 2,
                            'reduction_per_epoch' : 0.1,
                            'epochs_per_step' : 2,
                              }
                    p_combs.append(p_dict)
    df = pd.DataFrame(p_combs)
    
    #add id column:
    df.insert(10, 'id', '')

    #add final_losses column:
    df.insert(11, 'final_rec_test_loss', '')
    df.insert(12, 'final_kl_test_loss', '')
    
    df.to_csv(path)
        
    
    
def append_to_table(table_path, to_append):
    ## add new elements to the table:
    pass
    
    
    
    
    
def train_on_table(table_path, working_path):
    # batch size
    # hid size/ num_b
    # parameters
    
    transform = transforms.Compose(
        [transforms.ToTensor(),
         #   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
         ])
    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                            download=True, transform=transform)
    
    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                            download=True, transform=transform)

    optimizers = {'adam' : optim.Adam}
    losses = {'expectation_loss' : expectation_loss}

    
    df = pd.read_csv(table_path)
    print(df.head())
    
    df_todo = df[df['id'].isna()]
    for index, row in df_todo.iterrows():
        p_dict = dict(row)
        
        print('Now running ')
        print(p_dict)
        
        optimizer = optimizers[p_dict['optimizer']]
        loss = losses[p_dict['loss']]

        # create training loader:
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=p_dict['batch_size'],
                                                    shuffle=True, num_workers=1)
        testloader = torch.utils.data.DataLoader(testset, batch_size=p_dict['batch_size'],
                                                    shuffle=False, num_workers=1)
        
        #model:
        model = VAE_discrete(input_size=28*28, hidden_size=p_dict['hidden_size'], num_bernoullis=p_dict['num_bernoullis'])

    
        p = Parameters(
                path=working_path, 
                lr=p_dict['lr'], 
                optimizer=optimizer, 
                loss=loss, 
                epochs=p_dict['epochs'], 
                kl_rec_ratio=p_dict['kl_rec_ratio'], 
                reduction_per_epoch=p_dict['reduction_per_epoch'], 
                epochs_per_step=p_dict['epochs_per_step']
                )
        trainer = Model_Trainer(model, trainloader, testloader, p)    
        trainer.train(verbose=True, save_image_samples_interval=100)

    
        #add id,    
        df.loc[index, 'id'] = p.id
        #add losses: 
        last_epoch = p_dict['epochs']-1
        df.loc[index, 'final_rec_test_loss'] = trainer.test_rec_histories[last_epoch].mean()
        df.loc[index, 'final_kl_test_loss'] = trainer.test_kl_histories[last_epoch].mean()
        #save dataframe:
        df.to_csv(table_path)
    
    
if __name__ == "__main__":
    
    create_table('train_table')