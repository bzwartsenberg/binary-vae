#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  5 21:24:49 2020

@author: berend
"""
import matplotlib.pyplot as plt

def save_and_print_images(images, rec_images, savename, show=True):

    max_im = 4
    num_im = min(images.shape[0], max_im)
    fig,ax = plt.subplots(num_im,2, figsize = (1,2))
    for i in range(num_im):
        ax[i][0].imshow(images[i].reshape(28,28), cmap='Greys')
        ax[i][1].imshow(rec_images[i].detach().numpy().reshape(28,28), cmap='Greys')
        ax[i][0].set_xticks([])
        ax[i][0].set_yticks([])
        ax[i][1].set_xticks([])
        ax[i][1].set_yticks([])
        
    plt.savefig(savename)
    if show:
        plt.show()