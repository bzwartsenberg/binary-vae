#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  4 17:46:18 2020

@author: berend
"""

import torch.nn.functional as F
import torch
import itertools

def get_binaries(num_bernoullis):
    """
    Get all binary vectors for num_bernoullis
    Note: creates an array that grows exponentially in memory
    Args:
        num_bernoullis: number of Bernoulli variables
    Returns:
        array (1, num_bernoullis, 2^num_bernoullis)
    """
    return torch.tensor(list(itertools.product([0, 1], repeat = num_bernoullis))).unsqueeze(1).float()

#losses:
def neg_DKL(logit_q, logit_q_prior, b):
    """
    Return the argument of the negative DKL loss, i.e.
    log(p(b)/q(b|x)) = -log(q(b|x)/p(b)) = -DKL

    Args:
        logit_q: the logit probabilities according to q_phi(b | x)
        logit_q_prior: logit probabilities from the prior --> this can be a variable too, but indep. of x
        b: bernoulli vectors
            all shapes: (batch,num_bernoulli,num_binary_vecs)
    Returns:
        -DKL argument, shape: (batch,)"""

    #note:, below is equivalent to: torch.log(q*b + (1-b)*(1-q)), but using the logit of q, so should be numerically stable
    log_prob_q = -F.binary_cross_entropy_with_logits(logit_q, b, reduction='none')
    log_prob_prior = -F.binary_cross_entropy_with_logits(logit_q_prior, b, reduction='none')

    return (log_prob_prior - log_prob_q).mean(axis = 1) #return difference, summed over bernoullis, returns (batch, num_binary_vecs)

def log_likelihood(logits, x):
    """
    Log Likelihood term, or reconstruction
    Args:
        logits: predicted logit probability p(x|b)
        x: input data
            all shapes: (batch, in_shape, num_binary_vecs)

    """
    # note, BCE is: 
    #-x*torch.log(prob) - (1-x)*torch.log((1-prob))
    #for negative log-l we need negative that. Only sum over in_shape.
    # now using with logits because of underflow
    log_likelihood = -F.binary_cross_entropy_with_logits(logits, x, reduction='none').mean(axis=1)
    return log_likelihood


def loss_arg(x, b, logit_q, logit_q_prior, logit_p):
    """
    The loss argument term to take the expectation over
    Args:
        x: input data
            shape: (batch, in_shape, num_binary_vecs)
        b: the binary vector for which this sample is calculated
            shape: (batch,num_bernoulli,num_binary_vecs)
        logit_q: logit probabilities q(b|x). Note this is independent of b, 
            so should be calculated only once at the start of the loss loop
            shape: (batch,num_bernoulli,num_binary_vecs)
        logit_q_prior: logit probabilities of the prior
            shape: (batch,num_bernoulli,num_binary_vecs)
        logit_p: calculated logits for p(x|b)
            shape: (batch, in_shape, num_binary_vecs)
    Returns:
        rec_loss, kl_loss
            shape: (batch,)
    """
    rec_loss = log_likelihood(logit_p,x)
    kl_loss = neg_DKL(logit_q, logit_q_prior, b) 
    return rec_loss, kl_loss


def expectation_loss(x,logit_q,logit_q_prior,logit_p_x_given_b, kl_rec_ratio, f=loss_arg):
    """
    Args:
        x: input data
            shape: (batch,in_shape)
        logit_q: logit probability distribution of n bernoulli variables
            shape: (batch, num_bernoullis)
        logit_q_prior: logit probability distribution of n bernoulli variables for the prior
            shape: (1, num_bernoullis)
        logit_p_x_given_b: a function that returns logit_p(x|b)
        kl_rec_ratio: a number that specifies the ratio of kl_loss to rec_loss. Note that both are taken as a mean, not a sum.
                      i.e., for normal 28*28 input and 10 binary latents, use 10/(28*28)
        f: a function that takes x, b, q, q_prior, logit_p as argument, x is input,
            b is bernoulli vector, q is q(b|x), q_prior is the prior, logit_p is rec. probability logit
            shape: see "loss_arg"
            
            """
    binary_vecs = get_binaries(logit_q.shape[1]) #size of bernoulli's
    total_kl_loss = torch.zeros(logit_q.shape[0]) #size of batch
    total_rec_loss = torch.zeros(logit_q.shape[0]) 
    total_weight = torch.zeros(logit_q.shape[0])

    #first, need to get all the logit_p's for binary vecs. Therefore, need all binary vecs in 
    #dim[0], as (num_binary_vecs, num_binaries)
    logit_p = logit_p_x_given_b(binary_vecs.squeeze())
    #above gives logit_p as (num_binaries,in_size), but need (1,num_binaries,num_binary_vecs)
    logit_p = logit_p.transpose(1,0).unsqueeze(0)
 
    #same for binary_vecs:
    binary_vecs = binary_vecs.squeeze().transpose(1,0).unsqueeze(0)

    #add dimension for binary_vecs:
    logit_q = logit_q.unsqueeze(2)
    logit_q_prior = logit_q_prior.unsqueeze(2)
    x = x.unsqueeze(2)

    
    #expanding for KL loss:, because crossentropy function does not broadcast
    logit_q = logit_q.expand(-1,-1,binary_vecs.shape[2])
    binary_vecs = binary_vecs.expand(*logit_q.shape)
    logit_q_prior = logit_q_prior.expand(*logit_q.shape)
    
    #expanding for rec. loss:
    logit_p = logit_p.expand(x.shape[0],-1,-1)
    x = x.expand(-1,-1,logit_p.shape[2])    
    
    #the log_prob, this is equivalent to: prob = q*binary_vecs + (1-binary_vecs)*(1-q), but with logits
    log_prob = -F.binary_cross_entropy_with_logits(logit_q, binary_vecs, reduction='none')

    # and sum all the binary terms:
    log_prob = torch.sum(log_prob, dim=1, keepdims=False)
    
    # f should return rec_loss and kl_loss as:
    #(batch, num_binary_vecs)
    rec_loss, kl_loss = f(x, binary_vecs, logit_q, logit_q_prior, logit_p)

    #could sum to the logarithm if getting under/overflow issues
    total_rec_loss = (rec_loss*torch.exp(log_prob)).sum(axis=1)
    total_kl_loss = (kl_loss*torch.exp(log_prob)).sum(axis=1)

    #sanity check
    tol = 1e-5
    total_weight = torch.exp(log_prob).sum(axis=(1))
    if torch.any(torch.abs(total_weight - torch.ones_like(total_weight))>tol):
        print('Probs dont sum to one')
        print(total_weight)
        
    total_loss =  total_rec_loss + total_kl_loss*kl_rec_ratio
    return total_loss.mean(), total_rec_loss.mean(), total_kl_loss.mean()




