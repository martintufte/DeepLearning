# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 15:03:23 2022

@author: martigtu@stud.ntnu.no
"""


import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt



def plot_learning(model, logscale = False, to_file=''):
    '''
    Plot the learning curve over the epochs.
    '''
    loss = model.history.history['loss']            # calculated after every mini-batch
    val_loss = model.history.history['val_loss']    # calculated after every epoch
    
    total_batches = len(loss)
    epochs = len(val_loss)
    batches_per_epoch = int(total_batches / epochs)
    
    plt.plot(np.arange(1,total_batches+1)/batches_per_epoch, loss, label='training loss')
    plt.scatter(np.arange(1,epochs+1), val_loss, color='k', marker='x', label='validation loss')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    if logscale:
        plt.yscale('log') 
    if to_file != '':
        plt.savefig('./plots/'+to_file+'.pdf')
    plt.show()



def plot_prediction(model, x, y, start_idx, n_steps=24, n_before=60, to_file=''):
    '''
    Plot 2-hour forecast for the model
    '''
    # Check that the start index is good
    assert start_idx >= max(144, n_before), 'Too low start index.'
    assert start_idx <= x.shape[0]-n_steps, 'Too high start index.'
    
    
    # get prediction from model n_steps ahead
    prediction = model.n_in_1_out(x, start_idx, n_steps)
    
    plt.plot(range(1-n_before, 1), y[(start_idx-n_before):start_idx], label='hist')
    plt.plot(range(0,n_steps+1), y[(start_idx-1):(start_idx + n_steps)], label='target')
    plt.plot(range(1,n_steps+1), prediction, 'g', label='pred')
    plt.legend()
    plt.xlabel('Time (relative start index)')
    plt.ylabel('y')
    if to_file != '':
        plt.savefig('./plots/'+to_file+'.pdf')
    plt.show()



def plot_many_predictions(model, x, y, n_predictions, n_steps=24, n_before=60, to_file='', seed=1337):
    '''
    Plot multiple predictions at once with random starting indecies.
    '''
    if seed:
        np.random.seed(seed)
        
    random_indecies = np.random.randint(max(144, n_before), x.shape[0]-n_steps, n_predictions)
    
    for i in random_indecies:
        if to_file != '':
            plot_prediction(model, x, y, i, n_steps, n_before, to_file+'_'+str(i))
        else:
            plot_prediction(model, x, y, i, n_steps, n_before)



