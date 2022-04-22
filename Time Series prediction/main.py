# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 12:43:50 2022

@author: martigtu
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

    
# import preprocessing and sequence creation
from data_wrangle import preprocess, create_sequences
from RNN import RNN




if __name__=="__main__":
    # read in data
    train = pd.read_csv('./data/no1_train.csv')
    validation = pd.read_csv('./data/no1_validation.csv')
    
    # preprocess    
    preprocess(train, validation)
        
    # create into sequences
    inputs = ['hydro', 'micro', 'thermal', 'wind', 'river', 'total', 'sys_reg', 'flow', 'previous_y']
    n_features = len(inputs)
    n_seq = 144
    train_seq = create_sequences(train, n_seq, inputs, 'y')
    validation_seq = create_sequences(validation, n_seq, inputs, 'y')
    
    
    # create first RNN
    rnn = RNN(n_seq, n_features, force_learn=True, file_name="RNN")
    rnn.fit(train_seq, validation_seq, batch_size=64, epochs=4)
    
    # plot mse loss history
    plt.plot(np.arange(3515*4)/3515, rnn.history.history['loss'])
    plt.scatter(np.arange(4)+1, rnn.history.history['val_loss'], color='black', marker='x')
    plt.yscale('log')
    plt.show()
    
    
    
    
    