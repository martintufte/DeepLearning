# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 12:43:50 2022

@author: martigtu@stud.ntnu.no
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from RNN import RNN
from data_wrangle import preprocesser, create_sequences



if __name__=="__main__":
    
    ### Read the data
    train = pd.read_csv('./data/no1_train.csv')
    validation = pd.read_csv('./data/no1_validation.csv')
    #test = pd.read_csv('./data/no1_test.csv')
    
    
    ### Preprocess
    pre = preprocesser(use_clamped_y = True,
                       use_date_time_features = True,
                       use_lag_featues = True,
                       use_alternative = True)
    
    train = pre.fit_transform(train)
    validation = pre.transform(validation)
    #test = pre.transform(test)
    
    # input features, previous_y index
    inputs = list(train.columns)
    inputs = [i for i in inputs if i not in {'index', 'start_time', 'y'}]
    prev_y_idx = np.where(np.array(inputs)=='previous_y')[0]
    
    
    
    ### Create sequences -> np.genfromtxt('train_seq.csv', delimiter=',')
    x_train, y_train = create_sequences(train, 144, inputs)
    x_val, y_val = create_sequences(validation, 144, inputs)
    #x_test, y_test = create_sequences(test, n_seq = 144)
    _, n_seq, n_features = x_train.shape

    
    
    ### Hold-out data set

    
    
    ### LSTM with 1 hidden layer, 128 units, lrate 1e-4, 30 epochs, 64 in batch_size
    ###           some variables are min-maxed, others are standardized
    rnn2 = RNN(n_seq, n_features, force_learn=False, file_name="RNN_8", lrate=1e-4)
    rnn2.fit((x_train, y_train), (x_val, y_val), batch_size=64, epochs=20)
    
    
    
    
    
    for i in range(1000, 3000, 200):
        start_idx = i
        n_steps = 24
        
        prediction = rnn2.n_in_1_out(x_train, prev_y_idx, start_idx, n_steps)
        
        plt.plot(y_train[start_idx:(start_idx + n_steps)])
        plt.plot(prediction)
        plt.show()


