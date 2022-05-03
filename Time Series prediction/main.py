# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 12:43:50 2022

@author: martigtu@stud.ntnu.no
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# import preprocessing and sequence creation
from data_wrangle import preprocesser, create_sequences
from RNN import RNN



if __name__=="__main__":
    # read in data
    train = pd.read_csv('./data/no1_train.csv')
    validation = pd.read_csv('./data/no1_validation.csv')
    
    # preprocessing
    pre = preprocesser(clamp_y=True, use_date_time_features=True, use_lag_featues=True)
    train = pre.fit_transform(train)
    validation = pre.transform(validation)
    
    # create into sequences, with sequence inputs and length
    inputs = ['previous_y', 'hydro', 'micro', 'thermal', 'wind', 'river', 'total', 'sys_reg',
              'flow', 'is_winter', 'is_spring', 'is_summer', 'is_fall',
              'is_weekday', 'is_weekend', 'is_night', 'is_morning', 'is_midday',
              'is_evening', 'lag_one_day', 'lag_two_days', 'lag_one_week']
    n_seq = 144
    
    # time duration: (ca. 2 min), seq[0] is x, seq[1] is y
    train_seq = create_sequences(train, n_seq, inputs, outputs='y')
    validation_seq = create_sequences(validation, n_seq, inputs, outputs='y')
    
    
    # create first RNN
    rnn = RNN(n_seq, inputs, force_learn=True, file_name="RNN3")
    rnn.fit(train_seq, validation_seq, batch_size=128, epochs=10)
    
    # plot mse loss history
    #plt.plot(np.arange(1742*4)/1742, rnn.history.history['loss'])
    #plt.scatter(np.arange(4)+1, rnn.history.history['val_loss'], color='black', marker='x')
    #plt.yscale('log')
    #plt.show()
    
    
    # create predictions
    
    for i in range(1000, 3000, 200):
        start_idx = i
        
        prediction = rnn.multistep_prediction(train, inputs, start_idx)
        
        # plot prediction
        plt.plot(np.arange(start_idx-100, start_idx+25), train.loc[(start_idx-100):(start_idx+24), 'y'])
        plt.plot(np.arange(start_idx+1, start_idx+25), prediction)
        plt.show()




