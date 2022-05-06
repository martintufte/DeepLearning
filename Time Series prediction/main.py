# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 12:43:50 2022

@author: martigtu@stud.ntnu.no
"""


import numpy as np
import pandas as pd

from data_wrangle import preprocesser, create_sequences
from RNN import RNN
from plotting import plot_learning, plot_prediction, plot_many_predictions



if __name__=="__main__":
    
    ### READ DATA
    train = pd.read_csv('./data/no1_train.csv')
    val   = pd.read_csv('./data/no1_validation.csv')
    test  = pd.read_csv('./data/no1_test.csv')
    
    
    ### PREPROCESS
    pre = preprocesser(use_clamped_y   = True,
                       use_dt_features = True,
                       use_lag_featues = True,
                       use_alternative = False)
    input_features = pre.get_input_features()
    
    train = pre.fit_transform(train)
    val   = pre.transform(val)
    test  = pre.transform(test)
    
    # Create sequences
    n_seq = 144
    n_features = len(input_features)
    
    x_train, y_train = create_sequences(train, n_seq, input_features)
    x_val, y_val     = create_sequences(val, n_seq, input_features)
    x_test, y_test   = create_sequences(test, n_seq, input_features)
    
    
    
    
    ''' STANDARD FORECASTING '''

    rnn = RNN(n_seq, n_features, force_learn=False, file_name="RNN_standard", lrate=1e-3)
    rnn.fit((x_train, y_train), (x_val, y_val), batch_size=256, epochs=10)
    # -> Loss: 0.0152, Val_loss: 0.0165
    
    # Plot learning curve
    plot_learning(rnn, logscale=True, to_file='RNN_standard')
    
    # Plot 8 examples of predictions on the training and test set
    plot_many_predictions(rnn, x_train, y_train, n_predictions=8, n_steps=24, n_before=60, to_file='RNN_standard_train_pred')
    plot_many_predictions(rnn, x_test, y_test, n_predictions=8, n_steps=24, n_before=60, to_file='RNN_standard_test_pred')

    


    ''' TESTING ON THE STANDARD FORECAST '''
    # Testing lower learning rate (converges much slower)
    
    # LSTM with 1 hidden layer, 128 units, lrate 1e-5, 30 epochs, 64 in batch_size
    # LSTM with 1 hidden layer, 128 units, lrate 1e-4, 10 epochs, 128 in batch_size
    # LSTM with 1 hidden layer, 128 units, lrate 1e-3, 10 epochs, 256 in batch_size (fastest)
    
    rnn = RNN(n_seq, n_features, force_learn=True, file_name="RNN_standard2", lrate=1e-5)
    rnn.fit((x_train, y_train), (x_val, y_val), batch_size=64, epochs=10)
    print('\nTest with all features')
    print('Loss:', round(rnn.history.history['loss'][-1],5))
    print('Validation loss:', round(rnn.history.history['val_loss'][-1],5),'\n')
    # After 10 epochs: Loss: 0.02673 Validation loss: 0.02655
    # After 20 epochs: Loss: 0.02292 Validation loss: 0.02325
    # After 30 epochs: Loss: 0.02117 Validation loss: 0.02270
    
    
    # By increasing the learning rate and batch size, the learning become much faster.
    # Instead of using 20 minutes, now I can train for about 2.5 min to getbetter reuslts.
    # -> also better generality
    
    # I also tested with regulizer on the dense layer, but this didn't help
    
    
    ### Testing without engineered features (bit slower convergence)
    
    # without dt_features
    dt_indecies = np.in1d(range(len(input_features)), list(range(9,19)))
    x_train_dt  = x_train[:,:,np.invert(dt_indecies)]
    x_val_dt    = x_val[:,:,np.invert(dt_indecies)]
    n_features = x_train_dt.shape[-1]
    
    rnn_test1 = RNN(n_seq, n_features, force_learn=True, file_name="RNN_standard_no_dt", lrate=1e-4)
    rnn_test1.fit((x_train_dt, y_train), (x_val_dt, y_val), batch_size=64, epochs=10)
    print('\nTest with no date-time features')
    print('Loss:', round(rnn_test1.history.history['loss'][-1],5))
    print('Validation loss:', round(rnn_test1.history.history['val_loss'][-1],5),'\n')
    # After 10 epochs: Loss: 0.018 Validation loss: 0.02163 
    
    # without lag_features
    lag_indecies = np.in1d(range(len(input_features)), list(range(20,22)))
    x_train_lag  = x_train[:,:,np.invert(lag_indecies)]
    x_val_lag    = x_val[:,:,np.invert(lag_indecies)]
    n_features = x_train_lag.shape[-1]
    
    rnn_test2 = RNN(n_seq, n_features, force_learn=True, file_name="RNN_standard_no_lag", lrate=1e-4)
    rnn_test2.fit((x_train_lag, y_train), (x_val_lag, y_val), batch_size=64, epochs=10)
    print('\nTest with no lag features')
    print('Loss:', round(rnn_test2.history.history['loss'][-1],5))
    print('Validation loss:', round(rnn_test2.history.history['val_loss'][-1],5),'\n')
    # After 10 epochs: Loss: 0.01755 Validation loss: 0.02004 
    
    
    
    
    
    ''' ALTERNATIVE FORECAST '''
    
    ### READ DATA
    train = pd.read_csv('./data/no1_train.csv')
    val   = pd.read_csv('./data/no1_validation.csv')
    test  = pd.read_csv('./data/no1_test.csv')
    
    
    ### PREPROCESS
    alt_pre = preprocesser(use_clamped_y   = True,
                       use_dt_features = True,
                       use_lag_featues = True,
                       use_alternative = True)
    alt_input_features = alt_pre.get_input_features()
    
    alt_train = alt_pre.fit_transform(train)
    alt_val   = alt_pre.transform(val)
    alt_test  = alt_pre.transform(test)
    
    # Create sequences
    n_seq = 144
    n_features = len(alt_input_features)
    
    alt_x_train, alt_y_train = create_sequences(alt_train, n_seq, alt_input_features)
    alt_x_val, alt_y_val     = create_sequences(alt_val, n_seq, alt_input_features)
    alt_x_test, alt_y_test   = create_sequences(alt_test, n_seq, alt_input_features)
    
    
    
    ### Create sequences
    alt_rnn = RNN(n_seq, n_features, force_learn=False, file_name="RNN_alternative", lrate=1e-3)
    alt_rnn.fit((alt_x_train, alt_y_train), (alt_x_val, alt_y_val), batch_size=256, epochs=10)
    # -> Loss: 0.0142, Val_loss: 0.0171
    
    # Plot learning curve
    plot_learning(alt_rnn, logscale=True, to_file='RNN_alternative')
    
    # Plot 8 examples of predictions on the training and test set
    plot_many_predictions(alt_rnn, alt_x_train, alt_y_train, n_predictions=8, n_steps=24, n_before=60, to_file='RNN_alternative_train_pred')
    plot_many_predictions(alt_rnn, alt_x_test, alt_y_test, n_predictions=8, n_steps=24, n_before=60, to_file='RNN_alternative_test_pred')



