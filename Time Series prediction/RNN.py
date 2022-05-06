# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 15:02:58 2022

@author: martigtu@stud.ntnu.no
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM

from plotting import plot_learning


class LossHistory(keras.callbacks.Callback):
    ''' To get loss history for every batch. '''
    def on_train_begin(self, logs={}):
        self.history = {'loss':[],'val_loss':[]}

    def on_batch_end(self, batch, logs={}):
        self.history['loss'].append(logs.get('loss'))

    def on_epoch_end(self, epoch, logs={}):
        self.history['val_loss'].append(logs.get('val_loss'))



class RNN:
    def __init__(self, n_seq, n_features, force_learn = False, file_name = "RNN", lrate=1e-4):
        '''
        The model is a RNN for sequence data.
        '''
        self.force_relearn = force_learn
        self.done_training = False
        self.file_name = "./models/" + file_name
        self.history = LossHistory()
        self.n_seq = n_seq
        self.n_features = n_features
        
        
        ### RNN
        input_shape = (n_seq, self.n_features) # n_seq, n_features
        rnn_input = Input( shape=input_shape )
        x = LSTM(units=128, activation='tanh', return_sequences = False)(rnn_input)
        rnn_output = Dense(1, name="output")(x)
        
        #rnn_output = Dense(1, kernel_regularizer=tf.keras.regularizers.L1(l1=1e-3), name="output")(x) # with regulizer
        
        self.model = Model(rnn_input, rnn_output, name='RNN')
        
        self.model.compile(loss="MSE", optimizer=keras.optimizers.Adam(learning_rate=lrate))
        
        # Try reading the weights from file
        self.done_training = self.load_weights()
        
    
    def load_weights(self):
        # noinspection PyBroadException
        try:
            self.model.load_weights(filepath=self.file_name)
            # print(f"Read model from file, so I do not retrain")
            done_training = True

        except:
            print("Could not read weights for verification_net from file. Must retrain...")
            done_training = False

        return done_training
    
    
    def fit(self, data, val_data, batch_size=64, epochs=1, plot=True):
        """
        Train model if required. As we have a one-channel model we take care to
        only use the first channel of the data.
        """

        if self.force_relearn or self.done_training is False:
            # Get hold of data
            x_train, y_train = data
            x_val, y_val = val_data
            
            # Fit model
            self.model.fit(x_train, y_train, batch_size, epochs, validation_data=(x_val, y_val),
                           callbacks=[self.history])

            # Save weights and leave
            self.model.save_weights(filepath=self.file_name)
            self.done_training = True
            
            if plot:
                plot_learning(self)
    
    
    def predict(self, x):
        '''
        x has shape (1, n_seq, n_features)
        '''
        return self.model.predict(x)

    
    def n_in_1_out(self, sequences, start_idx, n_steps=24):
        '''
        Implements the n in 1 out multistep predictions.
        - sequences has shape (n_samples, n_seq, n_features)
        '''
        
        model_input = sequences[[start_idx]]
        forecasts = np.zeros(n_steps)
        forecasts[0] = self.model.predict(model_input)
        
        for i in range(1, n_steps):
            model_input = sequences[[start_idx + i]]
            model_input[0, -i:, 0] = forecasts[:i]
            
            forecasts[i] = self.model.predict(model_input)
            
        return forecasts
        
    
    '''
    def fix_input_format(self, x):
        #'
        Fix the input such that it is an array with shape (1, n_seq, n_features), float
        #'
        return np.array(x, dtype=float).reshape(1, self.n_seq, self.n_features)
    
    
    def n_in_1_out_old(self, df, inputs, start_idx, n_steps=24):
        #'
        Implementation of the n in 1 out multistep predictions.
        x has shape (n, n_features)
        #'
        
        df_copy = df.copy()
        
        model_input = self.fix_input_format( df_copy.loc[start_idx:(start_idx + self.n_seq - 1), inputs] )
        
        forecasts = np.zeros(n_steps)
        
        forecasts[0] = self.model.predict(model_input)
        
        for i in range(1, n_steps):
            df_copy.loc[start_idx + i + self.n_seq - 1, 'previous_y'] = forecasts[i-1]
            
            # Extract new input from copied x. Note that all prev_y from imbalance estimates in
            # the forecast window will be replaced with forecasts iteratively
            model_input = self.fix( df_copy.loc[(start_idx + i):(start_idx + i + self.n_seq - 1), inputs] )
            
            forecasts[i] = self.model.predict(model_input)
        
        return forecasts
    '''
    
    
    
    
    
    
    