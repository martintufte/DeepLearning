# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 15:02:58 2022

@author: martigtu@stud.ntnu.no
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Reshape, LSTM
    


class LossHistory(keras.callbacks.Callback):
    ''' To get loss history for every batch. '''
    def on_train_begin(self, logs={}):
        self.history = {'loss':[],'val_loss':[]}

    def on_batch_end(self, batch, logs={}):
        self.history['loss'].append(logs.get('loss'))

    def on_epoch_end(self, epoch, logs={}):
        self.history['val_loss'].append(logs.get('val_loss'))



class RNN:
    def __init__(self, n_seq, n_features, force_learn = False, file_name = "RNN"):
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
        
        input_shape = (n_seq, n_features) # n_seq, n_features
        rnn_input = Input( shape=input_shape )
        x = LSTM(units=32, activation='tanh', return_sequences = False)(rnn_input)
        rnn_output = Dense(1, name="output")(x)
        self.model = Model(rnn_input, rnn_output, name='RNN')
        
        #self.model.compile(loss="MSE", optimizer=keras.optimizers.SGD(learning_rate=1e-3))
        self.model.compile(loss="MSE", optimizer=keras.optimizers.Adam(learning_rate=1e-3))
        
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
    
    
    def fit(self, data, val_data, batch_size=64, epochs=1):
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
    
    
    def predict(self, x):
        return self.model.predict(x)


    def multistep_prediction(self, df, inputs, start_idx, n_steps=24):
        
        # TODO
        
        pass