# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 14:38:00 2022

@author: Teksle
"""

# Basic libraries
import numpy as np
import matplotlib.pyplot as plt

# Data generator
from stacked_mnist import StackedMNISTData, DataMode

# Tensorflow stuff
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K # standard practice

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, \
    Conv2DTranspose, Lambda, Reshape, MaxPooling2D, Dropout



class Data:
    '''
    Object that contains the training data
    '''
    def __init__(self, mode=DataMode.COLOR_BINARY_COMPLETE):
        ### Generate data from mode
        gen = StackedMNISTData(mode)
        
        ### Store training/test set
        self.x_train, self.y_train = gen.get_full_data_set(training=True)
        self.x_test, self.y_test = gen.get_full_data_set(training=False)
        # Input shape: (n_samples, height, width, n_channels)
        # Output shape: (n_smaples, )
        
        ### Store input dimensions
        self.input_shape = self.x_train.shape[1:]
        self.img_height = self.input_shape[0]
        self.img_width  = self.input_shape[1]
        self.n_channels = self.input_shape[2]


# Constants
data = Data()
latent_dim = 2



## DEFINING THE ENCODER

# Define the input to the encoder network
encoder_input = Input(shape=data.input_shape, name="Encoder_input")
# Convolution part of encoder network
X_enc = Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu')(encoder_input)
X_enc = Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu')(X_enc)
X_enc = Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu')(X_enc)

# Store convolution shape (n_samples, height, width, n_channels)
conv_shape = K.int_shape(X_enc)

# Flatten the image to a single array
X_enc = Flatten()(X_enc)

#X = Dense(32, activation='relu')(X)

# Latent mean and latent log variance
mu           = Dense(latent_dim, name='Latent_mean')(X_enc)
log_sigma_sq = Dense(latent_dim, name='Latent_log_variance')(X_enc)


def Z_sampler(inputs):
  '''
  Sample from the latent distribution Z using the reparameterization trick.
  "inputs" are on the form (mu, log_sigma_sq)
  
  '''  
  mu, log_sigma_sq = inputs
  
  # Error term
  eps = K.random_normal( shape = K.int_shape(mu)[1:] )
  
  return mu + K.exp(log_sigma_sq / 2) * eps


# The Z layer
Z_inputs = [mu, log_sigma_sq]
Z = Lambda(Z_sampler, output_shape = (latent_dim, ), name='Z')(Z_inputs)


# Encoder outputs
encoder_outputs = [mu, log_sigma_sq, Z]

encoder = Model(encoder_input, encoder_outputs, name='Encoder')
encoder.summary()



## DEFINING THE DECODER
decoder_input = Input(shape=(latent_dim, ), name="Decoder_input")

# Reshape the latent shape to the convolational shape (height, width, n_channels)
X_dec = Dense(conv_shape[1]*conv_shape[2]*conv_shape[3], activation='relu')(decoder_input)
X_dec = Reshape(target_shape=(conv_shape[1], conv_shape[2], conv_shape[3]))(X_dec)

# Convolution part of decoder network
X_dec = Conv2DTranspose(32, kernel_size=(3,3), padding='same', activation='relu')(X_dec)
X_dec = Conv2DTranspose(32, kernel_size=(3,3), padding='same', activation='relu')(X_dec)

# Use sigmoid activation to get predictions for each pixel
decoder_output = Conv2DTranspose(n_channels, kernel_size=(3,3), padding='same',
                                  activation='sigmoid', name="X_tilde")(X_dec)

decoder = Model(decoder_input, decoder_output, name='Decoder')
decoder.summary()



## DEFINING THE LOSS FUNCTION USING A CUSTOM LAYER
class Layer_customloss(keras.layers.Layer):
    '''
    A layer with custom loss function. (inherits from keras.layers.Layer)
    '''

    def vae_loss(self, X, X_tilde):
        # Flatten X and prediction X tilde
        X = K.flatten(X)
        X_tilde = K.flatten(X_tilde)
        
        # Reconstruction loss using binary crossentropy
        x_entropy_loss = keras.metrics.binary_crossentropy(X, X_tilde)
        
        # Kulback-Leibler divergence loss
        KL_loss = -5e-4 * K.mean(1 + log_sigma_sq - K.square(mu) - K.exp(log_sigma_sq), axis=-1)
        
        return K.mean(x_entropy_loss + KL_loss)

    # Add custom loss to the class, loss will be added in the feedforward process
    def call(self, inputs):
        # X is the first input
        X = inputs[0]
        # X_tilde is the second input
        X_tilde = inputs[1]
        
        # Calculate VAE loss
        loss = self.vae_loss(X, X_tilde)
        
        # Add loss to the layer
        self.add_loss(loss, inputs=inputs)
        
        return X


# Apply the custom loss to the input images and the decoded latent distribution sample
X_tilde = Layer_customloss()([encoder_input, decoder(Z)])

# The VAE
vae = Model(encoder_input, X_tilde, name='vae')

# Compile VAE
vae.compile(optimizer='adam', loss=None)
vae.summary()


# Train autoencoder
vae.fit(x_train, None, epochs = 10, batch_size = 32, validation_split = 0.2)

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    