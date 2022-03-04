# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 13:00:53 2022

@author: martigtu@stud.ntnu.no
"""

import numpy as np
import matplotlib.pyplot as plt

# Data generator
from tensorflow.keras.datasets import mnist

# Tensorflow stuff
from tensorflow import keras
from tensorflow.keras import backend as K # standard practice
import tensorflow_probability as tfp
import tensorflow as tf
tfd = tfp.distributions

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, \
    Conv2DTranspose, Lambda, Reshape, MaxPooling2D, Dropout



### As simple VAE as possible
latent_dim = 2
n_channels = 1
input_shape = (28, 28, n_channels)



### DEFINING THE ENCODER

# Function to sample from the latent distribution
def sample_Gaussian(params):
  ''' Sample from the latent Gaussian distribution Z. '''  
  # Parameters are the mean and log variance
  mu, log_sigma_sq = params
  # Error term from standard Gaussian
  eps = K.random_normal( shape = K.int_shape(mu)[1:] )
  return mu + K.exp(log_sigma_sq / 2) * eps


# Prior distribution for the Kullback Leibler divergence
#prior = tfd.Independent(tfd.Normal(loc=tf.zeros(latent_dim), scale=1),
#                        reinterpreted_batch_ndims=1)

# Encoder inputs
encoder_input = Input(shape=input_shape, name="Encoder_input")
# Convolution part of encoder network
x = Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu')(encoder_input)
#x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)
x = Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu')(x)
conv_shape = K.int_shape(x)[1:]
x = Flatten()(x)
mu = Dense(latent_dim, name='Latent_mean')(x)
log_sigma_sq = Dense(latent_dim, name='Latent_log_variance')(x)
# The stochastic layer Z
z_inputs = (mu, log_sigma_sq)
z = Lambda(sample_Gaussian, output_shape = (latent_dim, ), name='Z')(z_inputs)
# Encoder outputs
encoder_outputs = [mu, log_sigma_sq, z]
encoder = Model(encoder_input, encoder_outputs, name='Encoder')
encoder.summary()



### DEFINING THE DECODER

decoder_input = Input(shape=(latent_dim, ), name="Decoder_input")

# Reshape the latent shape to the convolational shape (height, width, n_channels)
x = Dense(conv_shape[0]*conv_shape[1]*conv_shape[2], activation='relu')(decoder_input)
x = Reshape(target_shape=(conv_shape[0], conv_shape[1], conv_shape[2]))(x)

# Convolution part of decoder network
x = Conv2DTranspose(32, kernel_size=(3,3), padding='same', activation='relu')(x)
x = Conv2DTranspose(32, kernel_size=(3,3), padding='same', activation='relu')(x)

# Use sigmoid activation to get predictions for each pixel
decoder_output = Conv2DTranspose(n_channels, kernel_size=(3,3), padding='same',\
                                 activation='sigmoid', name="X_tilde")(x)

decoder = Model(decoder_input, decoder_output, name='Decoder')
decoder.summary()



### DEFINING THE VARIATIONAL AUTOENCODER
class Layer_customloss(keras.layers.Layer):
    '''
    A layer with custom loss function. (inherits from keras.layers.Layer)
    '''

    def vae_loss(self, x, x_tilde):
        # Flatten X and prediction X tilde
        x = K.flatten(x)
        x_tilde = K.flatten(x_tilde)
        
        # Reconstruction loss using binary crossentropy
        cross_entropy_loss = keras.metrics.binary_crossentropy(x, x_tilde)
        
        # Kulback-Leibler divergence loss
        #KL_loss = -5e-4 * K.mean(1 + log_sigma_sq - K.square(mu) - K.exp(log_sigma_sq), axis=-1)
        
        return K.mean(cross_entropy_loss)

    # Add custom loss to the class, loss will be added in the feedforward process
    def call(self, inputs):
        # X is the first input
        x = inputs[0]
        # X_tilde is the second input
        x_tilde = inputs[1]
        
        # Calculate VAE loss
        loss = self.vae_loss(x, x_tilde)
        
        # Add loss to the layer
        self.add_loss(loss, inputs=inputs)  
        return x


# Apply the custom loss to the input images and the decoded latent distribution sample
x_tilde = Layer_customloss()([encoder_input, decoder(z)])

# The VAE
vae = Model(encoder_input, decoder(z), name='vae')

# Compile VAE
vae.compile(optimizer='adam', loss="binary_crossentropy")
vae.summary()



### Fit the VAE
vae.fit(x_train, x_train, epochs = 3, batch_size = 32, validation_split = 0.2)



### Display imgs
def display_imgs(x, n_rand=10):
    ''' Display 10 random imgs from x. '''
    if not isinstance(x, (np.ndarray, np.generic)):
        x = np.array(x)
    
    n = x.shape[0]
    fig, axs = plt.subplots(1, n, figsize=(1.5*n, 1.5))
    for i in range(n):
        axs.flat[i].imshow(1-x[i].squeeze(), cmap='gray')
        axs.flat[i].axis('off')
    plt.show()

X = x_train[:10]
Xhat = vae(X)
display_imgs(X)
display_imgs(Xhat)



mu, _, _ = encoder.predict(x_test)

# Plot dim1 and dim2 for mu
plt.figure(figsize=(10, 10))
plt.scatter(mu[:, 0], mu[:, 1], c=y_test, cmap='brg')
plt.xlabel('Latent dimension 1')
plt.ylabel('Latent dimension 2')
plt.colorbar()
plt.show()