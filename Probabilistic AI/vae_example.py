# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 16:41:13 2022

@author: Teksle
"""

#from stacked_mnist import StackedMNISTData, DataMode
from tensorflow.keras.datasets import mnist

from tensorflow import keras
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, \
    Conv2DTranspose, Lambda, Reshape
from tensorflow.keras import backend

import numpy as np
import matplotlib.pyplot as plt


def load_MNIST():
    ''' Loading the data '''
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    # Normalize
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train = x_train / 255
    x_test = x_test / 255
    
    # Reshape 
    img_width  = x_train.shape[1]
    img_height = x_train.shape[2]
    n_channels = 1
    
    x_train = x_train.reshape(x_train.shape[0], img_height, img_width, n_channels)
    x_test = x_test.reshape(x_test.shape[0], img_height, img_width, n_channels)
    
    return x_train, x_test, y_train, y_test, img_width, img_height, n_channels


# Build the variational autoencoder


### Encoder
x_train, x_test, y_train, y_test, img_width, img_height, n_channels = load_MNIST()

# Number of latent features
latent_dim = 2
input_shape = (img_width, img_height, n_channels)


# Method 1
input_img = Input(shape = input_shape, name = 'Encoder_input')
x = Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu')(input_img)
x = Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu',strides=(2, 2))(x)
x = Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu')(x)
x = Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu')(x)

# Shape of conv to be provided to decoder
conv_shape = backend.int_shape(x)

x = Flatten()(x)
x = Dense(32, activation='relu')(x)

# Two outputs, for latent mean and log variance
mu         = Dense(latent_dim, name='latent_mean')(x)
log_sigma2 = Dense(latent_dim, name='latent_log_variance')(x)

# REPARAMETERIZATION TRICK
def sample_Z(args):
  mu, log_sigma2 = args
  eps = backend.random_normal(shape=(backend.shape(mu)[0], backend.int_shape(mu)[1]))
  return mu + backend.exp(log_sigma2 / 2) * eps

z = Lambda(sample_Z, output_shape=(latent_dim, ), name='Z')([mu, log_sigma2])

encoder = Model(input_img, [mu, log_sigma2, z], name='Encoder')
print(encoder.summary())






### Decoder

# decoder takes the latent vector as input
decoder_input = Input(shape=(latent_dim, ), name='decoder_input')

# Need to start with a shape that can be remapped to original image shape as
# we want our final utput to be same shape original input.
# So, add dense layer with dimensions that can be reshaped to desired output shape
x = Dense(conv_shape[1]*conv_shape[2]*conv_shape[3], activation='relu')(decoder_input)
# reshape to the shape of last conv. layer in the encoder, so we can 
x = Reshape((conv_shape[1], conv_shape[2], conv_shape[3]))(x)
# upscale (conv2D transpose) back to original shape
# use Conv2DTranspose to reverse the conv layers defined in the encoder
x = Conv2DTranspose(32, 3, padding='same', activation='relu',strides=(2, 2))(x)
#Can add more conv2DTranspose layers, if desired. 
#Using sigmoid activation
x = Conv2DTranspose(num_channels, 3, padding='same', activation='sigmoid', name='decoder_output')(x)

# Define and summarize decoder model
decoder = Model(decoder_input, x, name='decoder')
decoder.summary()

# apply the decoder to the latent sample 
z_decoded = decoder(z)


# =========================
# Define custom loss
# VAE is trained using two loss functions reconstruction loss and KL divergence
class CustomLayer(keras.layers.Layer):
    ''' Let us add a class to define a custom layer with loss '''

    def vae_loss(self, x, z_decoded):
        x = K.flatten(x)
        z_decoded = K.flatten(z_decoded)
        
        # Reconstruction loss (as we used sigmoid activation we can use binarycrossentropy)
        recon_loss = tensorflow.keras.metrics.binary_crossentropy(x, z_decoded)
        
        # KL divergence
        kl_loss = -5e-4 * K.mean(1 + z_sigma - K.square(z_mu) - K.exp(z_sigma), axis=-1)
        return K.mean(recon_loss + kl_loss)

    # add custom loss to the class
    def call(self, inputs):
        x = inputs[0]
        z_decoded = inputs[1]
        loss = self.vae_loss(x, z_decoded)
        self.add_loss(loss, inputs=inputs)
        return x

# apply the custom loss to the input images and the decoded latent distribution sample
y = CustomLayer()([input_img, z_decoded])
# y is basically the original image after encoding input img to mu, sigma, z
# and decoding sampled z values.
#This will be used as output for vae

# The VAE
vae = Model(input_img, y, name='vae')

# Compile VAE
vae.compile(optimizer='adam', loss=None)
vae.summary()

# Train autoencoder
vae.fit(x_train, None, epochs = 10, batch_size = 32, validation_split = 0.2)

# =================
# Visualize results
# =================
#Visualize inputs mapped to the Latent space
#Remember that we have encoded inputs to latent space dimension = 2. 
#Extract z_mu --> first parameter in the result of encoder prediction representing mean

mu, _, _ = encoder.predict(x_test)

# Plot dim1 and dim2 for mu
plt.figure(figsize=(10, 10))
plt.scatter(mu[:, 0], mu[:, 1], c=y_test, cmap='brg')
plt.xlabel('dim 1')
plt.ylabel('dim 2')
plt.colorbar()
plt.show()


# Visualize images
#Single decoded image with random input latent vector (of size 1x2)
#Latent space range is about -5 to 5 so pick random values within this range
#Try starting with -1, 1 and slowly go up to -1.5,1.5 and see how it morphs from 
#one image to the other.
sample_vector = np.array([[1,-1]])
decoded_example = decoder.predict(sample_vector)
decoded_example_reshaped = decoded_example.reshape(img_width, img_height)
plt.imshow(decoded_example_reshaped)

#Let us automate this process by generating multiple images and plotting
#Use decoder to generate images by tweaking latent variables from the latent space
#Create a grid of defined size with zeros. 
#Take sample from some defined linear space. In this example range [-4, 4]
#Feed it to the decoder and update zeros in the figure with output.


n = 20  # generate 15x15 digits
figure = np.zeros((img_width * n, img_height * n, num_channels))

#Create a Grid of latent variables, to be provided as inputs to decoder.predict
#Creating vectors within range -5 to 5 as that seems to be the range in latent space
grid_x = np.linspace(-5, 5, n)
grid_y = np.linspace(-5, 5, n)[::-1]

# decoder for each square in the grid
for i, yi in enumerate(grid_y):
    for j, xi in enumerate(grid_x):
        z_sample = np.array([[xi, yi]])
        x_decoded = decoder.predict(z_sample)
        digit = x_decoded[0].reshape(img_width, img_height, num_channels)
        figure[i * img_width: (i + 1) * img_width,
               j * img_height: (j + 1) * img_height] = digit

plt.figure(figsize=(10, 10))
#Reshape for visualization
fig_shape = np.shape(figure)
figure = figure.reshape((fig_shape[0], fig_shape[1]))

plt.imshow(figure, cmap='gnuplot2')
plt.show()  