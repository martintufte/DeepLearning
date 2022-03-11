# -*- coding: utf-8 -*-
"""
Created on Tue Mar  10 20:06:00 2022

@author: martigtu@stud.ntnu.no
"""

import numpy as np

# Data generator
from stacked_mnist import StackedMNISTData, DataMode

# Tensorflow stuff
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, Conv2DTranspose, Reshape

# Own stuff
from functions import visualize, visualize_encoding, visualize_decoding, color_to_mono, mono_to_color




class AutoEncoder:
    def __init__(self, latent_dim = 2, force_learn = False, file_name = "AutoEncoder"):
        '''
        The model is a convolutional auto encoder (AE) used on the
        MNIST handwritten digits data set. It encodes the images in a latent space.
        '''
        self.force_relearn = force_learn
        self.done_training = False
        self.file_name = "./models/"+file_name
        self.latent_dim = latent_dim
        self.n_channels = 1
        
        input_shape = (28, 28, self.n_channels) # height, width, n_channels
        
        
        ### ENCODER
        encoder_input = Input( shape=input_shape )
        x = Conv2D(32, kernel_size=(5, 5), padding='valid', activation='relu')(encoder_input)
        x = Conv2D(64, kernel_size=(5, 5), padding='valid', activation='relu')(x)
        x = Conv2D(96, kernel_size=(5, 5), padding='valid', activation='relu')(x)
        conv_shape = (16, 16, 96) # shape of x
        x = Flatten()(x)
        # Encode the  input
        encoder_output = Dense(latent_dim, name="latent_representation")(x)
        self.encoder = Model(encoder_input, encoder_output, name='Encoder')
        

        ### DECODER
        decoder_input = Input( shape=(self.latent_dim,) )
        x = Dense(np.prod(conv_shape), activation='relu')(decoder_input) # shape (height*width*n_channels)
        x = Reshape(target_shape=(conv_shape[0], conv_shape[1], conv_shape[2]))(x) # shape (height, width, n_channels)
        x = Conv2DTranspose(64, kernel_size=(5,5), padding='valid', activation='relu')(x)
        x = Conv2DTranspose(32, kernel_size=(5,5), padding='valid', activation='relu')(x)
        # sigmoid activation to get predictions for each pixel
        decoder_output = Conv2DTranspose(1, kernel_size=(5,5), padding='valid', activation='sigmoid', name="x_recon")(x)
        self.decoder = Model(decoder_input, decoder_output, name='Decoder')
        # Decode the output from the encoder
        x_recon = self.decoder(encoder_output)
        
        ### AE
        self.model = Model(encoder_input, x_recon, name='AE')
        self.model.compile(loss="binary_crossentropy",
                           optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                           metrics = ['accuracy'] )
        
        # Try reading the weights from file if force_relearn is False
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
    
    
    def fit(self, generator: StackedMNISTData, batch_size=1024, epochs = 8):
        """
        Train model if required. As we have a one-channel model we take care to
        only use the first channel of the data.
        """

        if self.force_relearn or self.done_training is False:
            # Get hold of data
            x_train, _ = generator.get_full_data_set(training=True)
            x_test, _ = generator.get_full_data_set(training=False)
            
            # Adapting to 3 channels
            n_channels = x_train.shape[-1]
            if n_channels > 1:
                x_train = color_to_mono(x_train)
                x_test = color_to_mono(x_test)

            # Fit model
            self.model.fit(x=x_train, y=x_train, batch_size=batch_size, epochs=epochs,
                           validation_data=(x_test, x_test))

            # Save weights and leave
            self.model.save_weights(filepath=self.file_name)
            self.done_training = True
    
        
    def encode(self, x):
        return self.encoder(x)
    
    
    def decode(self, z):
        '''
        Decode the latent representation.
        shape is assumed to be (n_samples, latent_dim)
        or (n_samples, latent_dim, n_channels)
        '''
        # Monochrome
        if z.ndim==2: 
            return self.decoder(z)
        # RGB
        else:
            n_channels = z.shape[-1]
            return tf.concat( [self.decode(z[:,:,ch]) for ch in range(n_channels)], axis=-1 )
    
    
    def predict(self, x):
        ''' 
        Encode x into z and then decodes the encoding.
        This function allows for multiple channels, i.e. RGB images.
        '''
        n_channels = x.shape[-1]
        
        # RGB
        if n_channels > 1:
            return np.array( tf.concat( [self.predict(x[:,:,:,[ch]]) for ch in range(n_channels)], axis=-1 ) )
        # Monochrome
        return np.array( self.decode(self.encode(x)) )
        
    
        
if __name__=="__main__":
    # Train the Autoencoder on full dataset
    gen = StackedMNISTData(mode=DataMode.MONO_BINARY_COMPLETE)
    net = AutoEncoder(latent_dim=2, force_learn=False, file_name = "AutoEncoder")
    net.fit(generator=gen, batch_size=256, epochs=10)
    
    # Visualization
    x_test, y_test = gen.get_full_data_set(training=False)
    x_test_recon = net.predict(x_test)
    visualize(x_test, x_test_recon)
    
    #visualize_decoding(net, N=20, x_range=(-10,10), y_range=(-10,10))
    #visualize(x = x_test, x_ref = x_test_recon)
    #visualize_encoding(net, x_test, y_test, is_AE=True)
    

