# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 11:16:57 2022

@author: martigtu@stud.ntnu.no
"""

import numpy as np

# Data generator
from stacked_mnist import StackedMNISTData, DataMode

# Tensorflow stuff
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, Conv2DTranspose, Lambda, Reshape

# Own stuff
from functions import visualize, visualize_encoding, visualize_decoding




class VariationalAutoEncoder:
    def __init__(self, latent_dim = 2, force_learn = False, file_name = "VariationalAutoEncoder"):
        '''
        The model is a convolutional variational auto encoder (VAE) used on the
        MNIST handwritten digits data set. It encodes the images in a latent
        space represented by a mean and (logaritmic) variance.
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
        
        # defining the latent distribution
        z_mean = Dense(latent_dim, name="Z_mean")(x)
        z_log_var = Dense(latent_dim, name="Z_log_var")(x)
        
        # reparameterization trick to sample from the latent distribution
        z_sample = Lambda(self.sample_function, output_shape = (latent_dim,), name='z')((z_mean, z_log_var))
        encoder_outputs = (z_sample, z_mean, z_log_var)
        self.encoder = Model(encoder_input, encoder_outputs, name='Encoder')


        ### DECODER
        decoder_input = Input( shape=(self.latent_dim,) )
        x = Dense(np.prod(conv_shape), activation='relu')(decoder_input) # shape (height*width*n_channels)
        x = Reshape(target_shape=(conv_shape[0], conv_shape[1], conv_shape[2]))(x) # shape (height, width, n_channels)
        x = Conv2DTranspose(64, kernel_size=(5,5), padding='valid', activation='relu')(x)
        x = Conv2DTranspose(32, kernel_size=(5,5), padding='valid', activation='relu')(x)
        # sigmoid activation to get predictions for each pixel
        decoder_output = Conv2DTranspose(1, kernel_size=(5,5), padding='valid', activation='sigmoid', name="x_recon")(x)
        self.decoder = Model(decoder_input, decoder_output, name='Decoder')
        # reconstruction
        x_recon = self.decoder(z_sample)
        
        
        ### Calculating the negative ELBO as the loss function
        '''
        Specifically made for the VAE class.
        
        Custom loss function for the variational autoencoder. It implements the
        negative weighted Evidence lower bound (ELBO) as the loss function for the 
        variational autoencoder.
        
        omega: the Kullbakc Leibler weight
        
        x: input
        z: encoded(x)
        x_recon: decoded(z)
        lambda: variables in the encoder
        thetha: variables in the decoder
        
        - A prior distribution of Z is assumed standard multivariate normal.
            Z ~ p_theta(z) = N(0,1)
        
        - A priori distribution of estimated distribution
            q(z|x_i,lambda) = N(mu_lambda(x_i), sigma^2_lambda(x_i))
        where mu_lambda and sigma^2_lambda are from the encoder network.
        
        -ELBO(x_i) = E_q[log(q(z|x_i,lambda)/p_theta(z,x_i|theta))]
                   = KL(q(z|x_i, lambda)||p_theta(z)) + E_q[-log p_theta(x_i|z,theta)]
                                  (1)                               (2)
        where
        (1): penalizes a posterior over Z far from the prior p_theta(z)
        (2): penalizes a poor reconstruction ability, avgeraged over q(z|x_i,lambda)
        
        - Using the priors given above, the Kullback Leibler term equals
            (1) = -0.5*(1 + log(sigma^2) - sigma^2 - mu^2)
        
        - Since the decoder network uses the logit activation function, the
          reconstruction loss can be calculated using the binary crossentropy. The
          expectation is calculated using Monte Carlo with 1 sample.
        '''
        # Kullback Leibler loss (analytic solution when the prior is N(0,1))
        kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.exp(z_log_var) - tf.square(z_mean), axis=(-1))
        beta=1e-2
        
        # Reconstruction loss using binary crossentropy
        recon_loss = tf.reduce_mean(tf.keras.metrics.binary_crossentropy(encoder_input, x_recon), axis=(1,2))
        
        
        ### VAE
        #model_outputs = (x_recon, z_mean, z_log_var)
        self.model = Model(encoder_input, x_recon, name='VAE')
        self.model.add_loss(beta*kl_loss + recon_loss)
        self.model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), metrics = ['accuracy'] )
        
        # Try reading the weights from file if force_relearn is False
        self.done_training = self.load_weights()
        
        
    def sample_function(self, params):
        # sample from the latent distribution Z
        mean, log_var = params
        eps = tf.random.normal( shape=tf.shape(mean)[1:] )
        return mean + tf.exp(log_var / 2) * eps
     
    
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
            
            # "Translate": Only look at "red" channel; only use the last digit.s
            x_train = x_train[:, :, :, [0]]

            # Fit model
            self.model.fit(x=x_train, y=x_train, batch_size=batch_size, epochs=epochs,
                           validation_data=(x_test, x_test))

            # Save weights and leave
            self.model.save_weights(filepath=self.file_name)
            self.done_training = True
    
        
    def encode(self, x):
        mu, _, _ = self.encoder(x)
        return mu
    
    
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
        Encode x into z_mean and then decode the encoding.
        This function allows for multiple channels, i.e. RGB images.
        '''
        n_channels = x.shape[-1]
        
        # RGB
        if n_channels > 1:
            return tf.concat( [self.predict(x[:,:,:,[ch]]) for ch in range(n_channels)], axis=-1 )
        
        # Monochrome
        return np.array( self.decode(self.encode(x)) )
        
    
    
        
if __name__=="__main__":
    # Train the Autoencoder on full dataset
    gen = StackedMNISTData(mode=DataMode.MONO_BINARY_COMPLETE)
    net = VariationalAutoEncoder(latent_dim=2, force_learn=False, file_name = "VariationalAutoEncoder")
    net.fit(generator=gen, batch_size=256, epochs=10)
    
    # Train the Autoencoder on missing dataset
    gen2 = StackedMNISTData(mode=DataMode.MONO_BINARY_MISSING)
    net2 = VariationalAutoEncoder(latent_dim=2, force_learn=False, file_name = "VariationalAutoEncoder_missing")
    net2.fit(generator=gen2, batch_size=256, epochs=10)
    
    # Visualization
    x_test, y_test = gen.get_full_data_set(training=False)
    x_test_recon = net.predict(x_test)
    visualize(x_test, x_test_recon)
    
    visualize_decoding(net, N=20, x_range=(-10,10), y_range=(-10,10))
    visualize(x = x_test, x_ref = x_test_recon)
    visualize_encoding(net, x_test, y_test)
    

