# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 14:34:28 2022

@author: martigtu@stud.ntnu.no
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf



### CONVERT BETWEEN COLOR AND MONOCHROMATIC IMAGES

def color_to_mono(x):
    '''
    Flatten x from shape (n_samples, height, width, n_channels)
    to shape (n_channels*n_samples, height, width, 1) by stacking
    the color channels on under each other.
    '''
    n_channels = x.shape[-1]
    
    return tf.concat( [x[:,:,:,[ch]] for ch in range(n_channels)], axis=0 )
    


def mono_to_color(x, n_channels=3):
    '''
    Reshape x from shape (n_channels*n_samples, height, width, 1)
    to shape (n_samples, height, width, n_channels) by stacking
    the color channels on top of each other.
    '''
    assert x.shape[-1]==1, 'x is not monochromatic!'
    n_samples = int(x.shape[0] / n_channels)
    
    return tf.concat( [x[ch*n_samples:(ch+1)*n_samples] for ch in range(n_channels)], axis=-1 )



### VISUALIZATION FUNCTIONS

def visualize(x, x_ref='None', N=10, random=True):
    '''
    Display random imgs from x (upper) and y (lower).
    x (n_samples, height, width, n_channels)
    '''
    y = x_ref
    
    # Convert to numpy arrays
    if not isinstance(x, (np.ndarray, np.generic)):
        x = np.array(x)
    if not isinstance(y, (np.ndarray, np.generic)):
        y = np.array(y)
    
    # Number of samples
    n_samples = x.shape[0]
    n_channels = x.shape[-1]
    N = min(n_samples, N)
    
    p = np.random.permutation(n_samples) if random else np.arange(n_samples)
    
    # Monochrome imgs
    if n_channels == 1:
        if y == 'None':
            fig, axs = plt.subplots(1, N, figsize=(N, 1))
            for i in range(N):
                axs.flat[i].imshow(x[p[i]].squeeze(), cmap='gray_r')
                axs.flat[i].axis('off')
            plt.show()
        else:
            fig, axs = plt.subplots(2, N, figsize=(N, 2))
            for i in range(N):
                axs.flat[i].imshow(x[p[i]].squeeze(), cmap='gray_r')
                axs.flat[i].axis('off')
                
                axs.flat[i+N].imshow(y[p[i]].squeeze(), cmap='gray_r')
                axs.flat[i+N].axis('off')
            plt.show()
    # RGB imgs
    elif n_channels == 3:
        if y == 'None': 
            fig, axs = plt.subplots(1, N, figsize=(N, 1))
            for i in range(N):
                axs.flat[i].imshow(x[p[i]].astype(float))
                axs.flat[i].axis('off')
            plt.show()
        else:
            fig, axs = plt.subplots(2, N, figsize=(N, 2))
            for i in range(N):
                axs.flat[i].imshow(x[p[i]].astype(float))
                axs.flat[i].axis('off')
                
                axs.flat[i+N].imshow(y[p[i]])
                axs.flat[i+N].axis('off')
            plt.show()
    else:
        print('Number of channels is not in (1, 3), plotting not supported.')
            


def visualize_encoding(vae, x, y, is_AE=False):
    ''' Visualize the first two dimentions of the latent space. '''
    if is_AE:
        mu = vae.encoder(x)
    else:
        mu, _, _ = vae.encoder(x)
    
    df = pd.DataFrame({'dim1': mu[:,0], 'dim2': mu[:,1], 'label': y})
    groups = df.groupby('label')
    
    # Get hold of the minimum/maximum values for each feature dimension.
    min1 = np.amin(mu[:,0])
    max1 = np.amax(mu[:,0])
    min2 = np.amin(mu[:,1])
    max2 = np.amax(mu[:,1])
    
    # Choose a length scale for plotting the decodings
    d = min(max1-min1, max2-min2) / 30
    
    def on_click(event):
        if event.xdata != None:
            x, y = event.xdata, event.ydata
            mouse_input = np.array([[x, y]])
            mouse_img = vae.decoder.predict(mouse_input).reshape(28,28)
            event.inaxes.imshow(np.array(mouse_img), extent = [x-d, x+d, y-d, y+d], origin='upper', cmap='gray_r')
            event.inaxes.set_xlim([min1-1, max1+1])
            event.inaxes.set_ylim([min2-1, max2+1])
            event.canvas.draw()
            
    # Plot dim1 and dim2 for mu
    plt.connect('button_press_event', on_click)
    plt.axis('equal')
    for label, group in groups:
        plt.scatter(group.dim1, group.dim2, linewidths=1, alpha=0.05, label=label, zorder=0)
    plt.legend(title="Classes")
    plt.xlabel('dim 1')
    plt.ylabel('dim 2')
    plt.show()



def visualize_decoding(vae, N=20, x_range=(-3,3), y_range=(-3,3)):
    ''' Visualize a 2D grid of the decodings from the latent space. '''
    
    # Initialize an empty grid
    grid = np.zeros((N*28, N*28, 1))
    
    # Grid components
    grid_x = np.linspace(x_range[0], x_range[1], N)
    grid_y = np.linspace(y_range[0], y_range[1], N)[::-1]
    
    for i, y_i in enumerate(grid_y):
        for j, x_i in enumerate(grid_x):
            # create sample from coordinates
            z_sample = np.array([[x_i, y_i]])
            # get the decoded image, reshape it
            z_decoded = np.array(vae.decoder(z_sample)).reshape(28,28, 1)
            # insert decoded image into the grid
            grid[i*28:(i+1)*28, j*28:(j+1)*28] = z_decoded
    
    plt.axis('equal')
    plt.imshow(grid, cmap='gray_r')
    plt.axis('off')
    plt.show()



### EVALUATION OF THE GENERATIVE MODELS

def find_top_anomalies(x_true, x_pred, k=10):
    ''' Find the indecies for the top k anomalies using the reconstruction loss. '''
    
    # Use the binary crossentropy as reconstruction loss
    recon_loss = tf.reduce_mean( tf.keras.metrics.binary_crossentropy(x_true, x_pred), axis=(1,2) )
    recon_loss = np.array(recon_loss)
    
    # Indecies for the images with the highest anomalies
    idecies = np.argpartition(recon_loss, -k)[-k:]
    
    return list(idecies)



