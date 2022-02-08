# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 13:04:19 2022

@author: Teksle
"""

import numpy as np
from skimage.draw import line_aa
import matplotlib.pyplot as plt


def generate_imgs(N, n_samples, margin_tb_min=0.05, margin_tb_max=0.20,
                  displacement_y_max=0.20, difference_y_max=0.15):
    '''
    Function for generating training data + test data.
    Currently produces images of arrows pointing upwards.
    '''
    
    # Store images in this ndarray
    imgs = np.zeros((n_samples, N, N), dtype=np.uint8)
    
    # Generate one sample at a time
    for i in range(n_samples):
        # Generate line thickness
        line_thickness = min(0.25, np.random.random(1)/2)

        # Generate x-values
        x0 = margin_tb_min + np.random.random(1)*(margin_tb_max - margin_tb_min)
        x1 = 1 - margin_tb_min - np.random.random(1)*(margin_tb_max - margin_tb_min)

        # Generate y-values
        displacement = (2*np.random.random(1)-1)*displacement_y_max
        difference = (2*np.random.random(1)-1)*difference_y_max

        # Generate y-values
        y0 = 0.5 + displacement - difference/2
        y1 = 0.5 + displacement + difference/2

        # Calculate position of tips
        x_m = (3*x0+x1)/4
        y_m = (3*y0+y1)/4
        # Tip 1
        x_t1 = x_m - (y0-y_m)
        x_t2 = x_m + (y0-y_m)
        # Tip 2
        y_t1 = y_m + (x0-x_m) 
        y_t2 = y_m - (x0-x_m)

        # Convert proportions to pixel values in the image
        X0 = int(N*x0 + 0.5)
        X1 = int(N*x1 + 0.5)
        Y0 = int(N*y0 + 0.5)
        Y1 = int(N*y1 + 0.5)
        XT1 = int(N*x_t1 + 0.5)
        YT1 = int(N*y_t1 + 0.5)
        XT2 = int(N*x_t2 + 0.5)
        YT2 = int(N*y_t2 + 0.5)

        # Add main line in the arrow
        rr, cc, val = line_aa(X0, Y0, X1, Y1)
        imgs[i,rr,cc] = val > line_thickness

        # Add extra lines to make an actual arrow
        rr, cc, val = line_aa(X0, Y0, XT1, YT1)
        imgs[i,rr,cc] = val > line_thickness
        rr, cc, val = line_aa(X0, Y0, XT2, YT2)
        imgs[i,rr,cc] = val > line_thickness
        
    return imgs



def add_noise(imgs, prob):
    '''
    Add noise to random pixels in the image.
    '''
    noise = np.random.binomial(1, prob, imgs.shape)
    
    return np.bitwise_xor(imgs, noise)

    
    
def generate_data(N, n_samples, noise_prob = 0, flatten=False):
    '''
    Fuction for creating the actual training data w/ labels.
    '''
    # Generate imgs
    imgs = generate_imgs(N, n_samples)
    
    # Add noise
    if noise_prob > 0:
        imgs = add_noise(imgs, noise_prob)
    
    # Generate labels (0 = up, 1 = left, 2 = down, 3 = right)
    labels = np.random.randint(4, size=n_samples, dtype=np.uint8)
    
    # Create target vectors w/ one-hot encoding for labels
    targets = np.zeros((n_samples, 4))
    for i in range(4):    
        targets[labels==i, i] = 1
    
    # Rotate images according to their label
    for i in (1,2,3):
        imgs[labels==i] = np.rot90(imgs[labels==i], k=i, axes=(1,2))
    
    # If flatten is true, images are returned as 1D instead of 2D
    if flatten:
        return targets, imgs.reshape(n_samples, N*N)
    return targets, imgs



def create_data(N, n_samples, noise_prob = 0, flatten=False, train_prop=0.7, val_prop=0.2):
    targets, imgs = generate_data(N, n_samples, noise_prob, flatten)
    
    # Get number of samples
    n_samples = targets.shape[0]
    
    # Calculate number of samples for train, validation and test set
    n_train = int(n_samples * train_prop)
    n_val = int(n_samples * val_prop)
    
    # Split data into train, validation and test sets (and transpose)
    train_data = (targets[: n_train].T, imgs[: n_train].T)
    val_data = (targets[n_train : n_train+n_val].T, imgs[n_train : n_train+n_val].T)
    test_data = (targets[n_train+n_val :].T, imgs[n_train+n_val :].T)
    
    return train_data, val_data, test_data



def visualize(imgs, n_samples):
    '''
    Function for visualizing the images
    '''
    N = int(np.sqrt(len(imgs.T[0].flatten())))
    fig, axes = plt.subplots(1, n_samples, figsize=(N, N))
    for i, ax in enumerate(axes.flat):
        ax.imshow( imgs.T[i].reshape(N,N), cmap='Greys')
        ax.set_xlabel('t')
        ax.set_axis_off()
    plt.show()
    
    
    
if __name__=="__main__":
    train_data, val_data, test_data = create_data(N = 24, n_samples = 1000, noise_prob=0.005, flatten = True)
    
    visualize(train_data[1], 10)
    














