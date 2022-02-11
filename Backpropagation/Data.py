# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 21:20:08 2022

@author: marti
"""

import numpy as np
from skimage.draw import line_aa
import matplotlib.pyplot as plt



def generate_imgs(N, n_samples, noise_prob = 0, flatten=False):
    '''
    Function for generating data. Produces images of arrows poionting
    up, left, right, down.
    '''
    
    ### Additional parameters
    delta_x = 0.3
    delta_y = 0.2
    delta_theta = 25
    length_min = 0.4
    length_max = 0.8
    force_center = False
    
    
    ### Store images in this ndarray
    imgs = np.zeros((n_samples, N, N), dtype=np.uint8)
    
    
    ### Generate arrow in imgs (all pointing up)
    for i in range(n_samples):
        ### Find centerpoint, length and angle, line thickness
        if force_center:
            xc, yc = 0.5, 0.5
        else:
            xc = 0.5 + delta_x*np.random.uniform(-0.5, 0.5)
            yc = 0.5 + delta_y*np.random.uniform(-0.5, 0.5)
        length = np.random.uniform(length_min, length_max)
        delta_rad = np.pi*delta_theta/360
        angle = np.random.uniform(-delta_rad, delta_rad)
        line_thickness = min(0.25, np.random.random(1)/2)

        ### Generate end points of the arrow
        x0 = xc + length/2 * np.sin(angle)
        y0 = yc + length/2 * np.cos(angle)
        x1 = xc - length/2 * np.sin(angle)
        y1 = yc - length/2 * np.cos(angle)

        ### Calculate position of tips of arrow head
        xm = (3*x0+x1)/4
        ym = (3*y0+y1)/4
        # Tip 1
        x_t1 = xm - (y0-ym)
        x_t2 = xm + (y0-ym)
        # Tip 2
        y_t1 = ym + (x0-xm) 
        y_t2 = ym - (x0-xm)

        # Convert proportions to pixel values in the image
        def min_max_int(x):
            return max(min(int(x),N-1),0)
        
        X0 = min_max_int(N*x0 + 0.5)
        X1 = min_max_int(N*x1 + 0.5)
        Y0 = min_max_int(N*y0 + 0.5)
        Y1 = min_max_int(N*y1 + 0.5)
        XT1 = min_max_int(N*x_t1 + 0.5)
        YT1 = min_max_int(N*y_t1 + 0.5)
        XT2 = min_max_int(N*x_t2 + 0.5)
        YT2 = min_max_int(N*y_t2 + 0.5)

        ### Add main line in the arrow
        rr, cc, val = line_aa(X0, Y0, X1, Y1)
        imgs[i,rr,cc] = val > line_thickness

        ### Add extra lines to make an actual arrow
        rr, cc, val = line_aa(X0, Y0, XT1, YT1)
        imgs[i,rr,cc] = val > line_thickness
        rr, cc, val = line_aa(X0, Y0, XT2, YT2)
        imgs[i,rr,cc] = val > line_thickness
    
    
    ### Add noise
    if noise_prob > 0:
        noise = np.random.binomial(1, noise_prob, imgs.shape)
        imgs = np.bitwise_xor(imgs, noise)
    
    
    ### Generate labels (0 = up, 1 = left, 2 = down, 3 = right)
    labels = np.random.randint(4, size=n_samples, dtype=np.uint8)
    
    
    ### Create target vectors w/ one-hot encoding for labels
    targets = np.zeros((n_samples, 4))
    for i in range(4):    
        targets[labels==i, i] = 1
    
    
    ### Rotate images according to their label
    for i in (1,2,3):
        imgs[labels==i] = np.rot90(imgs[labels==i], k=i, axes=(1,2))
    
    
    ### If flatten is true, images are returned as 1D instead of 2D
    if flatten:
        return targets, imgs.reshape(n_samples, N*N)
    return targets, imgs



def create_data(N, n_samples, noise_prob = 0, flatten=False, train_prop=0.7, val_prop=0.2):
    targets, imgs = generate_imgs(N, n_samples, noise_prob, flatten)
    
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
    

    
class Data:
    def __init__(self, N, n_samples, noise_prob, flatten):
        self.N = N
        self.n_samples = n_samples
        
        all_data = create_data(N, n_samples, noise_prob, flatten)
        
        self.train, self.val, self.test = all_data


if __name__=="__main__":
    labels, imgs = generate_imgs(N=30, n_samples=5, noise_prob = 0.01, flatten=False)

    visualize(imgs, 5)











