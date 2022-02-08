# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 10:44:52 2022

@author: martigtu@stud.ntnu.no
"""

from Network import Network
from generator import create_data


if __name__=="__main__":
    
    ### Example to show in the class:
    
    # Create data
    train_data, val_data, test_data = create_data(N = 24, n_samples = 20000, noise_prob=0.05, flatten = True)
    
    # Create Neural Network
    # When using x-entropy, the learning rate should be smaller
    nn = Network('example')
    
    # Fit Neural Network
    nn.fit(train_data, val_data, batch_size=250, epochs=5)


