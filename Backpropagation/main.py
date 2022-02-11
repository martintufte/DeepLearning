# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 10:44:52 2022

@author: martigtu@stud.ntnu.no
"""

from Network import Network
from Data import Data


if __name__=="__main__":
    # Generate data
    data = Data(N = 30, n_samples = 10000, noise_prob=0.05, flatten = True)
    
    ### Example 1: 3 layers, mse, L1, relu + linear
    nn = Network('config_files/network.ini')
    
    #### Example 2: 5 layers, x-entropy, L1, sigmoid
    
    
    # Fit network to the data
    nn.fit(data, batch_size=250, epochs=5, plot_reg=True)


