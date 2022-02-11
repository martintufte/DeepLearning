# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 10:44:52 2022

@author: martigtu@stud.ntnu.no
"""

from Network import Network
from Data import Data
from functions import visualize


if __name__=="__main__":
    # To enable verbose
    #nn.forward_pass(data.train, verbose=True)
    
    
    ### Example 1: 3 layers, mse, L1, relu + linear
    data = Data(N = 30, n_samples = 10000, noise_prob=0.05, flatten = True)
    nn = Network('config_files/example1.ini')
    nn.fit(data, batch_size=250, epochs=5, plot_reg=True)
    

    #### Example 2: 2 layers, x-entropy, None, wr, sigmoid
# =============================================================================
#     data = Data(N = 30, n_samples = 20000, noise_prob=0.05, flatten = True)
#     nn = Network('config_files/example2.ini')
#     nn.fit(data, batch_size=250, epochs=5, plot_reg=True)
# =============================================================================
    
    
    ### Example 3: 5 layers, mse, L2, relu + tanh (May not work!)
# =============================================================================
#     data = Data(N = 30, n_samples = 10000, noise_prob=0.01, flatten = True)
#     nn = Network('config_files/example3.ini')
#     nn.fit(data, batch_size=250, epochs=25, plot_reg=True)
# =============================================================================
    
    
    ### Example 4: 3 layers, x-entropy, L1, relu, (500 size layer + noisy data)
# =============================================================================
#     data = Data(N = 30, n_samples = 40000, noise_prob=0.10, flatten = True)
#     nn = Network('config_files/example4.ini')
#     nn.fit(data, batch_size=500, epochs=3, plot_reg=True)
# =============================================================================
    

    # Example 5: Testing purposes
# =============================================================================
#     data = Data(N = 30, n_samples = 10000, noise_prob=0.01, flatten = True,
#                 train_prop=0.7, val_prop=0.2)
#     nn = Network('config_files/network.ini')
#     nn.fit(data, batch_size=200, epochs=15, plot_reg=False)
# =============================================================================
