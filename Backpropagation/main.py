# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 10:44:52 2022

@author: martigtu@stud.ntnu.no
"""

from Network import Network
from Data import Data


if __name__=="__main__":
    
    # Generate data
    data = Data(N = 30, n_samples = 20000, noise_prob=0.05, flatten = True)
    
    # Create neural network
    nn = Network('config_files/network.ini')
    
    # Fit network to the data
    nn.fit(data, batch_size=250, epochs=5)


