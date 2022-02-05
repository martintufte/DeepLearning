# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 10:44:52 2022

@author: Teksle
"""

import configparser
from ast import literal_eval

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


from functions import *
from generator import create_data



class Layer:
    def __init__(self, input_size, size, act, lrate, wr = (-0.1, 0.1), br = (0, 0)):
        # store information about size and learning rate
        self.input_size = input_size
        self.size = size
        self.lrate = lrate
        
        # initialize random weigths and biases
        self.W = wr[0] + (wr[1]-wr[0]) * np.random.random((size, input_size))
        self.b = br[0] + (br[1]-br[0]) * np.random.random((size, 1))
        
        # set activation function
        if act == 'linear':
            self.f = identity
            self.f_der = identity_der
        elif act == 'relu':
            self.f = relu
            self.f_der = relu_der
        elif act == 'tanh':
            self.f = tanh
            self.f_der = tanh_der
        elif act == 'logistic':
            self.f = logistic
            self.f_der = logistic_der
        else:
            raise TypeError("Type must be linear, relu, tanh or logistic.")
    
    
    def forward_pass(self, x):
        """
        Input shape: (parent.size, n_samples)
        Output shape: (size, n_samples)
        """
        if x.shape[0] != self.input_size:
            raise TypeError("Layer recieved wrong input size!")
        
        return self.f(self.W @ x + self.b) # Note: bias is added to each column
        
    
    def backward_pass(self):
        pass




class Network:
    def __init__(self, config_file_name):
        ''' parse the information from the configuration file '''
        
        config = configparser.ConfigParser()
        config.read(config_file_name + '.ini')
        
        
        ### GLOBALS
        
        # number of input neurons 
        if "input" in config['GLOBALS']:
            self.input_size = int(config['GLOBALS']["input"])
        else:
            self.input_size = 16 # TODO: make this compatible with input data
            
        # loss function ("mse" or "x_entropy")
        if "loss" in config['GLOBALS']:
            self.loss = config['GLOBALS']["loss"]
        else:
            self.loss = "mse"
            
        # Regularization ("none", "L1" or "L2")
        if "reg" in config['GLOBALS']:
            self.reg = config['GLOBALS']["reg"]
        else:
            self.reg = "none"
            
        # Weight used in regularization
        if "wreg" in config['GLOBALS']:
            self.wreg = float(config['GLOBALS']["wreg"])
        else:
            self.wreg = 0.0001
            
        # Use of softmax on output
        if "use_softmax" in config['GLOBALS']:
            self.use_softmax = literal_eval(config['GLOBALS']["use_softmax"])
        else:
            self.use_softmax = True
            
        # learning rate
        if "lrate" in config['GLOBALS']:
            self.lrate = float(config['GLOBALS']["lrate"])
        else:
            self.lrate = True
        
        
        ### LAYERS
        
        self.layers = []
        current_input_size = self.input_size
    
        for layer in config.sections()[1:]:
            # Size of layer
            if "size" in config[layer]:
                layer_size = int(config[layer]["size"])
            else:
                layer_size = current_input_size
            # Activation function
            if "act" in config[layer]:
                layer_act = config[layer]["act"]
            else:
                layer_act = "linear"
            # Layer learning rate
            if "lrate" in config[layer]:
                layer_lrate = float(config[layer]["lrate"])
            else:
                layer_lrate = self.lrate
            # Initial weight range
            if "wr" in config[layer]:
                layer_wr = literal_eval(config[layer]["wr"])
            else:
                layer_wr = (-0.1, 0.1)
            # Initial bias range
            if "br" in config[layer]:
                layer_br = literal_eval(config[layer]["br"])
            else:
                layer_br = (0, 0)
                
            self.layers.append(Layer(current_input_size,
                                     layer_size,
                                     layer_act, 
                                     layer_lrate, 
                                     layer_wr, 
                                     layer_br))
            
            # update current input size
            current_input_size = layer_size


    def forward_pass(self, x):
        ''' Do a forward pass of all columns in x. '''
        y = np.copy(x)
        for layer in self.layers:
            y = layer.forward_pass(y)
            
        if self.use_softmax == True:
            return softmax(y)
        return y
    
    
    
    def backward_pass(self):
        ''' Do a backwards pass.'''
        pass




if __name__=="__main__":
    # Create data
    train_data, val_data, test_data = create_data(N = 20, n_samples = 1000, noise_prob=0.005, flatten = True)
    
    # Create Neural Network
    nn = Network('example')
    
    # Test Network on training data
    train_targets, train_imgs = train_data
    
    # Forward the training imgs
    estimate = nn.forward_pass(train_imgs)
    
    






