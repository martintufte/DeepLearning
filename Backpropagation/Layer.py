# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 22:31:30 2022

@author: martigtu@stud.ntnu.no
"""

import numpy as np
from functions import identity, identity_der, relu, relu_der, tanh, tanh_der, \
    sigmoid, sigmoid_der


class Layer:
    def __init__(self, input_size, size, act, lrate, wr = (-0.1, 0.1), br = (0, 0)):
        # Store information about size and learning rate
        self.input_size = input_size
        self.size = size
        self.lrate = lrate
        
        # Initialize random weigths and biases
        self.W = wr[0] + (wr[1]-wr[0]) * np.random.random((size, input_size))
        self.b = br[0] + (br[1]-br[0]) * np.random.random((size, 1))
        
        # Activation function
        if act == 'linear':
            self.f = identity
            self.f_der = identity_der
        elif act == 'relu':
            self.f = relu
            self.f_der = relu_der
        elif act == 'tanh':
            self.f = tanh
            self.f_der = tanh_der
        elif act == 'sigmoid':
            self.f = sigmoid
            self.f_der = sigmoid_der
        else:
            raise TypeError("Type must be linear, relu, tanh or sigmoid!")
        
        # Information used in backpropagation and parameter updating
        self.last_input = np.zeros(input_size)
        self.JL = None # Size depends on number of samples in the batch.
        self.JL_W = np.zeros_like(self.W)
        self.JL_b = np.zeros_like(self.b)
        
        
    
    def forward(self, layer_input):
        '''
        Forward the layer input. Cache any necessary information used in BP.
        Input shape: (input_size, n_samples)
        Output shape: (size, n_samples)
        '''
        if layer_input.shape[0] != self.input_size:
            raise TypeError("Layer recieved wrong input size!")
        
        # Cache input, needed in backpropagation
        self.last_input = np.copy(layer_input)
        
        # Return output (note that bias is actually added to each column)
        return self.f(self.W @ layer_input + self.b)
    
    
    
    def predict(self, layer_input):
        '''
        Identical to forward, but do NOT cache any relevant information.
        '''
        if layer_input.shape[0] != self.input_size:
            raise TypeError("Layer recieved wrong input size!")
        
        return self.f(self.W @ layer_input + self.b)
    
    
    
    def backward(self, JL):
        '''
        JL is the Jacobian of loss function with respect to current layer.
        '''
        # Cache the Jacobian.
        self.JL = np.copy(JL)
        
        # Return Jacobian of loss function with respect to the previous layer.
        return self.W.T @ (JL * self.f_der(self.W @ self.last_input + self.b))
        
        
        
    def update_weights(self):
        '''
        Update weights and biases in the layer using gradient descent.
        '''
        n_samples = self.last_input.shape[1]
        
        # Jacobian of loss function wrt. layer weigths and biases
        M = self.JL * self.f_der(self.W @ self.last_input + self.b)
        self.JL_W = M @ self.last_input.T
        self.JL_b = M @ np.ones((n_samples,1))
        
        # Update weights using vanilla gradient descent
        self.W -= self.lrate * self.JL_W
        self.b -= self.lrate * self.JL_b
