# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 22:31:30 2022

@author: martigtu@stud.ntnu.no
"""

import numpy as np
from functions import identity, identity_der, relu, relu_der, tanh, tanh_der, sigmoid, sigmoid_der


class Layer:
    def __init__(self, input_size, size, act, lrate, wr, br, reg, regrate):
        # Input size and output size 
        self.input_size = input_size
        self.size = size
        # Learning rate
        self.lrate = lrate
        # Regulizer and regularization rate
        self.reg = reg
        self.regrate = regrate
        
        # Initialize random weigths and biases
        self.W = wr[0] + (wr[1]-wr[0]) * np.random.random((size, input_size))
        self.b = br[0] + (br[1]-br[0]) * np.random.random((size, 1))
        
        # Activation function, MUST be specified
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
        self.JL = None # (Size depends on number of samples in the batch.)
        self.JL_W = np.zeros_like(self.W)
        self.JL_b = np.zeros_like(self.b)
        
        
    
    def forward_pass(self, layer_input):
        '''
        Pass forward the layer input. Cache any information used in BP.
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
    
    
    def get_reg_loss(self):
        '''
        Returns the layer regularization loss.
        '''
        if self.reg == "L1":
            return self.regrate * np.sum(np.abs(self.W)) + np.sum(np.abs(self.b))
        elif self.reg == "L2":
            return self.regrate * np.sum(self.W**2) + np.sum(self.b**2)
        return 0
        
    
    def backward_pass(self, JL):
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
        
        # Calculate Jacobian of loss function wrt. layer weigths and biases
        M = self.JL * self.f_der(self.W @ self.last_input + self.b)
        self.JL_W = M @ self.last_input.T
        self.JL_b = M @ np.ones((n_samples,1))
        
        # Regularization term
        if self.reg == "L1":
            JL_W_reg = np.sign(self.W)
            JL_b_reg = np.sign(self.b)
        elif self.reg == "L2":
            JL_W_reg = self.W
            JL_b_reg = self.b
        else: # reg = None
            JL_W_reg = 0
            JL_b_reg = 0
        
        # Update weights using gradient descent
        self.W -= self.lrate * (self.JL_W + self.regrate * JL_W_reg)
        self.b -= self.lrate * (self.JL_b + self.regrate * JL_b_reg)



