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
        
        # Cache last input, derivative wrt. layer, weigths and biases
        self.last_input = None
        self.JL = None
        self.JL_W = None
        self.JL_b = None
        
    
    
    
    def forward_pass(self, x):
        """
        Input shape: (input_size, n_samples)
        Output shape: (size, n_samples)
        """
        if x.shape[0] != self.input_size:
            raise TypeError("Layer recieved wrong input size!")
        
        # Cache input, needed in backpropagation
        self.last_input = np.copy(x)
        
        # Return output (note that bias is added to each column)
        return self.f(self.W @ x + self.b)
        
    
    
    def backward_pass(self, JL):
        '''
        JL is the Jacobian of L with respect to next layer.
        
        '''
        self.JL = np.copy(JL)
        
        return self.W.T @ (JL * self.f_der(self.W @ self.last_input + self.b))
        
        
    
    
    
    def update_weights(self):
        '''
        Update weights and biases in the layer.

        '''
        n_samples = self.last_input.shape[1]
        
        # Derivative of loss with respect to weigths and biases
        self.JL_W = (self.JL * self.f_der(self.W @ self.last_input + self.b)) @ self.last_input.T
        self.JL_b = (self.JL * self.f_der(self.W @ self.last_input + self.b)) @ np.ones((n_samples,1))
        
        # Update weights using gradient descent
        self.W -= self.lrate * self.JL_W
        self.b -= self.lrate * self.JL_b
        


class Network:
    def __init__(self, config_file_name):
        ''' Parse the information from the configuration file '''
        
        ### Cache information in network
        self.last_output = None
        self.loss_history = []
        self.batch_size_history = []
        
        
        
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
            self.loss_type = config['GLOBALS']["loss"]
        else:
            self.loss_type = "mse"
        
        # Loss function and derivative
        if self.loss_type == "mse":
            self.loss = mse
            self.loss_der = mse_der
        elif self.loss_type == "x-entropy":
            self.loss = x_entropy
            self.loss_der = x_entropy_der
        else:
            print("Loss function name not recognized!")
        
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
        
        # Softmax output if specified
        if self.use_softmax == True:
            y = softmax(y)
        
        # Cache output
        self.last_output = np.copy(y)
        
        #self.batch_size_history.append(y.shape[1])
        
    
    
    def backward_pass(self, target):
        ''' Do a backwards pass.'''
        
        # Cache loss in history
        self.loss_history.append( self.loss(self.last_output, target) )
        
        n_samples = self.last_output.shape[1]
        
        # Derivative of loss with respect to inputs
        JL = self.loss_der(self.last_output, target)
        
        # Account for softmax:
        if self.use_softmax == True:
            for j in range(n_samples):
                output_j = self.last_output[:,j]
                
                # Multiply JL with softmamx derivative
                JL[:,j] = softmax_der(output_j) @ JL[:,j]
        
        
        # Backward pass trough each layer
        for layer in self.layers[::-1]:
            JL = layer.backward_pass(JL)
        
        
        
    def update_weights(self):
        ''' Update weights in the layers. '''
        
        for layer in self.layers:
            layer.update_weights()
            
    
    def do_iteration(self, batch_samples, batch_targets):
        ''' Feed minibatch through + update weights.'''
        self.forward_pass(batch_samples)
        self.backward_pass(batch_targets)
        self.update_weights()

        
    def fit(self, data, batch_size, epochs):
        '''
        Fit the network using SGD.
        '''
        targets, samples = data
        
        # Number of samples in data
        n_samples = samples.shape[1]
        
        # Number of batches per epoch (rounded down)
        n_batches = int(n_samples / batch_size)
        
        print('n_samples = ', n_samples)
        print('n_batches per epoch =', n_batches)
        
        # Iterate through the epochs
        for epoch in range(epochs):
            # Generate a random permutation (stochastic part)
            p = np.random.permutation(n_samples)
    
            for i in range(n_batches):
                # Start / end idx for batch permutation
                start = i*batch_size
                end = (i+1)*batch_size
                
                # Forward pass + Backward pass + update weights
                #print(samples[:, p[start:end]].shape)
                #print(targets[:, p[start:end]].shape)
                self.do_iteration(samples[:, p[start:end]], targets[:, p[start:end]])
    
        # plot the error after the training session
        plt.plot([np.sum(i) for i in self.loss_history])
    
    
    
    def test(self, data):
        '''
        Function for visualizing errors made by the Neural network.
        '''
        targets, samples = data
        
        # Forward the samples
        self.forward_pass(samples)
        
        # Number of correct classifications
        correct = self.last_output.argmax(axis=0) == targets.argmax(axis=0)
        print('Neural network succsevily classified', sum(correct), 'out of', correct.shape[0], 'samples.')
        
        # visualize wrong classifications
        wrong = (correct == False)
        if np.sum(wrong) != 0:
            wrong_pred = self.last_output.argmax(axis=0)[wrong]
            n = min(10, np.sum(wrong))
            
            N = int(np.sqrt(len(samples.T[0].flatten())))
            fig, axes = plt.subplots(1, n, figsize=(N, N))
            for i, ax in enumerate(axes.flat):
                ax.imshow(samples[:,wrong].T[i].reshape(N,N), cmap='Greys')
                ax.set_xlabel(['Up', 'Left', 'Down', 'Right'][wrong_pred[i]])
                ax.set_axis_off()
            plt.show()


if __name__=="__main__":
    # Create data
    train_data, val_data, test_data = create_data(N = 24, n_samples = 20000, noise_prob=0.01, flatten = True)
    
    # Create Neural Network
    nn = Network('example')
    
    # Fit Neural Network
    nn.fit(train_data, batch_size=200, epochs=15)


