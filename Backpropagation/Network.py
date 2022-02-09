# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 22:35:58 2022

@author: martigtu@stud.ntnu.no
"""

import numpy as np
import matplotlib.pyplot as plt
import configparser
from ast import literal_eval

# Internal file import
from Layer import Layer
from functions import mse, mse_grad, x_entropy, x_entropy_grad, softmax, softmax_grad


class Network:
    def __init__(self, config_file_name):
        ### Cache information about last output, training and validation loss.
        self.last_output = None
        self.train_loss = []
        self.val_loss = []
        self.reg_loss = []
        
        
        ### Parse the information from the configuration file
        config = configparser.ConfigParser()
        config.read(config_file_name + '.ini')
        
        # Number of input neurons, MUST be specified
        if "input_size" in config['GLOBALS']:
            self.input_size = int(config['GLOBALS']["input_size"])
        else:
            raise TypeError("Input size not specified!")
            
        # Loss function ("mse" or "x_entropy"), MUST be specified
        if "loss" in config['GLOBALS']:
            self.loss_type = config['GLOBALS']["loss"]
        else:
            self.loss_type = "mse"
        if self.loss_type == "mse":
            self.loss = mse
            self.loss_grad = mse_grad
        elif self.loss_type == "x-entropy":
            self.loss = x_entropy
            self.loss_grad = x_entropy_grad
        else:
            raise TypeError("Loss function type not recognized!")
        
        # Network regularization ("None", "L1" or "L2")
        if "reg" in config['GLOBALS']:
            self.reg = config['GLOBALS']["reg"]
        else:
            self.reg = "None"
            
        # Network regularization rate
        if "regrate" in config['GLOBALS']:
            self.regrate = float(config['GLOBALS']["regrate"])
        else:
            self.regrate = 0.0001
            
        # Boolean variable for wheter to use softmax on outputs
        if "use_softmax" in config['GLOBALS']:
            self.use_softmax = literal_eval(config['GLOBALS']["use_softmax"])
        else:
            self.use_softmax = True
            
        # Network learning rate
        if "lrate" in config['GLOBALS']:
            self.lrate = float(config['GLOBALS']["lrate"])
        else:
            self.lrate = 0.01
        
        
        ### Add hidden layers in the Neural Network
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
            
            # Layer regularization ("None", "L1" or "L2")
            if "reg" in config[layer]:
                layer_reg = config[layer]["reg"]
            else:
                layer_reg = self.reg
            
            # Layer regularization rate
            if "regrate" in config[layer]:
                layer_regrate = float(config[layer]["regrate"])
            else:
                layer_regrate = self.regrate
                
            # Append Layer object and update current input size
            self.layers.append(Layer(current_input_size,
                                     layer_size,
                                     layer_act, 
                                     layer_lrate, 
                                     layer_wr, 
                                     layer_br,
                                     layer_reg,
                                     layer_regrate))
            current_input_size = layer_size
            
            
    def forward_pass(self, batch):
        '''
        Do a forward_pass pass of all columns in x. Procedure:
            
        1. Make a copy of the batch.
        2. forward_pass the batch through each hidden layer.
        3. Softmax the output if specified.
        
        After the forward_pass pass, all necessary intermediate layer outputs are
        cached locally in the Layer objects.
        '''
        network_output = np.copy(batch)
        
        # Pass forward trough each layer
        for layer in self.layers:
            network_output = layer.forward_pass(network_output)
        
        # Softmax the output if specified
        if self.use_softmax == True:
            network_output = softmax(network_output)
        
        # Cache output and return it
        self.last_output = network_output
        return network_output
    
    
    def predict(self, batch):
        '''
        Identical to the forward_pass function, but after the forward_pass pass,
        NO intermediate information is cached.
        '''
        network_prediction = np.copy(batch)
        
        # forward_pass pass trough each layer
        for layer in self.layers:
            network_prediction = layer.predict(network_prediction)
        
        # Softmax output if specified
        if self.use_softmax == True:
            network_prediction = softmax(network_prediction)
        
        return network_prediction
    
    
    def get_reg_loss(self):
        '''
        Returns the Networks regularization loss.
        '''
        return sum([l.get_reg_loss() for l in self.layers])
    
    
    def backward_pass(self, target):
        '''
        Do a backward pass by calculating the gradient wrt the loss function.
        JL : Jacobian of loss function wrt. current layer in the BP. Procedure:
        
        1. JL is first sat to the gradient of L wrt. output layer of network.
        2. If the output was softmaxed, then each component JL_i is multiplied
           by the respective gradient of softmax(o_i) wrt. output o_i.
        3. JL is passed backward through each layer and updated such that each
           JL is the derivative of the loss function wrt. the current layer.
        
        After the bakwards pass, all necessary gradients are cached locally in 
        the Layer objects.
        '''

        # Compute initial Jacobian of loss with respect to output
        # Note: The Jacobian is sparse, so we represent only the
        # diagonal elements of it, as the off-diagonals are zero.
        JL = self.loss_grad(self.last_output, target)
        
        # Account for softmax
        if self.use_softmax == True:
            n_samples = self.last_output.shape[1]
            for j in range(n_samples):
                output_j = self.last_output[:,j]
                
                # Multiply JL with softmamx gradient
                JL[:,j] = softmax_grad(output_j) @ JL[:,j]

        # backward_pass_pass_pass pass trough each layer
        for layer in self.layers[::-1]:
            JL = layer.backward_pass(JL)
        
        
        
    def update_weights(self):
        '''
        Update all weights and biases in the hidden layers of the network.
        
        All necessary information is stored locally in the Layer objects.
        '''
        for layer in self.layers:
            layer.update_weights()
            
    
    
    def fit(self, train_data, val_data, batch_size, epochs):
        '''
        Fit the network using Stochastic Gradient Descent.
        '''
        train_targets, train_samples = train_data
        val_targets, val_samples = val_data
        
        # Number of samples in data
        n_samples = train_samples.shape[1]
        
        # Number of batches per epoch (rounded down)
        n_batches = int(n_samples / batch_size)
        
        # Plot regularization loss if any layer uses a regularization
        if any([l.reg != 'None' for l in self.layers]):
            have_reg = True
        else:
            have_reg = False
            
        
        # Iterate through the epochs
        for epoch in range(epochs):
            
            # Generate a random permutation to make the mini-batches stochastic
            p = np.random.permutation(n_samples)
            for i in range(n_batches):
                
                # Indecies for the random minibatch
                start = i*batch_size
                end = (i+1)*batch_size
                
                # Forward pass the minibatch through the network
                self.forward_pass(train_samples[:, p[start:end]])
                
                # Cache mean training loss in batch
                self.train_loss.append(np.mean(self.loss(self.last_output,
                    train_targets[:, p[start:end]])))
                
                # Cache mean validation loss 
                val_output = self.predict(val_samples)
                self.val_loss.append(np.mean(self.loss(val_output, val_targets)))
                
                # Cache regularization loss
                if have_reg:
                    self.reg_loss.append(self.get_reg_loss())
                else:
                    self.reg_loss.append(0)
                
                # Backpropagation and update weights in the network
                self.backward_pass(train_targets[:, p[start:end]])
                self.update_weights()
        
        
        # Plot the learning curve
        x_axis = np.arange(n_batches*epochs) / n_batches
        
        fig,ax = plt.subplots()
        ax.plot(x_axis, self.val_loss,
                 linewidth=1.5, color='r', label='Validation loss')
        ax.plot(x_axis, self.train_loss,
                 linewidth=1.5, color='b', label='Training loss')
        ax.legend()
        ax.set_xlabel('Epochs', fontsize=10)
        ax.set_ylabel(self.loss_type, color='black', fontsize=10)
        
        # Plot the regularization loss is aplicable
        if have_reg:
            ax2=ax.twinx()
            ax2.plot(x_axis, self.reg_loss,
                     linewidth=1.5, color='green', label='Regularization loss')
            ax2.set_ylabel(self.reg, color="green", fontsize=10)
        plt.show()
        
    
    
    def test(self, data):
        '''
        Function for visualizing errors made by the Neural network.
        '''
        targets, samples = data
        
        # forward_pass the samples
        prediction = self.predict(samples)
        
        # Number of correct classifications
        correct = prediction.argmax(axis=0) == targets.argmax(axis=0)
        print('Neural network succsevily classified', sum(correct), \
              'out of', correct.shape[0], 'samples.')
        
        # visualize wrong classifications
        wrong = (correct == False)
        if np.sum(wrong) != 0:
            wrong_pred = prediction.argmax(axis=0)[wrong]
            n = min(10, np.sum(wrong))
            
            N = int(np.sqrt(len(samples.T[0].flatten())))
            fig, axes = plt.subplots(1, n, figsize=(N, N))
            for i, ax in enumerate(axes.flat):
                ax.imshow(samples[:,wrong].T[i].reshape(N,N), cmap='Greys')
                ax.set_xlabel(['Up', 'Left', 'Down', 'Right'][wrong_pred[i]])
                #ax.set_axis_off()
            plt.show()