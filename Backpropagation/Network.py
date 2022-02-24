# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 22:35:58 2022

@author: martigtu@stud.ntnu.no
"""
# Necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import configparser
from ast import literal_eval

# Import Layer object
from Layer import Layer
# Import loss functions and softmax and their gradients
from functions import mse, mse_grad, x_entropy, x_entropy_grad, \
    softmax, softmax_grad, visualize


class Network:
    '''
    Neural Network object.
    '''
    def __init__(self, config_file):
        # Cache information about last output
        self.last_output = None
        # Cache training/validation/regularization loss history
        self.train_loss = []
        self.val_loss = []
        self.reg_loss = []
        
        
        ### Parse the information from the configuration file
        config = configparser.ConfigParser()
        config.read(config_file)
        
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
            
            
        
    def forward_pass(self, batch, verbose=False):
        '''
        Do a forward_pass pass of all columns in x. Procedure:
            
        1. Make a copy of the batch.
        2. forward_pass the batch through each hidden layer.
        3. Softmax the output if specified.
        
        After the forward_pass pass, all necessary intermediate layer outputs
        are cached locally in the Layer objects.
        '''
        # Check input type of batch
        if type(batch) == tuple: # I.e. batch = (targets, samples)
            network_input = batch[1]
        else:
            network_input = batch
        # Make a copy to forward pass
        network_output = np.copy(network_input)
        
        
        ### Pass forward trough each layer
        for layer in self.layers:
            network_output = layer.forward_pass(network_output)
        
        # Softmax the output if specified
        if self.use_softmax == True:
            network_output = softmax(network_output)
        
        # Cache output in the network
        self.last_output = network_output
        
        
        ### Verbose
        if verbose == True:
            '''
            Print information about the forward pass to the user.
            '''
            max_disp = min(network_input.shape[1], 6)
            
            print('\nNetwork input: shape', network_input.shape)
            print(network_input)
            
            print('\nNetwork output: shape', network_output.shape)
            print(np.round(network_output[:,:max_disp], 3))
            
            if type(batch) == tuple:
                print('\nTargets:')
                print(batch[0][:,:max_disp])
            
                print('\nLoss:', np.mean(self.loss(network_output, batch[0])))
                wrong_predictions = network_output.argmax(axis=0) != batch[0].argmax(axis=0)
                sum_error = sum(wrong_predictions)
                print('Error: ', sum_error, 'of', batch[0].shape[1])
                print('Accuracy: ', round((1-sum_error/batch[0].shape[1])*100,2), '%', sep='')
            
            ### Also visualize upwards of 10 prediction mistakes
            n = min(sum_error, 10)
            data_wrong = (network_output[:,wrong_predictions][:,0:n],
                          batch[1][:,wrong_predictions][:,0:n])
            visualize(data_wrong, n)
        else:
            return network_output
        
    
    
    def predict(self, batch):
        '''
        Identical to the forward_pass function, but after the forward pass,
        NO intermediate information is cached.
        '''
        network_prediction = np.copy(batch)
        
        ### Forward pass trough each layer
        for layer in self.layers:
            network_prediction = layer.predict(network_prediction)
        
        # Softmax output if specified
        if self.use_softmax == True:
            network_prediction = softmax(network_prediction)
        
        return network_prediction
    
    
    
    def get_reg_loss(self):
        '''
        Returns the Networks total regularization loss.
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

        ### Initial Jacobian of loss with respect to last output
        # Note: The Jacobian is sparse -> only represent diagonal elements
        JL = self.loss_grad(self.last_output, target)
        
        ### Account for softmax
        if self.use_softmax == True:
            n_samples = self.last_output.shape[1]
            for j in range(n_samples):
                output_j = self.last_output[:,j]
                
                # Multiply JL with softmamx gradient
                JL[:,j] = softmax_grad(output_j) @ JL[:,j]
                
        ### Backward pass trough each layer
        for layer in self.layers[::-1]:
            JL = layer.backward_pass(JL)
        
        
        
    def update_weights(self):
        '''
        Update all weights and biases in the hidden layers of the network.
        All necessary information is stored locally in the Layer objects.
        '''
        for layer in self.layers:
            layer.update_weights()
            
    
    
    def fit(self, data, batch_size, epochs, plot_reg = False):
        '''
        Fit the network to the data using Stochastic Gradient Descent.
        '''
        ### Fetch training and validation data
        train_targets, train_samples = data.train
        val_targets, val_samples = data.val
        
        # Number of samples in data / batches per epoch (rounded down)
        n_samples = train_samples.shape[1]
        n_batches = int(n_samples / batch_size)
        
        # Boolean to store whether the network has any regulizations
        have_regulizer = any([l.reg!='None' for l in self.layers])
            
        
        ### Iterate through the epochs
        for epoch in range(epochs):
            
            # Generate a random permutation to make the mini-batches stochastic
            p = np.random.permutation(n_samples)
            
            for i in range(n_batches):
                # Fetch batch from training data
                start_idx, end_idx = i*batch_size, (i+1)*batch_size
                batch_samples = train_samples[:, p[start_idx:end_idx]]
                batch_targets = train_targets[:, p[start_idx:end_idx]]
                
                # Forward pass the batch
                batch_prediction = self.forward_pass(batch_samples)
                
                # Cache mean training/validation loss in history
                self.train_loss.append(
                    np.mean(self.loss(batch_prediction, batch_targets)))
                self.val_loss.append(
                    np.mean(self.loss(self.predict(val_samples), val_targets)))
                
                # Cache regularization loss
                if have_regulizer:
                    self.reg_loss.append(self.get_reg_loss())
                else:
                    self.reg_loss.append(0)
                
                # Backpropagation and update weights
                self.backward_pass(batch_targets)
                self.update_weights()
        
        
        ### Plot the learning curve
        x_axis = np.arange(n_batches*epochs) / n_batches
        fig,ax = plt.subplots()
        # Plot training/validation loss
        ax.plot(x_axis, self.train_loss,
                 linewidth=0.8, color='k', label='Training loss')
        ax.plot(x_axis, self.val_loss,
                 linewidth=0.8, color='r', label='Validation loss')
        # Add test loss
        test_targets, test_samples = data.test
        test_loss = np.mean(self.loss(self.predict(test_samples), test_targets))
        ax.scatter(x_axis[-1], [test_loss], color='b', marker='x', label='Test loss')
        ax.set_ylabel('Loss: '+self.loss_type, color='black', fontsize=10)
        ax.set_xlabel('Epochs', fontsize=10)
        ax.legend(loc='upper right')
        
        
        ### Plot the regularization loss if aplicable
        if plot_reg:
            ax2=ax.twinx()
            ax2.plot(x_axis, self.reg_loss,
                     linewidth=0.8, color='green', label='Regularization loss')
            ax2.set_ylabel('Regularization: '+self.reg, color="black", fontsize=10)
            ax2.legend(loc='lower left')
        
        
        plt.show()
