# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 11:21:33 2022

@author: martigtu@stud.ntnu.no
"""

import numpy as np

### Activation functions and their derivatives

def identity(x):
    """Compute identity componentwise in x."""
    return x
def identity_der(x): 
    """Compute the derivative of identity componentwise in x."""
    return np.ones_like(x)


def relu(x):
    """Compute relu componentwise in x."""
    return np.maximum(0, x)
def relu_der(x):
    """Compute the derivative of relu componentwise in x."""
    return x>0


def tanh(x):
    """Compute tanh componentwise in x."""
    return np.tanh(x)
def tanh_der(x):
    """Compute the derivative of tanh componentwise in x."""
    return 1 / np.cosh(x)**2


def sigmoid(x):
    """Compute sigmoid componentwise in x."""
    return np.exp(x) / (np.exp(x) + 1)

def sigmoid_der(x):
    """Compute the derivative of sigmoid componentwise in x."""
    return np.exp(x) / (np.exp(x) + 1)**2


def softmax(x):
    """Compute softmax values for each column in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)
def softmax_grad(s):
    """Compute the gradient of s=softmax(x)."""
    # Note that s = softmax(x)
    return np.diag(s) - np.outer(s,s)



### Loss functions and their derivatives

def mse(x, t):
    """Compute mean squared error for each column in x."""
    return np.mean((x-t)**2, axis=0)

def mse_grad(x, t):
    """Derivative of mean squared error for each column in x."""
    ncol = x.shape[0]
    return 2*(x - t) / ncol


def x_entropy(x, t):
    """Compute cross entropy for each column in x."""
    return - np.sum(t * np.log(x), axis=0)

def x_entropy_grad(x, t):
    """Compute derivative of cross entropy for each column in x."""
    return - t/x



### Regulization function derivatives

def L1_der(w):
    pass

def L2_der(w):
    pass


