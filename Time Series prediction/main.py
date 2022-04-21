# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 12:43:50 2022

@author: martigtu
"""


import numpy as np
import pandas as pd
import tensorflow as tf


# read data
def prep_data(file_name):
    df = pd.read_csv("./data/" + file_name)
    return df





if __name__=="__main__":
    
    train      = prep_data("no1_train.csv")
    validation = prep_data("no1_validation.csv")
    
    train.head()
    
    
    print(train.columns)
