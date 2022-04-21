# Test import from .py - file

import sys
sys.path.insert(0, '/home/jupyter/DeepLearning/Time Series prediction')

import numpy as np
import pandas as pd

def read_data(file_name):
    df = pd.read_csv(sys.path[0] + '/data/' + file_name)
    return df
