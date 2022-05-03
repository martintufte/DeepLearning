# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 14:59:39 2022

@author: martigtu@stud.ntnu.no
"""

import numpy as np
import pandas as pd

from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler, StandardScaler



class preprocesser:
    '''
    Class for doing preprocessing of inputs
    '''
    def __init__(self, clamp_y=True, use_date_time_features=True, use_lag_featues=True):
        
        # boolean variables for feature engineering
        self.is_fitted = False
        self.clamp_y = clamp_y
        self.use_date_time_features = use_date_time_features
        self.use_lag_featues = use_lag_featues
        
        # normalizing / standardizing with sklearn preprocessing
        self.min_max_scaler = MinMaxScaler(feature_range=(0, 1))
        self.standard_scaler = StandardScaler()
        
        # set which values that are min-maxed and standardized
        self.min_max_var = ['hydro', 'micro', 'thermal', 'wind', 'river', 'total']
        self.standard_var = ['sys_reg', 'flow', 'y']
    

    def fit(self, df):
        if self.clamp_y:
            self.lower, self.upper = df['y'].quantile(0.005),  df['y'].quantile(0.995)
        df['y_not_clamped'] = df['y']
        
        df['y'].clip(self.lower, self.upper, inplace=True)
        
        self.min_max_scaler.fit(df[self.min_max_var])
        self.standard_scaler.fit(df[self.standard_var])
        
        # add back correct 'y'
        df['y'] = df['y_not_clamped']
        df.drop(columns='y_not_clamped')
        
        self.is_fitted = True
    
    
    def transform(self, df):
        if not self.is_fitted:
            print('Preprocessor is not fitted. Automatically fit to the input data.')
            self.fit(df)
        
        
        if self.clamp_y:
            df['y'].clip(self.lower, self.upper, inplace=True)
        
        
        # Transform with the min max scaler
        df[self.min_max_var] = self.min_max_scaler.transform(df[self.min_max_var])
        
        # Transform with the standard scaler
        df[self.standard_var] = self.standard_scaler.transform(df[self.standard_var])
        
        
        
        if self.use_date_time_features:
            # convert 'start time' to datetime
            df['start_time'] = pd.to_datetime(df['start_time'])
            
            # add one-hot encoding for time of day, time of week and time of year
            
            time_of_year = df.start_time.dt.month_name()
            df['is_winter'] = time_of_year.isin(['December', 'January', 'February'])
            df['is_spring'] = time_of_year.isin(['March', 'April', 'May'])
            df['is_summer'] = time_of_year.isin(['June', 'July', 'August'])
            df['is_fall']   = time_of_year.isin(['September', 'October', 'November'])
            
            time_of_week = df.start_time.dt.day_name()
            df['is_weekday'] = time_of_week.isin(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'])
            df['is_weekend'] = time_of_week.isin(['Saturday', 'Sunday'])
            
            time_of_day = df.start_time.dt.hour
            df['is_night'] = np.logical_and(0 <= time_of_day, time_of_day < 6)
            df['is_morning'] = np.logical_and(6 <= time_of_day, time_of_day < 12)
            df['is_midday'] = np.logical_and(12 <= time_of_day, time_of_day < 18)
            df['is_evening'] = np.logical_and(18 <= time_of_day, time_of_day < 24)
    
        
        if self.use_lag_featues:
            
            df['previous_y'] = df['y'].shift(1)
            
            df['lag_one_day'] = df['y'].shift(24*12)
            df['lag_two_days'] = df['y'].shift(2*24*12)
            df['lag_one_week'] = df['y'].shift(7*24*12)
        
        # drop rows containing NaN-values
        return df.dropna().reset_index()
    
    
    def fit_transform(self, df):
        self.fit(df)
        return self.transform(df)
    
    
    
    def inverse_transform(self, df):
        # Inverse normalize with the min max scaler
        df[self.min_max_var] = self.min_max_scaler.inverse_transform(df[self.min_max_var])
        
        # Inverse transform with the standard scaler
        df[self.standard_var] = self.standard_scaler.inverse_transform(df[self.standard_var])
        
        return df




def create_sequences(df, n_seq, inputs, outputs):
    ''' Create sequences data. '''
    
    n_samples = len(df) - n_seq
    sequences = np.zeros((n_samples, n_seq, len(inputs)))
    
    print('Shape of sequences:', sequences.shape)
    
    for i in tqdm(range(n_samples)):
        sequences[i,:,:] = df.loc[i:(i+n_seq-1), inputs]
        
    return sequences, np.array(df.loc[n_seq:, outputs])



