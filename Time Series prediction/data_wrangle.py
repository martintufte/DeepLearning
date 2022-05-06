# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 14:59:39 2022

@author: martigtu@stud.ntnu.no
"""

import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler, StandardScaler



class preprocesser:
    '''
    Class for doing preprocessing of inputs. The order is as follows:
        0. Fit the preprocessing parameters.
        1. Fix data types.
        2. Clean the target 'y'.
        3. If alternative forecast: calculate the sum of planned production
           and flow in the area and fit and interpolation to this. Then remove
           it from the target variable to explain some of the variance.
        4. Standardize.
        5. Add date time features.
        6. Add lag features.
        7. Remove potential starting columns with NaN-values. This occurs when
           adding lagged feautes.        
    '''
    
    def __init__(self, use_clamped_y=True, use_dt_features=True,
                 use_lag_featues=True, use_alternative=False):
        # boolean for knowing if the preprocessor is fitted        
        self.is_fitted = False
        # boolean variables for feature engineering
        self.use_clamped_y   = use_clamped_y
        self.use_dt_features = use_dt_features
        self.use_lag_features = use_lag_featues
        self.use_alternative = use_alternative
        # lower / upper values for clamping the target varrible
        self.y_lower = None
        self.y_upper = None
        # standardizing with sklearn preprocessing StandardScaler
        self.standard_scaler = StandardScaler()
        self.standard_var = ['hydro', 'micro', 'thermal', 'wind', 'river', 'total', 'sys_reg', 'flow', 'y']
    
        # normalizing with sklearn preprocessing MinMaxScaler
        #self.min_max_scaler = MinMaxScaler(feature_range=(-0.5, 0.5))
        #self.min_max_var = ['hydro', 'micro', 'thermal', 'wind', 'river', 'total', 'sys_reg', 'flow', 'y']
        

    def fit(self, df):
        if self.use_clamped_y:
            self.y_lower = df['y'].quantile(0.005)
            self.y_upper = df['y'].quantile(0.995)
            
            # store temporary 'y' and clamp the correct 'y'
            y_temp = df['y'].copy()
            df['y'].clip(self.y_lower, self.y_upper, inplace=True)
            
            # fit the normalizing / standardizing of variables
            self.standard_scaler.fit(df[self.standard_var])
            #self.min_max_scaler.fit(df[self.min_max_var])
            
            # add back correct 'y'
            df['y'] = y_temp
        else:
            # fit the normalizing / standardizing of variables
            self.standard_scaler.fit(df[self.standard_var])
            #self.min_max_scaler.fit(df[self.min_max_var])
            
        self.is_fitted = True
    
    
    def transform(self, df):
        # Check that the preprocessor is fitted
        assert self.is_fitted, 'Preprocessor is not fitted yet!'
        
        # 1. Fix data types: Set 'start time' to datetime type.
        df['start_time'] = pd.to_datetime(df['start_time'])
        
        # 2. Clean target 'y': Clamp upper and lower quantiles of y.
        if self.use_clamped_y:
            df['y'].clip(self.y_lower, self.y_upper, inplace=True)
        
        # 3. Alternative forecast: Remove the structural imbalance.
        if self.use_alternative:
            # find midpoints for 'total' and 'flow' (i.e. minute hand is at 30)
            minute = df["start_time"].dt.minute
            interp_points = np.array(minute.isin([30]))
            interp_points[[0,-1]] = True # add endpoints
            interp_indecies = interp_points.nonzero()[0]
            
            # define planned_comsumption = total - flow
            planned_consumption = df['total'] - df['flow']
            
            f = sp.interpolate.interp1d(interp_indecies, planned_consumption[interp_indecies],
                    kind = "cubic", fill_value = "extrapolate", assume_sorted = True)
            
            # calculate the structural imbalance, remove from the target variable
            interpolated_consumption = f(df.index)
            structural_imbalance = planned_consumption - interpolated_consumption
            # remove structural imbalance from the imbalance
            df['y'] -= structural_imbalance
            
            # verbose
            if True:
                # plot first 100 points in planned comsumption
                plt_indecies = interp_indecies[interp_indecies<=200]
                
                plt.plot(range(201),planned_consumption[0:201])
                plt.scatter(plt_indecies, planned_consumption[plt_indecies], s=2)
                plt.plot(range(201), interpolated_consumption[0:201])
                plt.savefig('test.pdf')
        
        # 4. Normalize / Standardize. Transform with the min-max and standard scaler
        df[self.standard_var] = self.standard_scaler.transform(df[self.standard_var])
        #df[self.min_max_var] = self.min_max_scaler.transform(df[self.min_max_var])
        
        # 5. Add date time features.
        if self.use_dt_features:
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
    
        # 6. Add lag features.
        df['previous_y'] = df['y'].shift(1) # always use this
        if self.use_lag_features:
            df['lag_one_day'] = df['y'].shift(24*12)
            df['lag_one_week'] = df['y'].shift(7*24*12)
        
        # Drop rows containing NaN-values
        return df.dropna().reset_index(drop=True)
    
    
    def fit_transform(self, df):
        ''' Fit the preprocessor to the data and transform. '''
        self.fit(df)
        return self.transform(df)
    
    
    def inverse_transform(self, df):
        ''' Inverse standardize with the scalers. '''
        #df[self.min_max_var] = self.min_max_scaler.inverse_transform(df[self.min_max_var])
        df[self.standard_var] = self.standard_scaler.inverse_transform(df[self.standard_var])
        return df


    def get_input_features(self):
        '''
        Return a list of the input features, not including the target variable.
        Note: This also makes sure that previous_y always has index 0.
        '''
        input_features = ['previous_y', 'hydro', 'micro', 'thermal', 'wind', 'river', 'total', 'sys_reg', 'flow']
        
        if self.use_dt_features:
            input_features += ['is_winter', 'is_spring', 'is_summer', 'is_fall',
                               'is_weekday', 'is_weekend',
                               'is_night', 'is_morning', 'is_midday', 'is_evening']
            
        if self.use_lag_features:
            input_features += ['lag_one_day', 'lag_one_week']
        
        return input_features




def create_sequences(df, n_seq, input_features):
    ''' Create sequences data. '''
    
    n_samples = len(df) - n_seq
    sequences = np.zeros((n_samples, n_seq, len(input_features)))
    
    print('Shape of sequences:', sequences.shape)
    
    for i in tqdm(range(n_samples)):
        sequences[i,:,:] = df.loc[i:(i+n_seq-1), input_features]
        
    return sequences, np.array(df.loc[n_seq:, 'y'])


