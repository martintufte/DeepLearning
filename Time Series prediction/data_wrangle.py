# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 14:59:39 2022

@author: martigtu@stud.ntnu.no
"""

import numpy as np
import pandas as pd

from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def preprocess(df, df_val):
    # convert start_time to datetime
    df['start_time'] = pd.to_datetime(df['start_time'])
    df_val['start_time'] = pd.to_datetime(df_val['start_time'])

    # clamp the target variable y
    lower, upper = df['y'].quantile(0.005),  df['y'].quantile(0.995)
    df['y'].clip(lower, upper, inplace=True)
    df_val['y'].clip(lower, upper, inplace=True)

    
    # Normalize with a min max scaler for the planned power production
    min_max_var = ['hydro', 'micro', 'thermal', 'wind', 'river', 'total']
    min_max_scaler = MinMaxScaler(feature_range=(0, 1))
    df[min_max_var] = min_max_scaler.fit_transform(df[min_max_var])
    df_val[min_max_var] = min_max_scaler.transform(df_val[min_max_var])
    
    
    # Normalize with a standard scaler for the system regulation, planned flow and imbalance predictions
    standard_var = ['sys_reg', 'flow', 'y']
    standard_scaler = StandardScaler()
    df[standard_var] = standard_scaler.fit_transform(df[standard_var])
    df_val[standard_var] = standard_scaler.transform(df_val[standard_var])
    
    
    # Add time features
    df['time_of_day']  = df.start_time.dt.hour
    df['time_of_week'] = df.start_time.dt.day_name()
    df['time_of_year'] = df.start_time.dt.month_name()
    df_val['time_of_day']  = df_val.start_time.dt.hour
    df_val['time_of_week'] = df_val.start_time.dt.day_name()
    df_val['time_of_year'] = df_val.start_time.dt.month_name()
    
    ## Add one-hot encoding for season and time of week
    
    #df['is_summer'] = df['time_of_year'].isin(['June', 'July', 'August'])
    df['is_fall'] = df['time_of_year'].isin(['September', 'October', 'November'])
    df['is_winter'] = df['time_of_year'].isin(['December', 'January', 'February'])
    df['is_spring'] = df['time_of_year'].isin(['March', 'April', 'May'])
    #df_val['is_summer'] = df_val['time_of_year'].isin(['June', 'July', 'August'])
    df_val['is_fall'] = df_val['time_of_year'].isin(['September', 'October', 'November'])
    df_val['is_winter'] = df_val['time_of_year'].isin(['December', 'January', 'February'])
    df_val['is_spring'] = df_val['time_of_year'].isin(['March', 'April', 'May'])
    
    #df['is_weekday'] = df['time_of_week'].isin(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'])
    df['is_weekend'] = df['time_of_week'].isin(['Saturday', 'Sunday'])
    #df_val['is_weekday'] = df_val['time_of_week'].isin(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'])
    df_val['is_weekend'] = df_val['time_of_week'].isin(['Saturday', 'Sunday'])
    
    #df['is_night'] = np.logical_and(0 <= df['time_of_day'], df['time_of_day'] < 6)
    df['is_morning'] = np.logical_and(6 <= df['time_of_day'], df['time_of_day'] < 12)
    df['is_midday'] = np.logical_and(12 <= df['time_of_day'], df['time_of_day'] < 18)
    df['is_evening'] = np.logical_and(18 <= df['time_of_day'], df['time_of_day'] <= 23)
    #df_val['is_night'] = np.logical_and(0 <= df_val['time_of_day'], df['time_of_day'] < 6)
    df_val['is_morning'] = np.logical_and(6 <= df_val['time_of_day'], df['time_of_day'] < 12)
    df_val['is_midday'] = np.logical_and(12 <= df_val['time_of_day'], df['time_of_day'] < 18)
    df_val['is_evening'] = np.logical_and(18 <= df_val['time_of_day'], df['time_of_day'] <= 23)
    

    ## Add previous y
    df['previous_y'] = df['y'].shift(1)
    df.loc[0,'previous_y'] = df.loc[1,'previous_y']
    df_val['previous_y'] = df_val['y'].shift(1)
    df_val.loc[0,'previous_y'] = df_val.loc[1,'previous_y']
    
    # Add lag features
    
    # 24 hour lag imbalance (= 288 periods of 5 min)
    df['lag_24_hours_y'] = df['y'].diff(periods=288)
    df.loc[0:287,'lag_24_hours_y'] = 0
    df_val['lag_24_hours_y'] = df_val['y'].diff(periods=288)
    df_val.loc[0:287,'lag_24_hours_y'] = 0




def create_sequences(df, n_seq, inputs, outputs):
    ''' Create sequences data. '''
    
    n_samples = len(df) - n_seq
    sequences = np.zeros((n_samples, n_seq, len(inputs)))
    
    for i in tqdm(range(n_samples)):
        sequences[i] = df.loc[i:(i+n_seq-1), inputs]
        
    return sequences, df.loc[n_seq:, outputs]



