# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 13:04:52 2022

@author: Teksle
"""

from configparser import ConfigParser

config = ConfigParser()
config.read('example.ini')


for el in config['GLOBALS']:
    print(el, config['GLOBALS'][el])