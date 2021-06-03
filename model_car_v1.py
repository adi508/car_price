# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 11:32:09 2021

@author: Adi
"""
import pandas as pd
import numpy as np
import pickle

filename = 'process_data'
infile = open(filename,'rb')
all_data = pickle.load(infile)
infile.close()



