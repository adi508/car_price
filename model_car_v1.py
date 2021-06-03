# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 11:32:09 2021

@author: Adi
"""
import pandas as pd
import numpy as np
import pickle

filename_data_in = 'process_data'
filename_dic_of_colu = 'dic_columns'

infile = open(filename_data_in,'rb')
all_data = pickle.load(infile)
infile.close()

infile = open(filename_dic_of_colu,'rb')
dic_of_colu = pickle.load(infile)
infile.close()

