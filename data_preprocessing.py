# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 10:07:00 2021

@author: Adi

"""

import pandas as pd
import numpy as np
from scipy.stats import rankdata

class My_norm():
    # norm of array by rank
    # array1 =[5,6,3,4]  =>  norm(3)=0, norm(6)=1,norm(2)=-1/3,norm(4.5)=0.5
    # array2 =[1,2,2,2,4,6]  =>  norm(1)=0, norm(6)=1,norm(2)=0.4,norm(3)=0.6
    def __init__(self):
        self.is_train = False
        
    def __call__(self,num):
        if self.is_train:
            return self.normfun1(num)
            
        else:
            print('error- norm didnt fit!!!')
            return np.NAN
    
    def fit(self,array):
        def find_rank_of_num(num,sort_array_u,rank_u):
            index = np.searchsorted(sort_array_u, num)
            # calculate liniar dif
            # if x0 = x1+(x2-x1)*t  then  y0 = y1+(y2-y1)*t
            # t = (x0-x1)/(x2-x1)
            if index == len(rank_u): 
                dif = (rank_u[-1]-rank_u[-2])*(num-sort_array_u[-1])/(sort_array_u[-1]-sort_array_u[-2])
                return rank_u[-1]+dif
            
            elif index == 0:
                dif = (rank_u[1]-rank_u[0])*(sort_array_u[0]-num)/(sort_array_u[1]-sort_array_u[0])
                return rank_u[0]-dif
            
            else:
                dif = (rank_u[index]-rank_u[index-1])*(sort_array_u[index]-num)/(sort_array_u[index]-sort_array_u[index-1])
                return rank_u[index]-dif
        
        sort_array = np.sort(array)   # sort values for searchsorted
        rank = rankdata(sort_array)   # calculate rank of array (rank([5,3,8])=[2,1,3])
        sort_array_u = np.unique(sort_array) # leave only uniqe values
        rank_u = np.unique(rank)           # leave only uniqe values 
        rank_u =(rank_u-1)/(rank_u[-1]-1)  # convert rank from [1..max] to [0..0]
        out_f = lambda x : find_rank_of_num(x,sort_array_u,rank_u)
        v_f = np.vectorize(out_f)
        self.is_train = True
        self.normfun1 = v_f
        
        


data_path = r'cars.csv'

data_raw = pd.read_csv(data_path)
#print(data_raw.head())

category_colum = ['manufacturer_name','model_name','color','engine_fuel','engine_type',
                  'body_type','state','drivetrain','location_region',]
num_colum = ['odometer_value','year_produced','engine_capacity','price_usd',
              'number_of_photos','up_counter','duration_listed']

bin_colum = ['feature_0',
              'feature_1','feature_2','feature_3','feature_4','feature_5','feature_6','feature_7',
              'feature_8','feature_9','is_exchangeable','has_warranty','engine_has_gas']

category2_colum = ['transmission']

# preproces data-

#binaryi data
data_proce = data_raw.copy()
for col in bin_colum:
    data_proce.loc[data_raw[col],col]=1
    data_proce.loc[data_raw[col]==False,col]=0

# num data
for col in num_colum:
    print('start norm',col)
    temp_norm = My_norm()
    temp_norm.fit(data_raw[col])
    data_proce[col] = temp_norm(data_raw[col])

#category data
for col in category_colum:
    print('convert category to int',col)
    category_name = np.unique(data_raw[col])
    #print(len(category_name))
    temp_dic  = {category_name[i]: i for i in range(0, len(category_name))}
    #print(temp_dic)
    data_proce = data_proce.replace({col: temp_dic})
    
print(data_proce.describe())
print(data_proce.head(10))























