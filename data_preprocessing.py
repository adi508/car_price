# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 10:07:00 2021

@author: Adi

"""

import pandas as pd
import numpy as np
from scipy.stats import rankdata
import pickle

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

# list of all category columns bame
category_colum = ['manufacturer_name','model_name','color','engine_fuel','engine_type',
                  'body_type','state','drivetrain','location_region','transmission']

# list of all columns with numeric value
num_colum = ['odometer_value','year_produced','engine_capacity','number_of_photos',
             'up_counter','duration_listed']

# list of all columns with binary value ( True/False)
bin_colum = ['feature_0','feature_1','feature_2','feature_3','feature_4','feature_5','feature_6',
             'feature_7','feature_8','feature_9','is_exchangeable','has_warranty','engine_has_gas']

target_colum = ['price_usd']

dic_of_colum_pickle = {'category_colum':category_colum,
                       'num_colum':num_colum,
                       'bin_colum':bin_colum,
                       'target_colum':target_colum}

print('test if sum of column is unchanged')
if  not len(bin_colum )+len(num_colum )+len(target_colum)+len(category_colum)==len(data_raw.columns):
    print('erorr!!!')
    

# binary data
data_proce = data_raw.copy()
for col in bin_colum:
    print('bin_colum:',col)
    #if value True, insert 1
    data_proce.loc[data_raw[col],col]=1
    #if value False, insert 0
    data_proce.loc[data_raw[col]==False,col]=0 
    data_proce[col] = pd.Series(data_proce[col], dtype='int32')


# num data
for col in (num_colum+target_colum):
    print('start norm',col)
    # define norm object
    temp_norm = My_norm()
    # fit norm to the colum data
    temp_norm.fit(data_raw[col])
    # calculale norm for all the column 
    data_proce[col] = temp_norm(data_raw[col]) 

#category data
for col in category_colum:
    print('convert category to int',col)
    # crate a list of all the unique category in a singal column
    category_name = np.unique(data_raw[col]) 
    # creat a dictionary: {category name:int} => [a,aa,b,ac] => {a:0,aa:1,b:2,ac:3}
    temp_dic  = {category_name[i]: i for i in range(0, len(category_name))}
    # convert category name in integer by dictionary
    data_proce = data_proce.replace({col: temp_dic})

#test all data and look for nan value
print(data_proce.describe())
print(data_proce.head(10))
print(data_proce.isnull().sum()) # 10 nan in "engine_capacity"
data_proce["engine_capacity"].fillna((data_proce["engine_capacity"].mean()), inplace=True)
print(data_proce.isnull().sum()) # 10 nan in "engine_capacity"
print(data_proce.dtypes) # all column are int or float


# pickle data for model
filename_data = 'process_data'
filename_dic_of_colu = 'dic_columns'

outfile = open(filename_data,'wb')
pickle.dump(data_proce,outfile)
outfile.close()

outfile = open(filename_dic_of_colu,'wb')
pickle.dump(dic_of_colum_pickle,outfile)
outfile.close()
























