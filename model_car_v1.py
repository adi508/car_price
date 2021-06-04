# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 11:32:09 2021

@author: Adi
"""
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf

# upload data from pickle
filename_data_in = 'process_data'
infile = open(filename_data_in,'rb')
all_data = pickle.load(infile)
infile.close()

# upload columns dic from pickle
filename_dic_of_colu = 'dic_columns'
infile = open(filename_dic_of_colu,'rb')
dic_of_colu = pickle.load(infile)
infile.close()

print(dic_of_colu.keys()) # dict_keys(['category_colum', 'num_colum', 'bin_colum', 'target_colum'])


models = [] # restart list of models
inputs = [] # restart lisy of inputs 


all_input_col_name = dic_of_colu['num_colum']+dic_of_colu['bin_colum']+dic_of_colu['category_colum']
input_dict ={}
for col in all_input_col_name:
    key_name = 'input_' + col
    input_dict[key_name] = all_data[col]
    

# input layers+embeddig for category data
for cat_colum in dic_of_colu['category_colum']:
    vocab_size = all_data[cat_colum].nunique()
    embedd_size = np.int(np.ceil(np.log2(vocab_size)))
    
    inpt = tf.keras.layers.Input(shape=(1,),
                                 name='input_' + '_'.join(cat_colum.split(' ')))
    
    embed = tf.keras.layers.Embedding(vocab_size, embedd_size,trainable=True,
                                      embeddings_initializer=tf.initializers.random_normal,
                                      name='embed_' + '_'.join(cat_colum.split(' ')))(inpt)
    
    embed_rehsaped = tf.keras.layers.Reshape(target_shape=(embedd_size,),
                                             name='Reshape_' + '_'.join(cat_colum.split(' ')))(embed)
    models.append(embed_rehsaped)
    inputs.append(inpt)

# input layers for numeric and binary data
for num_colum in (dic_of_colu['num_colum']+dic_of_colu['bin_colum']):
    inpt = tf.keras.layers.Input(shape=(1,),
                                 name='input_' + '_'.join(num_colum.split(' ')))
    models.append(inpt)
    inputs.append(inpt)
    
# merge all the input layers to one fully connected model
merge_models = tf.keras.layers.concatenate(models)
pre_preds = tf.keras.layers.Dense(4096,activation='sigmoid',name='Dense1_sigmoid')(merge_models)
pre_preds = tf.keras.layers.BatchNormalization(name='BatchNormalization1')(pre_preds)
pre_preds = tf.keras.layers.Dropout(0.3)(pre_preds)
pre_preds = tf.keras.layers.Dense(1024,activation='tanh',name='Dense2_sigmoid')(pre_preds)
pre_preds = tf.keras.layers.BatchNormalization(name='BatchNormalization2')(pre_preds)
pre_preds = tf.keras.layers.Dropout(0.1)(pre_preds)
pre_preds = tf.keras.layers.Dense(256,activation='linear',name='Dense3_tanh')(pre_preds)
#pre_preds = tf.keras.layers.BatchNormalization(name='BatchNormalization3')(pre_preds)
pred = tf.keras.layers.Dense(1,name='top_Danse_layers')(pre_preds)

model_full = tf.keras.models.Model(inputs = inputs,
                                   outputs = pred)
model_full.summary()

my_adam = tf.keras.optimizers.Adam(learning_rate=0.001)
model_full.compile(loss='log_cosh',
                   optimizer=my_adam)

#model_full.fit(input_dict,all_data['price_usd'],epochs=50,batch_size=256)

early_stopping = tf.keras.callbacks.EarlyStopping(
    patience=30,
    min_delta=0.0001,
    restore_best_weights=True,
    )

history = model_full.fit(input_dict,
                         all_data['price_usd'],
                         batch_size=512,
                         epochs=20,
                         validation_split=0.2,
                         callbacks=[early_stopping]
                         )

history = history.cumsum()
history.plot()














