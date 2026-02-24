#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Youngmin Cha
"""

# Paper: "Machine-learning-based prediction of key operational parameters in a
#         CH4/H2/air swirl combustor from a flame chemiluminescence spectrum"
# Terminology mapping:
#   FCS  = Flame Chemiluminescence Spectrum (coded as FES for legacy compatibility)
#   Vdot = Total combustion flow rate (V̇) [L/min] — coded as FR
#   phi  = Global equivalence ratio (φ) [-] — coded as EQ
#   XH2  = H2 blend ratio (X_H2) [mol%] — coded as MIX


import time
import numpy as np
import tensorflow as tf
from keras.models import Model
from hpbandster.core.worker import Worker
from hpbandster.optimizers import BOHB as BOHB_Optimizer
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import hpbandster.core.nameserver as hpns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import math
import random
import statistics
import warnings
import cv2
from pandas import set_option
from matplotlib import pyplot as plt
from math import floor
from numpy import sqrt
from numpy import arange
from numpy import array, hstack
from timeit import default_timer as timer
import scipy.io 
from scipy import interpolate
import seaborn as sns
import tensorflow as tf
from keras import optimizers
from keras import callbacks
from keras import layers, initializers
from keras.models import Sequential,load_model,Model
from tensorflow.python.keras.utils import np_utils
from tensorflow.keras.utils import plot_model
from keras.optimizers import Adam
from keras.layers import Activation, Dropout, Flatten, Dense, Bidirectional,BatchNormalization, concatenate 
from keras.layers import Conv1D, MaxPooling1D,LSTM, GRU, RepeatVector, TimeDistributed
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from tensorflow.python.client import device_lib

from sklearn.preprocessing import LabelEncoder, PowerTransformer
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold, mutual_info_regression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error as mse


# ===== BOHB =====

# FR model hyperparameter
class MyWorker(Worker):
    def __init__(self, *args, config_space=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.config_space = config_space

    def compute(self, config, budget, **kwargs):
        neurons_MIX = round(config['neurons_MIX'])
        batch_size_MIX = round(config['batch_size_MIX'])
        epochs_MIX = config['epochs_MIX']
        num_dense_layers = round(config['num_dense_layers'])
        first_decay_steps_MIX = config['first_decay_steps_MIX']
        t_mul_MIX = config['t_mul_MIX']
        alpha_MIX = config['alpha_MIX']
        m_mul_MIX = config['m_mul_MIX']
        initial_learning_rate_MIX = config['initial_learning_rate_MIX']
        l2 = config['l2']
        
        #model generate
        input_FES = tf.keras.layers.Input(shape=(1600,1),name='input_FES')
        encoded_FES = tf.keras.layers.Conv1D(filters = config['num_filters_2'],  activation ='relu', kernel_size = 4, padding = 'valid', strides = 4, kernel_initializer='he_normal', name ='encoder_1')(input_FES)
        encoded_FES = tf.keras.layers.Conv1D(filters = config['num_filters_3'],  activation ='relu', kernel_size = 5, padding = 'valid', strides = 5, kernel_initializer='he_normal', name ='encoder_2')(encoded_FES)
        encoded_FES = tf.keras.layers.Conv1D(filters = config['num_filters_4'],  activation ='relu', kernel_size = 8, padding = 'valid', strides = 8, kernel_initializer='he_normal', name ='encoder_3')(encoded_FES)
        encoded_FES = tf.keras.layers.Conv1D(filters = config['features'],  activation ='relu', kernel_size = 10,padding = 'valid', strides = 1, kernel_initializer='he_normal', name ='encoder_4')(encoded_FES)
            
        flatten=tf.keras.layers.Flatten()(encoded_FES)
        dense = tf.keras.layers.Dense(units=neurons_MIX, activation='relu', bias_initializer='zeros', use_bias=True,
                                          kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(l2),name='fc_1')(flatten)
        for i in range(num_dense_layers):
            dense = tf.keras.layers.Dense(units=neurons_MIX, activation='relu', bias_initializer='zeros', use_bias=True,
                                              kernel_initializer='he_normal',  kernel_regularizer=tf.keras.regularizers.l2(l2), name=f'fc_{i+2}')(dense)
        output_MIX_layer = tf.keras.layers.Dense(units=1, activation='linear', kernel_initializer='glorot_normal', name='output_layer')(dense)   
        
        BO_CAE_MIX_model = Model(inputs=input_FES, outputs=output_MIX_layer)
        BO_CAE_MIX_model.summary()
        lr_schedule_MIX = tf.keras.optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate_MIX,
            first_decay_steps=first_decay_steps_MIX,
            t_mul=t_mul_MIX,
            m_mul=m_mul_MIX,
            alpha=alpha_MIX
        )
        early_stop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                               restore_best_weights=True, patience=10, verbose=1)
        
        adam1 = tf.keras.optimizers.legacy.Adam(learning_rate=lr_schedule_MIX)
        BO_CAE_MIX_model.compile(optimizer=adam1, loss='mse', metrics=['mse'])
        hist_BO_CAE_MIX = BO_CAE_MIX_model.fit(norm_DataTrainSpec, MIX_Train,
                                           validation_data=(norm_DataValidSpec, MIX_Valid),
                                           epochs=epochs_MIX,
                                           callbacks=[early_stop_callback],
                                           batch_size=batch_size_MIX, shuffle=True, verbose=2) 
        
        BO_scores = hist_BO_CAE_MIX.history['val_loss'][-1]


        print(BO_scores)

        return {'loss': BO_scores, 
                'info':{
                'BO_scores':BO_scores}
                }




# Define the configuration space
def get_configspace():
        config_space = CS.ConfigurationSpace()
        config_space.add_hyperparameters([
        CSH.UniformFloatHyperparameter('alpha_MIX', lower=0.00001, upper=0.01),
        CSH.UniformIntegerHyperparameter('batch_size_MIX', lower=16, upper=256),
        CSH.UniformFloatHyperparameter('first_decay_steps_MIX', lower=1000, upper=10000),
        CSH.UniformFloatHyperparameter('initial_learning_rate_MIX', lower=0.00001, upper=0.01),
        CSH.UniformFloatHyperparameter('m_mul_MIX', lower=0.5, upper=0.96),
        CSH.UniformIntegerHyperparameter('neurons_MIX', lower=32, upper=512),
        CSH.UniformIntegerHyperparameter('num_dense_layers', lower=0, upper=5),
        CSH.UniformFloatHyperparameter('t_mul_MIX', lower=1, upper=9),
        CSH.UniformIntegerHyperparameter('epochs_MIX', lower=4, upper=64),
        CSH.UniformFloatHyperparameter('l2', lower= 0 , upper=0.0001 ),
        
        CSH.UniformIntegerHyperparameter('features', lower=8, upper=32),
        CSH.UniformIntegerHyperparameter('num_filters_2', lower=4, upper=128),
        CSH.UniformIntegerHyperparameter('num_filters_3', lower=4, upper=128),
        CSH.UniformIntegerHyperparameter('num_filters_4', lower=4, upper=128)

        ])
        return config_space
    
# Initialize and start the NameServer
name_host='127.0.0.1'
NS = hpns.NameServer(run_id='BOBH_CAE_MIX', host=name_host, port=None)
ns_host, ns_port = NS.start()

# Initialize the worker
worker = MyWorker(run_id='BOBH_CAE_MIX',
             host=name_host,
             nameserver=ns_host,
             nameserver_port=ns_port,
             config_space=get_configspace())
worker.run(background=True)

# Initialize the optimizer
bohb = BOHB_Optimizer(configspace=get_configspace(), 
                      host=name_host,
                      nameserver=ns_host,
                      nameserver_port=ns_port,  
                      run_id='BOBH_CAE_MIX',
                      min_budget = 1,
                      max_budget = 30)

res = bohb.run(n_iterations = 10)

# Shutdown
bohb.shutdown(shutdown_workers=True)
NS.shutdown()

# ===== Visualize =====
# Extract and plot the results
results_CAE_MIX = res.get_all_runs()
best_result_CAE_MIX = res.get_incumbent_id()
print(f"Best result: {best_result_CAE_MIX}")

# Get and print the best configuration
incumbent_config_CAE_MIX = res.get_id2config_mapping()[best_result_CAE_MIX]['config']
print(results_CAE_MIX)
print(f"Optimized configuration: {incumbent_config_CAE_MIX}")
