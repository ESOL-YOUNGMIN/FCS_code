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

    

#multi_model hyperparameter
class MyWorker(Worker):
    def __init__(self, *args, config_space=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.config_space = config_space

    def compute(self, config, budget, **kwargs):
        neurons_5 = round(config['neurons_5'])
        batch_size_5 = round(config['batch_size_5'])
        epochs_5 = config['epochs_5']
        num_dense_layers = round(config['num_dense_layers'])
        first_decay_steps_5 = config['first_decay_steps_5']
        t_mul_5 = config['t_mul_5']
        alpha_5 = config['alpha_5']
        m_mul_5 = config['m_mul_5']
        initial_learning_rate_5 = config['initial_learning_rate_5']
        l2 = config['l2']

        #model generate
        input_FES = CAE_model.get_layer('input_FES').input
        FES_1=CAE_model.get_layer('encoder_1')(input_FES)
        FES_2=CAE_model.get_layer('encoder_2')(FES_1)
        FES_3=CAE_model.get_layer('encoder_3')(FES_2)
        FES_4=CAE_model.get_layer('encoder_4')(FES_3)
        flatten=tf.keras.layers.Flatten()(FES_4)
        dense = tf.keras.layers.Dense(units=neurons_5, activation='relu', bias_initializer='zeros', use_bias=True,
                                          kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(l2),name='fc_1')(flatten)
        for i in range(num_dense_layers):
            dense = tf.keras.layers.Dense(units=neurons_5, activation='relu', bias_initializer='zeros', use_bias=True,
                                              kernel_initializer='he_normal',  kernel_regularizer=tf.keras.regularizers.l2(l2), name=f'fc_{i+2}')(dense)
        output_CAE_multi_layer = tf.keras.layers.Dense(units=3, activation='linear', kernel_initializer='glorot_normal', name='output_layer')(dense)   
        
        BO_CAE_multi_model = Model(inputs=input_FES, outputs=output_CAE_multi_layer)
        BO_CAE_multi_model.summary()
        lr_schedule5 = tf.keras.optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate_5,
            first_decay_steps=first_decay_steps_5,
            t_mul=t_mul_5,
            m_mul=m_mul_5,
            alpha=alpha_5
        )
        early_stop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', restore_best_weights=True, patience=10, verbose=1)
        
        adam5 = tf.keras.optimizers.legacy.Adam(learning_rate=lr_schedule5)
        BO_CAE_multi_model.compile(optimizer=adam5, loss='mse', metrics=['mse'])
        hist_BO_CAE_multi = BO_CAE_multi_model.fit(norm_DataTrainSpec, Total_Train,
                                            validation_data=(norm_DataValidSpec, Total_Valid),
                                            epochs=epochs_5,
                                            callbacks=[early_stop_callback],
                                            batch_size=batch_size_5, shuffle=True, verbose=2) 
        
        BO_scores = hist_BO_CAE_multi.history['val_loss'][-1]

        print(BO_scores)

        return {'loss': BO_scores, 
                'info':{
                'BO_scores':BO_scores}
                }


# Define the configuration space
def get_configspace():
        config_space = CS.ConfigurationSpace()
        config_space.add_hyperparameters([
        CSH.UniformFloatHyperparameter('alpha_5', lower=0.00001, upper=0.01),
        CSH.UniformIntegerHyperparameter('batch_size_5', lower=16, upper=256),
        CSH.UniformFloatHyperparameter('first_decay_steps_5', lower=1000, upper=10000),
        CSH.UniformFloatHyperparameter('initial_learning_rate_5', lower=0.00001, upper=0.01),
        CSH.UniformFloatHyperparameter('m_mul_5', lower=0.5, upper=0.96),
        CSH.UniformIntegerHyperparameter('neurons_5', lower=32, upper=512),
        CSH.UniformIntegerHyperparameter('num_dense_layers', lower=0, upper=5),
        CSH.UniformFloatHyperparameter('t_mul_5', lower=1, upper=9),
        CSH.UniformIntegerHyperparameter('epochs_5', lower=4, upper=64),
        CSH.UniformFloatHyperparameter('l2', lower= 0 , upper=0.0001 ),

        CSH.UniformIntegerHyperparameter('features', lower=8, upper=32),
        CSH.UniformIntegerHyperparameter('num_filters_2', lower=4, upper=128),
        CSH.UniformIntegerHyperparameter('num_filters_3', lower=4, upper=128),
        CSH.UniformIntegerHyperparameter('num_filters_4', lower=4, upper=128)

        ])
        return config_space
    
# Initialize and start the NameServer
name_host='127.0.0.1'
NS = hpns.NameServer(run_id='BOBH_CAE_multi', host=name_host, port=None)
ns_host, ns_port = NS.start()

# Initialize the worker
worker = MyWorker(run_id='BOBH_CAE_multi',
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
                      run_id='BOBH_CAE_multi',
                      min_budget = 1,
                      max_budget = 30)

res = bohb.run(n_iterations = 10)

# Shutdown
bohb.shutdown(shutdown_workers=True)
NS.shutdown()

# ===== Visualize =====
# Extract and plot the results
results_CAE_multi = res.get_all_runs()
best_result_CAE_multi = res.get_incumbent_id()
print(f"Best result: {best_result_CAE_multi}")

# Get and print the best configuration
incumbent_config_CAE_multi = res.get_id2config_mapping()[best_result_CAE_multi]['config']
print(f"Optimized configuration: {incumbent_config_CAE_multi}")


# Visualize the optimization history
losses = [r.loss for r in results_CAE_multi]
plt.figure(figsize=(15, 8))
plt.plot(range(len(losses)), losses, "o")
plt.grid(True)
plt.xlabel("Iteration", fontsize=14)
plt.ylabel("Validation Loss", fontsize=14)
plt.show()
