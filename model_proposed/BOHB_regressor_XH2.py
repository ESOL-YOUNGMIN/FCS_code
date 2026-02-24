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

device_lib.list_local_devices()
CUDA_VISIBLE_DEVICES=""
os.environ['CUDA_VISIBLE_DEVICES'] = ''

#data range
flowRate = [80, 90, 100, 110, 120, 130, 140]
equiRatio = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]
mixRatio = [0, 3.75, 7.5, 11.25, 15, 18.75, 22.5, 26.25, 30]
num_of_Files = 500

file_path_spectrum = "./data/OES_data"
Real_value=pd.read_excel("./data/Realvalue.xlsx")

 
Y=np.zeros(441*3).reshape(441,3)
#Y= Real_value

for k in range (0,7):                                  
    for j in range (0,7):                            
        for  i in range(0,9):                       
            for m in range (0,1):
                Y[63*k+9*j+i+m]=Real_value.loc[63*k+9*j+i]
                

# spectrum(X)

DataTrainSpec = []
DataTestSpec = []
DataValidSpec = []
  
# label(Y)
FR_Train=[]
EQ_Train=[]
MIX_Train=[]

FR_Test=[]
EQ_Test=[]
MIX_Test=[]

FR_Valid=[]
EQ_Valid=[]
MIX_Valid=[]
          
num_Data = 0
Valid_ratio = 0.4

for ii in range(len(flowRate)):
    for jj in range(len(equiRatio)):
        for kk in range(len(mixRatio)):
            print([ii, jj, kk])
            if   (ii+jj+kk) % 2 == 0 :
                num_Split = 0   #Train     
                
            elif (ii+jj+kk) % 2 == 1:
                num_Split = 1   #Test

            else:
                print('Data_split_error')
                
            filename0 = os.path.join(file_path_spectrum, str(flowRate[ii]), str(equiRatio[jj]), str(mixRatio[kk]), "*.xlsx")
            d = os.listdir(os.path.dirname(filename0))
            filename = os.path.join(file_path_spectrum, str(flowRate[ii]), str(equiRatio[jj]), str(mixRatio[kk]), d[0])

            spectra = pd.read_excel(filename, header=None).values
            spectra = spectra[5:1605, 1:num_of_Files+1]
            spectra[297, :] = (spectra[296, :] + spectra[298, :]) / 2.0
            spectra[1427, :] = (spectra[1426, :] + spectra[1428, :]) / 2.0
            Valid_score = random.random()    
            
            for idx in range(num_of_Files):
                #Train
                if num_Split == 0:
                    DataTrainSpec.append(spectra[:, idx])
                    FR_Train.append(Y[num_Data,0])
                    EQ_Train.append(Y[num_Data,1])
                    MIX_Train.append(Y[num_Data,2])
                   
                elif num_Split == 1:
                    if Valid_ratio >= Valid_score :
                        #Valid
                        DataValidSpec.append(spectra[:, idx])
                        FR_Valid.append(Y[num_Data,0])
                        EQ_Valid.append(Y[num_Data,1])
                        MIX_Valid.append(Y[num_Data,2])
                    else:  
                        #Test  
                        DataTestSpec.append(spectra[:, idx])
                        FR_Test.append(Y[num_Data,0])
                        EQ_Test.append(Y[num_Data,1])
                        MIX_Test.append(Y[num_Data,2])
            num_Data += 1        
            
print('Train case = ',int(len(DataTrainSpec)/500), '/','Test case = ',int(len(DataTestSpec)/500),'/' ,'Valid case = ',int(len(DataValidSpec)/500))

   
FR_Train=np.array(FR_Train)
FR_Test=np.array(FR_Test)
FR_Valid=np.array(FR_Valid)

EQ_Train=np.array(EQ_Train)
EQ_Test=np.array(EQ_Test)
EQ_Valid=np.array(EQ_Valid)

MIX_Train=np.array(MIX_Train)
MIX_Test=np.array(MIX_Test)
MIX_Valid=np.array(MIX_Valid)

DataTrainSpec=np.array(DataTrainSpec)
DataTestSpec=np.array(DataTestSpec)
DataValidSpec=np.array(DataValidSpec)

if np.max(DataTrainSpec)> 20000:
    max_index = np.argmax(DataTrainSpec)/1600
    DataTrainSpec[int(max_index)] = DataTrainSpec[int(max_index)-1]
    DataTrainSpec[int(max_index)+1] = DataTrainSpec[int(max_index)-1]
    print('Train')

elif np.max(DataTestSpec)> 20000:
    max_index = np.argmax(DataTestSpec)/1600
    DataTestSpec[int(max_index)] = DataTestSpec[int(max_index)-1]
    DataTestSpec[int(max_index)+1] = DataTestSpec[int(max_index)-1]
    print('Test')
elif np.max(DataValidSpec)> 20000:
    max_index= np.argmax(DataValidSpec)/1600
    DataValidSpec[int(max_index)] = DataValidSpec[int(max_index)-1]
    DataValidSpec[int(max_index)+1] = DataValidSpec[int(max_index)-1]    
    print('Valid')

#Noise Data 
del  Real_value, spectra,   max_index
# ===== Data Normalization =====

# Find max spectra value

max_Train_spectra=np.max(DataTrainSpec)
max_Test_spectra=np.max(DataTestSpec)
max_Valid_spectra=np.max(DataValidSpec)

max_spectra = np.max([max_Train_spectra, max_Test_spectra, max_Valid_spectra])

print(max_spectra)

# Normalize spectra data
def Nomalize_spectra(Data):
    Data=np.array(Data)
    norm_spectra=np.zeros(1600*len(Data)).reshape(len(Data),1600)
    for idx in range(len(Data)):
        norm_spectra[idx,:]=Data[idx,:]/max_spectra
    return norm_spectra   

norm_DataTrainSpec= Nomalize_spectra(DataTrainSpec)
norm_DataTestSpec= Nomalize_spectra(DataTestSpec)
norm_DataValidSpec= Nomalize_spectra(DataValidSpec)
      
# Find max and min values per combustion parameter
max_FR = np.max(Y[:,0])
min_FR = np.min(Y[:,0])
max_EQ = np.max(Y[:,1])
min_EQ = np.min(Y[:,1])
max_MIX = np.max(Y[:,2])
min_MIX = np.min(Y[:,2])
    
# Normalize combustion parameters

EQ_Train = (EQ_Train - min_EQ) / (max_EQ - min_EQ) 
EQ_Test = (EQ_Test - min_EQ) / (max_EQ - min_EQ) 
EQ_Valid = (EQ_Valid - min_EQ) / (max_EQ - min_EQ) 

FR_Train = (FR_Train - min_FR) / (max_FR - min_FR) 
FR_Test = (FR_Test - min_FR) / (max_FR - min_FR) 
FR_Valid = (FR_Valid - min_FR) / (max_FR - min_FR) 

MIX_Train = (MIX_Train - min_MIX) / (max_MIX - min_MIX) 
MIX_Test = (MIX_Test - min_MIX) / (max_MIX - min_MIX) 
MIX_Valid = (MIX_Valid - min_MIX) / (max_MIX - min_MIX) 


# Total = All labels (flow rate, equivalence ratio, blend ratio)
Total_Train=np.concatenate((FR_Train.reshape(len(FR_Train),1),
                            EQ_Train.reshape(len(EQ_Train),1),
                            MIX_Train.reshape(len(MIX_Train),1)),axis=1)

Total_Test=np.concatenate((FR_Test.reshape(len(FR_Test),1),
                           EQ_Test.reshape(len(EQ_Test),1),
                           MIX_Test.reshape(len(MIX_Test),1)),axis=1) 
 
Total_Valid=np.concatenate((FR_Valid.reshape(len(FR_Valid),1),
                        EQ_Valid.reshape(len(EQ_Valid),1),
                        MIX_Valid.reshape(len(MIX_Valid),1)),axis=1)

tf.keras.backend.clear_session()

del DataTrainSpec, DataTestSpec, DataValidSpec, Y
# ===== CAE Model =====

checkpoint_path_CAE= './checkpoint/CAE/cp.ckpt'

CAE_model = tf.keras.models.load_model(checkpoint_path_CAE)
CAE_model.summary()
CAE_model.trainable=False
reconstructed_spectra = CAE_model.predict(norm_DataTrainSpec)
reconstructed_spectra.reshape(len(norm_DataTrainSpec),1600)

original_spectra=np.array(norm_DataTrainSpec)
original_spectra=original_spectra.reshape(len(norm_DataTrainSpec),1600,1)

for ii in range(0,15000,1600):
    plt.plot(original_spectra[ii],'b',linewidth=1)
    plt.plot(reconstructed_spectra[ii],'--r',alpha = 1, linewidth=0.8)
plt.show()

# ===== BOHB =====

#MIX_model hyperparameter
class MyWorker(Worker):
    def __init__(self, *args, config_space=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.config_space = config_space

    def compute(self, config, budget, **kwargs):
        neurons3 = round(config['neurons3'])
        batch_size3 = round(config['batch_size3'])
        epochs_3 = config['epochs_3']
        num_dense_layers = round(config['num_dense_layers'])
        first_decay_steps3 = config['first_decay_steps3']
        t_mul_3 = config['t_mul_3']
        alpha_3 = config['alpha_3']
        m_mul_3 = config['m_mul_3']
        initial_learning_rate3 = config['initial_learning_rate3']
        l2 = config['l2']

        
        #model generate
        input_FES = CAE_model.get_layer('input_FES').input
        FES_1=CAE_model.get_layer('encoder_1')(input_FES)
        FES_2=CAE_model.get_layer('encoder_2')(FES_1)
        FES_3=CAE_model.get_layer('encoder_3')(FES_2)
        FES_4=CAE_model.get_layer('encoder_4')(FES_3)
        flatten=tf.keras.layers.Flatten()(FES_4)
        dense = tf.keras.layers.Dense(units=neurons3, activation='relu', bias_initializer='zeros', use_bias=True,
                                          kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(l2),name='fc_1')(flatten)
        for i in range(num_dense_layers):
            dense = tf.keras.layers.Dense(units=neurons3, activation='relu', bias_initializer='zeros', use_bias=True,
                                              kernel_initializer='he_normal',  kernel_regularizer=tf.keras.regularizers.l2(l2), name=f'fc_{i+2}')(dense)
        output_MIX_layer = tf.keras.layers.Dense(units=1, activation='linear', kernel_initializer='glorot_normal', name='output_layer')(dense)   
        
        BO_MIX_model = Model(inputs=CAE_model.input, outputs=output_MIX_layer)
        BO_MIX_model.summary()
        lr_schedule3 = tf.keras.optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate3,
            first_decay_steps=first_decay_steps3,
            t_mul=t_mul_3,
            m_mul=m_mul_3,
            alpha=alpha_3
        )
        early_stop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', restore_best_weights=True, patience=10, verbose=1)
        
        adam3 = tf.keras.optimizers.legacy.Adam(learning_rate=lr_schedule3)
        BO_MIX_model.compile(optimizer=adam3, loss='mse', metrics=['mse'])
        hist_BO_MIX = BO_MIX_model.fit(norm_DataTrainSpec, MIX_Train,
                                            validation_data=(norm_DataValidSpec, MIX_Valid),
                                            epochs=epochs_3,
                                            callbacks=[early_stop_callback],
                                            batch_size=batch_size3, shuffle=True, verbose=2) 
        
      #  predict_Valid_MIX = BO_MIX_model.predict(norm_DataValidSpec)
        BO_scores = hist_BO_MIX.history['val_loss'][-1]
        #-r2_score(MIX_Valid, predict_Valid_MIX)
        #hist_BO_MIX.history['val_loss'][-1]
        #

        print(BO_scores)
      #  print('Valid_r2score',r2_score(MIX_Valid, predict_Valid_MIX))

        return {'loss': BO_scores, 
                'info':{
            #    'Valid_r2score':r2_score(MIX_Valid, predict_Valid_MIX),
                'BO_scores':BO_scores}
                }

 
    
# Define the configuration space
def get_configspace():
        config_space = CS.ConfigurationSpace()
        config_space.add_hyperparameters([
        CSH.UniformFloatHyperparameter('alpha_3', lower=0.00001, upper=0.01),
        CSH.UniformIntegerHyperparameter('batch_size3', lower=16, upper=256),
        CSH.UniformFloatHyperparameter('first_decay_steps3', lower=1000, upper=10000),
        CSH.UniformFloatHyperparameter('initial_learning_rate3', lower=0.00001, upper=0.01),
        CSH.UniformFloatHyperparameter('m_mul_3',  lower=0.5, upper=0.96),
        CSH.UniformIntegerHyperparameter('neurons3', lower=32, upper=512),
        CSH.UniformIntegerHyperparameter('num_dense_layers', lower=0, upper=4),
        CSH.UniformFloatHyperparameter('t_mul_3', lower=1, upper=9),
        CSH.UniformIntegerHyperparameter('epochs_3', lower=4, upper=64),
        CSH.UniformFloatHyperparameter('l2', lower=0, upper=0.0001 ),


        ])
        return config_space
    
# Initialize and start the NameServer
name_host='127.0.0.1'
NS = hpns.NameServer(run_id='BOBH_MIX', host=name_host, port=None)
ns_host, ns_port = NS.start()

# Initialize the worker
worker = MyWorker(run_id='BOBH_MIX',
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
                      run_id='BOBH_MIX',
                      min_budget = 1,
                      max_budget = 30)

res = bohb.run(n_iterations = 10)

# Shutdown
bohb.shutdown(shutdown_workers=True)
NS.shutdown()

# ===== Visualize =====
# Extract and plot the results
results_MIX = res.get_all_runs()
best_result_MIX = res.get_incumbent_id()
print(f"Best result: {best_result_MIX}")

# Get and print the best configuration
incumbent_config_MIX = res.get_id2config_mapping()[best_result_MIX]['config']
print(f"Optimized configuration: {incumbent_config_MIX}")


# Visualize the optimization history
losses = [r.loss for r in results_MIX]
plt.figure(figsize=(15, 8))
plt.plot(range(len(losses)), losses, "o")
plt.grid(True)
plt.xlabel("Iteration", fontsize=14)
plt.ylabel("Validation Loss", fontsize=14)
plt.show()

