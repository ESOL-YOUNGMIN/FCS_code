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

#split Data case
#random split Data case
num_case = [[i] for i in range(441)] #0-440

#random_state=42 #seed fix
num_Train, num_Total = train_test_split(num_case, test_size=0.4, random_state=8) #Train60%, Total(Test+Valid)40%, 
num_Test, num_Valid = train_test_split(num_Total, test_size=0.5, random_state=8) #Test 20%, Valid 20%

print('Train case = ',len(num_Train), '/','Test case = ',len(num_Test),'/' ,'Valid case = ',len(num_Valid))

num_Train=np.array(num_Train) #random Train number
num_Test=np.array(num_Test) #random Test number
num_Valid=np.array(num_Valid) #random Valid number

DataTrainSpec=np.zeros(1600*500*len(num_Train)).reshape(len(num_Train)*500,1600)
DataTestSpec=np.zeros(1600*500*len(num_Test)).reshape(len(num_Test)*500,1600)
DataValidSpec=np.zeros(1600*500*len(num_Valid)).reshape(len(num_Valid)*500,1600)
   
Y=np.zeros(441*3).reshape(441,3)
#Y= Real_value

for k in range (0,7):                                  
    for j in range (0,7):                            
        for  i in range(0,9):                       
            for m in range (0,1):
                Y[63*k+9*j+i+m]=Real_value.loc[63*k+9*j+i]
                
#  FR_Train = Total flow rate label (Vdot)
#  EQ_Train = Equivalence ratio label (phi)
#  MIX_Train = H2 blend ratio label (XH2)

FR_Train=np.zeros(len(num_Train)*500)
EQ_Train=np.zeros(len(num_Train)*500)
MIX_Train=np.zeros(len(num_Train)*500)

FR_Test=np.zeros(len(num_Test)*500)
EQ_Test=np.zeros(len(num_Test)*500)
MIX_Test=np.zeros(len(num_Test)*500)

FR_Valid=np.zeros(len(num_Valid)*500)
EQ_Valid=np.zeros(len(num_Valid)*500)
MIX_Valid=np.zeros(len(num_Valid)*500)    
 
          
num_Data= 0 #0-440
num_Split=0

Data_Train=0
Data_Test=0
Data_Valid=0

for ii in range(len(flowRate)):
    for jj in range(len(equiRatio)):
        for kk in range(len(mixRatio)):
            print([ii, jj, kk])
            if num_Data in num_Train:
                num_Split = 0   #Train     
                
            elif num_Data in num_Test:
                num_Split = 1   #Test
                
            elif num_Data in num_Valid:
                num_Split = 2   #Valid
                
            else:
                print('Data_split_error')
                
            filename0 = os.path.join(file_path_spectrum, str(flowRate[ii]), str(equiRatio[jj]), str(mixRatio[kk]), "*.xlsx")
            d = os.listdir(os.path.dirname(filename0))
            filename = os.path.join(file_path_spectrum, str(flowRate[ii]), str(equiRatio[jj]), str(mixRatio[kk]), d[0])

            spectra = pd.read_excel(filename, header=None).values
            spectra = spectra[5:1605, 1:num_of_Files+1]
            spectra[297, :] = (spectra[296, :] + spectra[298, :]) / 2.0
            spectra[1427, :] = (spectra[1426, :] + spectra[1428, :]) / 2.0
                
            for idx in range(num_of_Files):
                #Train
                if num_Split == 0:
                    DataTrainSpec[Data_Train,:] = spectra[:, idx]
                    FR_Train[Data_Train]=Y[num_Data,0]
                    EQ_Train[Data_Train]=Y[num_Data,1]
                    MIX_Train[Data_Train]=Y[num_Data,2]
                    Data_Train = Data_Train+1  
                    
                #Test    
                elif num_Split == 1:
                    DataTestSpec[Data_Test,:] = spectra[:, idx]
                    FR_Test[Data_Test]=Y[num_Data,0]
                    EQ_Test[Data_Test]=Y[num_Data,1]
                    MIX_Test[Data_Test]=Y[num_Data,2]
                    Data_Test = Data_Test+1     
                    
                #Valid    
                elif num_Split == 2:
                    DataValidSpec[Data_Valid,:] = spectra[:, idx]
                    FR_Valid[Data_Valid]=Y[num_Data,0]
                    EQ_Valid[Data_Valid]=Y[num_Data,1]
                    MIX_Valid[Data_Valid]=Y[num_Data,2]
                    Data_Valid = Data_Valid+1  
            num_Data = num_Data+1
            
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
del  Real_value, spectra,  num_Train, num_Test,  num_Valid,num_Total, max_index
 
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
# ===== BOHB CAE Model =====


#CAE_model hyperparameter
class MyWorker(Worker):
    def __init__(self, *args, config_space=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.config_space = config_space
        
    def compute(self, config, budget, **kwargs):
        
        #model generate
        input_FES = tf.keras.layers.Input(shape=(1600,1),name='input_FES')
        encoded_FES = tf.keras.layers.Conv1D(filters = config['num_filters_2'],  activation ='relu', kernel_size = 4, padding = 'valid', strides = 4, kernel_initializer='he_normal', name ='encoder_1')(input_FES)
        encoded_FES = tf.keras.layers.Conv1D(filters = config['num_filters_3'],  activation ='relu', kernel_size = 5, padding = 'valid', strides = 5, kernel_initializer='he_normal', name ='encoder_2')(encoded_FES)
        encoded_FES = tf.keras.layers.Conv1D(filters = config['num_filters_4'],  activation ='relu', kernel_size = 8, padding = 'valid', strides = 8, kernel_initializer='he_normal', name ='encoder_3')(encoded_FES)
        encoded_FES = tf.keras.layers.Conv1D(filters = config['features'],  activation ='relu', kernel_size = 10,padding = 'valid', strides = 1, kernel_initializer='he_normal', name ='encoder_4')(encoded_FES)
            
        decoded_FES = tf.keras.layers.Conv1DTranspose(filters = config['num_filters_4'], activation ='relu', kernel_size = 10, padding = 'valid', strides =  1, kernel_initializer='he_normal', name ='decoder_1')(encoded_FES)
        decoded_FES = tf.keras.layers.Conv1DTranspose(filters = config['num_filters_3'], activation ='relu', kernel_size =  8, padding = 'valid', strides =  8 , kernel_initializer='he_normal', name ='decoder_2')(decoded_FES)
        decoded_FES = tf.keras.layers.Conv1DTranspose(filters = config['num_filters_2'], activation ='relu', kernel_size =  5, padding = 'valid', strides =  5 , kernel_initializer='he_normal', name ='decoder_3')(decoded_FES)
        decoded_FES = tf.keras.layers.Conv1DTranspose(filters = 1, activation ='relu', kernel_size =  4, padding = 'valid', strides =  4 , kernel_initializer='he_normal', name ='decoder_4')(decoded_FES)
    
        
        BO_CAE_model = Model(inputs=input_FES, outputs=decoded_FES)
        BO_CAE_model.summary()
       # print(config['num_layers'])

        lr_schedule4 = tf.keras.optimizers.schedules.CosineDecayRestarts(
            config['initial_learning_rate4'],
            first_decay_steps=config['first_decay_steps4'],
            t_mul=config['t_mul_4'],
            m_mul=config['m_mul_4'],
            alpha=config['alpha_4']
        )
        early_stop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', restore_best_weights=True, patience=10, verbose=1)
        
        adam4 = tf.keras.optimizers.legacy.Adam(learning_rate=lr_schedule4)
        BO_CAE_model.compile(optimizer=adam4, loss='mse', metrics=['mse'])
   
        hist_BO_CAE = BO_CAE_model.fit(norm_DataTrainSpec , norm_DataTrainSpec, 
                                            validation_data = (norm_DataValidSpec, norm_DataValidSpec),
                                            epochs=config['epochs_4'],
                                            callbacks=[early_stop_callback],
                                            batch_size=config['batch_size4'], shuffle=True, verbose=2) 
        
        BO_scores = hist_BO_CAE.history['val_loss'][-1]
        #-r2_score(CAE_Valid, predict_Valid_CAE)
       # print('Valid_r2score',r2_score(CAE_Valid, predict_Valid_CAE))

        return {'loss': BO_scores, 
                'info':{
                #'Valid_r2score':r2_score(CAE_Valid, predict_Valid_CAE),
                'BO_scores':BO_scores}
                }

# Define the configuration space
def get_configspace():
    
        config_space = CS.ConfigurationSpace()
        
        config_space.add_hyperparameters([
        CSH.UniformFloatHyperparameter('alpha_4', lower=0.00001, upper=0.01),
        CSH.UniformIntegerHyperparameter('batch_size4', lower=64, upper=256),
        CSH.UniformIntegerHyperparameter('epochs_4', lower=4, upper=64),
        CSH.UniformIntegerHyperparameter('features', lower=8, upper=24),
        CSH.UniformFloatHyperparameter('first_decay_steps4', lower=2000, upper=4000),
        CSH.UniformFloatHyperparameter('initial_learning_rate4', lower=0.00001, upper=0.01),
        CSH.UniformFloatHyperparameter('m_mul_4', lower=0.5, upper=0.9),   
        CSH.UniformFloatHyperparameter('t_mul_4', lower=5, upper=9),
        CSH.UniformIntegerHyperparameter('num_filters_2', lower=4, upper=32),
        CSH.UniformIntegerHyperparameter('num_filters_3', lower=4, upper=32),
        CSH.UniformIntegerHyperparameter('num_filters_4', lower=4, upper=32)
        ])
        
        
        return config_space
    
def print_configspace_info(config_space):
    print("Hyperparameters:")
    for hp in config_space.get_hyperparameters():
        print(f" - {hp.name} (Type: {hp.__class__.__name__}, Range: {hp.lower} to {hp.upper})")
    
    print("\nConditions:")
    for cond in config_space.get_conditions():
        print(f" - {cond}")

# Get and print configuration space
config_space = get_configspace()
print_configspace_info(config_space)

    
# Initialize and start the NameServer
name_host='127.0.0.1'
NS = hpns.NameServer(run_id='BOBH_CAE', host=name_host, port=None)
ns_host, ns_port = NS.start()

# Initialize the worker
worker = MyWorker(run_id='BOBH_CAE',
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
                      run_id='BOBH_CAE',
                      min_budget = 3,
                      max_budget = 30)

res = bohb.run(n_iterations = 1)

# Shutdown
bohb.shutdown(shutdown_workers=True)
NS.shutdown()

# ===== Visualize =====
# Extract and plot the results
results_CAE = res.get_all_runs()
best_result_CAE = res.get_incumbent_id()
print(f"Best result: {best_result_CAE}")

# Get and print the best configuration
incumbent_config_CAE = res.get_id2config_mapping()[best_result_CAE]['config']
print(f"Optimized configuration: {incumbent_config_CAE}")


# Visualize the optimization history
losses = [r.loss for r in results_CAE]
plt.figure(figsize=(15, 8))
plt.plot(range(len(losses)), losses, "o")
plt.grid(True)
plt.xlabel("Iteration", fontsize=14)
plt.ylabel("Validation Loss", fontsize=14)
plt.show()
tf.keras.backend.clear_session()


