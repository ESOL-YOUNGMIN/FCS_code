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

# Requires data_preprocessing.py to be executed first.
# Dependencies from data_preprocessing.py: file_path_spectrum, max_spectra, pd

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate
import tensorflow as tf
import os

# ===== Grad-RAM for XH2 =====

checkpoint_path_combine_MIX= './checkpoint/combine_MIX/MIX_model.h5'
MIX_model = tf.keras.models.load_model(checkpoint_path_combine_MIX)
MIX_model.summary()

flowRate = [80, 90, 100, 110, 120, 130, 140]
equiRatio = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]
mixRatio = [0, 3.75, 7.5, 11.25, 15, 18.75, 22.5, 26.25, 30]

num_data=10
grad_ram_nosum = np.zeros(1600*num_data).reshape(1600,num_data)
mean_grad_ram = np.zeros(1600)

grad_neg = []
grad_pos = []
normal_spectra =[]
index_MM = [0,4,8]

for MM in index_MM:
    for index in range(0, num_data):
        print(index)
        filename0 = os.path.join(file_path_spectrum, str(flowRate[3]), str(equiRatio[3]), str(mixRatio[MM]), "*.xlsx")
        d = os.listdir(os.path.dirname(filename0))
        filename = os.path.join(file_path_spectrum, str(flowRate[3]), str(equiRatio[3]), str(mixRatio[MM]), d[0])
        spectra = pd.read_excel(filename, header=None).values
        spectra = spectra[5:1605, 10*index]
        spectra[297] = (spectra[296] + spectra[298]) / 2.0
        spectra[1427] = (spectra[1426] + spectra[1428]) / 2.0

        spectra = spectra/max_spectra
        train_spectra = np.zeros(1600).reshape(1,1600,1)
        train_spectra[0,:,0] = spectra

        x = tf.convert_to_tensor(train_spectra)
        grad_model = tf.keras.models.Model([MIX_model.inputs],
                                           [MIX_model.get_layer('encoder_2').output, MIX_model.output])

        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            last_conv_layer_output, preds = grad_model(x)

        grads = tape.gradient(preds,last_conv_layer_output)
        grads2 = tf.reshape(grads,[80,42])
        conv_output=tf.reshape(last_conv_layer_output,[80,42])

        heat_map1=np.zeros(80)
        for ii in range(0,42):
            for jj in range(0,80):
                heat_map1[jj] = heat_map1[jj] + grads2[jj,ii]*conv_output[jj,ii]

        f_interpol = interpolate.interp1d(np.linspace(0,80,num=80), heat_map1, kind='quadratic')
        grad_ram = f_interpol(np.linspace(0,80,1600))
        grad_ram_nosum[:,index]=grad_ram


    for ii in range (1600):
        mean_grad_ram[ii] = np.mean(grad_ram_nosum[ii,:])

    mean_grad_ram = mean_grad_ram
    grad_ram_negative = abs(np.where(mean_grad_ram > 0, 0, mean_grad_ram))
    grad_ram_positive = np.where(mean_grad_ram < 0, 0, mean_grad_ram)

    grad_neg.append(grad_ram_negative)
    grad_pos.append(grad_ram_positive)
    normal_spectra.append(train_spectra[0,:,0])

    fig, ax1 = plt.subplots()
    ax1.plot(train_spectra[0,:,0],'k',linewidth='1')
    ax1.set_ylim(-0.05,1.1)
    ax1.set_yticks(np.arange(0, 1.1, 0.2))
    ax1.set_xlabel('Wavelength(nm)')
    ax1.set_ylabel('Normalized spectrum intensity (a.u.)')

    ax2 = ax1.twinx()
    ax2.plot(grad_ram_positive,'-r',markerfacecolor='white', label='positive',linewidth='1.5')
    ax2.plot(grad_ram_negative,'-b',markerfacecolor='white', label='negative',linewidth='1.5')
    ax2.set_ylabel('Grad-RAM score(a.u.)')
    ax2.set_ylim(-0.05,1.1)
    ax2.set_yticks(np.arange(0, 1.1, 0.2))
    ax2.legend(loc='upper right')
    plt.show()
