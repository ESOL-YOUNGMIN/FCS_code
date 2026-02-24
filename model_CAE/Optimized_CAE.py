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


num_filters_2 = 31
num_filters_3 = 42
num_filters_4 = 27
first_decay_steps4 =  3856.7075528077808
initial_learning_rate4 = 0.0017248711714185727
features = 22
t_mul_4 = 8.25010718825792
m_mul_4 = 0.6886803586044984
alpha_4 = 0.15976745011028634
epochs_4 = 35
batch_size4 = 45

def CAE_model():
    input_FES = tf.keras.layers.Input(shape=(1600,1),name='input_FES')
    encoded_FES = tf.keras.layers.Conv1D(filters = num_filters_2,  activation ='relu', kernel_size = 4, padding = 'valid', strides = 4, kernel_initializer='he_normal', name ='encoder_1')(input_FES)
    encoded_FES = tf.keras.layers.Conv1D(filters = num_filters_3,  activation ='relu', kernel_size = 5, padding = 'valid', strides = 5, kernel_initializer='he_normal', name ='encoder_2')(encoded_FES)
    encoded_FES = tf.keras.layers.Conv1D(filters = num_filters_4,  activation ='relu', kernel_size = 8, padding = 'valid', strides = 8, kernel_initializer='he_normal', name ='encoder_3')(encoded_FES)
    encoded_FES = tf.keras.layers.Conv1D(filters = features, activation ='relu', kernel_size = 10, padding = 'valid', strides = 1, kernel_initializer='he_normal', name ='encoder_4')(encoded_FES)
    
    decoded_FES = tf.keras.layers.Conv1DTranspose(filters = num_filters_4, activation ='relu', kernel_size = 10, padding = 'valid', strides = 1 , kernel_initializer='he_normal', name ='decoder_1')(encoded_FES)
    decoded_FES = tf.keras.layers.Conv1DTranspose(filters = num_filters_3, activation ='relu', kernel_size = 8, padding = 'valid', strides = 8 , kernel_initializer='he_normal', name ='decoder_2')(decoded_FES)
    decoded_FES = tf.keras.layers.Conv1DTranspose(filters = num_filters_2, activation ='relu', kernel_size = 5, padding = 'valid', strides = 5, kernel_initializer='he_normal', name ='decoder_3')(decoded_FES)
    decoded_FES = tf.keras.layers.Conv1DTranspose(filters = 1, activation ='relu', kernel_size = 4, padding = 'valid', strides = 4 , kernel_initializer='he_normal', name ='decoder_4')(decoded_FES)
    return Model(inputs = input_FES, outputs = decoded_FES)
CAE_model=CAE_model()
CAE_model.summary()

lr_schedule4 = tf.keras.optimizers.schedules.CosineDecayRestarts(
    initial_learning_rate4,
    first_decay_steps= first_decay_steps4,
    t_mul = t_mul_4,
    m_mul = m_mul_4,
    alpha = alpha_4
)
checkpoint_path_CAE= './checkpoint/CAE/cp.ckpt'
cp_callback_CAE = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path_CAE,
                                                      save_weights_only=True,
                                                      verbose=1)

early_stop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', restore_best_weights=True, patience=10, verbose=1)

adam4 = tf.keras.optimizers.legacy.Adam(learning_rate=lr_schedule4)
CAE_model.compile(optimizer=adam4, loss='mse', metrics=['mse'])

hist_CAE = CAE_model.fit(norm_DataTrainSpec , norm_DataTrainSpec, 
                                    validation_data = (norm_DataValidSpec, norm_DataValidSpec),
                                    epochs = epochs_4,
                                    callbacks=[cp_callback_CAE,early_stop_callback],
                                    batch_size = batch_size4, shuffle=True, verbose=2) 

CAE_model.save(checkpoint_path_CAE, save_format='tf')

reconstructed_spectra = CAE_model.predict(norm_DataTrainSpec)
reconstructed_spectra.reshape(len(norm_DataTrainSpec),1600)

original_spectra=np.array(norm_DataTrainSpec)
original_spectra=original_spectra.reshape(len(norm_DataTrainSpec),1600,1)

for ii in range(0,15000,1600):
    plt.plot(original_spectra[ii],'b',linewidth=1)
    plt.plot(reconstructed_spectra[ii],'--r',alpha = 1, linewidth=0.8)
plt.show()

