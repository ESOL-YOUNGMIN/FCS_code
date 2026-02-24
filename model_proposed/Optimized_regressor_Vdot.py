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

# ===== Optimized Combine FR Model =====

import time
import psutil
tf.keras.backend.clear_session()

# FR model hyperparameter
alpha_1 = 0.002995651123013279
batch_size1 = 51
epochs1 = 44
first_decay_steps1 = 3804.0744251682854
initial_learning_rate1 =  0.0006427289590883641
l2 = 1.9877731846446035e-07
m_mul_1 =  0.6083391561944947
neurons1 = 163
num_dense_layers = 3
t_mul_1 = 2.189076711051493

start_time = time.time()
process = psutil.Process(os.getpid())
memory_usage = process.memory_info().rss  # in bytes
 
# Start the TensorFlow Profiler
#tf.profiler.experimental.start(logdir)
#logdir = "./logdir"

def combine_FR_model():
    input_FES = CAE_model.get_layer('input_FES').input
    FES_1=CAE_model.get_layer('encoder_1')(input_FES)
    FES_2=CAE_model.get_layer('encoder_2')(FES_1)
    FES_3=CAE_model.get_layer('encoder_3')(FES_2)
    FES_4=CAE_model.get_layer('encoder_4')(FES_3)
    flatten=tf.keras.layers.Flatten()(FES_4)
    dense = tf.keras.layers.Dense(units=neurons1, activation='relu', bias_initializer='zeros', use_bias=True,
                                      kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(l2),name='fc_1')(flatten)
    for i in range(num_dense_layers):
        dense = tf.keras.layers.Dense(units=neurons1, activation='relu', bias_initializer='zeros', use_bias=True,
                                          kernel_initializer='he_normal',  kernel_regularizer=tf.keras.regularizers.l2(l2), name=f'fc_{i+2}')(dense)
    output_FR_layer = tf.keras.layers.Dense(units=1, activation='linear', kernel_initializer='glorot_normal', name='output_layer')(dense)   
    
    return Model(input_FES, output_FR_layer)

    
combine_FR_model=combine_FR_model()
combine_FR_model.summary()

lr_schedule1 = tf.keras.optimizers.schedules.CosineDecayRestarts(
    initial_learning_rate1,
    first_decay_steps=first_decay_steps1,
    t_mul=t_mul_1,
    m_mul=m_mul_1, 
    alpha=alpha_1, 
    name=None)
checkpoint_path_combine_FR= './checkpoint/combine_FR/FR_model.h5'
cp_callback_combine_FR = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path_combine_FR,
                                                     save_weights_only=True,
                                                     verbose=1)
early_stop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                       restore_best_weights=True, patience=10, verbose=2)

adam1 = tf.keras.optimizers.legacy.Adam(learning_rate= lr_schedule1)
combine_FR_model.compile(optimizer= adam1, loss='mse', metrics=['mse'])
hist_combine_FR = combine_FR_model.fit(norm_DataTrainSpec,FR_Train, 
                                validation_data = (norm_DataValidSpec, FR_Valid), 
                                epochs=epochs1,
                                callbacks=[cp_callback_combine_FR, early_stop_callback],
                                batch_size=batch_size1, shuffle=True,verbose=1)

end_time = time.time()
training_time = end_time - start_time
print(f"Training time: {training_time} seconds")
print(f"Memory usage: {memory_usage / 1024 ** 2} MB")

predict_test_FR = combine_FR_model.predict(norm_DataTestSpec)


plt.plot(((FR_Test)*(max_FR-min_FR))+min_FR, ((predict_test_FR)*(max_FR-min_FR))+min_FR,'r+',alpha=0.01)
plt.plot(((FR_Train)*(max_FR-min_FR))+min_FR ,((FR_Train)*(max_FR-min_FR))+min_FR, 'b')
plt.xlabel('FR')
plt.legend([f'r2_score: {r2_score(FR_Test, predict_test_FR):.4f}'], loc='upper left' )
plt.show()

combine_FR_model.save(checkpoint_path_combine_FR, save_format='tf')
combine_FR_model.save_weights('combine_FR_model')

