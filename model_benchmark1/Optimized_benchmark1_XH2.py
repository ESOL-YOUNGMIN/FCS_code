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


alpha_MIX = 0.000745065571146945
batch_size_MIX = 71
epochs_MIX = 50
features = 28
first_decay_steps_MIX = 9618.431096784807
initial_learning_rate_MIX =  0.0067238344733783805
l2 = 5.749020909423772e-06
m_mul_MIX =  0.6922095385003402
neurons_MIX = 396
num_dense_layers = 2
t_mul_MIX = 8.351341378572979
num_filters_2 = 50
num_filters_3 = 94
num_filters_4 = 57

start_time = time.time()
process = psutil.Process(os.getpid())
memory_usage = process.memory_info().rss  # in bytes


def CAE_MIX_model():
    input_FES = tf.keras.layers.Input(shape=(1600,1),name='input_FES')
    encoded_FES = tf.keras.layers.Conv1D(filters = num_filters_2,  activation ='relu', kernel_size = 4, padding = 'valid', strides = 4, kernel_initializer='he_normal', name ='encoder_1')(input_FES)
    encoded_FES = tf.keras.layers.Conv1D(filters = num_filters_3,  activation ='relu', kernel_size = 5, padding = 'valid', strides = 5, kernel_initializer='he_normal', name ='encoder_2')(encoded_FES)
    encoded_FES = tf.keras.layers.Conv1D(filters = num_filters_4,  activation ='relu', kernel_size = 8, padding = 'valid', strides = 8, kernel_initializer='he_normal', name ='encoder_3')(encoded_FES)
    encoded_FES = tf.keras.layers.Conv1D(filters = features,  activation ='relu', kernel_size = 10,padding = 'valid', strides = 1, kernel_initializer='he_normal', name ='encoder_4')(encoded_FES)
    
    flatten=tf.keras.layers.Flatten()(encoded_FES)
    dense = tf.keras.layers.Dense(units=neurons_MIX, activation='relu', bias_initializer='zeros', use_bias=True,
                                      kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(l2),name='fc_1')(flatten)
    for i in range(num_dense_layers):
        dense = tf.keras.layers.Dense(units=neurons_MIX, activation='relu', bias_initializer='zeros', use_bias=True,
                                          kernel_initializer='he_normal',  kernel_regularizer=tf.keras.regularizers.l2(l2), name=f'fc_{i+2}')(dense)
    output_MIX_layer = tf.keras.layers.Dense(units=1, activation='linear', kernel_initializer='glorot_normal', name='output_layer')(dense)   
    return Model(input_FES, output_MIX_layer)

CAE_MIX_model = CAE_MIX_model()
CAE_MIX_model.summary()
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
CAE_MIX_model.compile(optimizer=adam1, loss='mse', metrics=['mse'])
hist_CAE_MIX = CAE_MIX_model.fit(norm_DataTrainSpec, MIX_Train,
                                   validation_data=(norm_DataValidSpec, MIX_Valid),
                                   epochs=epochs_MIX,
                                   callbacks=[early_stop_callback],
                                   batch_size=batch_size_MIX, shuffle=True, verbose=2) 
end_time = time.time()
training_time = end_time - start_time
print(f"Training time: {training_time} seconds")
print(f"Memory usage: {memory_usage / 1024 ** 2} MB")
predict_test_CAE_MIX = CAE_MIX_model.predict(norm_DataTestSpec)

plt.plot(((MIX_Test)*(max_MIX-min_MIX))+min_MIX, ((predict_test_CAE_MIX)*(max_MIX-min_MIX))+min_MIX,'r+',alpha=0.01)
plt.plot(((MIX_Train)*(max_MIX-min_MIX))+min_MIX ,((MIX_Train)*(max_MIX-min_MIX))+min_MIX, 'b')
plt.xlabel('MIX')
plt.legend([f'r2_score: {r2_score(MIX_Test, predict_test_CAE_MIX):.4f}'], loc='upper left' )
plt.show()

# label_MIX = ((MIX_Test)*(max_MIX-min_MIX))+min_MIX
# preds_MIX = ((predict_test_CAE_MIX)*(max_MIX-min_MIX))+min_MIX 

# CAE_MIX_label=pd.DataFrame(label_MIX)
# CAE_MIX_preds=pd.DataFrame(preds_MIX)
# CAE_MIX_label.to_excel('./output/CAE_MIX_label.xlsx', index=False)
# CAE_MIX_preds.to_excel('./output/CAE_MIX_preds.xlsx', index=False)
