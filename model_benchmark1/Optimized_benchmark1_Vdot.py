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

tf.keras.backend.clear_session()

alpha_FR = 0.004064802312955727
batch_size_FR = 31
epochs_FR = 56
features = 25
first_decay_steps_FR = 2488.8133883869737
initial_learning_rate_FR =  0.0015618477402061335
l2 = 1.050357044722941e-05
m_mul_FR =  0.6406354959074829
neurons_FR = 117
num_dense_layers = 4
t_mul_FR = 4.075123280572459
num_filters_2 = 120
num_filters_3 = 94
num_filters_4 = 10

start_time = time.time()
process = psutil.Process(os.getpid())
memory_usage = process.memory_info().rss  # in bytes
 
def CAE_FR_model():
    input_FES = tf.keras.layers.Input(shape=(1600,1),name='input_FES')
    encoded_FES = tf.keras.layers.Conv1D(filters = num_filters_2,  activation ='relu', kernel_size = 4, padding = 'valid', strides = 4, kernel_initializer='he_normal', name ='encoder_1')(input_FES)
    encoded_FES = tf.keras.layers.Conv1D(filters = num_filters_3,  activation ='relu', kernel_size = 5, padding = 'valid', strides = 5, kernel_initializer='he_normal', name ='encoder_2')(encoded_FES)
    encoded_FES = tf.keras.layers.Conv1D(filters = num_filters_4,  activation ='relu', kernel_size = 8, padding = 'valid', strides = 8, kernel_initializer='he_normal', name ='encoder_3')(encoded_FES)
    encoded_FES = tf.keras.layers.Conv1D(filters = features,  activation ='relu', kernel_size = 10,padding = 'valid', strides = 1, kernel_initializer='he_normal', name ='encoder_4')(encoded_FES)
    
    flatten=tf.keras.layers.Flatten()(encoded_FES)
    dense = tf.keras.layers.Dense(units=neurons_FR, activation='relu', bias_initializer='zeros', use_bias=True,
                                      kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(l2),name='fc_1')(flatten)
    for i in range(num_dense_layers):
        dense = tf.keras.layers.Dense(units=neurons_FR, activation='relu', bias_initializer='zeros', use_bias=True,
                                          kernel_initializer='he_normal',  kernel_regularizer=tf.keras.regularizers.l2(l2), name=f'fc_{i+2}')(dense)
    output_FR_layer = tf.keras.layers.Dense(units=1, activation='linear', kernel_initializer='glorot_normal', name='output_layer')(dense)   
    return Model(input_FES, output_FR_layer)

CAE_FR_model = CAE_FR_model()
CAE_FR_model.summary()
lr_schedule_FR = tf.keras.optimizers.schedules.CosineDecayRestarts(
    initial_learning_rate_FR,
    first_decay_steps=first_decay_steps_FR,
    t_mul=t_mul_FR,
    m_mul=m_mul_FR,
    alpha=alpha_FR
)
early_stop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                       restore_best_weights=True, patience=10, verbose=1)

adam1 = tf.keras.optimizers.legacy.Adam(learning_rate=lr_schedule_FR)
CAE_FR_model.compile(optimizer=adam1, loss='mse', metrics=['mse'])
hist_CAE_FR = CAE_FR_model.fit(norm_DataTrainSpec, FR_Train,
                                   validation_data=(norm_DataValidSpec, FR_Valid),
                                   epochs=epochs_FR,
                                   callbacks=[early_stop_callback],
                                   batch_size=batch_size_FR, shuffle=True, verbose=2) 
end_time = time.time()
training_time = end_time - start_time
print(f"Training time: {training_time} seconds")
print(f"Memory usage: {memory_usage / 1024 ** 2} MB")

predict_test_CAE_FR = CAE_FR_model.predict(norm_DataTestSpec)

plt.plot(((FR_Test)*(max_FR-min_FR))+min_FR, ((predict_test_CAE_FR)*(max_FR-min_FR))+min_FR,'r+',alpha=0.01)
plt.plot(((FR_Train)*(max_FR-min_FR))+min_FR ,((FR_Train)*(max_FR-min_FR))+min_FR, 'b')
plt.xlabel('FR')
plt.legend([f'r2_score: {r2_score(FR_Test, predict_test_CAE_FR):.4f}'], loc='upper left' )
plt.show()




# label_FR = ((FR_Test)*(max_FR-min_FR))+min_FR
# preds_FR = ((predict_test_CAE_FR)*(max_FR-min_FR))+min_FR 

# CAE_FR_label=pd.DataFrame(label_FR)
# CAE_FR_preds=pd.DataFrame(preds_FR)
# CAE_FR_label.to_excel('./output/CAE_FR_label.xlsx', index=False)
# CAE_FR_preds.to_excel('./output/CAE_FR_preds.xlsx', index=False)
