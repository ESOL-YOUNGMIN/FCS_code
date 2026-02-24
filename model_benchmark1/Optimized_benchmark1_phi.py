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

alpha_EQ = 0.001360609322616897
batch_size_EQ = 69
epochs_EQ = 30
features = 25
first_decay_steps_EQ = 9162.70434169666
initial_learning_rate_EQ =  0.0021060347720764437
l2 = 2.7529264346304117e-05
m_mul_EQ =  0.7241195177944043
neurons_EQ = 459
num_dense_layers = 0
t_mul_EQ = 4.8157805458258265
num_filters_2 = 126
num_filters_3 = 109
num_filters_4 = 91

start_time = time.time()
process = psutil.Process(os.getpid())
memory_usage = process.memory_info().rss  # in bytes

def CAE_EQ_model():
    input_FES = tf.keras.layers.Input(shape=(1600,1),name='input_FES')
    encoded_FES = tf.keras.layers.Conv1D(filters = num_filters_2,  activation ='relu', kernel_size = 4, padding = 'valid', strides = 4, kernel_initializer='he_normal', name ='encoder_1')(input_FES)
    encoded_FES = tf.keras.layers.Conv1D(filters = num_filters_3,  activation ='relu', kernel_size = 5, padding = 'valid', strides = 5, kernel_initializer='he_normal', name ='encoder_2')(encoded_FES)
    encoded_FES = tf.keras.layers.Conv1D(filters = num_filters_4,  activation ='relu', kernel_size = 8, padding = 'valid', strides = 8, kernel_initializer='he_normal', name ='encoder_3')(encoded_FES)
    encoded_FES = tf.keras.layers.Conv1D(filters = features,  activation ='relu', kernel_size = 10,padding = 'valid', strides = 1, kernel_initializer='he_normal', name ='encoder_4')(encoded_FES)
    
    flatten=tf.keras.layers.Flatten()(encoded_FES)
    dense = tf.keras.layers.Dense(units=neurons_EQ, activation='relu', bias_initializer='zeros', use_bias=True,
                                      kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(l2),name='fc_1')(flatten)
    for i in range(num_dense_layers):
        dense = tf.keras.layers.Dense(units=neurons_EQ, activation='relu', bias_initializer='zeros', use_bias=True,
                                          kernel_initializer='he_normal',  kernel_regularizer=tf.keras.regularizers.l2(l2), name=f'fc_{i+2}')(dense)
    output_EQ_layer = tf.keras.layers.Dense(units=1, activation='linear', kernel_initializer='glorot_normal', name='output_layer')(dense)   
    return Model(input_FES, output_EQ_layer)

CAE_EQ_model = CAE_EQ_model()
CAE_EQ_model.summary()
lr_schedule_EQ = tf.keras.optimizers.schedules.CosineDecayRestarts(
    initial_learning_rate_EQ,
    first_decay_steps=first_decay_steps_EQ,
    t_mul=t_mul_EQ,
    m_mul=m_mul_EQ,
    alpha=alpha_EQ
)
early_stop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                       restore_best_weights=True, patience=10, verbose=1)

adam1 = tf.keras.optimizers.legacy.Adam(learning_rate=lr_schedule_EQ)
CAE_EQ_model.compile(optimizer=adam1, loss='mse', metrics=['mse'])
hist_CAE_EQ = CAE_EQ_model.fit(norm_DataTrainSpec, EQ_Train,
                                   validation_data=(norm_DataValidSpec, EQ_Valid),
                                   epochs=epochs_EQ,
                                   callbacks=[early_stop_callback],
                                   batch_size=batch_size_EQ, shuffle=True, verbose=2) 
end_time = time.time()
training_time = end_time - start_time
print(f"Training time: {training_time} seconds")
print(f"Memory usage: {memory_usage / 1024 ** 2} MB")

predict_test_CAE_EQ = CAE_EQ_model.predict(norm_DataTestSpec)

plt.plot(((EQ_Test)*(max_EQ-min_EQ))+min_EQ, ((predict_test_CAE_EQ)*(max_EQ-min_EQ))+min_EQ,'r+',alpha=0.01)
plt.plot(((EQ_Train)*(max_EQ-min_EQ))+min_EQ ,((EQ_Train)*(max_EQ-min_EQ))+min_EQ, 'b')
plt.xlabel('EQ')
plt.legend([f'r2_score: {r2_score(EQ_Test, predict_test_CAE_EQ):.4f}'], loc='upper left' )
plt.show()



# label_EQ = ((EQ_Test)*(max_EQ-min_EQ))+min_EQ
# preds_EQ = ((predict_test_CAE_EQ)*(max_EQ-min_EQ))+min_EQ 

# CAE_EQ_label=pd.DataFrame(label_EQ)
# CAE_EQ_preds=pd.DataFrame(preds_EQ)
# CAE_EQ_label.to_excel('./output/CAE_EQ_label.xlsx', index=False)
# CAE_EQ_preds.to_excel('./output/CAE_EQ_preds.xlsx', index=False)
