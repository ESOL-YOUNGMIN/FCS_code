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


alpha_5 = 0.00882252233344969
batch_size_5 = 21
epochs_5 = 38
features = 23
first_decay_steps_5 =  4243.568960953444
initial_learning_rate_5 = 0.0005262035069694276
l2 =1.494738180131976e-05
m_mul_5 =  0.8192755015489431
neurons_5 = 205
num_dense_layers = 5
t_mul_5 = 6.93335074668189
num_filters_2 = 128
num_filters_3 = 62
num_filters_4 = 87

start_time = time.time()
process = psutil.Process(os.getpid())
memory_usage = process.memory_info().rss  # in bytes


def CAE_multi_model():
    input_FES = tf.keras.layers.Input(shape=(1600,1),name='input_FES')
    encoded_FES = tf.keras.layers.Conv1D(filters = num_filters_2,  activation ='relu', kernel_size = 4, padding = 'valid', strides = 4, kernel_initializer='he_normal', name ='encoder_1')(input_FES)
    encoded_FES = tf.keras.layers.Conv1D(filters = num_filters_3,  activation ='relu', kernel_size = 5, padding = 'valid', strides = 5, kernel_initializer='he_normal', name ='encoder_2')(encoded_FES)
    encoded_FES = tf.keras.layers.Conv1D(filters = num_filters_4,  activation ='relu', kernel_size = 8, padding = 'valid', strides = 8, kernel_initializer='he_normal', name ='encoder_3')(encoded_FES)
    encoded_FES = tf.keras.layers.Conv1D(filters = features,  activation ='relu', kernel_size = 10,padding = 'valid', strides = 1, kernel_initializer='he_normal', name ='encoder_4')(encoded_FES)
    
    flatten=tf.keras.layers.Flatten()(encoded_FES)
    dense = tf.keras.layers.Dense(units=neurons_5, activation='relu', bias_initializer='zeros', use_bias=True,
                                      kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(l2),name='fc_1')(flatten)
    for i in range(num_dense_layers):
        dense = tf.keras.layers.Dense(units=neurons_5, activation='relu', bias_initializer='zeros', use_bias=True,
                                          kernel_initializer='he_normal',  kernel_regularizer=tf.keras.regularizers.l2(l2), name=f'fc_{i+2}')(dense)
    output_CAE_multi_layer = tf.keras.layers.Dense(units=3, activation='linear', kernel_initializer='glorot_normal', name='output_layer')(dense)   
    return Model(input_FES, output_CAE_multi_layer)

CAE_multi_model =CAE_multi_model()
CAE_multi_model.summary()
lr_schedule5 = tf.keras.optimizers.schedules.CosineDecayRestarts(
    initial_learning_rate_5,
    first_decay_steps=first_decay_steps_5,
    t_mul=t_mul_5,
    m_mul=m_mul_5,
    alpha=alpha_5
)
early_stop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', restore_best_weights=True, patience=10, verbose=1)

adam5 = tf.keras.optimizers.legacy.Adam(learning_rate=lr_schedule5)
CAE_multi_model.compile(optimizer=adam5, loss='mse', metrics=['mse'])
hist_CAE_multi = CAE_multi_model.fit(norm_DataTrainSpec, Total_Train,
                                    validation_data=(norm_DataValidSpec, Total_Valid),
                                    epochs=epochs_5,
                                    callbacks=[early_stop_callback],
                                    batch_size=batch_size_5, shuffle=True, verbose=2) 

predict_test_CAE_multi = CAE_multi_model.predict(norm_DataTestSpec)

plt.plot(((Total_Test[:,0])*(max_FR-min_FR))+min_FR, ((predict_test_CAE_multi[:,0])*(max_FR-min_FR))+min_FR,'r+',alpha=0.01)
plt.plot(((Total_Train[:,0])*(max_FR-min_FR))+min_FR ,((Total_Train[:,0])*(max_FR-min_FR))+min_FR, 'b')
plt.xlabel('FR')
plt.legend([f'r2_score: {r2_score(Total_Test[:,0], predict_test_CAE_multi[:,0]):.4f}'], loc='upper left' )
plt.show()

plt.plot(((Total_Test[:,1])*(max_EQ-min_EQ))+min_EQ, ((predict_test_CAE_multi[:,1])*(max_EQ-min_EQ))+min_EQ,'r+',alpha=0.01)
plt.plot(((Total_Train[:,1])*(max_EQ-min_EQ))+min_EQ ,((Total_Train[:,1])*(max_EQ-min_EQ))+min_EQ, 'b')
plt.xlabel('EQ')
plt.legend([f'r2_score: {r2_score(Total_Test[:,1], predict_test_CAE_multi[:,1]):.4f}'], loc='upper left' )
plt.show()


plt.plot(((Total_Test[:,2])*(max_MIX-min_MIX))+min_MIX, ((predict_test_CAE_multi[:,2])*(max_MIX-min_MIX))+min_MIX,'r+',alpha=0.01)
plt.plot(((Total_Train[:,2])*(max_MIX-min_MIX))+min_MIX ,((Total_Train[:,2])*(max_MIX-min_MIX))+min_MIX, 'b')
plt.xlabel('MIX')
plt.legend([f'r2_score: {r2_score(Total_Test, predict_test_CAE_multi):.4f}'], loc='upper left' )
plt.show()

end_time = time.time()
training_time = end_time - start_time
print(f"Training time: {training_time} seconds")
print(f"Memory usage: {memory_usage / 1024 ** 2} MB")

