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


# ===== Combine EQ Model =====
# EQ model hyperparameter


tf.keras.backend.clear_session()
alpha_2 =0.0007024277966283182
batch_size2 =  82
epochs2 = 45
first_decay_steps2 = 1983.0548550962853
initial_learning_rate2 = 0.0022479099592781445
l2 =  3.4984105619263376e-06
m_mul_2 =  0.6735220274814276
neurons2 = 241
t_mul_2 =  8.23213861301407
num_dense_layers = 2

#logdir = "./logdir"
# Start the TensorFlow Profiler
#tf.profiler.experimental.start(logdir)
process = psutil.Process(os.getpid())
memory_usage = process.memory_info().rss  # in bytes
start_time = time.time()

def combine_EQ_model():
    input_FES = CAE_model.get_layer('input_FES').input
    FES_1=CAE_model.get_layer('encoder_1')(input_FES)
    FES_2=CAE_model.get_layer('encoder_2')(FES_1)
    FES_3=CAE_model.get_layer('encoder_3')(FES_2)
    FES_4=CAE_model.get_layer('encoder_4')(FES_3)
    flatten=tf.keras.layers.Flatten()(FES_4)
    dense = tf.keras.layers.Dense(units=neurons2, activation='relu', bias_initializer='zeros', use_bias=True,
                                          kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(l2),name='fc_1')(flatten)
    for i in range(num_dense_layers):
        dense = tf.keras.layers.Dense(units=neurons2, activation='relu', bias_initializer='zeros', use_bias=True,
                                          kernel_initializer='he_normal',  kernel_regularizer=tf.keras.regularizers.l2(l2), name=f'fc_{i+2}')(dense)
    output_EQ_layer = tf.keras.layers.Dense(units=1, activation='linear', kernel_initializer='glorot_normal', name='output_layer')(dense)   
    
    return Model(input_FES, output_EQ_layer)

    
combine_EQ_model=combine_EQ_model()
combine_EQ_model.summary()

lr_schedule2 = tf.keras.optimizers.schedules.CosineDecayRestarts(
    initial_learning_rate2,
    
    first_decay_steps=first_decay_steps2,
    t_mul=t_mul_2,
    m_mul=m_mul_2, 
    alpha=alpha_2, 
    name=None)

early_stop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                       restore_best_weights=True, patience=10, verbose=1)

checkpoint_path_combine_EQ= './checkpoint/combine_EQ/EQ_model.h5'
cp_callback_combine_EQ = tf.keras.callbacks.ModelCheckpoint(filepath = checkpoint_path_combine_EQ,
                                                     save_weights_only = False,
                                                     verbose=1)

adam2 = tf.keras.optimizers.legacy.Adam(learning_rate = lr_schedule2,  clipnorm=1.0)
combine_EQ_model.compile(optimizer= adam2, loss='mse', metrics=['mse'])

hist_combine_EQ = combine_EQ_model.fit(norm_DataTrainSpec,EQ_Train, 
                    validation_data = (norm_DataValidSpec, EQ_Valid), 
                    epochs=epochs2,
                    callbacks=[early_stop_callback],
                    batch_size=batch_size2, shuffle=True, verbose=1)

end_time = time.time()
training_time = end_time - start_time
print(f"Training time: {training_time} seconds")
print(f"Memory usage: {memory_usage / 1024 ** 2} MB")

predict_test_EQ = combine_EQ_model.predict(norm_DataTestSpec)



plt.plot(((EQ_Test)*(max_EQ-min_EQ))+min_EQ, ((predict_test_EQ)*(max_EQ-min_EQ))+min_EQ,'r+',alpha=0.01)
plt.plot(((EQ_Train)*(max_EQ-min_EQ))+min_EQ ,((EQ_Train)*(max_EQ-min_EQ))+min_EQ, 'b')
plt.xlabel('EQ')
plt.legend([f'r2_score: {r2_score(EQ_Test, predict_test_EQ):.4f}'], loc='upper left' )

plt.show()

