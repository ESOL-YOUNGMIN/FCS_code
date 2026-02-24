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


# ===== Combine MIX Model =====
# MIX model hyperparameter
alpha_3 =  0.0005665758840752201
batch_size3 = 88
epochs3 = 28
first_decay_steps3 = 4886.086169405871
initial_learning_rate3 = 0.0017596840155768285
l2 = 1.6910834056282797e-06
m_mul_3 = 0.7888615380610965
neurons3 = 288
t_mul_3 =  6.855833798531949
num_dense_layers = 0

process = psutil.Process(os.getpid())
memory_usage = process.memory_info().rss  # in bytes
start_time = time.time()

def combine_MIX_model():
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
    return Model(input_FES, output_MIX_layer)

    
combine_MIX_model=combine_MIX_model()
combine_MIX_model.summary()

    
lr_schedule3 = tf.keras.optimizers.schedules.CosineDecayRestarts(
    initial_learning_rate3,
    first_decay_steps=first_decay_steps3,
    t_mul=t_mul_3,
    m_mul=m_mul_3, 
    alpha=alpha_3, 
    name=None)
early_stop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                       restore_best_weights=True, patience=10, verbose=1)

checkpoint_path_combine_MIX= './checkpoint/combine_MIX/MIX_model.h5'
cp_callback_combine_MIX = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path_combine_MIX,
                                                     save_weights_only=False,
                                                     verbose=1)

adam3 = tf.keras.optimizers.legacy.Adam(learning_rate= lr_schedule3)
combine_MIX_model.compile(optimizer= adam3, loss='mse', metrics=['mse'])

hist_combine_MIX = combine_MIX_model.fit(norm_DataTrainSpec,MIX_Train, 
                    validation_data = (norm_DataValidSpec, MIX_Valid), 
                    epochs=epochs3,
                    callbacks=[early_stop_callback],# cp_callback_combine_MIX],
                     batch_size=batch_size3, shuffle=True, verbose=1)

end_time = time.time()
training_time = end_time - start_time
print(f"Training time: {training_time} seconds")
print(f"Memory usage: {memory_usage / 1024 ** 2} MB")
predict_test_MIX = combine_MIX_model.predict(norm_DataTestSpec)

plt.plot(((MIX_Test)*(max_MIX-min_MIX))+min_MIX, ((predict_test_MIX)*(max_MIX-min_MIX))+min_MIX,'r+',alpha=0.01)
plt.plot(((MIX_Train)*(max_MIX-min_MIX))+min_MIX ,((MIX_Train)*(max_MIX-min_MIX))+min_MIX, 'b')
plt.xlabel('MIX')
plt.legend([f'r2_score: {r2_score(MIX_Test, predict_test_MIX):.4f}'], loc='upper left' )
plt.show()



combine_MIX_model.save(checkpoint_path_combine_MIX, save_format='tf')
tf.keras.backend.clear_session()
