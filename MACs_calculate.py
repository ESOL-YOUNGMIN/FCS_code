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

import tensorflow as tf

# Function to calculate MACs
def count_mac(model):
    macs = 0
    for layer in model.layers:
        if isinstance(layer, (tf.keras.layers.Conv1D, tf.keras.layers.Conv2D)):
            # Conv1D/Conv2D MACs: (kernel_size * input_channels * output_size * output_channels)
            output_shape = layer.output_shape
            input_shape = layer.input_shape
            filter_size = layer.kernel_size[0]
            if isinstance(layer, tf.keras.layers.Conv2D):
                filter_size *= layer.kernel_size[1]  # Multiply by second kernel size for Conv2D
            input_channels = input_shape[-1]
            output_channels = output_shape[-1]
            output_size = tf.reduce_prod(output_shape[1:-1])  # For Conv1D: output_length, Conv2D: output_height * output_width
            
            macs += filter_size * input_channels * output_size * output_channels

        elif isinstance(layer, tf.keras.layers.Dense):
            # Dense layer MACs: (input_units * output_units)
            input_units = layer.input_shape[-1]
            output_units = layer.units
            macs += input_units * output_units
    
    return macs


# Calculate MACs for the model
model = CAE_multi_model
macs = count_mac(model)
print(f"Total MACs: {macs}")
