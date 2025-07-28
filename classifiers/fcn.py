"""
The following code is inspired by
https://github.com/hfawaz/dl-4-tsc/blob/e0233efd886df8c6ca18e6f1b545d15aaf423627/classifiers/fcn.py
"""

import tensorflow as tf

def get_model(n_classes=32, input_shape=(None, 1), reduce=[tf.keras.layers.GlobalAveragePooling1D()]):
    input_layer = tf.keras.layers.Input(input_shape)
    
    conv_1 = tf.keras.layers.Conv1D(filters=128, kernel_size=8, padding='same')(input_layer)
    conv_1 = tf.keras.layers.BatchNormalization()(conv_1)
    conv_1 = tf.keras.layers.Activation(activation='relu')(conv_1)
    
    conv_2 = tf.keras.layers.Conv1D(filters=256, kernel_size=5, padding='same')(conv_1)
    conv_2 = tf.keras.layers.BatchNormalization()(conv_2)
    conv_2 = tf.keras.layers.Activation('relu')(conv_2)
    
    conv_3 = tf.keras.layers.Conv1D(filters=128, kernel_size=3, padding='same')(conv_2)
    conv_3 = tf.keras.layers.BatchNormalization()(conv_3)
    conv_3 = tf.keras.layers.Activation('relu')(conv_3)

    reduced_outputs = [reduce_fn(conv_3) for reduce_fn in reduce]
    if len(reduced_outputs) > 1:
        concatenated = tf.keras.layers.Concatenate()(reduced_outputs)
    else:
        concatenated = reduced_outputs[0]

    output_layer = tf.keras.layers.Dense(units=n_classes, activation='softmax')(concatenated)
    model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)

    return model
