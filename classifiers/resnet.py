"""
The following code is inspired by
https://github.com/hfawaz/dl-4-tsc/blob/e0233efd886df8c6ca18e6f1b545d15aaf423627/classifiers/resnet.py
"""

import tensorflow as tf

def get_model(n_classes=32, input_shape=(None, 1), reduce=tf.keras.layers.GlobalAveragePooling1D()):
    n_feature_maps = 64
    
    input_layer = tf.keras.layers.Input(input_shape)
    
    # block 1
    conv_x = tf.keras.layers.Conv1D(filters=n_feature_maps, kernel_size=8, padding='same')(input_layer)
    conv_x = tf.keras.layers.BatchNormalization()(conv_x)
    conv_x = tf.keras.layers.Activation('relu')(conv_x)

    conv_y = tf.keras.layers.Conv1D(filters=n_feature_maps, kernel_size=5, padding='same')(conv_x)
    conv_y = tf.keras.layers.BatchNormalization()(conv_y)
    conv_y = tf.keras.layers.Activation('relu')(conv_y)

    conv_z = tf.keras.layers.Conv1D(filters=n_feature_maps, kernel_size=3, padding='same')(conv_y)
    conv_z = tf.keras.layers.BatchNormalization()(conv_z)

    # expand channels for the sum
    shortcut_y = tf.keras.layers.Conv1D(filters=n_feature_maps, kernel_size=1, padding='same')(input_layer)
    shortcut_y = tf.keras.layers.BatchNormalization()(shortcut_y)

    output_block_1 = tf.keras.layers.add([shortcut_y, conv_z])
    output_block_1 = tf.keras.layers.Activation('relu')(output_block_1)

    # block 2
    conv_x = tf.keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same')(output_block_1)
    conv_x = tf.keras.layers.BatchNormalization()(conv_x)
    conv_x = tf.keras.layers.Activation('relu')(conv_x)

    conv_y = tf.keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=5, padding='same')(conv_x)
    conv_y = tf.keras.layers.BatchNormalization()(conv_y)
    conv_y = tf.keras.layers.Activation('relu')(conv_y)

    conv_z = tf.keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=3, padding='same')(conv_y)
    conv_z = tf.keras.layers.BatchNormalization()(conv_z)

    # expand channels for the sum
    shortcut_y = tf.keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=1, padding='same')(output_block_1)
    shortcut_y = tf.keras.layers.BatchNormalization()(shortcut_y)

    output_block_2 = tf.keras.layers.add([shortcut_y, conv_z])
    output_block_2 = tf.keras.layers.Activation('relu')(output_block_2)

    # block 3
    conv_x = tf.keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same')(output_block_2)
    conv_x = tf.keras.layers.BatchNormalization()(conv_x)
    conv_x = tf.keras.layers.Activation('relu')(conv_x)

    conv_y = tf.keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=5, padding='same')(conv_x)
    conv_y = tf.keras.layers.BatchNormalization()(conv_y)
    conv_y = tf.keras.layers.Activation('relu')(conv_y)

    conv_z = tf.keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=3, padding='same')(conv_y)
    conv_z = tf.keras.layers.BatchNormalization()(conv_z)

    # no need to expand channels because they are equal
    shortcut_y = tf.keras.layers.BatchNormalization()(output_block_2)

    output_block_3 = tf.keras.layers.add([shortcut_y, conv_z])
    output_block_3 = tf.keras.layers.Activation('relu')(output_block_3)

    # final
    reduced_outputs = [reduce_fn(output_block_3) for reduce_fn in reduce]
    if len(reduced_outputs) > 1:
        concatenated = tf.keras.layers.Concatenate()(reduced_outputs)
    else:
        concatenated = reduced_outputs[0]

    output_layer = tf.keras.layers.Dense(units=n_classes, activation='softmax')(concatenated)
    model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)

    return model
