"""
The following code is inspired by
https://github.com/MSD-IRIMAS/LITE/blob/e2c2680fda0e0f4f9d1384ef959e33580cc7f473/classifiers/lite.py
"""

import tensorflow as tf
import numpy as np


def hybird_layer(input_tensor, input_channels, kernel_sizes=[2, 4, 8, 16, 32, 64], multivariate=False):
    conv_list = []

    for kernel_size in kernel_sizes:
        filter_ = np.ones(shape=(kernel_size, input_channels, 1))
        indices_ = np.arange(kernel_size)

        filter_[indices_ % 2 == 0] *= -1

        if multivariate:
            conv = tf.keras.layers.DepthwiseConv1D(
                kernel_size=kernel_size,
                padding="same",
                use_bias=False,
                depthwise_initializer=tf.keras.initializers.Constant(filter_.tolist()),
                trainable=False,
            )(input_tensor)
        else:
            conv = tf.keras.layers.Conv1D(
                filters=1,
                kernel_size=kernel_size,
                padding="same",
                use_bias=False,
                kernel_initializer=tf.keras.initializers.Constant(filter_),
                trainable=False,
            )(input_tensor)

        conv_list.append(conv)

    for kernel_size in kernel_sizes:
        filter_ = np.ones(shape=(kernel_size, input_channels, 1))
        indices_ = np.arange(kernel_size)

        filter_[indices_ % 2 > 0] *= -1

        if multivariate:
            conv = tf.keras.layers.DepthwiseConv1D(
                kernel_size=kernel_size,
                padding="same",
                use_bias=False,
                depthwise_initializer=tf.keras.initializers.Constant(filter_.tolist()),
                trainable=False,
            )(input_tensor)
        else:
            conv = tf.keras.layers.Conv1D(
                filters=1,
                kernel_size=kernel_size,
                padding="same",
                use_bias=False,
                kernel_initializer=tf.keras.initializers.Constant(filter_),
                trainable=False,
            )(input_tensor)

        conv_list.append(conv)

    for kernel_size in kernel_sizes[1:]:
        filter_ = np.zeros(
            shape=(kernel_size + kernel_size // 2, input_channels, 1)
        )

        xmash = np.linspace(start=0, stop=1, num=kernel_size // 4 + 1)[1:].reshape(
            (-1, 1, 1)
        )

        filter_left = xmash**2
        filter_right = filter_left[::-1]

        filter_[0 : kernel_size // 4] = -filter_left
        filter_[kernel_size // 4 : kernel_size // 2] = -filter_right
        filter_[kernel_size // 2 : 3 * kernel_size // 4] = 2 * filter_left
        filter_[3 * kernel_size // 4 : kernel_size] = 2 * filter_right
        filter_[kernel_size : 5 * kernel_size // 4] = -filter_left
        filter_[5 * kernel_size // 4 :] = -filter_right

        if multivariate:
            conv = tf.keras.layers.DepthwiseConv1D(
                kernel_size=kernel_size + kernel_size // 2,
                padding="same",
                use_bias=False,
                depthwise_initializer=tf.keras.initializers.Constant(filter_.tolist()),
                trainable=False,
            )(input_tensor)
        else:
            conv = tf.keras.layers.Conv1D(
                filters=1,
                kernel_size=kernel_size + kernel_size // 2,
                padding="same",
                use_bias=False,
                kernel_initializer=tf.keras.initializers.Constant(filter_),
                trainable=False,
            )(input_tensor)

        conv_list.append(conv)

    hybird_layer = tf.keras.layers.Concatenate(axis=2)(conv_list)
    hybird_layer = tf.keras.layers.Activation(activation="relu")(hybird_layer)

    return hybird_layer


def inception_module(input_tensor, dilation_rate, n_filters, kernel_size, stride=1, activation="linear", use_hybird_layer=False, use_multiplexing=True, multivariate=False):
    input_inception = input_tensor

    if not use_multiplexing:
        n_convs = 1
        n_filters = n_filters * 3
    else:
        n_convs = 3
        n_filters = n_filters

    kernel_size_s = [kernel_size // (2**i) for i in range(n_convs)]
    conv_list = []
    for i in range(len(kernel_size_s)):
        if multivariate:
            conv_list.append(
                tf.keras.layers.SeparableConv1D(
                    filters=n_filters,
                    kernel_size=kernel_size_s[i],
                    strides=stride,
                    padding="same",
                    dilation_rate=dilation_rate,
                    activation=activation,
                    use_bias=False,
                )(input_inception)
            )
        else:
            conv_list.append(
                tf.keras.layers.Conv1D(
                    filters=n_filters,
                    kernel_size=kernel_size_s[i],
                    strides=stride,
                    padding="same",
                    dilation_rate=dilation_rate,
                    activation=activation,
                    use_bias=False,
                )(input_inception)
            )

    if use_hybird_layer:
        hybird = hybird_layer(
            input_tensor=input_tensor, input_channels=input_tensor.shape[-1], multivariate=multivariate
        )
        conv_list.append(hybird)

    if len(conv_list) > 1:
        x = tf.keras.layers.Concatenate(axis=2)(conv_list)
    else:
        x = conv_list[0]
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(activation="relu")(x)

    return x


def fcn_module(input_tensor, kernel_size, dilation_rate, n_filters, stride=1, activation="relu",):
        x = tf.keras.layers.SeparableConv1D(
            filters=n_filters,
            kernel_size=kernel_size,
            padding="same",
            strides=stride,
            dilation_rate=dilation_rate,
            use_bias=False,
        )(input_tensor)

        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation(activation=activation)(x)

        return x


def get_model(n_classes=32, input_shape=(None, 1), reduce=[tf.keras.layers.GlobalAveragePooling1D()],
              n_filters=32, use_custom_filters=True, use_dilation=True, use_multiplexing=True, kernel_size=41):
    kernel_size = kernel_size - 1

    multivariate = False
    if input_shape[1] > 1:
        multivariate = True

    input_layer = tf.keras.layers.Input(input_shape)

    inception = inception_module(
        input_tensor=input_layer,
        dilation_rate=1,
        n_filters=n_filters,
        kernel_size=kernel_size,
        use_hybird_layer=use_custom_filters,
        multivariate=multivariate
    )
    input_tensor = inception

    kernel_size //= 2
    dilation_rate = 1
 
    for i in range(2):
        if use_dilation:
            dilation_rate = 2 ** (i + 1)
 
        x = fcn_module(
            input_tensor=input_tensor,
            kernel_size=kernel_size // (2**i),
            n_filters=n_filters,
            dilation_rate=dilation_rate,
        )
        input_tensor = x
 

    reduced_outputs = [reduce_fn(input_tensor) for reduce_fn in reduce]
    if len(reduced_outputs) > 1:
        concatenated = tf.keras.layers.Concatenate()(reduced_outputs)
    else:
        concatenated = reduced_outputs[0]

    output_layer = tf.keras.layers.Dense(units=n_classes, activation='softmax')(concatenated)
    model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)

    return model
