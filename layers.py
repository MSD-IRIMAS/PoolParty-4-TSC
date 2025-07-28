import tensorflow as tf
from tensorflow.keras.saving import register_keras_serializable

# Proportion of positive values
@register_keras_serializable()
class HardPPV(tf.keras.layers.Layer):
    def __init__(self, axis=1, **kwargs):
        super(HardPPV, self).__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        return tf.reduce_mean(tf.nn.relu(tf.sign(inputs)), axis=self.axis)


@register_keras_serializable()
class SoftPPV(tf.keras.layers.Layer):
    def __init__(self, axis=1, **kwargs):
        super(SoftPPV, self).__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        return tf.reduce_mean(tf.sigmoid(10000.0 * (inputs - 0.001)), axis=self.axis)

# HardPPV with Straight-Through Estimator
@register_keras_serializable()
class HardPPVSTE(tf.keras.layers.Layer):
    def __init__(self, axis=1, **kwargs):
        super(HardPPVSTE, self).__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        return self._skip_gradient(inputs)

    @tf.custom_gradient
    def _skip_gradient(self, inputs):
        output = tf.reduce_mean(tf.nn.relu(tf.sign(inputs)), axis=self.axis)

        def grad(dy):
            input_shape = tf.shape(inputs)
            dy_broadcast = tf.broadcast_to(tf.expand_dims(dy, axis=self.axis), input_shape)
            return dy_broadcast

        return output, grad

PPV = SoftPPV

# Mean of positive values
@register_keras_serializable()
class HardMPV(tf.keras.layers.Layer):
    def __init__(self, axis=1, **kwargs):
        super(HardMPV, self).__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        positive_mask = tf.cast(inputs > 0.0, tf.float32)
        positive_sum = tf.reduce_sum(inputs * positive_mask, axis=self.axis)
        positive_count = tf.reduce_sum(positive_mask, axis=self.axis)
        return tf.math.divide_no_nan(positive_sum, positive_count)


@register_keras_serializable()
class SoftMPV(tf.keras.layers.Layer):
    def __init__(self, axis=1, **kwargs):
        super(SoftMPV, self).__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        positive_mask = tf.sigmoid(10000.0 * (inputs - 0.001))
        positive_sum = tf.reduce_sum(inputs * positive_mask, axis=self.axis)
        positive_count = tf.reduce_sum(positive_mask, axis=self.axis)
        return positive_sum / (positive_count + 1e-8)

MPV = HardMPV

# Mean of Indices of Positive Values
@register_keras_serializable()
class HardMIPV(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(HardMIPV, self).__init__(**kwargs)
        self.axis = 1

    def call(self, inputs):
        positive_mask = tf.cast(inputs > 0.0, tf.float32)
        indices = tf.range(tf.shape(inputs)[self.axis], dtype=tf.float32)

        # if self.axis == 1:
        indices = tf.expand_dims(indices, axis=0)
        indices = tf.expand_dims(indices, axis=-1)

        # No UserWarning Gradients trick
        positive_indices_sum = tf.reduce_sum(indices * tf.math.divide_no_nan(inputs * positive_mask, inputs), axis=self.axis)
        positive_count = tf.reduce_sum(positive_mask, axis=self.axis)
        return tf.math.divide_no_nan(positive_indices_sum, positive_count)
        # UserWarning: Gradients do not exist for variables [previous layers]
        positive_indices_sum = tf.reduce_sum(indices * positive_mask, axis=self.axis)
        positive_count = tf.reduce_sum(positive_mask, axis=self.axis)
        return tf.math.divide_no_nan(positive_indices_sum, positive_count)


@register_keras_serializable()
class SoftMIPV(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(SoftMIPV, self).__init__(**kwargs)
        self.axis = 1

    def call(self, inputs):
        positive_mask = tf.sigmoid(10000.0 * (inputs - 0.001))
        indices = tf.range(tf.shape(inputs)[self.axis], dtype=tf.float32)

        # if self.axis == 1:
        indices = tf.expand_dims(indices, axis=0)
        indices = tf.expand_dims(indices, axis=-1)

        # positive_indices_sum = tf.reduce_sum(indices * (inputs * positive_mask) / (inputs + 1e-8), axis=self.axis)
        positive_indices_sum = tf.reduce_sum(indices * positive_mask, axis=self.axis)
        positive_count = tf.reduce_sum(positive_mask, axis=self.axis)
        return positive_indices_sum / (positive_count + 1e-8)

MIPV = SoftMIPV


# Multiple Static Average Pooling
@register_keras_serializable()
class MultiAveragePooling1D(tf.keras.layers.Layer):
    def __init__(self, multi=5, **kwargs):
        super(MultiAveragePooling1D, self).__init__(**kwargs)
        self.multi = multi

    def build(self, input_shape):
        stride = input_shape[1]//self.multi + (0 if input_shape[1] % self.multi == 0 else 1)
        pool   = stride
        self.pool = tf.keras.layers.AveragePooling1D(pool_size=pool, strides=stride, padding='same')
        # self.pool = tf.keras.layers.AveragePooling1D(input_shape[1]//self.multi)
        if input_shape[1] / stride > self.multi - 1:
            self.reshape = tf.keras.layers.Reshape((self.multi*input_shape[2],))
        else:
            print('WARNING: MultiAveragePooling1D multi argument is too large, new value =', input_shape[1] / stride)
            self.reshape = tf.keras.layers.Reshape((-1,))  # usefull for large multi values

    def call(self, inputs):
        return self.reshape(self.pool(inputs))


# Multiple Static Max Pooling
@register_keras_serializable()
class MultiMaxPooling1D(tf.keras.layers.Layer):
    def __init__(self, multi=5, **kwargs):
        super(MultiMaxPooling1D, self).__init__(**kwargs)
        self.multi = multi

    def build(self, input_shape):
        stride = input_shape[1]//self.multi + (0 if input_shape[1] % self.multi == 0 else 1)
        pool   = stride
        self.pool = tf.keras.layers.MaxPool1D(pool_size=pool, strides=stride, padding='same')
        # self.pool = tf.keras.layers.AveragePooling1D(input_shape[1]//self.multi)
        if input_shape[1] / stride > self.multi - 1:
            self.reshape = tf.keras.layers.Reshape((self.multi*input_shape[2],))
        else:
            print('WARNING: MultiMaxPooling1D multi argument is too large, new value =', input_shape[1] / stride)
            self.reshape = tf.keras.layers.Reshape((-1,))  # usefull for large multi values

    def call(self, inputs):
        return self.reshape(self.pool(inputs))
