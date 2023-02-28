import tensorflow as tf

from tensorflow.keras import initializers, layers, optimizers, Model


class Generator:

    def __init__(self, learning_rate, conv_blocks_total, initial_width, initial_height, channels):
        self._FEATURE_MAP = 64
        self._learning_rate = learning_rate
        self._conv_blocks_total = conv_blocks_total
        self._initial_width = initial_width
        self._initial_height = initial_height
        self._channels = channels
        self._optimizer = optimizers.Adam(learning_rate=self._learning_rate, beta_1=0, beta_2=0.9)
        self._model = self._create_model()
        self._model.summary()

    def _create_model(self):
        inpt = layers.Input(shape=(100,))

        output = layers.Dense(self._initial_width * self._initial_height * (self._FEATURE_MAP * 2), use_bias=False, activation='linear')(inpt)
        output = layers.Reshape((self._initial_width, self._initial_height, (self._FEATURE_MAP * 2)))(output)

        for i in range(self._conv_blocks_total, 0, -1):
            output = self._conv_block(output, 2**(i-1))

        output = layers.Conv2DTranspose(self._channels, kernel_size=(5, 5), strides=(1, 1), padding='same', use_bias=False, activation='tanh',
                                        kernel_initializer=initializers.RandomNormal(mean=0., stddev=0.02))(output)

        model = Model(inpt, output, name='Generator')

        return model

    def _conv_block(self, inpt, features_multiplier):
        output = layers.Conv2DTranspose(self._FEATURE_MAP * features_multiplier, kernel_size=(5, 5), strides=(2, 2), padding='same', use_bias=False,
                                        kernel_initializer=initializers.RandomNormal(mean=0., stddev=0.02))(inpt)
        output = layers.BatchNormalization()(output)
        output = layers.ReLU()(output)

        return output

    def get_model(self):
        return self._model

    def loss(self, fake_output):
        return -1 * tf.reduce_mean(fake_output)

    def optimizer(self):
        return self._optimizer
