import tensorflow as tf

from tensorflow.keras import initializers, layers, optimizers, Sequential


class Generator:

    def __init__(self, learning_rate):
        self._FEATURE_MAP = 64
        self._learning_rate = learning_rate
        self._optimizer = optimizers.Adam(learning_rate=self._learning_rate, beta_1=0, beta_2=0.9)
        self._model = self._create_model()
        self._model.summary()

    def _create_model(self):
        model = Sequential(name='Generator')
        model.add(layers.Dense(7 * 7 * (self._FEATURE_MAP * 2), input_shape=(100,), use_bias=False, activation='linear'))

        model.add(layers.Reshape((7, 7, (self._FEATURE_MAP * 2))))

        model.add(layers.Conv2DTranspose(self._FEATURE_MAP * 2, kernel_size=(5, 5), strides=(1, 1), padding='same', use_bias=False,
                                         kernel_initializer=initializers.RandomNormal(mean=0., stddev=0.02)))
        model.add(layers.LayerNormalization())
        model.add(layers.ReLU())

        model.add(layers.Conv2DTranspose(self._FEATURE_MAP * 1, kernel_size=(5, 5), strides=(2, 2), padding='same', use_bias=False,
                                         kernel_initializer=initializers.RandomNormal(mean=0., stddev=0.02)))
        model.add(layers.LayerNormalization())
        model.add(layers.ReLU())

        model.add(layers.Conv2DTranspose(1, kernel_size=(5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh',
                                         kernel_initializer=initializers.RandomNormal(mean=0., stddev=0.02)))

        return model

    def get_model(self):
        return self._model

    def loss(self, fake_output):
        return -1 * tf.reduce_mean(fake_output)

    def optimizer(self):
        return self._optimizer
