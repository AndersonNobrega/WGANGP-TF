import tensorflow as tf

from tensorflow.keras import initializers, layers, optimizers, Sequential


class Critic:

    def __init__(self, learning_rate):
        self._FEATURE_MAP = 64
        self._learning_rate = learning_rate
        self._optimizer = optimizers.Adam(learning_rate=self._learning_rate, beta_1=0, beta_2=0.9)
        self._model = self._create_model()
        self._model.summary()

    def _create_model(self):
        model = Sequential(name='Critic')
        model.add(layers.Conv2D(self._FEATURE_MAP * 1, input_shape=[28, 28, 1], kernel_size=(5, 5), strides=(2, 2), padding='same',
                                use_bias=False, kernel_initializer=initializers.RandomNormal(mean=0., stddev=0.02)))
        model.add(layers.LayerNormalization())
        model.add(layers.LeakyReLU(0.2))

        model.add(layers.Conv2D(self._FEATURE_MAP * 2, kernel_size=(5, 5), strides=(2, 2), padding='same', use_bias=False,
                                kernel_initializer=initializers.RandomNormal(mean=0., stddev=0.02)))
        model.add(layers.LayerNormalization())
        model.add(layers.LeakyReLU(0.2))

        model.add(layers.Flatten())
        model.add(layers.Dense(1, activation='linear'))

        return model

    def get_model(self):
        return self._model

    def loss(self, real_output, fake_output, grad_penalty, lambda_grad_penalty):
        total_loss = (-1 * (tf.reduce_mean(real_output) - tf.reduce_mean(fake_output))) + (grad_penalty * lambda_grad_penalty)
        return total_loss

    def optimizer(self):
        return self._optimizer
