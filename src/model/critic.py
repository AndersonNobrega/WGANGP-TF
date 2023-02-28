import tensorflow as tf

from tensorflow.keras import initializers, layers, optimizers, Model


class Critic:

    def __init__(self, learning_rate):
        self._FEATURE_MAP = 64
        self._learning_rate = learning_rate
        self._optimizer = optimizers.Adam(learning_rate=self._learning_rate, beta_1=0, beta_2=0.9)
        self._model = self._create_model()
        self._model.summary()

    def _create_model(self):
        inpt = layers.Input(shape=(28, 28, 1))

        output = self._conv_block(inpt, 1)
        output = self._conv_block(output, 2)

        output = layers.Flatten()(output)
        output = layers.Dense(1, activation='linear')(output)

        model = Model(inpt, output, name='Critic')

        return model

    def _conv_block(self, inpt, downscaling_factor):
        output = layers.Conv2D(self._FEATURE_MAP * downscaling_factor, input_shape=[28, 28, 1], kernel_size=(5, 5), strides=(2, 2), padding='same',
                               use_bias=False, kernel_initializer=initializers.RandomNormal(mean=0., stddev=0.02))(inpt)
        output = layers.LayerNormalization()(output)
        output = layers.LeakyReLU(0.2)(output)

        return output

    def get_model(self):
        return self._model

    def loss(self, real_output, fake_output, grad_penalty, lambda_grad_penalty):
        total_loss = (-1 * (tf.reduce_mean(real_output) - tf.reduce_mean(fake_output))) + (grad_penalty * lambda_grad_penalty)
        return total_loss

    def optimizer(self):
        return self._optimizer
