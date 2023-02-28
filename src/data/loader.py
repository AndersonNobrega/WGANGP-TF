from abc import ABC, abstractmethod

import numpy as np


class Loader(ABC):
    def __init__(self, data_path, batch_size, conv_block_total, width, height, channels):
        self._data_path = data_path
        self._batch_size = batch_size
        self._conv_block_total = conv_block_total
        self._width = width
        self._height = height
        self._channels = channels
        self._image_shape = (self._width, self._height, self._channels)
        self._total_size = None

    @abstractmethod
    def _resize(self, image, width, height):
        pass

    @abstractmethod
    def _normalize(self, image):
        pass

    @abstractmethod
    def denormalize(self, image):
        pass

    def get_conv_block_total(self):
        return self._conv_block_total

    def get_image_shape(self):
        return self._image_shape

    def get_image_width(self):
        return self._width

    def get_image_height(self):
        return self._height

    def get_image_channels(self):
        return self._channels

    def get_initial_generator_width(self):
        exp = np.log2(self._width) - self._conv_block_total
        return int(2 ** exp)

    def get_initial_generator_height(self):
        exp = np.log2(self._height) - self._conv_block_total
        return int(2 ** exp)

    def get_batches_amount(self):
        return self._total_size

    def _batch(self, iterable, n=64):
        iterable_len = len(iterable)
        for ndx in range(0, iterable_len, n):
            yield iterable[ndx:min(ndx + n, iterable_len)]

    @abstractmethod
    def get_batch(self, idx):
        pass
