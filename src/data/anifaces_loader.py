import pathlib

import numpy as np
import tensorflow as tf
from PIL import Image

from .loader import Loader


class AnimeFacesLoader(Loader):
    def __init__(self, data_path, batch_size, conv_block_total, width, height, channels):
        super().__init__(data_path, batch_size, conv_block_total, width, height, channels)
        self._data_dir = pathlib.Path(self._data_path)
        self._data = [x for x in self._batch(list(self._data_dir.glob('*.jpg')), self._batch_size)]
        self._total_size = len(self._data)

    def _resize(self, image, width, height):
        image = tf.image.resize(image, size=(width, height), method=tf.image.ResizeMethod.LANCZOS3)
        return tf.cast(image, np.uint8).numpy()

    def _normalize(self, image):
        return (image - 127.5) / 127.5  # Normalize the images to [-1, 1]

    def denormalize(self, image):
        return tf.cast((image * 127.5) + 127.5, np.uint8).numpy()

    def get_batch(self, idx):
        return np.array([
            self._normalize(self._resize(np.asarray(Image.open(file_path)), self._width, self._height))
            for file_path in self._data[idx]
        ], dtype='float32')
