import tensorflow as tf


class CheckpointManager:
    def __init__(self, generator, critic, path, checkpoint_epoch):
        self._path = path
        self._checkpoint_epoch = checkpoint_epoch
        self._checkpoint = tf.train.Checkpoint(generator_optimizer=generator.optimizer(),
                                               critic_optimizer=critic.optimizer(),
                                               generator=generator.get_model(),
                                               critic=critic.get_model())
        self._manager = tf.train.CheckpointManager(self._checkpoint, directory=self._path, max_to_keep=5)

    def save(self, epoch):
        if self._checkpoint_epoch > 0 and epoch % self._checkpoint_epoch == 0:
            self._manager.save()
