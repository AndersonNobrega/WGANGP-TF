import datetime
import os
import pathlib
import time
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import matplotlib.pyplot as plt

# Remove Tensorflow log spam
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

# Check for available GPU
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    print('Using GPU for model training.\n')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
else:
    print('No GPU available for model training. Using CPU instead.\n')

from model import Generator, Critic
from utils import create_gif


def get_args():
    parser = ArgumentParser(allow_abbrev=False, description='', formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('-b', '--batch_size', type=int, help='Batch size for the training dataset.', default=256)
    parser.add_argument('-e', '--epochs', type=int, help='Amount of epochs to train model.', default=1)
    parser.add_argument('-l', '--learning_rate', type=float, help='Learning rate for both the generator and critic models.', default=3e-4)
    parser.add_argument('-n', '--noise_dim', type=int, help='Dimension for noise vector used by the generator.', default=100)
    parser.add_argument('-u', '--num_generate', type=int, help='Dimension for noise vector used by the generator.', default=16)
    parser.add_argument('-s', '--buffer_size', type=int, help='Buffer size for dataset shuffle.', default=60000)
    parser.add_argument('-c', '--critic_iterations', type=int, help='Number of critic iterations per generator iteration', default=5)

    return vars(parser.parse_args())


def load_dataset(buffer_size, batch_size):
    (train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()

    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
    train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]

    # Batch and shuffle the data
    return tf.data.Dataset.from_tensor_slices(train_images).shuffle(buffer_size).batch(batch_size)


def generate_and_save_images(model, epoch, test_input, img_path):
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    plt.savefig('{}/image_at_epoch_{:04d}.png'.format(img_path, epoch))
    plt.close(fig)


@tf.function
def critic_train_step(generator, critic, real_input_batch, noise_dim):
    noise = tf.random.normal([real_input_batch.shape[0], noise_dim])

    with tf.GradientTape() as critic_tape:
        with tf.GradientTape() as gp_tape:
            generator_fake_input = generator.get_model()(noise, training=True)

            epsilon = tf.random.uniform(shape=[real_input_batch.shape[0], 1, 1, 1], minval=0, maxval=1)
            mixed_output = (epsilon * real_input_batch) + ((1 - epsilon) * generator_fake_input)
            mixed_predictions = critic.get_model()(mixed_output, training=True)

        grad = gp_tape.gradient(mixed_predictions, mixed_output)
        grad_norm = tf.sqrt(tf.reduce_sum(tf.square(grad)))
        grad_penalty = tf.reduce_mean(tf.square(grad_norm - 1))

        critic_real_output = critic.get_model()(real_input_batch, training=True)
        critic_fake_output = critic.get_model()(generator_fake_input, training=True)

        critic_loss = critic.loss(critic_real_output, critic_fake_output, grad_penalty, 10)

    gradients_of_critic = critic_tape.gradient(critic_loss, critic.get_model().trainable_variables)
    critic.optimizer().apply_gradients(zip(gradients_of_critic, critic.get_model().trainable_variables))

    return critic_loss


@tf.function
def generator_train_step(generator, critic, batch_size, noise_dim):
    noise = tf.random.normal([batch_size, noise_dim])

    with tf.GradientTape() as gen_tape:
        generator_fake_input = generator.get_model()(noise, training=True)
        critic_fake_output = critic.get_model()(generator_fake_input, training=True)

        gen_loss = generator.loss(critic_fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.get_model().trainable_variables)
    generator.optimizer().apply_gradients(zip(gradients_of_generator, generator.get_model().trainable_variables))

    return gen_loss


def train(generator, critic, dataset, img_path, epochs, num_generate, noise_dim, critic_iterations):
    print("\n---------- Starting training loop... ----------\n")

    seed = tf.random.normal([num_generate, noise_dim])

    generator_loss_hist = []
    critic_loss_hist = []

    for epoch in range(epochs):
        start = time.time()

        for image_batch in dataset:
            temp_loss = []
            for _ in range(critic_iterations):
                temp_loss.append(critic_train_step(generator, critic, image_batch, noise_dim))
            critic_loss_hist.append(tf.reduce_mean(temp_loss))
            generator_loss_hist.append(generator_train_step(generator, critic, image_batch.shape[0], noise_dim))

        # Produce images for the GIF as you go
        generate_and_save_images(generator.get_model(), epoch + 1, seed, img_path)

        print('Epoch: {}/{} - Time: {:.2f} seconds - Critic Loss: {:.4f} - Generator Loss: {:.4f}'.format(epoch + 1, epochs, time.time() - start,
                                                                                                          critic_loss_hist[-1],
                                                                                                          generator_loss_hist[-1]))

    print("\n---------- Training loop finished. ----------\n")


def main():
    # Get CLI args
    args = get_args()

    # Create directory for images
    img_path = pathlib.Path(__file__).resolve().parents[1] / pathlib.Path("img") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    img_path.mkdir(parents=True)

    # Create Generator and Critic models
    generator = Generator(learning_rate=args['learning_rate'])
    critic = Critic(learning_rate=args['learning_rate'])

    # Train loop
    train(generator,
          critic,
          load_dataset(args['buffer_size'],
                       args['batch_size']),
          img_path,
          args['epochs'],
          args['num_generate'],
          args['noise_dim'],
          args['critic_iterations'])

    # Create gif from all images created during training
    create_gif('{}/dcgan.gif'.format(img_path), '{}/image*.png'.format(img_path), delete_file=True)


if __name__ == '__main__':
    main()
