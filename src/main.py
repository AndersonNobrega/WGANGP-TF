import datetime
import io
import os
import pathlib
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from tqdm import tqdm

import matplotlib.pyplot as plt

# Remove Tensorflow log spam
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

# Check for available GPU
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tqdm.write('Using GPU for model training.\n')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
else:
    tqdm.write('No GPU available for model training. Using CPU instead.\n')

from data import AnimeFacesLoader
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
    parser.add_argument('-d', '--dataset', type=str, required=True, help='Image dataset that is going to be used (mnist, anime_faces)')
    parser.add_argument('-p', '--dataset_path', type=str, help='Path for the dataset that is going to be used', default=None)

    return vars(parser.parse_args())


def load_dataset(buffer_size, batch_size):
    (train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()

    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
    train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]

    # Batch and shuffle the data
    return tf.data.Dataset.from_tensor_slices(train_images).shuffle(buffer_size).batch(batch_size)


def get_dataset_loader(dataset_name, batch_size, dataset_path=None):
    if dataset_name == 'mnist':
        # TODO: Create loader for MNIST
        pass
    elif dataset_name == 'anime_faces':
        if dataset_path is None:
            raise IOError('Dataset path is required.')
        return AnimeFacesLoader(dataset_path, batch_size=batch_size, conv_block_total=3, width=64, height=64, channels=3)
    else:
        raise ValueError('Invalid dataset. if you wish to use a new one, please implement a loader for it.')


def generate_and_save_images(model, test_input, loader, epoch=None, img_path=None):
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        if loader.get_image_channels() == 1:
            plt.imshow(loader.denormalize(predictions[i, :, :, 0]), cmap='gray')
        else:
            plt.imshow(loader.denormalize(predictions[i, :, :, :]))
        plt.axis('off')

    if epoch is not None and img_path is not None:
        plt.savefig('{}/image_at_epoch_{:04d}.png'.format(img_path, epoch))

    # Step needed to be compatible with tensorboard
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)

    image = tf.image.decode_png(buf.getvalue(), channels=4)
    image = tf.expand_dims(image, 0)

    return image


def write_tensorboard_logs(file_writer, label, content, step, content_type='scalar'):
    with file_writer.as_default():
        if content_type == 'scalar':
            tf.summary.scalar(label, content, step=step)
        elif content_type == 'image':
            tf.summary.image(label, content, step=step)


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
        grad_norm = tf.sqrt(tf.reduce_sum(tf.square(grad), axis=[1, 2, 3]))
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


def train(generator, critic, loader, img_path, epochs, num_generate, noise_dim, critic_iterations, critic_file_writer, generator_file_writer):
    tqdm.write("\n---------- Starting training loop... ----------\n")

    seed = tf.random.normal([num_generate, noise_dim])

    generator_loss_hist = []
    critic_loss_hist = []
    step = 0

    for epoch in range(epochs):
        tqdm.write('Epoch: {}/{}'.format(epoch + 1, epochs))

        for batch_idx in tqdm(range(loader.get_batches_amount())):
            image_batch = loader.get_batch(batch_idx)
            temp_loss = []

            for _ in range(critic_iterations):
                temp_loss.append(critic_train_step(generator, critic, image_batch, noise_dim))
            critic_loss_hist.append(tf.reduce_mean(temp_loss))
            generator_loss_hist.append(generator_train_step(generator, critic, image_batch.shape[0], noise_dim))

            if batch_idx % 100 == 0 and batch_idx > 0:
                write_tensorboard_logs(critic_file_writer, 'Loss', tf.reduce_mean(critic_loss_hist), step, 'scalar')
                write_tensorboard_logs(generator_file_writer, 'Loss', tf.reduce_mean(generator_loss_hist), step, 'scalar')

                write_tensorboard_logs(generator_file_writer, 'Generated Images', generate_and_save_images(generator.get_model(), seed, loader), step,
                                       'image')

                step += 1

        # Produce images for the GIF as you go
        generate_and_save_images(generator.get_model(), seed, loader, epoch + 1, img_path)

        tqdm.write(('Critic Loss: {:.4f} - Generator Loss: {:.4f}'.format(tf.reduce_mean(critic_loss_hist), tf.reduce_mean(generator_loss_hist))))

        generator_loss_hist.clear()
        critic_loss_hist.clear()

    tqdm.write("\n---------- Training loop finished. ----------\n")


def main():
    # Get CLI args
    args = get_args()

    # Create directory for images
    current_time = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

    img_path = pathlib.Path(__file__).resolve().parents[1] / pathlib.Path("img") / current_time
    img_path.mkdir(parents=True)

    # Tensorboard file writer for training logs
    critic_log_dir = "logs/{}/critic/".format(current_time)
    generator_log_dir = "logs/{}/generator/".format(current_time)

    critic_file_writer = tf.summary.create_file_writer(critic_log_dir)
    generator_file_writer = tf.summary.create_file_writer(generator_log_dir)

    # Get loader for specified dataset
    loader = get_dataset_loader(args['dataset'], args['batch_size'], args['dataset_path'])

    # Create Generator and Critic models
    generator = Generator(learning_rate=args['learning_rate'],
                          conv_blocks_total=loader.get_conv_block_total(),
                          initial_width=loader.get_initial_generator_width(),
                          initial_height=loader.get_initial_generator_height(),
                          channels=loader.get_image_channels())

    critic = Critic(learning_rate=args['learning_rate'],
                    conv_blocks_total=loader.get_conv_block_total(),
                    input_shape=loader.get_image_shape())

    # Train loop
    train(generator,
          critic,
          loader,
          img_path,
          args['epochs'],
          args['num_generate'],
          args['noise_dim'],
          args['critic_iterations'],
          critic_file_writer,
          generator_file_writer)

    # Create gif from all images created during training
    create_gif('{}/wgan_gp.gif'.format(img_path), '{}/image*.png'.format(img_path), delete_file=True)


if __name__ == '__main__':
    main()
