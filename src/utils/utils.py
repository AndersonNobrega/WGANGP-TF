import glob
import os

import imageio


def create_gif(output_file, glob_path, delete_file=False):
    with imageio.get_writer(output_file, mode='I') as writer:
        filenames = glob.glob(glob_path)
        filenames = sorted(filenames)
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
            if delete_file:
                remove_file(filename)


def remove_file(file_path):
    try:
        os.remove(file_path)
    except OSError as e:
        print("Error trying to delete file: {} - {}.".format(e.filename, e.strerror))
