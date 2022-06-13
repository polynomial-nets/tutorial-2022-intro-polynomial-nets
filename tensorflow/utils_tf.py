"""
Dataloader for Cifar 10
Reference: https://github.com/lionelmessi6410/tensorflow2-cifar
"""
from os.path import splitext, basename, dirname
import sys
import tensorflow as tf
from tensorflow.keras import datasets
import numpy as np
padding = 4
image_size = 32
target_size = 32 + padding * 2
def load_module(fn, name):
    mod_name = splitext(basename(fn))[0]
    mod_path = dirname(fn)
    sys.path.insert(0, mod_path)
    return getattr(__import__(mod_name), name)

def load_model(model_fn, model_name, args=None):
    model = load_module(model_fn, model_name)
    model1 = model(**args) if args else model()
    return model1

def MultiStepLR(initial_learning_rate, lr_steps, lr_rate):
    """Multi-steps learning rate scheduler."""
    lr_steps_value = [initial_learning_rate]
    for _ in range(len(lr_steps)):
        lr_steps_value.append(lr_steps_value[-1] * lr_rate)
    return tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries=lr_steps, values=lr_steps_value)

def return_loaders(batch_size, **kwargs):
    """
    Return the loader for the data. This is used both for training and for
    validation. Currently, hardcoded to CIFAR10.
    :param batch_size: (int) The batch size for training.
    :param kwargs:
    :return: The train and validation time loaders.
    """
    train_images, train_labels, test_images, test_labels = get_dataset()
    mean, std = get_mean_and_std(train_images)
    train_images = normalize(train_images, mean, std)
    test_images = normalize(test_images, mean, std)
    train_ds = dataset_generator(train_images, train_labels,batch_size)
    test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels)). \
        batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return train_ds, test_ds

def get_dataset():
    """Download, parse and process a dataset to unit scale and one-hot labels."""
    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
    # Normalize pixel values to be between 0 and 1
    train_images, test_images = train_images / 255.0, test_images / 255.0
    # One-hot labels
    train_labels = _one_hot(train_labels, 10)
    test_labels = _one_hot(test_labels, 10)
    return train_images, train_labels, test_images, test_labels

def get_mean_and_std(images):
    """Compute the mean and std value of dataset."""
    mean = np.mean(images, axis=(0, 1, 2))
    std = np.std(images, axis=(0, 1, 2))
    return mean, std

def normalize(images, mean, std):
    """Normalize data with mean and std."""
    return (images - mean) / std

def dataset_generator(images, labels, batch_size):
    ds = tf.data.Dataset.from_tensor_slices((images, labels))
    ds = ds.map(_augment_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.shuffle(len(images)).batch(batch_size)
    ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return ds

def _one_hot(train_labels, num_classes, dtype=np.float32):
    """Create a one-hot encoding of labels of size num_classes."""
    return np.array(train_labels == np.arange(num_classes), dtype)

def _augment_fn(images, labels):
    images = tf.image.pad_to_bounding_box(images, padding, padding, target_size, target_size)
    images = tf.image.random_crop(images, (image_size, image_size, 3))
    images = tf.image.random_flip_left_right(images)
    return images, labels