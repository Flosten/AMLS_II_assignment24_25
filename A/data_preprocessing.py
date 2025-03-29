import re
from functools import partial

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

import A.visualising as vs


def data_acquisition(GCS_path, train_file_path, image_size):
    trainfile, valfile = split_data(GCS_path, train_file_path)
    train_dataset = load_trainset(trainfile, image_size=image_size, labeled=True)
    val_dataset = load_valset(valfile, image_size=image_size, labeled=True)
    # visualise the images
    # fig1, fig2 = vs.origin_image_plot(train_dataset)
    # fig2.savefig("figures/Healthy Leaf.png")
    # fig1.savefig("figures/Diseases Leaf.png")
    return train_dataset, val_dataset


def decode_raw_image(image_data, image_shape):
    image = tf.image.decode_jpeg(image_data, channels=3)
    image = tf.image.resize(image, image_shape)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.reshape(image, [*image_shape, 3])
    return image


def read_tfrecord(example, image_size, labeled):
    tfrecord_format = (
        {
            "image": tf.io.FixedLenFeature([], tf.string),
            "target": tf.io.FixedLenFeature([], tf.int64),
        }
        if labeled
        else {
            "image": tf.io.FixedLenFeature([], tf.string),
            "image_name": tf.io.FixedLenFeature([], tf.string),
        }
    )
    example = tf.io.parse_single_example(example, tfrecord_format)
    images = decode_raw_image(example["image"], image_shape=image_size)
    if labeled:
        labels = tf.cast(example["target"], tf.int32)
        return images, labels
    image_name = example["image_name"]
    return images, image_name


def split_data(GCS_path, train_file_path):
    train_filename, val_filename = train_test_split(
        tf.io.gfile.glob(GCS_path + train_file_path), train_size=0.8, random_state=711
    )
    return train_filename, val_filename


def load_trainset(filenames, image_size, labeled, ordered=False):
    ignore_order = tf.data.Options()
    if not ordered:
        ignore_order.experimental_deterministic = True  # disable order, increase speed
    dataset = tf.data.TFRecordDataset(
        filenames,
        num_parallel_reads=tf.data.experimental.AUTOTUNE,
    )  # automatically interleaves reads from multiple files
    dataset = dataset.with_options(
        ignore_order
    )  # uses data as soon as it streams in, rather than in its original order
    dataset = dataset.map(
        partial(read_tfrecord, image_size=image_size, labeled=labeled),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    return dataset


def load_valset(filenames, image_size, labeled, ordered=False):
    ignore_order = tf.data.Options()
    if not ordered:
        ignore_order.experimental_deterministic = False  # disable order, increase speed
    dataset = tf.data.TFRecordDataset(
        filenames,
        num_parallel_reads=tf.data.experimental.AUTOTUNE,
    )  # automatically interleaves reads from multiple files
    dataset = dataset.with_options(
        ignore_order
    )  # uses data as soon as it streams in, rather than in its original order
    dataset = dataset.map(
        partial(read_tfrecord, image_size=image_size, labeled=labeled),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    return dataset


def count_data_items(trainset):
    sample_count = sum(1 for _ in trainset)
    return sample_count
