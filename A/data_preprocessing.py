import re
from functools import partial

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

import A.visualising as vs


def UNet_preprocessing_pro(dataset, batch_size, gen):
    healthy_leaf_set = get_healthy_image(dataset)
    unet_dataset = build_Unet_dataset(healthy_leaf_set)
    cbb_leaf_set = get_CBB_image(dataset)
    cbb_dataset = build_cbb_dataset(cbb_leaf_set)

    # fig = vs.plot_cive_result(unet_dataset)
    # fig.savefig("figures/Image after CIVE.png")

    image_list = []
    mask_list = []

    for img, msk in unet_dataset:
        image_list.append(img)
        mask_list.append(msk)

    for img, msk in cbb_dataset:
        image_list.append(img)
        mask_list.append(msk)
        if gen.uniform(()) > 0.8:
            img_flip = tf.image.flip_left_right(img)
            msk_flip = tf.image.flip_left_right(msk)
            image_list.append(img_flip)
            mask_list.append(msk_flip)
        if gen.uniform(()) > 0.8:
            img_flip = tf.image.flip_up_down(img)
            msk_flip = tf.image.flip_up_down(msk)
            image_list.append(img_flip)
            mask_list.append(msk_flip)

    image_list = np.array(image_list)
    mask_list = np.array(mask_list)

    print(image_list.shape)

    x_train, x_val, y_train, y_val = train_test_split(
        image_list, mask_list, test_size=0.2, random_state=711
    )

    unet_trainset = (
        tf.data.Dataset.from_tensor_slices((x_train, y_train))
        .shuffle(1000)
        .batch(batch_size)
    )
    unet_valset = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(batch_size)

    return unet_trainset, unet_valset


def build_cbb_dataset(train_dataset):
    cbb_dataset = train_dataset.map(
        create_mask_pair_cbb, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    return cbb_dataset


def UNet_preprocessing(dataset, batch_size):
    healthy_leaf_set = get_healthy_image(dataset)
    unet_dataset = build_Unet_dataset(healthy_leaf_set)

    # fig = vs.plot_cive_result(unet_dataset)
    # fig.savefig("figures/Image after CIVE.png")

    image = []
    mask = []

    for img, msk in unet_dataset:
        image.append(img)
        mask.append(msk)

    image = np.array(image)
    mask = np.array(mask)

    print(image.shape)

    x_train, x_val, y_train, y_val = train_test_split(
        image, mask, test_size=0.2, random_state=711
    )

    unet_trainset = (
        tf.data.Dataset.from_tensor_slices((x_train, y_train))
        .shuffle(1000)
        .batch(batch_size)
    )
    unet_valset = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(batch_size)

    return unet_trainset, unet_valset


def compute_cive(image):
    # CIVE = 0.441 * R - 0.811 * G + 0.385 * B + 18.78745
    R = image[:, :, 0]
    G = image[:, :, 1]
    B = image[:, :, 2]
    cive = 0.441 * R - 0.811 * G + 0.385 * B + 18.78745
    return cive


def compute_cive_cbb(image):
    # CIVE = 5.95 * R + 0.6 * G - 1.885 * B - 48.787
    R = image[:, :, 0]
    G = image[:, :, 1]
    B = image[:, :, 2]
    cive = 10.441 * R + 0.611 * G - 1.885 * B - 48.787
    return cive


def get_mask(cive, threshold=None):
    if threshold is None:
        # threshold = tf.reduce_mean(cive)
        threshold = 18.7  # threshold is set to 18.7
    mask = tf.cast(cive < threshold, tf.float32)
    mask = tf.expand_dims(mask, axis=-1)
    return mask


def get_mask_cbb(cive, threshold=None):
    if threshold is None:
        threshold = tf.reduce_mean(cive)
        # threshold = -48.63
    mask = tf.cast(cive > threshold, tf.float32)
    mask = tf.expand_dims(mask, axis=-1)
    return mask


def create_mask_pair(image, label):
    cive = compute_cive(image)
    mask = get_mask(cive, threshold=None)
    return image, mask


def create_mask_pair_cbb(image, label):
    cive = compute_cive_cbb(image)
    mask = get_mask_cbb(cive, threshold=None)
    return image, mask


def build_Unet_dataset(train_dataset):
    Unet_dataset = train_dataset.map(
        create_mask_pair, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    return Unet_dataset


def which_classes(image, label, classes):
    return tf.equal(label, classes)


def get_healthy_image(train_dataset):
    filter_fn = partial(which_classes, classes=4)
    healthy_dataset = train_dataset.filter(filter_fn)
    # healthy_dataset_num = count_data_items(healthy_dataset)
    # print(f"Number of healthy images: {healthy_dataset_num}")
    return healthy_dataset


def get_all_diseases_image(train_dataset):
    cbb_dataset = get_CBB_image(train_dataset)
    cbsd_dataset = get_CBSD_image(train_dataset)
    cgm_dataset = get_CGM_image(train_dataset)
    cmd_dataset = get_CMD_image(train_dataset)
    return cbb_dataset, cbsd_dataset, cgm_dataset, cmd_dataset


def get_CBB_image(train_dataset):
    filter_fn = partial(which_classes, classes=0)
    cbb_dataset = train_dataset.filter(filter_fn)
    # cbb_dataset_num = count_data_items(cbb_dataset)
    # print(f"Number of CBB images: {cbb_dataset_num}")
    return cbb_dataset


def get_CBSD_image(train_dataset):
    filter_fn = partial(which_classes, classes=1)
    cbsd_dataset = train_dataset.filter(filter_fn)
    # cbsd_dataset_num = count_data_items(cbsd_dataset)
    # print(f"Number of CBSD images: {cbsd_dataset_num}")
    return cbsd_dataset


def get_CGM_image(train_dataset):
    filter_fn = partial(which_classes, classes=2)
    cgm_dataset = train_dataset.filter(filter_fn)
    # cgm_dataset_num = count_data_items(cgm_dataset)
    # print(f"Number of CGM images: {cgm_dataset_num}")
    return cgm_dataset


def get_CMD_image(train_dataset):
    filter_fn = partial(which_classes, classes=3)
    cmd_dataset = train_dataset.filter(filter_fn)
    # cmd_dataset_num = count_data_items(cmd_dataset)
    # print(f"Number of CMD images: {cmd_dataset_num}")
    return cmd_dataset


def data_acquisition(GCS_path, train_file_path, image_size):
    trainfile, valfile = split_data(GCS_path, train_file_path)
    train_dataset = load_trainset(trainfile, image_size=image_size, labeled=True)
    val_dataset = load_valset(valfile, image_size=image_size, labeled=True)
    # visualise the images
    # fig1, fig2 = vs.origin_image_plot(train_dataset)
    # fig2.savefig("figures/Healthy Leaf.png")
    # fig1.savefig("figures/Diseases Leaf.png")
    return train_dataset, val_dataset


def data_acquisition_classification(
    GCS_path, train_file_path, image_size, train_ratio, val_ratio
):
    trainfile, valfile, testfile = split_data_classification(
        GCS_path, train_file_path, train_ratio, val_ratio
    )
    train_dataset = load_trainset(trainfile, image_size=image_size, labeled=True)
    val_dataset = load_valset(valfile, image_size=image_size, labeled=True)
    test_dataset = load_testset(testfile, image_size=image_size, labeled=True)
    return train_dataset, val_dataset, test_dataset


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


def split_data_classification(GCS_path, train_file_path, train_ratio, val_ratio):

    train_filename, val_test_filename = train_test_split(
        tf.io.gfile.glob(GCS_path + train_file_path),
        train_size=train_ratio,
        random_state=711,
    )

    val_test_ratio = val_ratio / (1 - train_ratio)
    val_filename, test_filename = train_test_split(
        val_test_filename, train_size=val_test_ratio, random_state=711
    )

    return train_filename, val_filename, test_filename


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


def load_testset(filenames, image_size, labeled, ordered=False):
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


def count_data_items(trainset):
    sample_count = sum(1 for _ in trainset)
    return sample_count
