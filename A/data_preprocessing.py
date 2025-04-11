"""
This file contains all the functions to preprocess the dataset for
the cassava leaf image classification task.
The functions include:
- `image acquisition`
- `image preprocessing`
- `CIVE calculation`
- `mask creation`
- `dataset preprocessing for image segmentation`
- `dataset creation`
- `tfrecord saving`
- `dataset loading`
- `dataset preprocessing for classification`
"""

# import re
from functools import partial

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

import A.visualising as vs


def UNet_preprocessing_pro(dataset, batch_size, gen):
    """
    Preprocess the dataset including healthy and CBB images for UNet training.

    Args:
        dataset: tf.data.Dataset object.
        batch_size: int, batch size for the dataset.
        gen: tf.random.Generator object.

    Returns:
        unet_trainset: tf.data.Dataset object for training.
        unet_valset: tf.data.Dataset object for validation.
    """
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
    """
    Build the dataset for CBB images.

    Args:
        train_dataset: tf.data.Dataset object.

    Returns:
        cbb_dataset: tf.data.Dataset object for CBB images.
    """
    cbb_dataset = train_dataset.map(
        create_mask_pair_cbb, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    return cbb_dataset


def UNet_preprocessing(dataset, batch_size):
    """
    Preprocess the dataset including healthy images for UNet training.

    Args:
        dataset: tf.data.Dataset object.
        batch_size: int, batch size for the dataset.

    Returns:
        unet_trainset: tf.data.Dataset object for training.
        unet_valset: tf.data.Dataset object for validation.
    """
    healthy_leaf_set = get_healthy_image(dataset)
    unet_dataset = build_Unet_dataset(healthy_leaf_set)

    fig = vs.plot_cive_result(unet_dataset)
    fig.savefig("figures/Image after CIVE.png")

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
    """
    Compute the CIVE index for a given healthy cassava leaf image.

    Args:
        image: np.ndarray, input image.

    Returns:
        cive: np.ndarray, CIVE index.
    """
    # CIVE = 0.441 * R - 0.811 * G + 0.385 * B + 18.78745
    R = image[:, :, 0]
    G = image[:, :, 1]
    B = image[:, :, 2]
    cive = 0.441 * R - 0.811 * G + 0.385 * B + 18.78745
    return cive


def compute_cive_cbb(image):
    """
    Compute the CIVE index for a given CBB cassava leaf image.

    Args:
        image: np.ndarray, input image.

    Returns:
        cive: np.ndarray, CIVE index.
    """
    # CIVE = 10.441 * R + 0.611 * G - 1.885 * B - 48.787
    R = image[:, :, 0]
    G = image[:, :, 1]
    B = image[:, :, 2]
    cive = 10.441 * R + 0.611 * G - 1.885 * B - 48.787
    return cive


def get_mask(cive, threshold=None):
    """
    Create the mask based on the CIVE index.

    Args:
        cive: np.ndarray, CIVE index.
        threshold: float, threshold value for mask creation.

    Returns:
        mask: np.ndarray, binary mask.
    """
    if threshold is None:
        # threshold = tf.reduce_mean(cive)
        threshold = 18.7  # threshold is set to 18.7
    mask = tf.cast(cive < threshold, tf.float32)
    mask = tf.expand_dims(mask, axis=-1)
    return mask


def get_mask_cbb(cive, threshold=None):
    """
    Create the mask based on the CIVE index for CBB images.

    Args:
        cive: np.ndarray, CIVE index.
        threshold: float, threshold value for mask creation.

    Returns:
        mask: np.ndarray, binary mask.
    """
    if threshold is None:
        threshold = tf.reduce_mean(cive)
        # threshold = -48.63
    mask = tf.cast(cive > threshold, tf.float32)
    mask = tf.expand_dims(mask, axis=-1)
    return mask


def create_mask_pair(image, label):
    """
    Create a mask pair for the given image and label.

    Args:
        image: np.ndarray, input image.
        label: np.ndarray, input label.

    Returns:
        image: np.ndarray, input image.
        mask: np.ndarray, binary mask.
    """
    cive = compute_cive(image)
    mask = get_mask(cive, threshold=None)
    return image, mask


def create_mask_pair_cbb(image, label):
    """
    Create a mask pair for the given CBB image and label.

    Args:
        image: np.ndarray, input image.
        label: np.ndarray, input label.

    Returns:
        image: np.ndarray, input image.
        mask: np.ndarray, binary mask.
    """
    cive = compute_cive_cbb(image)
    mask = get_mask_cbb(cive, threshold=None)
    return image, mask


def build_Unet_dataset(train_dataset):
    """
    Build the dataset including images and masks for UNet training.

    Args:
        train_dataset: tf.data.Dataset object.

    Returns:
        Unet_dataset: tf.data.Dataset object for UNet training.
    """
    unet_dataset = train_dataset.map(
        create_mask_pair, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    return unet_dataset


def which_classes(image, label, classes):
    """
    Filter the dataset based on the given classes.

    Args:
        image: np.ndarray, input image.
        label: np.ndarray, input label.
        classes: int, class to filter.

    Returns:
        bool: True if the label matches the classes, False otherwise.
    """
    return tf.equal(label, classes)


def get_healthy_image(train_dataset):
    """
    Get the healthy images from the dataset.

    Args:
        train_dataset: tf.data.Dataset object.

    Returns:
        healthy_dataset: tf.data.Dataset object for healthy images.
    """
    filter_fn = partial(which_classes, classes=4)
    healthy_dataset = train_dataset.filter(filter_fn)
    # healthy_dataset_num = count_data_items(healthy_dataset)
    # print(f"Number of healthy images: {healthy_dataset_num}")
    return healthy_dataset


def get_all_diseases_image(train_dataset):
    """
    Get all the diseases images including CBB, CBSD, CGM, and CMD from the dataset.

    Args:
        train_dataset: tf.data.Dataset object.

    Returns:
        cbb_dataset: tf.data.Dataset object for CBB images.
        cbsd_dataset: tf.data.Dataset object for CBSD images.
        cgm_dataset: tf.data.Dataset object for CGM images.
        cmd_dataset: tf.data.Dataset object for CMD images.
    """
    cbb_dataset = get_CBB_image(train_dataset)
    cbsd_dataset = get_CBSD_image(train_dataset)
    cgm_dataset = get_CGM_image(train_dataset)
    cmd_dataset = get_CMD_image(train_dataset)
    return cbb_dataset, cbsd_dataset, cgm_dataset, cmd_dataset


def get_CBB_image(train_dataset):
    """
    Get the CBB images from the dataset.

    Args:
        train_dataset: tf.data.Dataset object.

    Returns:
        cbb_dataset: tf.data.Dataset object for CBB images.
    """
    filter_fn = partial(which_classes, classes=0)
    cbb_dataset = train_dataset.filter(filter_fn)
    # cbb_dataset_num = count_data_items(cbb_dataset)
    # print(f"Number of CBB images: {cbb_dataset_num}")
    return cbb_dataset


def get_CBSD_image(train_dataset):
    """
    Get the CBSD images from the dataset.

    Args:
        train_dataset: tf.data.Dataset object.

    Returns:
        cbsd_dataset: tf.data.Dataset object for CBSD images.
    """
    filter_fn = partial(which_classes, classes=1)
    cbsd_dataset = train_dataset.filter(filter_fn)
    # cbsd_dataset_num = count_data_items(cbsd_dataset)
    # print(f"Number of CBSD images: {cbsd_dataset_num}")
    return cbsd_dataset


def get_CGM_image(train_dataset):
    """
    Get the CGM images from the dataset.

    Args:
        train_dataset: tf.data.Dataset object.

    Returns:
        cgm_dataset: tf.data.Dataset object for CGM images.
    """
    filter_fn = partial(which_classes, classes=2)
    cgm_dataset = train_dataset.filter(filter_fn)
    # cgm_dataset_num = count_data_items(cgm_dataset)
    # print(f"Number of CGM images: {cgm_dataset_num}")
    return cgm_dataset


def get_CMD_image(train_dataset):
    """
    Get the CMD images from the dataset.

    Args:
        train_dataset: tf.data.Dataset object.

    Returns:
        cmd_dataset: tf.data.Dataset object for CMD images.
    """
    filter_fn = partial(which_classes, classes=3)
    cmd_dataset = train_dataset.filter(filter_fn)
    # cmd_dataset_num = count_data_items(cmd_dataset)
    # print(f"Number of CMD images: {cmd_dataset_num}")
    return cmd_dataset


def data_acquisition(gcs_path, train_file_path, image_size):
    """
    Acquire the training and validation datasets for image segmentation.

    Args:
        gcs_path: str, path to the GCS bucket.
        train_file_path: str, path to the training file.
        image_size: list, size of the images.

    Returns:
        train_dataset: tf.data.Dataset object for training.
        val_dataset: tf.data.Dataset object for validation.
    """
    trainfile, valfile = split_data(gcs_path, train_file_path)
    train_dataset = load_trainset(trainfile, image_size=image_size, labeled=True)
    val_dataset = load_valset(valfile, image_size=image_size, labeled=True)
    # visualise the images
    fig1, fig2 = vs.origin_image_plot(train_dataset)
    fig2.savefig("figures/Healthy Leaf.png")
    fig1.savefig("figures/Diseases Leaf.png")
    return train_dataset, val_dataset


def data_acquisition_classification(
    gcs_path, train_file_path, image_size, train_ratio, val_ratio
):
    """
    Acquire the training, validation, and test datasets to build a new dataset for classification.

    Args:
        gcs_path: str, path to the GCS bucket.
        train_file_path: str, path to the training file.
        image_size: list, size of the images.
        train_ratio: float, ratio of training data.
        val_ratio: float, ratio of validation data.

    Returns:
        train_dataset: tf.data.Dataset object for training.
        val_dataset: tf.data.Dataset object for validation.
        test_dataset: tf.data.Dataset object for testing.
    """
    trainfile, valfile, testfile = split_data_classification(
        gcs_path, train_file_path, train_ratio, val_ratio
    )
    train_dataset = load_trainset(trainfile, image_size=image_size, labeled=True)
    val_dataset = load_valset(valfile, image_size=image_size, labeled=True)
    test_dataset = load_testset(testfile, image_size=image_size, labeled=True)
    return train_dataset, val_dataset, test_dataset


def decode_raw_image(image_data, image_shape):
    """
    Decode the raw image data and resize it to the specified shape.

    Args:
        image_data: bytes, raw image data.
        image_shape: list, shape of the image.

    Returns:
        image: tf.Tensor, decoded and resized image.
    """
    image = tf.image.decode_jpeg(image_data, channels=3)
    image = tf.image.resize(image, image_shape)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.reshape(image, [*image_shape, 3])
    return image


def read_tfrecord(example, image_size, labeled):
    """
    Read a single TFRecord example and decode the image data.

    Args:
        example: tf.train.Example, TFRecord example.
        image_size: list, size of the images.
        labeled: bool, whether the dataset is labeled.

    Returns:
        images: tf.Tensor, decoded and resized image.
        labels: tf.Tensor, labels for the images (if labeled).
    """
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


def split_data(gcs_path, train_file_path):
    """
    Split the tfrecords into training and validation use.

    Args:
        gcs_path: str, path to the GCS bucket.
        train_file_path: str, path to the training file.

    Returns:
        train_filename: list, list of training filenames.
        val_filename: list, list of validation filenames.
    """
    train_filename, val_filename = train_test_split(
        tf.io.gfile.glob(gcs_path + train_file_path), train_size=0.8, random_state=711
    )
    return train_filename, val_filename


def split_data_classification(gcs_path, train_file_path, train_ratio, val_ratio):
    """
    Split the tfrecords into training, validation, and test use.

    Args:
        gcs_path: str, path to the GCS bucket.
        train_file_path: str, path to the training file.
        train_ratio: float, ratio of training data.
        val_ratio: float, ratio of validation data.

    Returns:
        train_filename: list, list of training filenames.
        val_filename: list, list of validation filenames.
        test_filename: list, list of test filenames.
    """
    train_filename, val_test_filename = train_test_split(
        tf.io.gfile.glob(gcs_path + train_file_path),
        train_size=train_ratio,
        random_state=711,
    )

    val_test_ratio = val_ratio / (1 - train_ratio)
    val_filename, test_filename = train_test_split(
        val_test_filename, train_size=val_test_ratio, random_state=711
    )

    return train_filename, val_filename, test_filename


def load_trainset(filenames, image_size, labeled, ordered=False):
    """
    Load the training dataset from TFRecord files.

    Args:
        filenames: list, list of TFRecord filenames.
        image_size: list, size of the images.
        labeled: bool, whether the dataset is labeled.
        ordered: bool, whether to load the dataset in order.

    Returns:
        dataset: tf.data.Dataset object for training.
    """
    ignore_order = tf.data.Options()
    if not ordered:
        ignore_order.experimental_deterministic = True
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
    """
    Load the validation dataset from TFRecord files.

    Args:
        filenames: list, list of TFRecord filenames.
        image_size: list, size of the images.
        labeled: bool, whether the dataset is labeled.
        ordered: bool, whether to load the dataset in order.

    Returns:
        dataset: tf.data.Dataset object for validation.
    """
    ignore_order = tf.data.Options()
    if not ordered:
        ignore_order.experimental_deterministic = True
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
    """
    Load the test dataset from TFRecord files.

    Args:
        filenames: list, list of TFRecord filenames.
        image_size: list, size of the images.
        labeled: bool, whether the dataset is labeled.
        ordered: bool, whether to load the dataset in order.

    Returns:
        dataset: tf.data.Dataset object for testing.
    """
    ignore_order = tf.data.Options()
    if not ordered:
        ignore_order.experimental_deterministic = True
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
    """
    Count the number of samples in the dataset.

    Args:
        trainset: tf.data.Dataset object.

    Returns:
        sample_count: int, number of samples in the dataset.
    """
    sample_count = sum(1 for _ in trainset)
    return sample_count


# --------preprocessing for classification task--------


def create_classification_dataset_batch(model, dataset, batch_size=64, threshold=0.4):
    """
    Create a new dataset after image segmentation.

    Args:
        model: tf.keras.Model object, trained UNet model for segmentation.
        dataset: tf.data.Dataset object, input dataset.
        batch_size: int, batch size for the dataset.
        threshold: float, threshold value for mask creation.

    Returns:
        dataset: tf.data.Dataset object, new dataset after segmentation.
    """
    images, labels = [], []

    for image, label in dataset:
        images.append(image)
        labels.append(label)

    images = tf.stack(images)
    labels = tf.convert_to_tensor(labels)

    # predict the mask
    masks = model.predict(images, batch_size=batch_size)
    masks = masks > threshold
    masks = tf.cast(masks, tf.float32)

    # segment the image
    segmented_images = images * masks

    return tf.data.Dataset.from_tensor_slices((segmented_images, labels))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def create_example(image, label, image_type=tf.float32):
    """
    Create a TFRecord example from the image and label.

    Args:
        image: tf.Tensor, input image.
        label: tf.Tensor, input label.
        image_type: tf.DType, type of the image.

    Returns:
        example: tf.train.Example, TFRecord example.
    """
    # image_raw = tf.io.serialize_tensor(tf.cast(image, image_type)).numpy()
    image_uint8 = tf.image.convert_image_dtype(image, tf.uint8)
    image_raw = tf.image.encode_jpeg(image_uint8).numpy()
    label = int(label.numpy())

    feature = {
        "image": _bytes_feature(image_raw),
        "target": _int64_feature(label),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def save_tfrecord(dataset, filename):
    """
    Save the dataset to a TFRecord file.

    Args:
        dataset: tf.data.Dataset object, input dataset.
        filename: str, path to the output TFRecord file.
    """
    with tf.io.TFRecordWriter(filename) as writer:
        for image, label in dataset:
            example = create_example(image, label)
            writer.write(example.SerializeToString())
    print(f"{filename} saved")


def count_data_items_from_tfrecord(filenames):
    """
    Count the number of samples in the TFRecord files.

    Args:
        filenames: list, list of TFRecord filenames.

    Returns:
        count: int, number of samples in the TFRecord files.
    """
    count = 0
    for f in filenames:
        for _ in tf.data.TFRecordDataset(f):
            count += 1
    return count


def preprocess_image_for_classification(
    gcs_path,
    train_file_path,
    val_file_path,
    test_file_path,
    image_size,
    batch_size,
    autotune,
):
    """
    Preprocess the dataset for cassava leaf image classification based on new dataset.

    Args:
        gcs_path: str, path to the GCS bucket.
        train_file_path: str, path to the training file.
        val_file_path: str, path to the validation file.
        test_file_path: str, path to the test file.
        image_size: list, size of the images.
        batch_size: int, batch size for the dataset.
        autotune: tf.data.experimental.AUTOTUNE object.

    Returns:
        trainset: tf.data.Dataset object for training.
        valset: tf.data.Dataset object for validation.
        testset: tf.data.Dataset object for testing.
        num_trainset: int, number of samples in the training dataset.
        num_valset: int, number of samples in the validation dataset.
    """
    trainset_path = gcs_path + train_file_path
    trainset_files = tf.io.gfile.glob(trainset_path)
    num_trainset = count_data_items_from_tfrecord(trainset_files)
    valset_path = gcs_path + val_file_path
    valset_files = tf.io.gfile.glob(valset_path)
    num_valset = count_data_items_from_tfrecord(valset_files)
    testset_path = gcs_path + test_file_path

    trainset = load_trainset(trainset_files, image_size=image_size, labeled=True)
    # trainset = trainset.repeat()
    trainset = trainset.shuffle(2048, seed=711)
    trainset = trainset.repeat()
    trainset = trainset.batch(batch_size)
    trainset = trainset.prefetch(autotune)
    valset = load_valset(valset_path, image_size=image_size, labeled=True)
    valset = valset.repeat().batch(batch_size).prefetch(autotune)
    testset = load_testset(testset_path, image_size=image_size, labeled=True)
    testset = testset.batch(batch_size).prefetch(autotune)

    return trainset, valset, testset, num_trainset, num_valset


def preprocess_image_for_compare(
    gcs_path,
    train_file_path,
    image_size,
    train_ratio,
    val_ratio,
    batch_size,
    autotune,
):
    """
    Preprocess the dataset for cassava leaf image classification based on original dataset.

    Args:
        gcs_path: str, path to the GCS bucket.
        train_file_path: str, path to the training file.
        image_size: list, size of the images.
        train_ratio: float, ratio of training data.
        val_ratio: float, ratio of validation data.
        batch_size: int, batch size for the dataset.
        autotune: tf.data.experimental.AUTOTUNE object.

    Returns:
        trainset: tf.data.Dataset object for training.
        valset: tf.data.Dataset object for validation.
        testset: tf.data.Dataset object for testing.
        num_trainset: int, number of samples in the training dataset.
        num_valset: int, number of samples in the validation dataset.
    """
    trainfile, valfile, testfile = split_data_classification(
        gcs_path, train_file_path, train_ratio, val_ratio
    )
    num_trainset = count_data_items_from_tfrecord(trainfile)
    num_valset = count_data_items_from_tfrecord(valfile)

    trainset = load_trainset(trainfile, image_size=image_size, labeled=True)
    trainset = trainset.repeat()
    trainset = trainset.shuffle(2048, seed=711)
    trainset = trainset.batch(batch_size)
    trainset = trainset.prefetch(autotune)
    valset = load_valset(valfile, image_size=image_size, labeled=True)
    valset = valset.repeat().batch(batch_size).prefetch(autotune)
    testset = load_testset(testfile, image_size=image_size, labeled=True)
    testset = testset.batch(batch_size).prefetch(autotune)

    return trainset, valset, testset, num_trainset, num_valset
