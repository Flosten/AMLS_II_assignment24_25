"""
This file contains functions to build and train the U-Net model for image segmentation
and the EfficientNetB0 model for cassava disease classification.

The functions include:
- `set_seed`: Set the seed for reproducibility.
- `Unet_Arch`: Define the U-Net architecture.
- `Unet_train`: Train the U-Net model.
- `Unet_test`: Test the U-Net model.
- `EfficientNetB0_Arch`: Build a EfficientNetB0 model based on transfer learning.
- `Effi_B0_train`: Train the EfficientNetB0 model.
- `Effi_B0_test`: Test the EfficientNetB0 model.
"""

import os
import random

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import initializers, layers, models
from tensorflow.keras.applications import EfficientNetB1

import A.data_preprocessing as dp
import A.visualising as vs


def set_seed(seed=711):
    """
    Set the seed for reproducibility.

    Args:
        seed (int): The seed value to set.

    Returns:
        tf.random.Generator: A TensorFlow random generator with the specified seed.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"

    return tf.random.Generator.from_seed(seed)


def Unet_Arch(dropout_num, input_shape=(256, 256, 3)):
    """
    Define the U-Net architecture.

    Args:
        dropout_num (float): Dropout rate.
        input_shape (tuple): Shape of the input image.

    Returns:
        tf.keras.Model: U-Net model.
    """
    inputs = keras.Input(shape=input_shape)
    conv_init = initializers.HeNormal(seed=711)
    bias_init = initializers.Zeros()

    # contracting path
    c1 = layers.Conv2D(
        16,
        (3, 3),
        activation="relu",
        kernel_initializer=conv_init,
        bias_initializer=bias_init,
        padding="same",
    )(
        inputs
    )  # 256 x 256 x 16
    c2 = layers.Conv2D(
        16,
        (3, 3),
        activation="relu",
        kernel_initializer=conv_init,
        bias_initializer=bias_init,
        padding="same",
    )(
        c1
    )  # 256 x 256 x 16
    p1 = layers.MaxPooling2D((2, 2))(c2)  # 128 x 128 x 16
    p1 = layers.Dropout(dropout_num)(p1)

    c3 = layers.Conv2D(
        32,
        (3, 3),
        activation="relu",
        kernel_initializer=conv_init,
        bias_initializer=bias_init,
        padding="same",
    )(
        p1
    )  # 128 x 128 x 32
    c4 = layers.Conv2D(
        32,
        (3, 3),
        activation="relu",
        kernel_initializer=conv_init,
        bias_initializer=bias_init,
        padding="same",
    )(
        c3
    )  # 128 x 128 x 32
    p2 = layers.MaxPooling2D((2, 2))(c4)  # 64 x 64 x 32
    p2 = layers.Dropout(dropout_num)(p2)

    # bottleneck
    c5 = layers.Conv2D(
        64,
        (3, 3),
        activation="relu",
        kernel_initializer=conv_init,
        bias_initializer=bias_init,
        padding="same",
    )(
        p2
    )  # 64 x 64 x 64
    c6 = layers.Conv2D(
        64,
        (3, 3),
        activation="relu",
        kernel_initializer=conv_init,
        bias_initializer=bias_init,
        padding="same",
    )(
        c5
    )  # 64 x 64 x 64

    # expansive path
    u1 = layers.UpSampling2D((2, 2))(c6)  # 128 x 128 x 64
    u1 = layers.Concatenate()([u1, c4])  # 128 x 128 x 96
    u1 = layers.Dropout(dropout_num)(u1)
    c7 = layers.Conv2D(
        32,
        (3, 3),
        activation="relu",
        kernel_initializer=conv_init,
        bias_initializer=bias_init,
        padding="same",
    )(
        u1
    )  # 128 x 128 x 32
    c8 = layers.Conv2D(
        32,
        (3, 3),
        activation="relu",
        kernel_initializer=conv_init,
        bias_initializer=bias_init,
        padding="same",
    )(
        c7
    )  # 128 x 128 x 32

    u2 = layers.UpSampling2D((2, 2))(c8)  # 256 x 256 x 32
    u2 = layers.Concatenate()([u2, c2])  # 256 x 256 x 48
    u2 = layers.Dropout(dropout_num)(u2)
    c9 = layers.Conv2D(
        16,
        (3, 3),
        activation="relu",
        kernel_initializer=conv_init,
        bias_initializer=bias_init,
        padding="same",
    )(
        u2
    )  # 256 x 256 x 16
    c10 = layers.Conv2D(
        16,
        (3, 3),
        activation="relu",
        kernel_initializer=conv_init,
        bias_initializer=bias_init,
        padding="same",
    )(
        c9
    )  # 256 x 256 x 16

    outputs = layers.Conv2D(1, (1, 1), activation="sigmoid")(c10)  # 256 x 256 x 1

    model = models.Model(inputs=[inputs], outputs=[outputs])
    return model


def Unet_train(model, train, val, epochs=10, learning_rate=0.001):
    """
    Build the U-Net model.

    Args:
        model (tf.keras.Model): U-Net model.
        train (tf.data.Dataset): Training dataset.
        val (tf.data.Dataset): Validation dataset.
        epochs (int): Number of epochs to train.
        learning_rate (float): Learning rate.

    Returns:
        tf.keras.Model: Trained U-Net model.
        matplotlib.figure.Figure: Learning curve figure.
    """
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.BinaryAccuracy()],
    )
    history = model.fit(
        train,
        validation_data=val,
        epochs=epochs,
    )

    fig = vs.learning_curve(history)

    return model, fig


def Unet_test(model, dataset):
    """
    Test the U-Net model.

    Args:
        model (tf.keras.Model): U-Net model.
        dataset (tf.data.Dataset): Dataset to test.

    Returns:
        matplotlib.figure.Figure: Figure showing the original image, mask, and processed image.
    """
    cbb_dataset, cbsd_dataset, cgm_dataset, cmd_dataset = dp.get_all_diseases_image(
        dataset
    )
    # get the test sample
    for image, _ in cbb_dataset.skip(56).take(1):  # skip(56).take(1)  11 71 116
        cbb_image_test = image
        cbb_image_test_batch = tf.expand_dims(cbb_image_test, axis=0)

    for image, _ in cbsd_dataset.skip(55).take(1):
        cbsd_image_test = image
        cbsd_image_test_batch = tf.expand_dims(cbsd_image_test, axis=0)

    for image, _ in cgm_dataset.skip(45).take(1):  # 45
        cgm_image_test = image
        cgm_image_test_batch = tf.expand_dims(cgm_image_test, axis=0)

    for image, _ in cmd_dataset.take(1):
        cmd_image_test = image
        cmd_image_test_batch = tf.expand_dims(cmd_image_test, axis=0)

    # predict the test sample
    cbb_mask_test = model.predict(cbb_image_test_batch)
    cbb_mask_test = tf.squeeze(cbb_mask_test, axis=0)
    cbb_mask_test = (cbb_mask_test > 0.4).numpy().astype("uint8")
    cbsd_mask_test = model.predict(cbsd_image_test_batch)
    cbsd_mask_test = tf.squeeze(cbsd_mask_test, axis=0)
    cbsd_mask_test = (cbsd_mask_test > 0.4).numpy().astype("uint8")
    cgm_mask_test = model.predict(cgm_image_test_batch)
    cgm_mask_test = tf.squeeze(cgm_mask_test, axis=0)
    cgm_mask_test = (cgm_mask_test > 0.4).numpy().astype("uint8")
    cmd_mask_test = model.predict(cmd_image_test_batch)
    cmd_mask_test = tf.squeeze(cmd_mask_test, axis=0)
    cmd_mask_test = (cmd_mask_test > 0.4).numpy().astype("uint8")

    # plot the test sample
    fig = vs.origin_Unet_result_plot(
        [cbb_image_test, cbsd_image_test, cgm_image_test, cmd_image_test],
        [cbb_mask_test, cbsd_mask_test, cgm_mask_test, cmd_mask_test],
        ["CBB", "CBSD", "CGM", "CMD"],
    )

    return fig


def EfficientNetB1_Arch(
    dropout_ratio, layers_freezed, input_shape=(224, 224, 3), classes=5
):
    """
    Build a EfficientNetB1 model based on transfer learning.

    Args:
        dropout_ratio (float): Dropout rate.
        layers_freezed (int): Number of layers to freeze.
        input_shape (tuple): Shape of the input image.
        classes (int): Number of classes.

    Returns:
        tf.keras.Model: EfficientNetB1 model.
    """
    base_model = EfficientNetB1(
        include_top=False, weights="imagenet", input_shape=input_shape
    )
    base_model.trainable = True

    # Freeze some layers
    # cheak the number of layers in the base model
    print(len(base_model.layers))
    # Freeze the first 50 layers
    for layer in base_model.layers[:layers_freezed]:
        layer.trainable = False

    model = models.Sequential(
        [
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(dropout_ratio),
            layers.Dense(classes, activation="softmax"),
        ]
    )

    return model


def Effi_B1_train(
    model,
    trainset,
    valset,
    epochs,
    learning_scheduler,
    num_samples,
    num_valset,
    batch_size,
):
    """
    Train the EfficientNetB1 model.

    Args:
        model (tf.keras.Model): EfficientNetB1 model.
        trainset (tf.data.Dataset): Training dataset.
        valset (tf.data.Dataset): Validation dataset.
        epochs (int): Number of epochs to train.
        learning_scheduler: Learning rate schedule.
        num_samples (int): Number of samples in the training set.
        num_valset (int): Number of samples in the validation set.
        batch_size (int): Batch size.

    Returns:
        tf.keras.Model: Trained EfficientNetB1 model.
        matplotlib.figure.Figure: Learning curve figure.
        tf.keras.callbacks.History: Training history.
    """
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=learning_scheduler, epsilon=0.001
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"],
    )

    history = model.fit(
        trainset,
        validation_data=valset,
        epochs=epochs,
        steps_per_epoch=num_samples // batch_size,
        validation_steps=num_valset // batch_size,
    )

    fig = vs.learning_curve(history)

    return model, fig, history


def Effi_B1_test(model, testset):
    """
    Test the EfficientNetB1 model.

    Args:
        model (tf.keras.Model): EfficientNetB1 model.
        testset (tf.data.Dataset): Test dataset.

    Returns:
        matplotlib.figure.Figure: Figure of confusion matrix.
        float: Classification accuracy.
    """
    y_true = []
    y_pred = []

    # get the image and label from the testset
    for images, labels in testset:
        preds = model.predict(images)
        preds = np.argmax(preds, axis=1)
        y_true.extend(labels.numpy())
        y_pred.extend(preds)

    # convert to numpy array
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # get the classification result
    fig, acc = vs.classfication_result(y_true, y_pred)

    return fig, acc
