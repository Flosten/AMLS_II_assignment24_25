import os
import random

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import initializers, layers, models
from tensorflow.keras.applications import EfficientNetB0

import A.data_preprocessing as dp
import A.visualising as vs


def set_seed(seed=711):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    return tf.random.Generator.from_seed(seed)


def Unet_Arch(dropout_num, input_shape=(256, 256, 3)):
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
