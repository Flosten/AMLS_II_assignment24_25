"""
This file contains the main function to run the project.
"""

import math
import os
import re
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import optimizers
from tensorflow.keras.models import load_model

import A.data_preprocessing as dp
import A.Modelling as mdl


# Task A
def task_A():
    """
    This function is the main function for cassava leaf disease classification task.

    Returns:
        acc_a_train: The accuracy of the EfficientNet B1 model on the train set.
        acc_a_test: The accuracy of the EfficientNet B1 model on the test set.
    """
    # set parameters
    autotune = tf.data.experimental.AUTOTUNE
    gcs_path = "./Datasets"
    trainset_path = "/train_tfrecords/ld_train*.tfrec"
    # parameters for image segmentation
    batch_size = 256
    batch_size_unet = 64
    image_size = [256, 256]
    classes = ["0", "1", "2", "3", "4"]
    epochs = 18
    epochs_pro = 26
    unet_lr = 0.0005
    unet_lr_pro = 0.0002
    dropout_num_pro = 0.12
    train_ratio_cnn = 0.8
    val_ratio_cnn = 0.1
    learning_scheduler = optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.001, decay_steps=10000, decay_rate=0.9
    )  # ...
    model_dir = "./models"
    new_dataset_path = "./Datasets/New_Dataset"
    fig_path = "./figures"

    # parameters for image classification
    trainset_path_enet = "/New_Dataset/trainset_*.tfrec"
    valset_path_enet = "/New_Dataset/validation_set.tfrec"
    testset_path_enet = "/New_Dataset/test_set.tfrec"

    image_size_enet = [240, 240]
    dropout_ratio_ori = 0.7
    dropout_ratio_enet = 0.6
    layer_freezed = 50
    epochs_ori = 12
    epochs_enet = 12
    batch_size_enet = 64
    learning_scheduler_ori = optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.00008, decay_steps=500, decay_rate=0.4
    )
    learning_scheduler_enet = optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.0001, decay_steps=500, decay_rate=0.5
    )

    # # image preprocessing(image segmentation)
    # # Load the dataset
    # trainset, valset = dp.data_acquisition(gcs_path, trainset_path, image_size)
    # # build Unet dataset
    # unet_trainset, unet_valset = dp.UNet_preprocessing(trainset, batch_size_unet)

    # # model training takes a long time, so we directly load the model
    # # build Unet model
    # unet_model_origin = mdl.Unet_Arch(dropout_num=0, input_shape=[*image_size, 3])
    # # train Unet model
    # unet_model_origin_done, unet_lr_fig = mdl.Unet_train(
    #     unet_model_origin,
    #     unet_trainset,
    #     unet_valset,
    #     epochs,
    #     unet_lr,
    # )
    # # save Unet model
    # unet_model_origin_done.save(os.path.join(model_dir, "Unet_model_origin.h5"))
    # unet_model_origin_done.save(os.path.join(model_dir, "Unet_model_origin.keras"))
    # # load the unet model
    # unet_model_origin_done = load_model(os.path.join(model_dir, "Unet_model_origin.h5"))
    # fig_image_seg_origin = mdl.Unet_test(
    #     unet_model_origin_done,
    #     trainset,
    # )

    # # modify CIVE calculation
    # gen = mdl.set_seed()
    # # create cbb dataset
    # cbb_trainset, cbb_valset = dp.UNet_preprocessing_pro(
    #     trainset,
    #     batch_size_unet,
    #     gen,
    # )

    # # build Unet model
    # unet_model_pro = mdl.Unet_Arch(
    #     dropout_num=dropout_num_pro, input_shape=(*image_size, 3)
    # )
    # # train Unet model
    # Unet_model_pro_done, Unet_lr_fig_pro = mdl.Unet_train(
    #     unet_model_pro,
    #     cbb_trainset,
    #     cbb_valset,
    #     epochs=epochs_pro,
    #     learning_rate=unet_lr_pro,
    # )
    # # save Unet model
    # Unet_model_pro_done.save(os.path.join(model_dir, "Unet_model_improve.h5"))
    # Unet_model_pro_done.save(os.path.join(model_dir, "Unet_model_improve.keras"))
    # # load the unet model
    # unet_model_pro_done = load_model(os.path.join(model_dir, "Unet_model_improve.h5"))
    # fig_image_seg_pro = mdl.Unet_test(
    #     unet_model_pro_done,
    #     trainset,
    # )

    # # create and save new dataset
    # # split the origin dataset into train, validation and test set
    # trainset_cls, valset_cls, testset_cls = dp.data_acquisition_classification(
    #     gcs_path,
    #     trainset_path,
    #     image_size,
    #     train_ratio=train_ratio_cnn,
    #     val_ratio=val_ratio_cnn,
    # )
    # # deal with the train dataset
    # # set the number of parts
    # total_parts = 8
    # # get the length of the dataset
    # dataset_length = sum(1 for _ in trainset_cls)
    # batch_size = dataset_length // total_parts

    # # deal with the train dataset
    # for i in range(total_parts):
    #     start = i * batch_size
    #     end = start + batch_size if i < total_parts - 1 else dataset_length

    #     partial_dataset = trainset_cls.skip(start).take(end - start)
    #     print(f"Processing part {i + 1} of the dataset, length: {end - start}")

    #     result = dp.create_classification_dataset_batch(
    #         unet_model_pro_done, partial_dataset
    #     )
    #     dp.save_tfrecord(
    #         result, os.path.join(new_dataset_path, f"trainset_{i+1}.tfrec")
    #     )
    # # deal with the validation dataset
    # valset_use_cls = dp.create_classification_dataset_batch(
    #     unet_model_pro_done, valset_cls
    # )
    # dp.save_tfrecord(
    #     valset_use_cls, os.path.join(new_dataset_path, "validation_set.tfrec")
    # )
    # # deal with the test dataset
    # testset_use_cls = dp.create_classification_dataset_batch(
    #     unet_model_pro_done, testset_cls
    # )
    # dp.save_tfrecord(testset_use_cls, os.path.join(new_dataset_path, "test_set.tfrec"))

    # --------------create the classification model--------------
    # get the trainset, validation set and test set
    # original dataset
    trainset_ori, valset_ori, testset_ori, num_train_ori, num_val_ori = (
        dp.preprocess_image_for_compare(
            gcs_path,
            trainset_path,
            image_size_enet,
            train_ratio_cnn,
            val_ratio_cnn,
            batch_size_enet,
            autotune,
        )
    )

    # new dataset
    trainset_enet, valset_enet, testset_enet, num_train_enet, num_val_enet = (
        dp.preprocess_image_for_classification(
            gcs_path,
            trainset_path_enet,
            valset_path_enet,
            testset_path_enet,
            image_size_enet,
            batch_size_enet,
            autotune,
        )
    )

    # # build the EfficientNet_B1 model
    # eff_model_enet = mdl.EfficientNetB1_Arch(
    #     dropout_ratio=dropout_ratio_enet,
    #     layers_freezed=layer_freezed,
    #     input_shape=(*image_size_enet, 3),
    #     classes=5,
    # )

    # # train the EfficientNet B1 model
    # enet_model_done, enet_lr_fig, _ = mdl.Effi_B1_train(
    #     model=eff_model_enet,
    #     trainset=trainset_enet,
    #     valset=valset_enet,
    #     epochs=epochs_enet,
    #     learning_scheduler=learning_scheduler_enet,
    #     num_samples=num_train_enet,
    #     num_valset=num_val_enet,
    #     batch_size=batch_size_enet,
    # )

    # # save the EfficientNet B1 model
    # enet_model_done.save(os.path.join(model_dir, "EfficientNetB1_model.h5"))
    # enet_model_done.save(os.path.join(model_dir, "EfficientNetB1_model.keras"))
    # # save learning curve
    # enet_lr_fig.savefig(
    #     os.path.join(model_dir, "EfficientNetB1_model_learning_curve.png")
    # )

    # load the EfficientNet B1 model
    enet_model_done = load_model(os.path.join(model_dir, "EfficientNetB1_model.h5"))
    # test the EfficientNet B1 model
    fig_enet, acc_enet = mdl.Effi_B1_test(enet_model_done, testset_enet)
    # fig_enet.savefig(os.path.join(fig_path, "EfficientNetB1_model_test_result.png"))
    # output the results
    trainset_data_path = gcs_path + trainset_path_enet
    trainset_file = tf.io.gfile.glob(trainset_data_path)
    trainset_output = dp.load_trainset(
        filenames=trainset_file, image_size=image_size_enet, labeled=True
    )
    trainset_output = trainset_output.batch(batch_size_enet).prefetch(autotune)
    _, acc_a_train = mdl.Effi_B1_test(enet_model_done, trainset_output)
    acc_a_test = acc_enet

    # # compare to the original dataset
    # # build the EfficientNet B1 model
    # eff_model_ori = mdl.EfficientNetB1_Arch(
    #     dropout_ratio=dropout_ratio_ori,
    #     layers_freezed=layer_freezed,
    #     input_shape=(*image_size_enet, 3),
    #     classes=5,
    # )

    # # train the EfficientNet B1 model
    # eff_model_ori_done, ori_lr_fig, _ = mdl.Effi_B1_train(
    #     model=eff_model_ori,
    #     trainset=trainset_ori,
    #     valset=valset_ori,
    #     epochs=epochs_ori,
    #     learning_scheduler=learning_scheduler_ori,
    #     num_samples=num_train_ori,
    #     num_valset=num_val_ori,
    #     batch_size=batch_size_enet,
    # )

    # # save the EfficientNet B1 model
    # eff_model_ori_done.save(os.path.join(model_dir, "EfficientNetB1_model_ori.h5"))
    # eff_model_ori_done.save(os.path.join(model_dir, "EfficientNetB1_model_ori.keras"))

    # load the EfficientNet B1 model
    eff_model_ori_done = load_model(
        os.path.join(model_dir, "EfficientNetB1_model_ori.h5")
    )
    # test the EfficientNet B1 model
    _, acc_ori_test = mdl.Effi_B1_test(eff_model_ori_done, testset_ori)
    # # get the trainset
    # trainset_data_path_ori = gcs_path + trainset_path
    # trainset_file_ori = tf.io.gfile.glob(trainset_data_path_ori)
    # trainset_output_ori = dp.load_trainset(
    #     filenames=trainset_file_ori, image_size=image_size_enet, labeled=True
    # )
    # trainset_output_ori = trainset_output_ori.batch(batch_size_enet).prefetch(autotune)
    # _, acc_ori_train = mdl.Effi_B1_test(eff_model_ori_done, trainset_output_ori)

    # baseline model
    baseline_model = load_model(
        os.path.join(model_dir, "baseline_model.h5"),
        custom_objects={
            "preprocess_input": tf.keras.applications.resnet50.preprocess_input
        },
    )
    # test the baseline model
    _, acc_baseline_test = mdl.Effi_B1_test(baseline_model, testset_ori)

    # print the results
    print(
        "The accuracy of the EfficientNet B1 model without image segmentation is: ",
        acc_ori_test,
    )
    print("The accuracy of the baseline model is: ", acc_baseline_test)

    return acc_a_train, acc_a_test


def task_B():
    """
    This function is empty because this project involves only one task

    Returns:
        acc_b_train: TBD
        acc_b_test: TBD
    """
    acc_b_train = "TBD"
    acc_b_test = "TBD"
    return acc_b_train, acc_b_test


if __name__ == "__main__":
    # set seed for reproducibility
    mdl.set_seed(711)

    # create folders
    # create figures folder
    if not os.path.exists("figures"):
        os.makedirs("figures")

    # create models folder
    if not os.path.exists("models"):
        os.makedirs("models")

    # create datasets folder
    if not os.path.exists("Datasets/New_Dataset"):
        os.makedirs("Datasets/New_Dataset")

    acc_A_train, acc_A_test = task_A()
    acc_B_train, acc_B_test = task_B()

    # print the results
    print("TA:{},{};TB:{},{};".format(acc_A_train, acc_A_test, acc_B_train, acc_B_test))
