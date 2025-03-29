import math
import os
import re
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

import A.data_preprocessing as dp

# print("Tensorflow version " + tf.__version__)


# set variables
AUTOTUNE = tf.data.experimental.AUTOTUNE
GCS_PATH = "./Datasets"
TRAINING_FILE_PATH = "/train_tfrecords/ld_train*.tfrec"
BATCH_SIZE = 256
IMAGE_SIZE = [256, 256]
CLASSES = ["0", "1", "2", "3", "4"]
EPOCHS = 25

# create figures folder
if not os.path.exists("figures"):
    os.makedirs("figures")

# create models folder
if not os.path.exists("models"):
    os.makedirs("models")

trainset = dp.data_acquisition(GCS_PATH, TRAINING_FILE_PATH, IMAGE_SIZE)

# # set variables
# AUTOTUNE = tf.data.experimental.AUTOTUNE
# # ======================================================================================================================
# # Data preprocessing
# data_train, data_val, data_test = data_preprocessing(args...)

# # ======================================================================================================================
# # Task A
# model_A = A(args...)                 # Build model object.
# acc_A_train = model_A.train(args...) # Train model based on the training set (you should fine-tune your model based on validation set.)
# acc_A_test = model_A.test(args...)   # Test model based on the test set.
# Clean up memory/GPU etc...             # Some code to free memory if necessary.


# # ======================================================================================================================
# ## Print out your results with following format:
# print('TA:{},{};TB:{},{};'.format(acc_A_train, acc_A_test,
#                                                         acc_B_train, acc_B_test))

# # If you are not able to finish a task, fill the corresponding variable with 'TBD'. For example:
# # acc_A_train = 'TBD'
# # acc_B_test = 'TBD'
