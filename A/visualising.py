"""
This file contains functions to visualize the results of the whole process
of cassava disease classification task.

The functions include:
- `origin_image_plot`: Plot the original images from the dataset.
- `learning_curve`: Plot the learning curve of the model.
- `plot_cive_result`: Plot the original image, mask image, and processed image.
- `origin_Unet_result_plot`: Plot the original image and processed image including all the diseases.
- `classfication_result`: Plot confusion matrix and calculate accuracy of the EfficientNetB0 model.
"""

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix


def origin_image_plot(train_dataset):
    """
    Plot the original images from the dataset.

    Args:
        train_dataset: The dataset containing the images and labels.

    Returns:
        fig1: The figure containing different disease images.
        fig2: The figure containing the healthy image.
    """
    fig1, axes = plt.subplots(2, 2, figsize=(10, 10))
    fig2, ax2 = plt.subplots(1, 1, figsize=(10, 10))

    labels = [0, 1, 2, 3, 4]
    labels_name = {
        "0": "Cassava Bacterial Blight (CBB)",
        "1": "Cassava Brown Streak Disease (CBSD)",
        "2": "Cassava Green Mottle (CGM)",
        "3": "Cassava Mosaic Disease (CMD)",
        "4": "Healthy",
    }
    found_images = {}

    for image, label in train_dataset:
        label = label.numpy()
        if label in labels and label not in found_images:
            found_images[label] = image
            labels.remove(label)
        if len(labels) == 0:
            break

    # plot the healthy image
    ax2.imshow(found_images[4])
    ax2.set_xlabel(labels_name["4"], fontsize=25)
    # ax2.set_title("")
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.spines["top"].set_visible(False)
    ax2.spines["bottom"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    found_images.pop(4)

    # plot the rest of the images
    for i, (label, image) in enumerate(found_images.items()):
        row = i // 2
        col = i % 2
        axes[row, col].imshow(image)
        axes[row, col].set_xlabel(labels_name[str(label)], fontsize=15)
        # axes[row, col].set_title("")
        axes[row, col].set_xticks([])
        axes[row, col].set_yticks([])
        axes[row, col].spines["top"].set_visible(False)
        axes[row, col].spines["bottom"].set_visible(False)
        axes[row, col].spines["left"].set_visible(False)
        axes[row, col].spines["right"].set_visible(False)

    return fig1, fig2


def learning_curve(history):
    """
    Plot the learning curve of the model.

    Args:
        history: The history object returned by the model's fit method.

    Returns:
        fig: The figure containing the learning curve.
    """
    history_frame = pd.DataFrame(history.history)

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    epochs = range(1, len(history_frame) + 1)

    ax.plot(epochs, history_frame["loss"], label="Training Loss")
    ax.plot(epochs, history_frame["val_loss"], label="Validation Loss")
    ax.legend()
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.set_xticks(range(1, len(history_frame) + 1, 2))

    return fig


def plot_cive_result(dataset):
    """
    Plot the original image, mask image, and processed image.

    Args:
        dataset: The dataset containing the images and masks.

    Returns:
        fig: The figure containing the original image, mask image, and processed image.
    """
    fig, axes = plt.subplots(1, 3, figsize=(12, 8))

    for image, mask in dataset.skip(14).take(1):
        # for image, mask in dataset.take(3):
        axes[0].imshow(image)
        axes[0].set_xlabel("Original Image", fontsize=15)
        axes[0].set_xticks([])
        axes[0].set_yticks([])
        axes[0].spines["top"].set_visible(False)
        axes[0].spines["bottom"].set_visible(False)
        axes[0].spines["left"].set_visible(False)
        axes[0].spines["right"].set_visible(False)

        axes[1].imshow(mask)
        axes[1].set_xlabel("Mask Image", fontsize=15)
        axes[1].set_xticks([])
        axes[1].set_yticks([])
        axes[1].spines["top"].set_visible(False)
        axes[1].spines["bottom"].set_visible(False)
        axes[1].spines["left"].set_visible(False)
        axes[1].spines["right"].set_visible(False)

        image_pro = image * mask
        axes[2].imshow(image_pro)
        axes[2].set_xlabel("Processed Image", fontsize=15)
        axes[2].set_xticks([])
        axes[2].set_yticks([])
        axes[2].spines["top"].set_visible(False)
        axes[2].spines["bottom"].set_visible(False)
        axes[2].spines["left"].set_visible(False)
        axes[2].spines["right"].set_visible(False)

    return fig


def origin_Unet_result_plot(origin_image: list, mask_image: list, image_name: list):
    """
    Plot the original image and the processed image including all the diseases.

    Args:
        origin_image: The original images.
        mask_image: The mask images.
        image_name: The names of the images.

    Returns:
        fig: The figure containing the original image and the processed image.
    """
    fig, axes = plt.subplots(2, 4, figsize=(10, 6))
    for i in range(4):
        axes[0, i].imshow(origin_image[i])
        axes[0, i].set_xlabel(image_name[i], fontsize=15)
        axes[0, i].set_xticks([])
        axes[0, i].set_yticks([])
        axes[0, i].spines["top"].set_visible(False)
        axes[0, i].spines["bottom"].set_visible(False)
        axes[0, i].spines["left"].set_visible(False)
        axes[0, i].spines["right"].set_visible(False)

        axes[1, i].imshow(origin_image[i] * mask_image[i])
        axes[1, i].set_xlabel(f"Processed {image_name[i]}", fontsize=15)
        axes[1, i].set_xticks([])
        axes[1, i].set_yticks([])
        axes[1, i].spines["top"].set_visible(False)
        axes[1, i].spines["bottom"].set_visible(False)
        axes[1, i].spines["left"].set_visible(False)
        axes[1, i].spines["right"].set_visible(False)
        plt.tight_layout(pad=0.3)

    return fig


def classfication_result(y_true, y_pred):
    """
    Plot the confusion matrix and calculate the accuracy of the EfficientNetB1 model.

    Args:
        y_true: The true labels.
        y_pred: The predicted labels.

    Returns:
        fig: The confusion matrix figure.
        acc: The accuracy of the model.
    """
    # calculate the accuracy
    acc = accuracy_score(y_true, y_pred)

    # calculate the confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # plot the confusion matrix
    fig, ax = plt.subplots(figsize=(10, 10))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax, cmap="Blues", values_format=".4g")
    ax.set_xlabel("Predicted Label", fontsize=15)
    ax.set_ylabel("True Label", fontsize=15)

    # set the x and y ticks
    xticks = ["CBB", "CBSD", "CGM", "CMD", "Healthy"]
    yticks = ["CBB", "CBSD", "CGM", "CMD", "Healthy"]
    ax.set_xticks(range(len(xticks)))
    ax.set_xticklabels(xticks, fontsize=12)
    ax.set_yticks(range(len(yticks)))
    ax.set_yticklabels(yticks, fontsize=12)

    plt.tight_layout(pad=0.3)

    return fig, acc
