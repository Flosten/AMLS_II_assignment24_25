import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix


def origin_image_plot(train_dataset):

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
    history_frame = pd.DataFrame(history.history)

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.plot(history_frame["loss"], label="Training Loss")
    ax.plot(history_frame["val_loss"], label="Validation Loss")
    ax.legend()
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")

    return fig


def plot_cive_result(dataset):
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


# def origin_Unet_result_plot(origin_image: list, mask_image: list, image_name: list):
#     fig, axes = plt.subplots(4, 2, figsize=(6, 10))
#     for i in range(4):
#         axes[i, 0].imshow(origin_image[i])
#         axes[i, 0].set_xlabel(image_name[i], fontsize=15)
#         axes[i, 0].set_xticks([])
#         axes[i, 0].set_yticks([])
#         axes[i, 0].spines["top"].set_visible(False)
#         axes[i, 0].spines["bottom"].set_visible(False)
#         axes[i, 0].spines["left"].set_visible(False)
#         axes[i, 0].spines["right"].set_visible(False)

#         axes[i, 1].imshow(origin_image[i] * mask_image[i])
#         axes[i, 1].set_xlabel(f"Processed {image_name[i]}", fontsize=15)
#         axes[i, 1].set_xticks([])
#         axes[i, 1].set_yticks([])
#         axes[i, 1].spines["top"].set_visible(False)
#         axes[i, 1].spines["bottom"].set_visible(False)
#         axes[i, 1].spines["left"].set_visible(False)
#         axes[i, 1].spines["right"].set_visible(False)
#         plt.tight_layout(pad=0.3)

#     return fig


def origin_Unet_result_plot(origin_image: list, mask_image: list, image_name: list):
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

    ax.tight_layout(pad=0.3)

    return fig, acc
