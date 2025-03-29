import matplotlib.pyplot as plt
import pandas as pd


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
