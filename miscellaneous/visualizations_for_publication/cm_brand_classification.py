import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn


def plot_confusion_matrix(cm_matrix, save_to_dir=None):

    # Creating labels for the plot
    x_ticks = [''] * len(cm_matrix)
    y_ticks = [''] * len(cm_matrix)
    for i in np.arange(0, len(cm_matrix)):
        x_ticks[i] = str(i + 1)
        y_ticks[i] = str(i + 1)
    num_classes = 13
    df_cm = pd.DataFrame(cm_matrix, range(1, num_classes + 1), range(1, num_classes + 1))
    plt.figure(figsize=(30, 20))
    sn.set(font_scale=3.5)  # for label size
    ax = sn.heatmap(df_cm,
                    annot=True,
                    xticklabels=x_ticks, yticklabels=y_ticks,
                    annot_kws={"size": 28}, fmt='d',
                    square=True,
                    vmin=0, vmax=cm_matrix.max(),
                    cbar_kws={'label': 'No. of images'})  # font size

    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    plt.title("Confusion Matrix", pad=30)
    plt.ylabel('Ground Truth', labelpad=30)
    plt.xlabel('Predictions', labelpad=30)
    # plt.savefig(save_to_dir.joinpath("publication_brand_cm.png"))

    plt.show()
    plt.clf()
    plt.cla()
    plt.close()


if __name__ == '__main__':

    # 400 patches
    # 18_models_hierarchical_8\fold_1\scd_majority_vote_use_contributing_patches_False.log
    # cm = np.array(
    #     [[172, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #      [0, 173, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    #      [0, 0, 210, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #      [0, 0, 0, 434, 0, 0, 0, 0, 0, 9, 0, 0, 0],
    #      [0, 0, 0, 0, 661, 0, 0, 0, 0, 3, 0, 0, 0],
    #      [0, 0, 0, 0, 0, 186, 0, 0, 0, 1, 0, 0, 0],
    #      [0, 0, 0, 0, 0, 0, 415, 0, 0, 0, 0, 0, 0],
    #      [0, 0, 0, 0, 0, 0, 0, 157, 0, 1, 0, 0, 0],
    #      [0, 0, 0, 0, 0, 0, 0, 0, 184, 1, 0, 0, 0],
    #      [0, 0, 0, 0, 0, 0, 0, 0, 0, 165, 0, 0, 0],
    #      [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 178, 0, 0],
    #      [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 412, 0],
    #      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 668]]
    # )

    # 200 patches
    # 18_models_hcal_09\test_200\fold_1\scd_majority_vote_use_contributing_patches_False.log
    cm = np.array(
        [[172, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 173, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
         [0, 0, 210, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 433, 0, 0, 0, 0, 0, 10, 0, 0, 0],
         [0, 0, 0, 0, 661, 0, 0, 0, 0, 3, 0, 0, 0],
         [0, 0, 0, 0, 0, 186, 0, 0, 0, 1, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 415, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 157, 0, 1, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 184, 1, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 165, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 178, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 412, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 668]]
    )

    plot_confusion_matrix(cm_matrix=cm)
