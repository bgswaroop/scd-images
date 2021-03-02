import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn


def plot_confusion_matrix(cm_matrix, num_classes, save_to_dir=None):
    # Creating labels for the plot
    x_ticks = [''] * len(cm_matrix)
    y_ticks = [''] * len(cm_matrix)
    for i in np.arange(0, len(cm_matrix)):
        x_ticks[i] = str(i + 1)
        y_ticks[i] = str(i + 1)
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
    plt.ylabel('Ground Truths', labelpad=30)
    plt.xlabel('Predictions', labelpad=30)
    # plt.savefig(save_to_dir.joinpath("publication_brand_cm.png"))

    plt.show()
    plt.clf()
    plt.cla()
    plt.close()


if __name__ == '__main__':
    # 18_models_hierarchical_8\fold_1\scd_majority_vote_use_contributing_patches_False.log
    cm = np.array(
        [[172, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 173, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
         [0, 0, 210, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 434, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 164, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 334, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 163, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 186, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 415, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 157, 0, 1, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 184, 1, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 165, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 178, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 214, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 198, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 284, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 184, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 18, 0, 182]]
    )

    plot_confusion_matrix(cm_matrix=cm, num_classes=18)
