import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn


def plot_confusion_matrix(cm_matrix, num_classes, title, y_ticks, save_to_dir=None):
    # Creating labels for the plot

    x_ticks = [''] * len(cm_matrix)
    for i in np.arange(0, len(cm_matrix)):
        x_ticks[i] = str(i + 1)

    df_cm = pd.DataFrame(cm_matrix, range(1, num_classes + 1), range(1, num_classes + 1))
    plt.figure(figsize=(10, 6))
    sn.set(font_scale=2.8)  # for label size
    ax = sn.heatmap(df_cm,
                    annot=True,
                    xticklabels=x_ticks, yticklabels=y_ticks,
                    annot_kws={"size": 24}, fmt='d',
                    square=True,
                    vmin=0, vmax=400,
                    cbar_kws={'label': '# images'})  # font size

    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    plt.title(title, pad=30)
    plt.ylabel('Ground Truths', labelpad=30)
    plt.xlabel('Predictions', labelpad=30)
    # plt.savefig(save_to_dir.joinpath("publication_brand_cm.png"))

    plt.tight_layout()
    plt.show()
    plt.clf()
    plt.cla()
    plt.close()


if __name__ == '__main__':
    # # 18_models_hierarchical_8\fold_1\scd_majority_vote_use_contributing_patches_False.log
    # cm_nikon = np.array(
    #     [[166, 0, 0],
    #      [2, 333, 0],
    #      [0, 0, 163]]
    # )
    #
    # cm_samsung = np.array(
    #     [[216, 0],
    #      [0, 198]]
    # )
    #
    # cm_sony = np.array(
    #     [[284, 0, 0],
    #      [0, 184, 0],
    #      [20, 0, 180]]
    # )

    # 18_models_hcal_09\test_200\fold_1\scd_majority_vote_use_contributing_patches_False.log
    cm_nikon = np.array(
        [[166, 0, 0],
         [2, 333, 0],
         [0, 0, 163]]
    )

    cm_samsung = np.array(
        [[216, 0],
         [0, 198]]
    )

    cm_sony = np.array(
        [[284, 0, 0],
         [0, 184, 0],
         [20, 0, 180]]
    )

    plot_confusion_matrix(
        cm_matrix=cm_nikon,
        num_classes=3,
        title="Nikon",
        y_ticks=['Nikon_CoolPixS710 - 1', 'Nikon_D200 - 2', 'Nikon_D70 - 3']
    )

    plot_confusion_matrix(
        cm_matrix=cm_samsung,
        num_classes=2,
        title="Samsung",
        y_ticks=['Samsung_L74wide - 1', 'Samsung_NV15 - 2']
    )

    plot_confusion_matrix(
        cm_matrix=cm_sony,
        num_classes=3,
        title="Sony",
        y_ticks=['Sony_DSC-H50 - 1', 'Sony_DSC-T77 - 2', 'Sony_DSC-W170 - 3']
    )

    plot_confusion_matrix(
        cm_matrix=cm_nikon,
        num_classes=3,
        title="Nikon",
        y_ticks=['1', '2', '3']
    )
