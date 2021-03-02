from collections import namedtuple
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

Scores = namedtuple('Scores', 'img_acc, img_f1, patch_acc', defaults=(None, None, None))
Results = namedtuple('Results', 'brands, nikon, samsung, sony, hierarchical', defaults=(None, None, None, None, None))


def extract_scores_from_log_file(filepath):
    """
    Extract the last set of scores from the provided file. The scores will be set to None if they are missing.
    :param filepath: a Path object
    :return: tuple of scores
    """

    result_brands = Scores()
    result_nikon = Scores()
    result_samsung = Scores()
    result_sony = Scores()
    result_hierarchy = Scores()

    context = 'hierarchy'
    with open(filepath, 'r') as f:
        for line in f:
            # find the context ('brands, nikon, samsung, sony, hierarchical')
            # if 'utils.training_utils - INFO - Best trained model is found at' in line:
            #     context = line.split('/')[-2].split('_')[-1].lower()
            if '0            Can' in line:
                context = 'brands'
            elif '0      Nikon_CoolPixS7' in line:
                context = 'nikon'
            elif '0      Samsung_L74wi' in line:
                context = 'samsung'
            elif '0       Sony_DSC-H' in line:
                context = 'sony'
            elif '17           Sony_DSC-W170' in line:
                context = 'hierarchy'

            # find the three scores ('img_acc, img_f1, patch_acc')
            if 'signature_net.sig_net_flow - INFO - Test accuracy:' in line:
                patch_acc = float(line.split(' ')[-1])
            elif 'utils.evaluation_metrics - INFO - Accuracy' in line:
                img_acc = float(line.split(' ')[-1])
            elif 'utils.evaluation_metrics - INFO - Macro f1-score' in line:
                img_f1 = float(line.split(' ')[-1])

                # update the context variable
                if context == 'brands':
                    result_brands = Scores(img_acc, img_f1, patch_acc)
                elif context == 'nikon':
                    result_nikon = Scores(img_acc, img_f1, patch_acc)
                elif context == 'samsung':
                    result_samsung = Scores(img_acc, img_f1, patch_acc)
                elif context == 'sony':
                    result_sony = Scores(img_acc, img_f1, patch_acc)
                elif context == 'hierarchy':
                    result_hierarchy = Scores(img_acc, img_f1)

                context = 'hierarchy'

    return Results(result_brands, result_nikon, result_samsung, result_sony, result_hierarchy)


def plot_accuracy_vs_num_patches(results_dict, title):
    """
    :param results_dict: the dictionary containing the results from all the folds
    :param title: title of the plot
    """

    labels = ['1', '5', '10', '20', '40', '100', '200', '400']
    hcal_acc_avg = []
    brands_acc_avg = []
    nikon_acc_avg = []
    samsung_acc_avg = []
    sony_acc_avg = []

    for patch_count in results_dict:
        hcal_acc = []
        brands_acc = []
        nikon_acc = []
        samsung_acc = []
        sony_acc = []

        for fold in results_dict[patch_count]:
            hcal_acc += [results_dict[patch_count][fold].hierarchical.img_acc]
            brands_acc += [results_dict[patch_count][fold].brands.img_acc]
            nikon_acc += [results_dict[patch_count][fold].nikon.img_acc]
            samsung_acc += [results_dict[patch_count][fold].samsung.img_acc]
            sony_acc += [results_dict[patch_count][fold].sony.img_acc]

        hcal_acc_avg += [np.mean(hcal_acc)]
        brands_acc_avg += [np.mean(brands_acc)]
        nikon_acc_avg += [np.mean(nikon_acc)]
        samsung_acc_avg += [np.mean(samsung_acc)]
        sony_acc_avg += [np.mean(sony_acc)]

    plt.figure(figsize=(6, 4), dpi=300)
    plt.plot(labels, hcal_acc_avg, marker='.', alpha=0.80, label='Overall (Hierarchical)')
    plt.plot(labels, brands_acc_avg, marker='.', alpha=0.80, label='Brands')
    plt.plot(labels, nikon_acc_avg, marker='.', alpha=0.80, label='Nikon')
    plt.plot(labels, samsung_acc_avg, marker='.', alpha=0.80, label='Samsung')
    plt.plot(labels, sony_acc_avg, marker='.', alpha=0.80, label='Sony')

    plt.legend()
    plt.title(title)
    plt.xlabel('Number of Patches')
    plt.ylabel('Average Accuracy')
    plt.grid(axis='y')
    plt.show()
    plt.clf()
    plt.cla()
    plt.close()


def plot_f1_vs_num_patches(results_dict):
    """
    :param results_dict:
    :return:
    """

    labels = ['1', '5', '10', '20', '40', '100', '200']
    hcal_f1_avg = []
    brands_f1_avg = []
    nikon_f1_avg = []
    samsung_f1_avg = []
    sony_f1_avg = []

    for patch_count in results_dict:
        hcal_f1 = []
        brands_f1 = []
        nikon_f1 = []
        samsung_f1 = []
        sony_f1 = []

        for fold in results_dict[patch_count]:
            hcal_f1 += [results_dict[patch_count][fold].hierarchical.img_f1]
            brands_f1 += [results_dict[patch_count][fold].brands.img_f1]
            nikon_f1 += [results_dict[patch_count][fold].nikon.img_f1]
            samsung_f1 += [results_dict[patch_count][fold].samsung.img_f1]
            sony_f1 += [results_dict[patch_count][fold].sony.img_f1]

        hcal_f1_avg += [np.mean(hcal_f1)]
        brands_f1_avg += [np.mean(brands_f1)]
        nikon_f1_avg += [np.mean(nikon_f1)]
        samsung_f1_avg += [np.mean(samsung_f1)]
        sony_f1_avg += [np.mean(sony_f1)]

    plt.figure(figsize=(6, 4), dpi=300)
    plt.plot(labels, hcal_f1_avg, marker='.', label='Overall (Hierarchical)')
    plt.plot(labels, brands_f1_avg, marker='.', label='Brands')
    plt.plot(labels, nikon_f1_avg, marker='.', label='Nikon')
    plt.plot(labels, samsung_f1_avg, marker='.', label='Samsung')
    plt.plot(labels, sony_f1_avg, marker='.', label='Sony')

    plt.legend()
    plt.title('Image-level predictions on Homogeneous Patches')
    plt.xlabel('Number of Patches')
    plt.ylabel('Average macro F1-score')
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()
    plt.clf()
    plt.cla()
    plt.close()


if __name__ == '__main__':
    # # Homogeneous patches
    # base_directory = Path(r'/scratch/p288722/runtime_data/scd_pytorch/18_models_hcal_09/')
    # filename = 'scd_majority_vote_use_contributing_patches_False.log'
    #
    # results = {}
    # for num_patches in [1, 5, 10, 20, 40, 100, 200, 400]:
    #     if num_patches not in results:
    #         results[f'num_patches_{num_patches}'] = {}
    #     for fold_id in [1, 2, 3, 4, 5]:
    #         result_directory = base_directory.joinpath(rf'test_{num_patches}/fold_{fold_id}/')
    #         results[f'num_patches_{num_patches}'][f'fold_{fold_id}'] = \
    #             extract_scores_from_log_file(filepath=result_directory.joinpath(filename))
    #
    # plot_accuracy_vs_num_patches(results, title='Image-level predictions on Homogeneous Patches')

    # Random patches
    base_directory = Path(r'/scratch/p288722/runtime_data/scd_pytorch/18_models_hcal_random_2/')
    filename = 'scd_majority_vote_use_contributing_patches_False.log'

    results = {}
    for num_patches in [1, 5, 10, 20, 40, 100, 400, 200]:
        if num_patches not in results:
            results[f'num_patches_{num_patches}'] = {}
        for fold_id in [1, 2, 3, 4, 5]:
            result_directory = base_directory.joinpath(rf'test_{num_patches}/fold_{fold_id}/')
            results[f'num_patches_{num_patches}'][f'fold_{fold_id}'] = \
                extract_scores_from_log_file(filepath=result_directory.joinpath(filename))

    plot_accuracy_vs_num_patches(results, title='Image-level predictions on Random Patches')

    # Flat Classifier
    base_directory = Path(r'/scratch/p288722/runtime_data/scd_pytorch/18_models_hcal_random_2/')
    filename = 'scd_majority_vote_use_contributing_patches_False.log'

    results = {}
    for num_patches in [1, 5, 10, 20, 40, 100, 400, 200]:
        if num_patches not in results:
            results[f'num_patches_{num_patches}'] = {}
        for fold_id in [1]:
            result_directory = base_directory.joinpath(rf'test_{num_patches}/fold_{fold_id}/')
            results[f'num_patches_{num_patches}'][f'fold_{fold_id}'] = \
                extract_scores_from_log_file(filepath=result_directory.joinpath(filename))
