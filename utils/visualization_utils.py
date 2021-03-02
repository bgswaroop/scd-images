from collections import namedtuple
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import sklearn.metrics

from utils.training_utils import Utils

AE_Sample = namedtuple('AE_Sample', ['loss', 'input', 'output', 'label'])


class VisualizationUtils:
    def __init__(self):
        pass

    @staticmethod
    def plot_class_wise_data_distribution(dataset_dir, save_to_dir):

        # Prepare the data
        data_root = Path(dataset_dir)
        class_list = [f for f in data_root.glob("*")]
        class_wise_distribution = np.zeros(len(class_list))
        class_names = [''] * len(class_list)
        for idx, item in enumerate(class_list):
            class_wise_distribution[idx] = len([f for f in item.glob("*")])
            class_names[idx] = item.stem

        # Sort the data for better visualization
        class_names = np.array(class_names)
        sorted_indices = np.argsort(class_wise_distribution)

        # Generate the plot
        # plt.figure(figsize=(24, 24))
        plt.figure(figsize=(15, 8))
        plt.style.use('fivethirtyeight')
        plt.barh(class_names[sorted_indices], class_wise_distribution[sorted_indices])
        plt.title('Class wise distribution')
        plt.rc('ytick', labelsize=1)
        plt.xlabel('No. of training images')
        plt.tight_layout()
        plt.savefig(str(Path(save_to_dir).joinpath("data_distribution.png")))
        plt.clf()
        plt.cla()
        plt.close()

    @staticmethod
    def plot_training_history(save_to_dir, history):

        mpl.rcParams.update(mpl.rcParamsDefault)

        # Set up the paths for saving the plots
        acc_filename = str(Path(save_to_dir).joinpath("training_accuracy.png"))
        loss_filename = str(Path(save_to_dir).joinpath("training_loss.png"))

        # Initialize the plot data
        acc = history['accuracy']
        val_acc = history['val_accuracy']
        loss = history['loss']
        val_loss = history['val_loss']
        epochs = history['epochs']
        selected_epoch = Utils.choose_best_epoch_from_history(history) + 1

        plt.figure()
        plt.plot(epochs, acc, 'r', label='Train')
        plt.plot(epochs, val_acc, 'b', label='Test')
        if 'val_acc2' in history:
            plt.plot(epochs, history['val_acc2'], 'g', label='Test known models unknown devices')
        if 'val_acc3' in history:
            plt.plot(epochs, history['val_acc3'], 'c', label='Test unknown models unknown devices')
        plt.axvline(x=selected_epoch, label='Selected model', linestyle='--', alpha=0.40, c='r')

        plt.title('Accuracy plot')
        plt.xlabel('epochs')
        plt.ylabel('accuracy')
        plt.legend()
        plt.savefig(acc_filename)
        plt.clf()
        plt.cla()
        plt.close()

        plt.figure()
        plt.plot(epochs, loss, 'r', label='Train')
        plt.plot(epochs, val_loss, 'b', label='Test')
        if 'val_loss2' in history:
            plt.plot(epochs, history['val_loss2'], 'g', label='Test known models unknown devices')
        if 'val_loss3' in history:
            plt.plot(epochs, history['val_loss3'], 'c', label='Test unknown models unknown devices')
        plt.axvline(x=selected_epoch, label='Selected model', linestyle='--', alpha=0.40, c='r')

        plt.title('Loss plot')
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.legend()
        plt.savefig(loss_filename)
        plt.clf()
        plt.cla()
        plt.close()

    @classmethod
    def plot_learning_rate(cls, save_to_dir, history):
        mpl.rcParams.update(mpl.rcParamsDefault)
        filename = str(Path(save_to_dir).joinpath("optimizer_leaning_rate.png"))

        # Initialize the plot data
        lr = history['learning_rate']
        epochs = history['epochs']

        plt.figure()
        plt.plot(epochs, lr, 'r', label='Leaning Rate')
        plt.title('Learning Rate Decay')
        plt.xlabel('epochs')
        plt.ylabel('learning rate')
        plt.legend()
        plt.savefig(filename)
        plt.clf()
        plt.cla()
        plt.close()

    @staticmethod
    def plot_learning_statistics(history, save_to_dir):
        VisualizationUtils.plot_training_history(save_to_dir=save_to_dir, history=history)
        VisualizationUtils.plot_learning_rate(save_to_dir=save_to_dir, history=history)

    @classmethod
    def plot_confusion_matrix(cls, ground_truths, predictions, one_hot, save_to_dir):
        if one_hot:
            ground_truths = [np.argmax(x) for x in ground_truths]
            predictions = [np.argmax(x) for x in predictions]

        cm_matrix = sklearn.metrics.confusion_matrix(ground_truths, predictions)

        # Creating labels for the plot
        x_ticks = [''] * len(cm_matrix)
        y_ticks = [''] * len(cm_matrix)
        for i in np.arange(0, len(cm_matrix), 2):
            x_ticks[i] = str(i + 1)
            y_ticks[i] = str(i + 1)
        num_classes = max(max(ground_truths), max(predictions)) + 1
        df_cm = pd.DataFrame(cm_matrix, range(1, num_classes + 1), range(1, num_classes + 1))
        plt.figure(figsize=(30, 20))
        sn.set(font_scale=2.5)  # for label size
        ax = sn.heatmap(df_cm,
                        annot=True,
                        xticklabels=x_ticks, yticklabels=y_ticks,
                        annot_kws={"size": 16}, fmt='d',
                        square=True,
                        vmin=0, vmax=cm_matrix.max(),
                        cbar_kws={'label': 'No. of images'})  # font size

        # This is to fix an issue with matplotlib==3.1.1
        # bottom, top = ax.get_ylim()
        # ax.set_ylim(bottom + 0.5, top - 0.5)
        plt.xticks(rotation=0)
        plt.yticks(rotation=0)
        plt.title("Confusion Matrix - Signature Net", pad=30)
        plt.ylabel('Ground Truth', labelpad=30)
        plt.xlabel('Predictions', labelpad=30)
        plt.savefig(save_to_dir.joinpath("sig_net_test_cm.png"))
        # plt.show()
        plt.clf()
        plt.cla()
        plt.close()

        norm_cm = cm_matrix / (cm_matrix.sum(1).repeat(len(cm_matrix)).reshape(cm_matrix.shape))
        df_cm = pd.DataFrame(norm_cm, range(1, num_classes + 1), range(1, num_classes + 1))
        plt.figure(figsize=(30, 20))
        sn.set(font_scale=2.5)  # for label size
        ax = sn.heatmap(df_cm,
                        xticklabels=x_ticks, yticklabels=y_ticks,
                        square=True,
                        vmin=0, vmax=1,
                        cbar_kws={'label': 'Normalized num images per class'})  # font size
        # This is to fix an issue with matplotlib==3.1.1
        # bottom, top = ax.get_ylim()
        # ax.set_ylim(bottom + 0.5, top - 0.5)
        plt.xticks(rotation=0)
        plt.yticks(rotation=0)
        plt.title("Confusion Matrix (Normalized) - Signature Net", pad=30)
        plt.ylabel('Ground Truth', labelpad=30)
        plt.xlabel('Predictions', labelpad=30)
        plt.savefig(save_to_dir.joinpath("sig_net_test_cm_normalized.png"))

        # plt.show()
        plt.clf()
        plt.cla()
        plt.close()

    # https://github.com/jeffheaton/t81_558_deep_learning/blob/47f5b87342fab61e19c0ee3ff46a3930cca41b1e/t81_558_class_04_2_multi_class.ipynb
    @classmethod
    def plot_roc(cls, ground_truths, predictions, save_to_dir):
        mpl.rcParams.update(mpl.rcParamsDefault)
        fpr, tpr, _ = sklearn.metrics.roc_curve(ground_truths, predictions)
        roc_auc = sklearn.metrics.auc(fpr, tpr)
        print(roc_auc)
        plt.figure()
        plt.plot(fpr, tpr, label='+ve label for same sources (area = %0.3f)' % roc_auc)

        # fpr, tpr, _ = sklearn.metrics.roc_curve(ground_truth, predictions, pos_label=0)
        # roc_auc = sklearn.metrics.auc(fpr, tpr)
        # plt.plot(fpr, tpr, label='+ve label for different sources (area = %0.3f)' % roc_auc)

        plt.plot([0, 1], [0, 1], 'k--', label='No Skill')
        plt.xlim([-0.05, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend(loc="lower right")

        plt.savefig(save_to_dir.joinpath("roc_plot.png"))
        # plt.show()
        plt.clf()
        plt.cla()
        plt.close()

    @classmethod
    def plot_scores_with_thresholds(cls, ground_truths, predictions, save_to_dir):
        mpl.rcParams.update(mpl.rcParamsDefault)
        f1_scores = []
        mcc_scores = []
        num_thresholds = 100.0
        thresholds = list(np.arange(start=min(predictions),
                                    stop=max(predictions),
                                    step=(max(predictions) - min(predictions)) / num_thresholds))
        for th in thresholds:
            pred = (predictions >= th) * 1
            f1_scores.append(sklearn.metrics.f1_score(ground_truths, pred, pos_label=1))
            mcc_scores.append(sklearn.metrics.matthews_corrcoef(ground_truths, pred))
        plt.figure()
        plt.plot(thresholds, f1_scores, label='F1 score')
        plt.plot(thresholds, mcc_scores, label='MCC score')
        plt.xlim([min(thresholds), max(thresholds)])
        plt.ylim([min(f1_scores + mcc_scores) - 0.05, 1.05])

        hist_bins = np.histogram(ground_truths, 2)

        rounded_scores = np.round(f1_scores, 2)
        index_max_scores = np.where(np.max(rounded_scores) == rounded_scores)[0]
        f1_based_threshold = thresholds[index_max_scores[len(index_max_scores) // 2]]
        print('Max f1 score is at threshold : {}'.format(f1_based_threshold))
        plt.axvline(x=f1_based_threshold, label='Max F1-based threshold', linestyle='--', alpha=0.40, c='r')

        rounded_scores = np.round(mcc_scores, 2)
        index_max_scores = np.where(np.max(rounded_scores) == rounded_scores)[0]
        mcc_based_threshold = thresholds[index_max_scores[len(index_max_scores) // 2]]
        print('Max mcc score is at threshold : {}'.format(mcc_based_threshold))
        plt.axvline(x=mcc_based_threshold, label='Max MCC-based threshold', linestyle='--', alpha=0.40, c='g')

        plt.title('Performance on test data ' +
                  '\n {} same sources '.format(hist_bins[0][1]) +
                  '& {} different sources'.format(hist_bins[0][0]))
        plt.xlabel('Thresholds')
        plt.ylabel('Scores')
        plt.legend()  # loc="lower right"

        plt.savefig(save_to_dir.joinpath("scores_with_thresholds.png"))
        # plt.show()
        plt.clf()
        plt.cla()
        plt.close()

        return mcc_based_threshold

    @classmethod
    def plot_similarity_matrix(cls, sim_matrix, save_to_dir):

        # Creating labels for the plot
        x_ticks = [''] * len(sim_matrix)
        y_ticks = [''] * len(sim_matrix)
        for i in np.arange(0, len(sim_matrix), 2):
            x_ticks[i] = str(i + 1)
            y_ticks[i] = str(i + 1)

        # df_cm = pd.DataFrame(sim_matrix, source_cameras, source_cameras)
        # df_cm = pd.DataFrame(sim_matrix, x_ticks, y_ticks)
        plt.figure(figsize=(30, 20))
        sn.set(font_scale=3.5)  # for label size
        ax = sn.heatmap(sim_matrix, square=True,
                        xticklabels=x_ticks, yticklabels=y_ticks,
                        vmin=0, vmax=1,
                        cbar_kws={'label': 'Similarity Score'})  # font size

        # This is to fix an issue with matplotlib==3.1.1
        # bottom, top = ax.get_ylim()
        # ax.set_ylim(bottom + 0.5, top - 0.5)

        plt.xticks(rotation=0)
        plt.yticks(rotation=0)

        plt.title("Similarity Matrix")
        plt.xlabel('Camera Device 1')
        plt.ylabel('Camera Device 2')

        plt.savefig(save_to_dir.joinpath("similarity_matrix.png"))
        # plt.show()
        plt.clf()
        plt.cla()
        plt.close()

    @classmethod
    def plot_similarity_scores_distribution(cls, similarity_scores, ground_truths, threshold, save_to_dir):
        mpl.rcParams.update(mpl.rcParamsDefault)
        positive_samples = np.where(ground_truths == 1)
        negative_samples = np.where(ground_truths == 0)
        positive_scores, negative_scores = similarity_scores[positive_samples], similarity_scores[negative_samples]

        num_bins = 20
        _, bin_edges = np.histogram(similarity_scores, bins=num_bins)
        positive_score_distribution, _ = np.histogram(positive_scores, bins=bin_edges)
        negative_score_distribution, _ = np.histogram(negative_scores, bins=bin_edges)

        num_positive_samples = np.sum(positive_score_distribution)
        num_negative_samples = np.sum(negative_score_distribution)

        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bar_width = 0.4 * (bin_edges[-1] - bin_edges[0]) / num_bins
        plt.figure()
        plt.bar(bin_centers - bar_width / 2, positive_score_distribution, width=bar_width, color='#444444',
                label='+ve class (num samples {})'.format(num_positive_samples), alpha=0.7)
        plt.bar(bin_centers + bar_width / 2, negative_score_distribution, width=bar_width, color='#e5ae38',
                label='-ve class (num samples {})'.format(num_negative_samples), alpha=0.7)
        plt.axvline(x=threshold, label='Selected threshold', linestyle='--', alpha=0.70, color='#000000')

        # Describe plot attributes
        plt.title('Similarity scores distribution')
        plt.xlabel('Similarity Scores')
        plt.ylabel('Count (Log Scale)')
        plt.yscale('log')
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_to_dir.joinpath("scores_distribution.png"))
        # plt.show()
        plt.clf()
        plt.cla()
        plt.close()


if __name__ == "__main__":
    VisualizationUtils.plot_class_wise_data_distribution(
        dataset_dir=r'/data/p288722/dresden/source_models/natural/',
        save_to_dir=r'/data/p288722/dresden/source_models/natural/'
    )
    # VisualizationUtils.save_avg_fourier_images()
