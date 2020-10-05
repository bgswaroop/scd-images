import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

logger = logging.getLogger(__name__)


class Score:
    def __init__(self):
        self.tp = int(0)
        self.tn = int(0)
        self.fp = int(0)
        self.fn = int(0)
        self.tpr = None
        self.tnr = None
        self.balanced_accuracy = None
        self.precision = None
        self.recall = None
        self.accuracy = None
        self.mcc = None
        self.f1 = None

    def compute_evaluation_metrics(self):
        """
        :return: dictionary with scores for true positives, true negatives, false positives and false negatives
        """

        return {"true_positive": self.tp, "true_negative": self.tn, "false_positive": self.fp,
                "false_negative": self.fn}

    def compute_precision_recall(self, evaluation_metrics=None):
        """
        Computes precision and recall
        precision = tp/(tp+fp)
        recall = tp/(tp+fn)

        It is recommended to call the method compute_evaluation_metrics before invoking this method.
        Note, this method does not update the tp, fp, tn and fn rates of the members.

        :param evaluation_metrics: (optional) This is a dictionary of the form
        {"true_positive": tp, "true_negative": tn, "false_positive": fp, "false_negative": fn}

        :return: precision, recall
        """

        if evaluation_metrics is None:
            tp = self.tp
            fp = self.fp
            fn = self.fn
        else:
            tp = evaluation_metrics["true_positive"]
            fp = evaluation_metrics["false_positive"]
            fn = evaluation_metrics["false_negative"]

        self.precision = tp / (tp + fp)
        self.recall = tp / (tp + fn)

        return {"precision": self.precision, "recall": self.recall}

    def compute_accuracy(self):
        num_correct_predictions = self.tp + self.tn
        num_samples = self.tp + self.tn + self.fp + self.fn
        self.accuracy = num_correct_predictions / num_samples

        self.tpr = self.tp / (self.tp + self.fn)
        self.tnr = self.tn / (self.tn + self.fp)
        self.balanced_accuracy = (self.tpr + self.tnr) / 2

        return {"accuracy": self.accuracy, "balanced_accuracy": self.balanced_accuracy}

    def compute_f1_score(self):
        self.f1 = 2 * self.precision * self.recall / (self.precision + self.recall)

    def compute_mcc(self):
        tp = int(self.tp)
        tn = int(self.tn)
        fp = int(self.fp)
        fn = int(self.fn)
        self.mcc = (tp * tn - fp * fn) / math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    def evaluate_all_metrics(self):
        self.compute_evaluation_metrics()
        self.compute_accuracy()
        self.compute_precision_recall()
        self.compute_f1_score()
        self.compute_mcc()

    def log_scores(self, print_func=logger.info):
        print_func("------------------- Global Metrics ignoring duplicate --------------------")
        print_func("True Positives     : {}".format(self.tp))
        print_func("True Negatives     : {}".format(self.tn))
        print_func("False Positives    : {}".format(self.fp))
        print_func("False Negatives    : {}".format(self.fn))
        print_func("True Positive Rate : {}".format(self.tpr))
        print_func("True Negative Rate : {}".format(self.tnr))
        print_func("Accuracy           : {}".format(self.accuracy))
        print_func("Balanced Accuracy  : {}".format(self.balanced_accuracy))
        print_func("Precision          : {}".format(self.precision))
        print_func("Recall             : {}".format(self.recall))
        print_func("MCC                : {}".format(self.mcc))
        print_func("F1                 : {}".format(self.f1))
        print_func("--------------------------------------------------------------------------")


class SimilarityMatrixScores(Score):
    def __init__(self, similarity_matrix, num_samples_per_experiment=100, consider_upper_diagonal=True):
        super().__init__()
        self.similarity_matrix = similarity_matrix
        self.consider_upper_diagonal = consider_upper_diagonal
        self.num_samples = num_samples_per_experiment  # this could be a NxN matrix (todo)

        self.evaluate_all_metrics()

    def compute_evaluation_metrics(self):
        """
        similarity_matrix: This is a NxN ndarray of similarity scores between [0, 1]
        where all the diagonal entries represent the similarity scores between pairs of same cameras
        and all the non-diagonal entries represent the similarity between pairs of different cameras.

        num_samples: This parameter represents the number of image pairs used for each experiment.
        this could be a single number, representing same number of pairs for all the experiments
        this could be a NxN matrix (todo)

        consider_upper_diagonal: This parameter which is True by default uses only the upper diagonal elements
        for the computation.

        :return: dictionary with scores for true positives, true negatives, false positives and false negatives
        """
        start = 0
        num_samples_matrix = np.multiply(np.ones_like(self.similarity_matrix), self.num_samples)

        n = len(self.similarity_matrix)
        for row in np.arange(0, n):
            if self.consider_upper_diagonal:
                start = row
            for col in np.arange(start, n):
                if row == col:
                    self.tp += (self.similarity_matrix[row][col] * num_samples_matrix[row][col])
                    self.fn += ((1 - self.similarity_matrix[row][col]) * num_samples_matrix[row][col])
                else:
                    self.fp += (self.similarity_matrix[row][col] * num_samples_matrix[row][col])
                    self.tn += ((1 - self.similarity_matrix[row][col]) * num_samples_matrix[row][col])

        self.tp = int(self.tp)
        self.tn = int(self.tn)
        self.fp = int(self.fp)
        self.fn = int(self.fn)

        return {"true_positive": self.tp, "true_negative": self.tn, "false_positive": self.fp,
                "false_negative": self.fn}


class BinaryClassificationScores(Score):

    def __init__(self, ground_truths, predictions):
        super().__init__()
        self.ground_truths = ground_truths
        self.predictions = predictions

        self.evaluate_all_metrics()

    def compute_evaluation_metrics(self):
        flipped_gt = 1 - self.ground_truths
        flipped_pr = 1 - self.predictions

        self.tp = np.dot(self.ground_truths, self.predictions)
        self.tn = np.dot(flipped_gt, flipped_pr)
        self.fp = np.dot(flipped_gt, self.predictions)
        self.fn = np.dot(self.ground_truths, flipped_pr)

        return {"true_positive": self.tp, "true_negative": self.tn, "false_positive": self.fp,
                "false_negative": self.fn}


class MultinomialClassificationScores(Score):
    def __init__(self, ground_truths, predictions, one_hot, camera_names):
        super().__init__()

        self.camera_names = camera_names
        if one_hot:
            self.ground_truths = [np.argmax(x) for x in ground_truths]
            self.predictions = [np.argmax(x) for x in predictions]
        else:
            self.ground_truths = ground_truths
            self.predictions = predictions

        self.confusion_matrix = None
        self.evaluate_all_metrics()

    def evaluate_all_metrics(self):
        self.compute_evaluation_metrics()
        self.compute_accuracy()
        # self.compute_precision_recall()
        # self.compute_f1_score()
        # self.compute_mcc()

    def compute_accuracy(self):
        self.accuracy = sklearn.metrics.accuracy_score(self.ground_truths, self.predictions)

    def compute_evaluation_metrics(self):
        self.confusion_matrix = sklearn.metrics.confusion_matrix(self.ground_truths, self.predictions)

        # sklearn.metrics.f1_score
        # start = 0
        #
        # n = len(self.confusion_matrix)
        # for row in np.arange(0, n):
        #     if self.consider_upper_diagonal:
        #         start = row
        #     for col in np.arange(start, n):
        #         if row == col:
        #             self.tp += self.confusion_matrix[row][col]
        #             self.fn += (1 - self.confusion_matrix[row][col])
        #         else:
        #             self.fp += self.confusion_matrix[row][col]
        #             self.tn += ((1 - self.confusion_matrix[row][col]) * num_samples_matrix[row][col])
        #
        # self.tp = int(self.tp)
        # self.tn = int(self.tn)
        # self.fp = int(self.fp)
        # self.fn = int(self.fn)

    def log_scores(self, print_func=logger.info):
        print_func("Cameras List : \n\n{}\n".format(
            print_numpy_array_with_index(self.camera_names)
        ))
        print_func("Confusion Matrix : \n\n{}\n".format(
            print_numpy_array_with_index(self.confusion_matrix)
        ))

        print_func("---------------------------- Global Metrics  -----------------------------")
        print_func("True Positives     : {}".format(self.tp))
        print_func("True Negatives     : {}".format(self.tn))
        print_func("False Positives    : {}".format(self.fp))
        print_func("False Negatives    : {}".format(self.fn))
        print_func("True Positive Rate : {}".format(self.tpr))
        print_func("True Negative Rate : {}".format(self.tnr))
        print_func("Accuracy           : {}".format(self.accuracy))
        print_func("Balanced Accuracy  : {}".format(self.balanced_accuracy))
        print_func("Precision          : {}".format(self.precision))
        print_func("Recall             : {}".format(self.recall))
        print_func("MCC                : {}".format(self.mcc))
        print_func("F1                 : {}".format(self.f1))
        print_func("--------------------------------------------------------------------------")


class ScoreUtils:
    def __init__(self, source_device_labels, predictions, camera_names):
        self.ground_truth_count_matrix = None
        self.true_prediction_count_matrix = None
        self.accuracy_matrix = None
        self.similarity_matrix = None
        self.camera_names = camera_names
        self.class_wise_scores = None
        self.global_scores = {}
        self.source_device_labels = source_device_labels
        self.predictions = predictions
        self.evaluate_all_metrics()

    def evaluate_all_metrics(self):
        self.prepare_similarity_matrix()
        self.compute_class_wise_scores()
        self.compute_macro_micro_scores()

    def prepare_similarity_matrix(self):

        label_pairs = [np.array(x) for x in zip(*self.source_device_labels)]
        ground_truths = np.where(label_pairs[0] == label_pairs[1], 1, 0)
        _, indices_0 = np.unique(label_pairs[0], return_inverse=True)
        _, indices_1 = np.unique(label_pairs[1], return_inverse=True)

        num_cameras = len(self.camera_names)
        self.ground_truth_count_matrix = np.zeros((num_cameras, num_cameras), dtype=object)
        self.true_prediction_count_matrix = np.zeros((num_cameras, num_cameras), dtype=object)

        for item in zip(zip(indices_0, indices_1), ground_truths, self.predictions):
            self.ground_truth_count_matrix[item[0][0]][item[0][1]] += 1
            if item[1] == item[2]:
                self.true_prediction_count_matrix[item[0][0]][item[0][1]] += 1

        # Make the matrix symmetric
        self.ground_truth_count_matrix += self.ground_truth_count_matrix.transpose() - np.diag(
            np.diag(self.ground_truth_count_matrix))
        self.true_prediction_count_matrix += self.true_prediction_count_matrix.transpose() - np.diag(
            np.diag(self.true_prediction_count_matrix))

        # Compute accuracy matrix (element wise division)
        self.accuracy_matrix = (self.true_prediction_count_matrix + np.finfo(float).eps) / (
                self.ground_truth_count_matrix + np.finfo(float).eps)

        # Compute similarity matrix
        self.similarity_matrix = np.diag(np.diag(self.accuracy_matrix))
        self.similarity_matrix += (1 - self.accuracy_matrix) * (1 - np.eye(num_cameras))

        return self.similarity_matrix

    def compute_class_wise_scores(self):
        true_pr_count = self.true_prediction_count_matrix
        false_pr_count = self.ground_truth_count_matrix - self.true_prediction_count_matrix

        tp = np.diag(true_pr_count)
        tn = np.sum(true_pr_count, axis=0) - tp
        fn = np.diag(false_pr_count)
        fp = np.sum(false_pr_count, axis=0) - fn

        # Avoiding division by zero - by setting the result to 0
        # precision = tp / (tp + fp)
        # recall = tp / (tp + fn)
        # f1_score = 2 * precision * recall / (precision + recall)
        precision = tp / np.where(tp + fp == 0, 1, tp + fp)
        recall = tp / np.where(tp + fn == 0, 1, tp + fn)
        f1_score = 2 * precision * recall / np.where(precision + recall == 0, 1, precision + recall)

        # Mathews Correlation Coefficient
        # mcc = (tp * tn - fp * fn) / (np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))

        mcc_denominator = np.where((tp + fp) == 0, 1, tp + fp) * np.where((tp + fn) == 0, 1, tp + fn)
        mcc_denominator *= np.where((tn + fp) == 0, 1, tn + fp) * np.where((tn + fn) == 0, 1, tn + fn)
        mcc_denominator = np.array([math.sqrt(x) for x in mcc_denominator], dtype=object)
        mcc = (tp * tn - fp * fn) / mcc_denominator

        self.class_wise_scores = pd.DataFrame({
            "class name": np.array(self.camera_names),
            "true_positives": tp.astype(np.int),
            "true_negatives": tn.astype(np.int),
            "false_positives": fp.astype(np.int),
            "false_negatives": fn.astype(np.int),
            "precision": precision.astype(np.float),
            "recall": recall.astype(np.float),
            "f1_score": f1_score.astype(np.float),
            "mcc_score": mcc.astype(np.float)
        })

    def compute_macro_micro_scores(self):
        self.global_scores["Macro F1"] = np.nanmean(self.class_wise_scores["f1_score"].values)
        self.global_scores["Macro Precision"] = np.nanmean(self.class_wise_scores["precision"].values)
        self.global_scores["Macro Recall"] = np.nanmean(self.class_wise_scores["recall"].values)
        self.global_scores["Macro MCC"] = np.nanmean(self.class_wise_scores["mcc_score"].values)

        # class_wise_weights = np.sum(self.ground_truth_count_matrix, axis=0)
        # logger.info("Micro scores are weighted by TP+TN+FP+FN")
        class_wise_weights = \
            self.class_wise_scores["true_positives"].values + self.class_wise_scores["false_negatives"].values
        logger.info("Micro scores are weighted by TP+FN")

        num_samples = np.nansum(class_wise_weights)
        self.global_scores["Micro F1"] = \
            np.nansum(self.class_wise_scores["f1_score"].values * class_wise_weights) / num_samples
        self.global_scores["Micro Precision"] = \
            np.nansum(self.class_wise_scores["precision"].values * class_wise_weights) / num_samples
        self.global_scores["Micro Recall"] = \
            np.nansum(self.class_wise_scores["recall"].values * class_wise_weights) / num_samples
        self.global_scores["Micro MCC"] = \
            np.nansum(self.class_wise_scores["mcc_score"].values * class_wise_weights) / num_samples

    def log_scores(self, print_func=logger.info):
        print_func("Cameras List : \n\n{}\n".format(
            print_numpy_array_with_index(self.camera_names)
        ))
        print_func("ground_truth_count_matrix : \n\n{}\n".format(
            print_numpy_array_with_index(self.ground_truth_count_matrix)
        ))
        print_func("true_prediction_count_matrix : \n\n{}\n".format(
            print_numpy_array_with_index(self.true_prediction_count_matrix)
        ))
        print_func("accuracy_matrix : \n\n{}\n".format(
            print_numpy_array_with_index(self.accuracy_matrix)
        ))
        print_func("similarity_matrix : \n\n{}\n".format(
            print_numpy_array_with_index(self.similarity_matrix)
        ))

        # print class wise scores
        self.class_wise_scores.loc['Total'] = \
            self.class_wise_scores.select_dtypes(include=[np.int, np.float]).sum()
        print_func("Class wise scores: \n\n{}\n".format(
            print_numpy_array_with_index(self.class_wise_scores)
        ))

        # print global statistics

        # print macro and micro scores
        print_func("Macro F1           : {}".format(self.global_scores["Macro F1"]))
        print_func("Macro Precision    : {}".format(self.global_scores["Macro Precision"]))
        print_func("Macro Recall       : {}".format(self.global_scores["Macro Recall"]))
        print_func("Macro MCC          : {}".format(self.global_scores["Macro MCC"]))
        print_func("Micro F1           : {}".format(self.global_scores["Micro F1"]))
        print_func("Micro Precision    : {}".format(self.global_scores["Micro Precision"]))
        print_func("Micro Recall       : {}".format(self.global_scores["Micro Recall"]))
        print_func("Micro MCC          : {}".format(self.global_scores["Micro MCC"]))


def print_numpy_array_with_index(numpy_array):
    pd_array = pd.DataFrame(numpy_array)
    pd.options.display.float_format = '{:,.2f}'.format
    return pd_array.to_string()
