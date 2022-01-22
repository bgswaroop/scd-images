import logging
from pathlib import Path

import numpy as np
from sklearn import neighbors, preprocessing, model_selection, decomposition

from _old_strcture.signature_net.sig_net_flow import SigNetFlow

logger = logging.getLogger(__name__)


class ClassifySignatures(object):

    @classmethod
    def sklearn_knn_using_train_test_set(cls):
        """
        Method to classify the signatures using KNN Algorithm
        :return: None
        """
        sig_pairs_train = SigNetFlow.extract_signatures(config_mode='train')
        train_sig = [np.array(x[0]) for x in sig_pairs_train]
        train_labels = [Path(x[1]).parent.name for x in sig_pairs_train]

        sig_pairs_test = SigNetFlow.extract_signatures(config_mode='test')
        test_sig = [np.array(x[0]) for x in sig_pairs_test]
        test_labels = [Path(x[1]).parent.name for x in sig_pairs_test]

        # # normalize all samples to unit l2-norm
        # train_sig = preprocessing.normalize(train_sig, norm='l2')
        # test_sig = preprocessing.normalize(test_sig, norm='l2')

        # standardize the data - zero mean and unit variance for each feature dimension
        scalar = preprocessing.StandardScaler()
        scalar.fit(train_sig)
        # logger.info(scalar.mean_, scalar.var_)
        train_sig = scalar.transform(train_sig, copy=True)
        test_sig = scalar.transform(test_sig, copy=True)

        logger.info('Num train samples: {}'.format(len(train_sig)))
        logger.info('Num test samples: {}'.format(len(test_sig)))

        accuracy = {}
        for weights in ['uniform', 'distance']:
            for n_neighbors in [3, 5, 7, 10, 15]:
                clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)
                clf.fit(train_sig, train_labels)
                predictions = clf.predict(test_sig)
                accuracy[n_neighbors] = sum(test_labels == predictions) / len(test_labels)

            logger.info(weights)
            logger.info(accuracy)

    @classmethod
    def knn_using_leave_one_out_cross_validation(cls):
        """
        Method to classify the signatures using KNN, performing n fold cross validation
        :param n_folds: number of folds. If None then perform leave one out cross validation.
        :return: None
        """
        sig_pairs = SigNetFlow.extract_signatures(config_mode='train', images_dir=Configure.data)
        signatures = np.array([np.array(x[0]) for x in sig_pairs])
        labels = np.array([Path(x[1]).parent.name for x in sig_pairs])

        # data pre-processing
        scalar = preprocessing.StandardScaler()
        scalar.fit(signatures)
        signatures = scalar.transform(signatures, copy=True)

        # # normalize all samples to unit l2-norm
        # signatures = preprocessing.normalize(signatures, norm='l2')

        pca = decomposition.PCA(n_components=0.95, svd_solver='full')
        signatures = pca.fit_transform(signatures)

        logger.info('Num train samples: {}'.format(len(signatures) - 1))
        logger.info('Num test samples: {}'.format(1))
        logger.info('Feature dimensionality: {}'.format(signatures.shape[1]))

        distance_matrix = np.zeros((len(signatures), len(signatures)))
        for idx1, s1 in enumerate(signatures):
            for idx2, s2 in enumerate(signatures):
                if idx1 == idx2:
                    distance_matrix[idx1][idx2] = float('inf')
                else:
                    distance_matrix[idx1][idx2] = np.linalg.norm(s1 - s2)
        inverse_distance_matrix = 1 / distance_matrix

        accuracy = {}
        for weights in ['distance', 'uniform']:
            logger.info(weights)

            for n_neighbors in [3, 5, 7, 10, 15]:
                assert n_neighbors < len(signatures), \
                    'n_neighbours = {}, num_signatures = {}'.format(n_neighbors, len(signatures))
                accuracy[(weights, n_neighbors)] = 0

                for idx in range(len(signatures)):
                    min_distance_indices = np.argsort(distance_matrix[idx])[:n_neighbors]

                    if weights == 'distance':
                        w = inverse_distance_matrix[idx][min_distance_indices]
                    elif weights == 'uniform':
                        w = np.ones_like(inverse_distance_matrix[idx][min_distance_indices])
                    else:
                        raise ValueError('Invalid weights')
                    l = labels[min_distance_indices]

                    prediction_scores = {}
                    for var_label, var_weight in zip(l, w):
                        if var_label in prediction_scores:
                            prediction_scores[var_label] += var_weight
                        else:
                            prediction_scores[var_label] = var_weight

                    prediction = max(prediction_scores, key=lambda x: prediction_scores[x])
                    if prediction == labels[idx]:
                        accuracy[(weights, n_neighbors)] += 1
                accuracy[(weights, n_neighbors)] /= len(signatures)

                logger.info('{}\t{}'.format(n_neighbors, accuracy[(weights, n_neighbors)]))

        logger.info(accuracy)

    @classmethod
    def knn_using_train_test_set(cls):
        """
        Method to classify the signatures using KNN Algorithm
        :return: None
        """
        sig_pairs_train = SigNetFlow.extract_signatures(config_mode='train')
        train_sig = np.array([np.array(x[0]) for x in sig_pairs_train])
        train_labels = np.array([Path(x[1]).parent.name for x in sig_pairs_train])

        sig_pairs_test = SigNetFlow.extract_signatures(config_mode='test')
        test_sig = np.array([np.array(x[0]) for x in sig_pairs_test])
        test_labels = np.array([Path(x[1]).parent.name for x in sig_pairs_test])

        # normalize all samples to unit l2-norm
        train_sig = preprocessing.normalize(train_sig, norm='l2')
        test_sig = preprocessing.normalize(test_sig, norm='l2')

        # # standardize the data - zero mean and unit variance for each feature dimension
        # scalar = preprocessing.StandardScaler()
        # scalar.fit(train_sig)
        # # logger.info(scalar.mean_, scalar.var_)
        # train_sig = scalar.transform(train_sig, copy=True)
        # test_sig = scalar.transform(test_sig, copy=True)

        # https: // stackoverflow.com / a / 47325158 / 2709971
        pca = decomposition.PCA(n_components=0.95, svd_solver='full')
        train_sig = pca.fit_transform(train_sig)
        test_sig = pca.transform(test_sig)

        logger.info('Num train samples: {}'.format(len(train_sig)))
        logger.info('Num test samples: {}'.format(len(test_sig)))
        logger.info('Feature dimensionality: {}'.format(train_sig.shape[1]))

        distance_matrix = np.zeros((len(test_sig), len(train_sig)))
        for idx1, s1 in enumerate(test_sig):
            for idx2, s2 in enumerate(train_sig):
                distance_matrix[idx1][idx2] = np.linalg.norm(s1 - s2)
        inverse_distance_matrix = 1 / distance_matrix

        accuracy = {}
        for weights in ['uniform', 'distance']:
            logger.info(weights)
            for n_neighbors in [3, 5, 7, 10, 15]:
                # clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)
                # clf.fit(train_sig, train_labels)
                # predictions = clf.predict(test_sig)
                accuracy[(weights, n_neighbors)] = 0
                for idx in range(len(test_sig)):
                    min_distance_indices = np.argsort(distance_matrix[idx])[:n_neighbors]
                    if weights == 'distance':
                        w = inverse_distance_matrix[idx][min_distance_indices]
                    elif weights == 'uniform':
                        w = np.ones_like(inverse_distance_matrix[idx][min_distance_indices])
                    else:
                        raise ValueError('Invalid weights')
                    l = train_labels[min_distance_indices]

                    prediction_scores = {}
                    for temp_label, temp_weight in zip(l, w):
                        if temp_label in prediction_scores:
                            prediction_scores[temp_label] += temp_weight
                        else:
                            prediction_scores[temp_label] = temp_weight

                    prediction = max(prediction_scores, key=lambda x: prediction_scores[x])
                    if prediction == test_labels[idx]:
                        accuracy[(weights, n_neighbors)] += 1
                accuracy[(weights, n_neighbors)] /= len(test_sig)

                logger.info('{}\t{}'.format(n_neighbors, accuracy[(weights, n_neighbors)]))
            #
            # logger.info(weights)
            # logger.info(accuracy)

    @classmethod
    def knn_using_n_fold_cross_validation(cls, n_folds=None):
        """
        Method to classify the signatures using KNN, performing n fold cross validation
        :param n_folds: number of folds. If None then perform leave one out cross validation.
        :return: None
        """
        sig_pairs = SigNetFlow.extract_signatures(config_mode='train', images_dir=Configure.data)
        signatures = np.array([np.array(x[0]) for x in sig_pairs])
        labels = np.array([Path(x[1]).parent.name for x in sig_pairs])

        if n_folds:
            k_fold = model_selection.KFold(n_splits=n_folds)
        else:
            k_fold = model_selection.KFold(n_splits=len(signatures))

        accuracy = {}
        for weights in ['uniform', 'distance']:
            logger.info(weights)

            for n_neighbors in [3, 5, 7, 10, 15]:
                accuracy[(weights, n_neighbors)] = 0

                for train_index, test_index in k_fold.split(signatures):
                    # train and test data
                    train_sig, test_sig = signatures[train_index], signatures[test_index]
                    train_labels, test_labels = labels[train_index], labels[test_index]

                    # data pre-processing
                    scalar = preprocessing.StandardScaler()
                    scalar.fit(train_sig)
                    train_sig = scalar.transform(train_sig, copy=True)
                    test_sig = scalar.transform(test_sig, copy=True)

                    # train and evaluate
                    clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)
                    clf.fit(train_sig, train_labels)
                    predictions = clf.predict(test_sig)

                    # noinspection PyTypeChecker
                    accuracy[(weights, n_neighbors)] += sum(test_labels == predictions)
                accuracy[(weights, n_neighbors)] /= len(signatures)

                logger.info('{}\t{}'.format(n_neighbors, accuracy[(weights, n_neighbors)]))

        logger.info(accuracy)


if __name__ == '__main__':
    from _old_strcture.utils.logging import SetupLogger
    from _old_strcture.configure import Configure

    SetupLogger(Configure.runtime_dir.joinpath('scd.log'))
    try:
        # ClassifySignatures.knn_using_train_test_set()
        # ClassifySignatures.knn_using_leave_one_out_cross_validation()
        ClassifySignatures.knn_using_n_fold_cross_validation(n_folds=10)
        # ClassifySignatures.knn_using_train_test_set()
    except Exception as e:
        logger.error(e)
