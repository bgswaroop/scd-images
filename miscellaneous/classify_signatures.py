from sklearn import neighbors, preprocessing
import numpy as np
from pathlib import Path
from signature_net.sig_net_flow import SigNetFlow


def signatures():
    pass


def classify_signatures():
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
    # print(scalar.mean_, scalar.var_)
    train_sig = scalar.transform(train_sig, copy=True)
    test_sig = scalar.transform(test_sig, copy=True)

    print('Num train samples: {}'.format(len(train_sig)))
    print('Num test samples: {}'.format(len(test_sig)))

    accuracy = {}
    for weights in ['uniform', 'distance']:
        for n_neighbors in [3, 5, 7, 10, 15]:
            clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)
            clf.fit(train_sig, train_labels)
            predictions = clf.predict(test_sig)

            accuracy[n_neighbors] = sum(test_labels == predictions) / len(test_labels)

        print(weights)
        print(accuracy)


if __name__ == '__main__':
    classify_signatures()
