import argparse
import logging
import pickle

import numpy as np
from matplotlib import pyplot as plt

from configure import Configure, SigNet, SimNet
from utils.logging import SetupLogger, log_running_time

logger = logging.getLogger(__name__)

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


@log_running_time
def run_flow():
    """
    A high level function to run the train and evaluation flows
    :return: None
    """

    # # Seed that will determine the sampling in the data sets
    # random.seed(123)
    #
    # # Step 0: Create a temp dir to save the dataset views
    # temp_dir = Configure.runtime_dir.joinpath('temp')
    # temp_dir.mkdir(exist_ok=True, parents=True)
    # train_view, test_view = Configure.train_data_config, Configure.test_data_config
    #
    # # Step 1: Training the auto-encoders at the models level
    # target_level = 'model'
    # Configure.train_data_config = temp_dir.joinpath(rf'train_{target_level}_fold_{fold_id}.json')
    # models_dataset_train = hierarchical_to_flat_view_with_balancing(source_view=train_view,
    #                                                                 dest_level=target_level,
    #                                                                 dest_view=Configure.train_data_config)
    # Configure.test_data_config = temp_dir.joinpath(rf'test_{target_level}_fold_{fold_id}.json')
    # hierarchical_to_flat_view(source_view=test_view,
    #                           dest_level=target_level,
    #                           dest_view=Configure.test_data_config)

    # SigNetFlow.train()

    # Train a SVM classifier
    # sig_train = SigNetFlow.extract_signatures('train')
    # sig_test = SigNetFlow.extract_signatures('test')

    # with open(Configure.runtime_dir.joinpath('train_features.pkl'), 'wb+') as f:
    #     pickle.dump(sig_train, f)
    # with open(Configure.runtime_dir.joinpath('test_features.pkl'), 'wb+') as f:
    #     pickle.dump(sig_test, f)

    with open(Configure.runtime_dir.joinpath('train_features.pkl'), 'rb') as f:
        sig_train = pickle.load(f)
    with open(Configure.runtime_dir.joinpath('test_features.pkl'), 'rb') as f:
        sig_test = pickle.load(f)

    # train_data = [x[0] / np.linalg.norm(x[0]) for x in sig_train]
    train_labels = [x[1] for x in sig_train]
    ordering = np.argsort(train_labels)
    train_data = [sig_train[x][0] for x in ordering]

    # clf = svm.SVC()
    # clf.fit(train_data, train_labels)

    test_data = [x[0] / np.linalg.norm(x[0]) for x in sig_test]
    test_labels = [x[1] for x in sig_test]

    train_matrix = np.stack(train_data)
    plt.imshow(train_matrix)
    plt.colorbar()
    plt.title('Feature Matrix Visualization - Train Set')
    plt.xlabel('Feature dimension')
    plt.ylabel('Examples')
    plt.show()

    # plt.imshow(train_matrix)
    # plt.colorbar()
    # plt.title('Feature Matrix Visualization - Test Set')
    # plt.xlabel('Feature dimension')
    # plt.ylabel('Examples')
    # plt.show()

    # predictions = clf.predict(test_data)
    #
    # num_correct_predictions = sum(predictions == test_labels)
    # accuracy = num_correct_predictions / len(test_labels)

    # unique_labels = set(train_labels)
    # for idx, label in enumerate(unique_labels):
    #
    #     indices = [ind for ind, x in enumerate(train_labels) if x==label]
    #     per_class_train_data = [train_data[x] for x in indices]
    #     per_class_test_data = [test_data[x] for x in indices]
    #
    #     x = np.arange(1500)
    #     plt.figure()
    #     for y in per_class_train_data:
    #         plt.scatter(x, y, s=1, c=idx)
    #     plt.title(f'(Train); Class - {label}; num_samples - {len(per_class_train_data)}')
    #     plt.xlabel('Feature dimension')
    #     plt.ylabel('Values')
    #     plt.show()
    #
    #     plt.figure()
    #     for y in per_class_test_data:
    #         plt.scatter(x, y, s=1, c=idx)
    #     plt.title(f'(Test); Class - {label}; num_samples - {len(per_class_test_data)}')
    #     plt.xlabel('Feature dimension')
    #     plt.ylabel('Values')
    #     plt.show()
    #
    #     break

    print()

    # print(f'Number of samples - {len(data)}')
    # x = np.arange(1500)
    # from matplotlib import pyplot as plt
    # plt.figure()
    # for y in data:
    #     plt.scatter(x, y, s=1)
    # plt.title('Scatter plot of normalized features - Test Set')
    # plt.xlabel('Feature dimension')
    # plt.ylabel('Values')
    # plt.show()

    # Similarity Net
    # SimNetFlow.train()
    # SimNet.balance_classes = False
    # SimNetFlow.classify()


if __name__ == '__main__':
    import inspect
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-fold', type=int, help='enter fold number')
    args = parser.parse_args()
    fold_id = args.fold

    Configure.train_data_config = Path(rf'/data/p288722/dresden/train/18_models_image_level/fold_{fold_id}.json')
    Configure.test_data_config = Path(rf'/data/p288722/dresden/test/18_models_image_level/fold_{fold_id}.json')
    Configure.dataset_folder = Path(rf'/data/p288722/dresden/source_devices/natural')
    Configure.runtime_dir = Path(rf'/scratch/p288722/runtime_data/scd_ae/18_models_baseline/fold_{fold_id}')
    Configure.runtime_dir.mkdir(exist_ok=True, parents=True)
    Configure.update()

    SetupLogger(log_file=Configure.runtime_dir.joinpath(f'scd.log'))
    for item in [Configure, SigNet, SimNet]:
        logger.info(f'---------- {item.__name__} ----------')
        for name, value in inspect.getmembers(item)[26:]:
            logger.info(f'{name.ljust(20)}: {value}')
    try:
        run_flow()
    except Exception as e:
        logger.error(e)
        print(e)
        raise e
