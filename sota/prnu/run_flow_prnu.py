import argparse
import json
import logging
import pickle
from pathlib import Path

import numpy as np
from PIL import Image

from configure import Configure
from sota.prnu import functions as prnu
from utils.evaluation_metrics import MultinomialClassificationScores
from utils.logging import SetupLogger, log_running_time
from utils.visualization_utils import VisualizationUtils

logger = logging.getLogger(__name__)


@log_running_time
def run_train(fold_id):
    filename = str(Configure.fold_dir.joinpath(f'gt_prnu_{fold_id}.pkl'))
    if Path(filename).exists():
        with open(filename, 'rb') as f:
            prnu_signature = pickle.load(f)
    else:
        with open(Configure.train_data_config, 'r') as f:
            train_images_paths = json.load(f)['file_paths']

        prnu_signature = {}
        for device in train_images_paths:
            logger.info(f'Generating GT for {device}')
            prnu_signature[device] = []
            for img_path in train_images_paths[device]:
                img = np.asarray(Image.open(img_path))
                img = prnu.center_crop(img, (500, 500, 3))
                prnu_signature[device].append(img)
            prnu_signature[device] = prnu.extract_multiple(prnu_signature[device])

        with open(filename, 'wb+') as f:
            pickle.dump(prnu_signature, f)
        logger.info(f'Saved the ground truth PRNUs to: {filename}')

    return prnu_signature


@log_running_time
def run_classify(prnu_signature, fold_id):
    with open(Configure.test_data_config, 'r') as f:
        test_images_paths = json.load(f)['file_paths']
        devices_list = {x: idx for idx, x in enumerate(sorted(test_images_paths.keys()))}

    # def compute_corellation(device, img_path):
    #     img = np.asarray(Image.open(img_path))
    #     img = prnu.center_crop(img, (500, 500, 3))
    #     prnu_img = prnu.extract_single(img)
    #
    #     prediction = predict(prnu_signature, prnu_img)
    #     gt = devices_list[device]
    #     pred = devices_list[prediction]
    #     return gt, pred

    # items_list = []
    # for device in test_images_paths:
    #     for img_path in test_images_paths[device]:
    #         items_list.append((device, img_path))
    #
    # logger.info('Started predictions - multiprocessing')
    # pool = multiprocessing.Pool(4)
    # ground_truths, predictions = zip(*pool.map(compute_corellation, items_list))

    ground_truths, predictions = [], []
    for device in test_images_paths:
        logger.info(f'Performing predictions for {device}')
        for idx, img_path in enumerate(test_images_paths[device]):
            img = np.asarray(Image.open(img_path))
            img = prnu.center_crop(img, (500, 500, 3))
            prnu_img = prnu.extract_single(img)

            prediction = predict(prnu_signature, prnu_img)
            ground_truths.append(devices_list[device])
            predictions.append(devices_list[prediction])

    with open(str(Configure.fold_dir.joinpath(f'ground_truths_{fold_id}.pkl')), 'wb+') as f:
        pickle.dump(ground_truths, f)
    with open(str(Configure.fold_dir.joinpath(f'predictions_{fold_id}.pkl')), 'wb+') as f:
        pickle.dump(predictions, f)

    scores = MultinomialClassificationScores(ground_truths, predictions, one_hot=False,
                                             camera_names=sorted(test_images_paths.keys()))
    scores.log_scores()
    with open(str(Configure.fold_dir.joinpath(f'scores_{fold_id}.pkl')), 'wb+') as f:
        pickle.dump(scores, f)

    VisualizationUtils.plot_confusion_matrix(ground_truths, predictions,
                                             one_hot=False, save_to_dir=Configure.fold_dir)


def predict(gt_signatures, test_sig):
    predictions = {}
    for device in gt_signatures:
        predictions[device] = [{}] * 4
        k = gt_signatures[device]
        for rot_idx in range(4):
            cc = prnu.crosscorr_2d(k, np.rot90(test_sig, rot_idx))
            predictions[device][rot_idx] = prnu.pce(cc)

        best_pce = np.max([p['pce'] for p in predictions[device]])
        predictions[device] = best_pce

    # return the key with the maximum pce
    return max(predictions, key=lambda key: predictions[key])


@log_running_time
def run_flow(fold_id):
    """
    A high level function to run the train and evaluation flows
    :return: None
    """

    Configure.train_data_config = rf'/data/p288722/dresden/train/nat_images_18/fold_{fold_id}.json'
    Configure.test_data_config = rf'/data/p288722/dresden/test/nat_images_18/fold_{fold_id}.json'
    Configure.fold_dir = Configure.runtime_dir.joinpath(f'fold_{fold_id}')
    Configure.fold_dir.mkdir(exist_ok=True, parents=True)

    prnu_signature = run_train(fold_id)
    run_classify(prnu_signature, fold_id)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-fold', type=str, help='enter fold number')
    args = parser.parse_args()
    fold_id = args.fold

    SetupLogger(log_file=Configure.runtime_dir.joinpath(f'prnu_{fold_id}.log'))
    logger.info('Running the flow for PRNU')
    try:
        run_flow(fold_id)
    except Exception as e:
        logger.error(e)
        raise e
