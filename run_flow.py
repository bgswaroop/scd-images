import argparse
import json
import logging
import pickle
import shutil

import numpy as np

from configure import Configure, SigNet, SimNet
from miscellaneous.prepare_image_and_patch_data import level_balanced_from_hierarchical_dataset, \
    level_from_hierarchical_dataset
from signature_net.sig_net_flow import SigNetFlow
from similarity_net.sim_net_flow import SimNetFlow
from utils.evaluation_metrics import MultinomialClassificationScores
from utils.logging import SetupLogger, log_running_time
from utils.visualization_utils import VisualizationUtils

logger = logging.getLogger(__name__)


@log_running_time
def run_flow(fold_id):
    """
    A high level function to run the train and evaluation flows
    :return: None
    """

    # Signature Net
    Configure.sig_net_name = f'signature_net_{fold_id}'
    Configure.update()

    SigNetFlow.train()
    SigNetFlow.classify(aggregation_method='majority_vote')
    SigNetFlow.classify(aggregation_method='prediction_score_sum')
    SigNetFlow.classify(aggregation_method='log_scaled_prediction_score_sum')

    # Utils.visualize_ae_input_output_pairs()
    # Utils.save_avg_fourier_images()

    # Configure.train_data = r'/data/p288722/dresden/train/nat_patches_bal_kmkd_1/'
    # Configure.test_data = r'/data/p288722/dresden/test/nat_patches_bal_kmkd_1/'

    # Similarity Net
    # SimNetFlow.train()

    # Configure.train_data = r'/data/p288722/dresden/train/nat_patches_bal_kmkd_10/'
    # Configure.test_data = r'/data/p288722/dresden/test/nat_patches_bal_kmkd_10/'
    #
    # SimNet.balance_classes = False
    # SimNetFlow.classify()

    # SimNetFlow.classify()


@log_running_time
def run_cross_validation_flow():
    for fold_id in range(1, 6):
        # Signature Net
        Configure.train_data = rf'/data/p288722/dresden/train/nat_patches_18_models_128x128_15/fold_{fold_id}.json'
        Configure.test_data = rf'/data/p288722/dresden/test/nat_patches_18_models_128x128_15/fold_{fold_id}.json'
        Configure.sig_net_name = f'signature_net_{fold_id}'
        Configure.update()

        SigNetFlow.train()
        SigNetFlow.classify()

        # Similarity Net
        Configure.train_data = rf'/data/p288722/dresden/train/nat_patches_18_models_128x128_3/fold_{fold_id}.json'
        Configure.test_data = rf'/data/p288722/dresden/test/nat_patches_18_models_128x128_3/fold_{fold_id}.json'
        Configure.sim_net_name = f'similarity_net_{fold_id}'
        Configure.update()

        SimNet.balance_classes = True
        SimNetFlow.train()
        SimNet.balance_classes = False
        SimNetFlow.classify()


@log_running_time
def run_flow_hierarchical():
    # Step 0: Create a temp dir to save the dataset views
    temp_dir = Configure.runtime_dir.joinpath('temp')
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(exist_ok=True, parents=True)
    hierarchical_train_data_filepath, hierarchical_test_data_filepath = Configure.train_data, Configure.test_data

    # Step 1: Brand level training and classification
    target_level = 'brand'
    Configure.train_data = temp_dir.joinpath(rf'train_{target_level}_2000_fold_{fold_id}.json')
    brands_dataset_train = level_balanced_from_hierarchical_dataset(source_view=hierarchical_train_data_filepath,
                                                                    dest_level=target_level,
                                                                    max_patches=2000, dest_view=Configure.train_data)
    Configure.test_data = temp_dir.joinpath(rf'test_{target_level}_fold_{fold_id}.json')
    brands_dataset_test = level_from_hierarchical_dataset(source_view=hierarchical_test_data_filepath,
                                                          dest_level=target_level,
                                                          dest_view=Configure.test_data)
    SigNet.update_model(num_classes=len(brands_dataset_train), is_constrained=False)
    Configure.sig_net_name = f'signature_net_brands'
    Configure.update()

    SigNetFlow.train()
    SigNetFlow.classify()

    # Step 2: Model level training and classification
    with open(hierarchical_train_data_filepath, 'r') as f:
        hierarchical_train_data = json.load(f)['file_paths']
    with open(hierarchical_test_data_filepath, 'r') as f:
        hierarchical_test_data = json.load(f)['file_paths']

    target_level = 'model'
    for current_brand in ['Nikon', 'Samsung', 'Sony']:
        Configure.train_data = temp_dir.joinpath(rf'train_{target_level}_{current_brand}_500_fold_{fold_id}.json')
        models_dataset_train = level_balanced_from_hierarchical_dataset(
            source_view={current_brand: hierarchical_train_data[current_brand]}, dest_level=target_level,
            max_patches=500, dest_view=Configure.train_data
        )
        Configure.test_data = temp_dir.joinpath(rf'test_{target_level}_{current_brand}_fold_{fold_id}.json')
        models_dataset_test = level_from_hierarchical_dataset(
            source_view={current_brand: hierarchical_test_data[current_brand]}, dest_level=target_level,
            dest_view=Configure.test_data
        )
        SigNet.update_model(num_classes=len(models_dataset_train), is_constrained=False)
        Configure.sig_net_name = f'signature_net_{current_brand}'
        Configure.update()

        SigNetFlow.train()
        SigNetFlow.classify()

    # Step 3: Classify at brand level
    target_level = 'brand'
    Configure.sig_net_name = f'signature_net_brands'
    Configure.test_data = temp_dir.joinpath(rf'test_{target_level}_fold_{fold_id}.json')
    Configure.update()
    SigNet.update_model(num_classes=len(brands_dataset_train), is_constrained=False)
    SigNet.model.eval()
    image_paths, brand_predictions, brand_pred_scores = SigNetFlow.predict(Configure.test_data)

    brand_names = brands_dataset_train.keys()
    target_level = 'model'
    model_predictions = np.zeros_like(brand_predictions)
    for brand_name, brand_id in [(x, brand_names.index(x)) for x in ['Nikon', 'Samsung', 'Sony']]:
        indices = np.where(brand_predictions == brand_id)[0]
        if len(indices) > 0:
            filtered_image_paths = np.array(image_paths)[indices.astype(int)]
            Configure.sig_net_name = f'signature_net_{brand_name}'
            Configure.test_data = temp_dir.joinpath(rf'test_{target_level}_{brand_name}_fold_{fold_id}.json')
            Configure.update()
            with open(Configure.test_data, 'r') as f:
                num_classes = len(json.load(f)['file_paths'])
            SigNet.update_model(num_classes, is_constrained=False)
            filtered_image_paths = [str(x) for x in filtered_image_paths]
            SigNet.model.eval()
            _, predictions, pred_scores = SigNetFlow.predict(filtered_image_paths)
            for idx, item in zip(indices, predictions):
                model_predictions[idx] = item

    # Merge brand and model level predictions
    predictions = brand_predictions * 10 + model_predictions
    model_ids = []
    for brand_id, brand in enumerate(sorted(hierarchical_train_data.keys())):
        for model_id, model in enumerate(sorted(hierarchical_train_data[brand].keys())):
            model_ids.append(10*brand_id + model_id)
    labels_correction_map = {pr: idx for idx, pr in enumerate(model_ids)}
    predictions = [labels_correction_map[x] for x in predictions]

    model_path_to_gt_dict = {}
    with open(hierarchical_test_data_filepath, 'r') as f:
        test_data_dict = json.load(f)['file_paths']
        model_name_to_gt_label = {model_name: idx for idx, model_name in enumerate(sorted(test_data_dict.keys()))}
        for model_name in test_data_dict.keys():
            for path in test_data_dict[model_name]:
                model_path_to_gt_dict[path] = model_name_to_gt_label[model_name]

    ground_truths = [model_path_to_gt_dict[x] for x in image_paths]

    # fixme: predicted_scores is not being passed correctly
    ground_truths, predictions = SigNetFlow.patch_to_image(ground_truths, predictions, image_paths,
                                                           aggregation_method='majority_vote')

    # computing classification scores
    scores = MultinomialClassificationScores(ground_truths, predictions, one_hot=False,
                                             camera_names=sorted(test_data_dict.keys()))
    scores.log_scores()
    with open(str(Configure.signet_dir.joinpath('scores.pkl')), 'wb+') as f:
        pickle.dump(scores, f)
    VisualizationUtils.plot_confusion_matrix(ground_truths, predictions,
                                             one_hot=False, save_to_dir=Configure.runtime_dir.joinpath('signature_net'))


def run_flow_hierarchical_balanced():
    pass


if __name__ == '__main__':
    import inspect
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-fold', type=str, help='enter fold number')
    parser.add_argument('-num_patches', type=str, help='enter num patches')

    args = parser.parse_args()
    # fold_id = args.fold
    # num_patches = args.num_patches
    fold_id = 1
    # num_patches = 100
    #
    # Configure.runtime_dir = Path(
    #     rf'/scratch/p288722/runtime_data/scd_pytorch/18_models_hierarchial_4/fold_{fold_id}')
    # Configure.train_data = rf'/data/p288722/dresden/train/nat_patches_18_models_128x128_100/fold_{fold_id}.json'
    # Configure.test_data = rf'/data/p288722/dresden/test/nat_patches_18_models_128x128_100/fold_{fold_id}.json'
    # Configure.update()

    # import tarfile
    # with tarfile.open(Configure.tar_file, 'r:') as tar:
    #     member = tar.getmember('Rollei_RCP-7325XS_2/Rollei_RCP-7325XS_2_43235_004.JPG')
    #     pass

    SetupLogger(log_file=Configure.runtime_dir.joinpath(f'scd_{fold_id}_exp.log'))
    for item in [Configure, SigNet, SimNet]:
        logger.info(f'---------- {item.__name__} ----------')
        for name, value in inspect.getmembers(item)[26:]:
            logger.info(f'{name.ljust(20)}: {value}')
    try:
        # run_cross_validation_flow()
        # run_flow(fold_id)
        run_flow_hierarchical()
    except Exception as e:
        logger.error(e)
        print(e)
        raise e
