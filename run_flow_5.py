# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse
import json
import logging
import pickle
import shutil
from shutil import copy2

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
def run_flow():
    """
    A high level function to run the train and evaluation flows
    :return: None
    """

    # Signature Net
    SigNetFlow.train()
    SigNetFlow.classify(aggregation_method='majority_vote')
    SigNetFlow.classify(aggregation_method='prediction_score_sum')
    SigNetFlow.classify(aggregation_method='log_scaled_prediction_score_sum')

    # Similarity Net
    SimNetFlow.train()
    SimNet.balance_classes = False
    SimNetFlow.classify()


@log_running_time
def run_flow_hierarchical():
    # Config
    patch_aggregation = SigNet.patch_aggregation

    # # Step 0: Create a temp dir to save the dataset views
    temp_dir = Configure.runtime_dir.joinpath('temp')
    # if temp_dir.exists():
    #     shutil.rmtree(temp_dir)
    temp_dir.mkdir(exist_ok=True, parents=True)
    hierarchical_train_data_filepath, hierarchical_test_data_filepath = \
        Configure.train_data_config, Configure.test_data_config
    #
    # # Step 1: Brand level training and classification
    target_level = 'brand'
    Configure.train_data_config = temp_dir.joinpath(rf'train_{target_level}_fold_{fold_id}.json')
    brands_dataset_train = level_balanced_from_hierarchical_dataset(source_view=hierarchical_train_data_filepath,
                                                                    dest_level=target_level,
                                                                    max_patches=SigNet.samples_per_class,
                                                                    dest_view=Configure.train_data_config)
    Configure.test_data_config = temp_dir.joinpath(rf'test_{target_level}_fold_{fold_id}.json')
    brands_dataset_test = level_from_hierarchical_dataset(source_view=hierarchical_test_data_filepath,
                                                          dest_level=target_level,
                                                          dest_view=Configure.test_data_config)
    SigNet.update_model(num_classes=len(brands_dataset_train), is_constrained=False)
    Configure.sig_net_name = f'signature_net_brands'
    Configure.update()

    # SigNetFlow.train()
    # SigNetFlow.classify(aggregation_method=patch_aggregation)

    # Step 2: Model level training and classification
    with open(hierarchical_train_data_filepath, 'r') as f:
        hierarchical_train_data = json.load(f)['file_paths']
    with open(hierarchical_test_data_filepath, 'r') as f:
        hierarchical_test_data = json.load(f)['file_paths']

    target_level = 'model'
    for current_brand in ['Nikon', 'Samsung', 'Sony']:
        Configure.train_data_config = temp_dir.joinpath(rf'train_{target_level}_{current_brand}_fold_{fold_id}.json')
        models_dataset_train = level_balanced_from_hierarchical_dataset(
            source_view={current_brand: hierarchical_train_data[current_brand]},
            dest_level=target_level,
            max_patches=SigNet.samples_per_class,
            dest_view=Configure.train_data_config
        )
        Configure.test_data_config = temp_dir.joinpath(rf'test_{target_level}_{current_brand}_fold_{fold_id}.json')
        models_dataset_test = level_from_hierarchical_dataset(
            source_view={current_brand: hierarchical_test_data[current_brand]}, dest_level=target_level,
            dest_view=Configure.test_data_config
        )
        SigNet.update_model(num_classes=len(models_dataset_train), is_constrained=False)
        Configure.sig_net_name = f'signature_net_{current_brand}'
        Configure.update()

        # SigNetFlow.train()
        # SigNetFlow.classify(aggregation_method=patch_aggregation)

    # Step 3: Classify at brand level
    target_level = 'brand'
    Configure.sig_net_name = f'signature_net_brands'
    Configure.test_data_config = temp_dir.joinpath(rf'test_{target_level}_fold_{fold_id}.json')
    Configure.update()
    SigNet.update_model(num_classes=len(brands_dataset_train), is_constrained=False)
    SigNet.model.eval()
    image_paths, brand_predictions, brand_pred_scores, std_devs = SigNetFlow.predict(Configure.test_data_config)

    if SigNet.use_contributing_patches:
        brand_name_to_label_map = {brand_name: idx for idx, brand_name in enumerate(sorted(brands_dataset_test.keys()))}
        ground_truths = [brand_name_to_label_map[Path(x).name.split('_')[0]] for x in image_paths]
        aggregation = SigNetFlow.patch_to_image(ground_truths, brand_predictions, image_paths,
                                                pred_scores=brand_pred_scores,
                                                aggregation_method=patch_aggregation,
                                                std_devs=std_devs)
        indices = np.where(aggregation.patch_contribution)[0].astype(int)
        image_paths = np.array(image_paths)[indices]
        brand_predictions = np.array(brand_predictions)[indices]
        brand_pred_scores = np.array(brand_pred_scores)[indices]
        std_devs = np.array(std_devs)[indices]

    brand_names = list(brands_dataset_train.keys())
    target_level = 'model'
    model_predictions = np.zeros_like(brand_predictions)
    prediction_scores = np.copy(brand_pred_scores)
    for brand_name, brand_id in [(x, brand_names.index(x)) for x in ['Nikon', 'Samsung', 'Sony']]:
        indices = np.where(brand_predictions == brand_id)[0]
        if len(indices) > 0:
            filtered_image_paths = np.array(image_paths)[indices.astype(int)]
            Configure.sig_net_name = f'signature_net_{brand_name}'
            Configure.test_data_config = temp_dir.joinpath(rf'test_{target_level}_{brand_name}_fold_{fold_id}.json')
            Configure.update()
            with open(Configure.test_data_config, 'r') as f:
                num_classes = len(json.load(f)['file_paths'])
            SigNet.update_model(num_classes, is_constrained=False)
            filtered_image_paths = [str(x) for x in filtered_image_paths]
            SigNet.model.eval()
            _, predictions, pred_scores, model_std_devs = SigNetFlow.predict(filtered_image_paths)
            for idx, pr, pr_s, sd in zip(indices, predictions, pred_scores, model_std_devs):
                model_predictions[idx] = pr
                prediction_scores[idx] = pr_s

    # Merge brand and model level predictions
    predictions = brand_predictions * 10 + model_predictions
    model_ids = []
    for brand_id, brand in enumerate(sorted(hierarchical_train_data.keys())):
        for model_id, model in enumerate(sorted(hierarchical_train_data[brand].keys())):
            model_ids.append(10 * brand_id + model_id)
    labels_correction_map = {pr: idx for idx, pr in enumerate(model_ids)}
    predictions = [labels_correction_map[x] for x in predictions]

    test_dict = level_from_hierarchical_dataset(source_view=hierarchical_test_data_filepath, dest_level='model')
    model_name_to_label_map = {model_name: idx for idx, model_name in enumerate(sorted(test_dict.keys()))}
    image_name_to_label_map = {}
    for model in test_dict:
        label = model_name_to_label_map[model]
        image_name_to_label_map.update({x: label for x in test_dict[model]})
    ground_truths = [image_name_to_label_map[x] for x in image_paths]

    aggregation = SigNetFlow.patch_to_image(ground_truths, predictions, image_paths,
                                            pred_scores=prediction_scores,
                                            std_devs=std_devs,
                                            aggregation_method=patch_aggregation)

    # computing classification scores
    scores = MultinomialClassificationScores(aggregation.image_ground_truths, aggregation.image_predictions,
                                             one_hot=False, camera_names=sorted(test_dict.keys()))
    scores.log_scores()
    with open(str(Configure.signet_dir.joinpath('scores.pkl')), 'wb+') as f:
        pickle.dump(scores, f)
    VisualizationUtils.plot_confusion_matrix(aggregation.image_ground_truths, aggregation.image_predictions,
                                             one_hot=False,
                                             save_to_dir=Configure.runtime_dir.joinpath('signature_net'))


def run_flow_hierarchical_balanced():
    pass


if __name__ == '__main__':
    import inspect
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-fold', type=int, help='enter fold number')
    parser.add_argument('-num_patches', type=int, help='enter num patches')
    parser.add_argument('-patch_aggregation', type=str, help='enter patch_aggregation')
    parser.add_argument('-use_contributing_patches', type=int, help='enter use_contributing_patches')
    parser.add_argument('-patches_type', type=str, help='choose between either "random" or "non_homo"')

    args = parser.parse_args()
    fold_id = args.fold
    num_patches = args.num_patches
    patch_aggregation = args.patch_aggregation
    use_contributing_patches = bool(args.use_contributing_patches)
    patches_type = args.patches_type

    if patches_type == "random" or patches_type == "non_homo":
        Configure.test_data_config = Path(
            rf'/data/p288722/dresden/test/18_models_{patches_type}_128x128_{num_patches}/fold_{fold_id}.json')
        Configure.train_data_config = Configure.test_data_config
        Configure.dataset_folder = \
            rf'/scratch/p288722/datasets/dresden/source_devices/nat_patches_{patches_type}_128x128_400'
        Configure.runtime_dir = Path(
            rf'/scratch/p288722/runtime_data/scd_pytorch/18_models_hcal_{patches_type}_2/test_{num_patches}/fold_{fold_id}')
        Configure.update()
        source_root = Path(rf'/scratch/p288722/runtime_data/scd_pytorch/18_models_hcal_{patches_type}_1/fold_{fold_id}/')

    else:
        Configure.test_data_config = Path(
            rf'/data/p288722/dresden/test/18_models_128x128_{num_patches}/fold_{fold_id}.json')
        Configure.train_data_config = Configure.test_data_config
        Configure.dataset_folder = rf'/scratch/p288722/datasets/dresden/source_devices/nat_patches_128x128_400'
        Configure.runtime_dir = Path(
            rf'/scratch/p288722/runtime_data/scd_pytorch/18_models_hcal_09/test_{num_patches}/fold_{fold_id}')
        Configure.update()
        source_root = Path(rf'/scratch/p288722/runtime_data/scd_pytorch/18_models_hcal_08/fold_{fold_id}/')

    # Copy the necessary files
    for f in list(source_root.glob("*.pt")):
        copy2(f, Configure.runtime_dir)

    for folder in ['signature_net_brands', 'signature_net_Nikon', 'signature_net_Samsung',
                   'signature_net_Sony']:
        for f in list(source_root.joinpath(folder).glob("*")):
            copy2(f, Configure.runtime_dir.joinpath(folder))

    SigNet.use_contributing_patches = use_contributing_patches
    SigNet.patch_aggregation = patch_aggregation

    SetupLogger(log_file=Configure.runtime_dir.joinpath(
        f'scd_{SigNet.patch_aggregation}_use_contributing_patches_{SigNet.use_contributing_patches}.log'))
    for item in [Configure, SigNet, SimNet]:
        logger.info(f'---------- {item.__name__} ----------')
        for name, value in inspect.getmembers(item)[26:]:
            logger.info(f'{name.ljust(20)}: {value}')

    try:
        run_flow_hierarchical()
    except Exception as e:
        logger.error(e)
        print(e)
        raise e
