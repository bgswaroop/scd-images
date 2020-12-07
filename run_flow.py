import logging
import pickle

from configure import Configure, SigNet, SimNet
from signature_net.sig_net_flow import SigNetFlow
from similarity_net.sim_net_flow import SimNetFlow
from utils.evaluation_metrics import MultinomialClassificationScores
from utils.logging import SetupLogger, log_running_time
from utils.visualization_utils import VisualizationUtils

logger = logging.getLogger(__name__)
import argparse


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


def run_flow_hierarchial():
    import json
    from pathlib import Path
    import datetime
    import shutil

    temp_dir = Configure.runtime_dir.joinpath('temp')
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(exist_ok=True, parents=True)

    # Step 1 : Brand level training and classification
    def create_brands_dataset_from_models(filename):
        with open(filename, 'r') as f:
            models_dict = json.load(f)['file_paths']
        brands_dict = {}
        for model_name in sorted(models_dict.keys()):
            brand_name = model_name.split('_')[0]
            if brand_name not in brands_dict:
                brands_dict[brand_name] = []
            brands_dict[brand_name].extend(models_dict[model_name])
        timestamp = str(datetime.datetime.now()).replace(' ', '').replace('-', '').replace(':', '').replace('.', '')
        modified_json_file = temp_dir.joinpath(f'{timestamp}_brands_{Path(filename).name}')
        with open(modified_json_file, 'w+') as f:
            json.dump({'file_paths': brands_dict}, f, indent=2)
        return modified_json_file

    source_train_data, source_test_data = Configure.train_data, Configure.test_data
    Configure.train_data = create_brands_dataset_from_models(source_train_data)
    Configure.test_data = create_brands_dataset_from_models(source_test_data)

    with open(Configure.train_data, 'r') as f:
        brands_dataset = json.load(f)['file_paths']
        num_classes = len(brands_dataset)
        SigNet.update_model(num_classes, is_constrained=False)

    Configure.sig_net_name = f'signature_net_brands'
    Configure.update()

    SigNetFlow.train()
    SigNetFlow.classify()

    # Step 2: model level training and classification
    def create_multi_models_datasets_from_models_dataset(filename):
        with open(filename, 'r') as f:
            models_dict = json.load(f)['file_paths']
        brands_dict = {}
        for model_name in sorted(models_dict.keys()):
            brand_name = model_name.split('_')[0]
            if brand_name not in brands_dict:
                brands_dict[brand_name] = {model_name: models_dict[model_name]}
            else:
                brands_dict[brand_name][model_name] = models_dict[model_name]

        modified_files = []
        for brand_name in brands_dict:
            if len(brands_dict[brand_name]) > 1:
                timestamp = str(datetime.datetime.now()).replace(' ', '').replace('-', '').replace(':', '').replace('.',
                                                                                                                    '')
                modified_json_file = temp_dir.joinpath(f'{timestamp}_{brand_name}_{Path(filename).name}')
                with open(modified_json_file, 'w+') as f:
                    json.dump({'file_paths': brands_dict[brand_name]}, f, indent=2)
                modified_files.append(modified_json_file)

        return modified_files

    train_data_models = create_multi_models_datasets_from_models_dataset(source_train_data)
    test_data_models = create_multi_models_datasets_from_models_dataset(source_test_data)

    camera_model_ids = {}
    camera_brand_names = {brand_name: idx for idx, brand_name in enumerate(sorted(brands_dataset.keys()))}
    camera_brand_ids = {idx: brand_name for idx, brand_name in enumerate(sorted(brands_dataset.keys()))}

    for train_data, test_data in zip(train_data_models, test_data_models):
        Configure.train_data = train_data
        Configure.test_data = test_data
        with open(Configure.train_data, 'r') as f:
            num_classes = len(json.load(f)['file_paths'])
        SigNet.update_model(num_classes, is_constrained=False)

        brand_name = train_data.stem.split("_")[1]
        Configure.sig_net_name = f'signature_net_{brand_name}'
        camera_model_ids[camera_brand_names[brand_name]] = Configure.sig_net_name
        Configure.update()

        SigNetFlow.train()
        SigNetFlow.classify()

    # Classify at brand level
    Configure.sig_net_name = f'signature_net_brands'
    Configure.train_data = create_brands_dataset_from_models(source_train_data)
    Configure.test_data = create_brands_dataset_from_models(source_test_data)
    Configure.update()
    with open(Configure.train_data, 'r') as f:
        num_classes = len(json.load(f)['file_paths'])
    SigNet.update_model(num_classes, is_constrained=False)
    SigNet.model.eval()
    image_paths, brand_predictions, brand_pred_scores = SigNetFlow.predict(Configure.test_data)

    import numpy as np
    model_predictions = np.zeros_like(brand_predictions)
    for idx, brand_id in enumerate(camera_model_ids):
        indices = np.where(brand_predictions == brand_id)[0]
        if len(indices) > 0:
            filtered_image_paths = np.array(image_paths)[indices.astype(int)]
            Configure.sig_net_name = f'signature_net_{camera_brand_ids[brand_id]}'
            Configure.test_data = test_data_models[idx]
            Configure.train_data = train_data_models[idx]
            Configure.update()
            with open(Configure.train_data, 'r') as f:
                num_classes = len(json.load(f)['file_paths'])
            SigNet.update_model(num_classes, is_constrained=False)
            filtered_image_paths = [str(x) for x in filtered_image_paths]
            SigNet.model.eval()
            _, predictions, pred_scores = SigNetFlow.predict(filtered_image_paths)
            for idx, item in zip(indices, predictions):
                model_predictions[idx] = item

    # Merge brand and model level predictions
    predictions = brand_predictions * 10 + model_predictions
    labels_correction_map = {pr: idx for idx, pr in
                             enumerate(sorted(set(predictions)))}  # this kind of labelling is misl
    predictions = [labels_correction_map[x] for x in predictions]

    model_path_to_gt_dict = {}

    with open(source_test_data, 'r') as f:
        test_data_dict = json.load(f)['file_paths']
        model_name_to_gt_label = {model_name: idx for idx, model_name in enumerate(sorted(test_data_dict.keys()))}
        for model_name in test_data_dict.keys():
            for path in test_data_dict[model_name]:
                model_path_to_gt_dict[path] = model_name_to_gt_label[model_name]

    ground_truths = []
    for path in image_paths:
        ground_truths.append(model_path_to_gt_dict[path])

    # fixme: predicted scores is not being passed correctly
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
    num_patches = 100

    Configure.runtime_dir = Path(
        rf'/scratch/p288722/runtime_data/scd_pytorch/18_models_hierarchial_4/fold_{fold_id}')
    Configure.train_data = rf'/data/p288722/dresden/train/nat_patches_18_models_128x128_100/fold_{fold_id}.json'
    Configure.test_data = rf'/data/p288722/dresden/test/nat_patches_18_models_128x128_100/fold_{fold_id}.json'
    Configure.update()

    SetupLogger(log_file=Configure.runtime_dir.joinpath(f'scd_{fold_id}_exp.log'))
    for item in [Configure, SigNet, SimNet]:
        logger.info(f'---------- {item.__name__} ----------')
        for name, value in inspect.getmembers(item)[26:]:
            logger.info(f'{name.ljust(20)}: {value}')
    try:
        # run_cross_validation_flow()
        # run_flow(fold_id)
        run_flow_hierarchial()
    except Exception as e:
        logger.error(e)
        raise e
