import logging
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from configure import Configure, SigNet
from signature_net.data_rgb import Data
from utils.evaluation_metrics import MultinomialClassificationScores
from utils.logging import log_running_time
from utils.training_utils import Utils
from utils.visualization_utils import VisualizationUtils

logger = logging.getLogger(__name__)


class SigNetFlow(object):

    @staticmethod
    def train_batch(inputs, targets):
        SigNet.optimizer.zero_grad()
        outputs = SigNet.model(inputs)
        loss = SigNet.criterion(outputs, targets)
        loss.backward()
        SigNet.optimizer.step()

        predictions = torch.max(outputs, dim=1).indices
        ground_truths = torch.max(targets, dim=1).indices
        accuracy = torch.mean(torch.as_tensor(ground_truths == predictions, dtype=torch.float32))

        return loss.item(), accuracy.item()

    @staticmethod
    @torch.no_grad()
    def test_batch(inputs, targets, return_predictions=False):
        outputs = SigNet.model(inputs)
        loss = SigNet.criterion(outputs, targets)

        predictions = torch.max(outputs, dim=1).indices
        ground_truths = torch.max(targets, dim=1).indices
        accuracy = torch.mean(torch.as_tensor(ground_truths == predictions, dtype=torch.float32))

        if return_predictions:
            return loss.item(), accuracy.item(), ground_truths.cpu().numpy(), predictions.cpu().numpy()
        else:
            return loss.item(), accuracy.item()

    @classmethod
    @log_running_time
    def train(cls):
        # Prepare the data
        train_loader = Data.load_data(dataset=Configure.train_data, config_mode='train')
        test_loader = Data.load_data(dataset=Configure.test_data, config_mode='test')

        init_epoch, history, model = Utils.prepare_for_training(Configure.signet_dir, SigNet.model)
        for epoch in range(init_epoch, SigNet.epochs + 1):
            # Train
            SigNet.model.train()
            num_batches = len(train_loader)
            loss, acc = np.zeros(num_batches), np.zeros(num_batches)

            epoch_start_time = perf_counter()
            for i, (input_images, (target_labels, _)) in enumerate(train_loader):
                loss[i], acc[i] = cls.train_batch(input_images.to(Configure.device), target_labels.to(Configure.device))

            train_acc, train_loss = np.mean(acc), np.mean(loss)
            lr = SigNet.scheduler.get_last_lr()
            SigNet.scheduler.step()
            epoch_end_time = perf_counter()

            # Validate
            SigNet.model.eval()
            num_batches = len(test_loader)
            loss, acc = np.zeros(num_batches), np.zeros(num_batches)
            for i, (input_images, (target_labels, _)) in enumerate(test_loader):
                loss[i], acc[i] = cls.test_batch(input_images.to(Configure.device), target_labels.to(Configure.device))

            val_acc, val_loss = np.mean(acc), np.mean(loss)

            # Log epoch statistics
            Utils.update_history(history, epoch, train_loss, val_loss, train_acc, val_acc, lr, Configure.signet_dir)
            VisualizationUtils.plot_learning_statistics(history, Configure.signet_dir)
            Utils.save_model_on_epoch_end(SigNet.model, history, Configure.signet_dir)

            logger.info(f"epoch : {epoch}/{SigNet.epochs}, "
                        f"train_loss = {train_loss:.6f}, val_loss = {val_loss:.6f}, "
                        f"train_acc = {train_acc:.3f}, val_acc = {val_acc:.3f}, "
                        f"time = {epoch_end_time - epoch_start_time:.2f} sec")

        Utils.save_best_model(pre_trained_models_dir=Configure.signet_dir,
                              destination_dir=Configure.runtime_dir,
                              history=history, name=SigNet.name)

    @classmethod
    @log_running_time
    def extract_signatures(cls, config_mode, images_dir=None, pre_trained_model_path=None):
        """
        Method to extract signatures and labels
        :param pre_trained_model_path: (optional) Pre-trained model path
        :param images_dir: (optional) Directory path containing images
        :param config_mode: string - train / test
        :return: list of labelled signatures
        """

        if not pre_trained_model_path:
            pre_trained_model_path = Configure.runtime_dir.joinpath('{}.pt'.format(SigNet.name))
        SigNet.model = torch.load(pre_trained_model_path)

        if config_mode == 'train' and not images_dir:
            data_loader = Data.load_data(dataset=Configure.train_data, config_mode=config_mode)
        elif config_mode == 'test' and not images_dir:
            data_loader = Data.load_data(dataset=Configure.test_data, config_mode=config_mode)
        elif images_dir:
            data_loader = Data.load_data(dataset=images_dir, config_mode=config_mode)
        else:
            raise ValueError('Invalid config_mode')

        SigNet.model.eval()
        signatures = []
        for input_images, (_, input_img_paths) in data_loader:
            input_images = input_images.to(Configure.device)
            features = SigNet.model.extract_features(input_images).to(torch.device("cpu")).detach()
            signatures += list(zip(features, input_img_paths))

        logger.info(f'Number of extracted signatures: {len(signatures)}')
        return signatures

    @classmethod
    @log_running_time
    def classify(cls, config_mode='test', pre_trained_model_path=None):
        """
        Method to extract signatures and labels
        :param config_mode: string - train / test
        :param pre_trained_model_path: (optional) Pre-trained model path
        :return: list of labelled signatures
        """
        if not pre_trained_model_path:
            pre_trained_model_path = Configure.runtime_dir.joinpath('{}.pt'.format(SigNet.name))
        SigNet.model = torch.load(pre_trained_model_path)

        if config_mode == 'train':
            data_loader = Data.load_data(dataset=Configure.train_data, config_mode=config_mode)
            devices_list = list(sorted([x.name for x in Path(Configure.train_data).glob('*')]))
        elif config_mode == 'test':
            data_loader = Data.load_data(dataset=Configure.test_data, config_mode=config_mode)
            devices_list = list(sorted([x.name for x in Path(Configure.test_data).glob('*')]))
        else:
            raise ValueError('Invalid config_mode')

        num_batches = len(data_loader)
        loss, acc = np.zeros(num_batches), np.zeros(num_batches)
        ground_truths, predictions = [None] * num_batches, [None] * num_batches
        image_paths = []

        for batch_id, (input_images, (target_labels, img_paths)) in enumerate(data_loader):
            loss[batch_id], acc[batch_id], ground_truths[batch_id], predictions[batch_id] = \
                cls.test_batch(input_images.to(Configure.device), target_labels.to(Configure.device),
                               return_predictions=True)
            image_paths += img_paths

        logger.info(f'Test loss: {np.mean(loss)}')
        logger.info(f'Test accuracy: {np.mean(acc)}')

        ground_truths, predictions = np.concatenate(ground_truths), np.concatenate(predictions)
        ground_truths, predictions = SigNetFlow.patch_to_image(ground_truths, predictions, image_paths)

        MultinomialClassificationScores(ground_truths, predictions, one_hot=False,
                                        camera_names=devices_list).log_scores()
        VisualizationUtils.plot_confusion_matrix(ground_truths, predictions,
                                                 one_hot=False, save_to_dir=Configure.signet_dir)

        if Configure.compute_model_level_stats:
            # Compute model level scores
            # Combine device-wise predictions to model-level predictions
            # Warning: The following code assumes that the device folder name is as: <camera_model>_<device_index>
            # Note that the last two characters of device folder name must uniquely identify the device for a specific
            # camera model
            logger.info('Computing model level statistics')
            results_dir = Configure.signet_dir.joinpath('model_level')
            results_dir.mkdir(exist_ok=True, parents=True)
            models_list = list(sorted(set([x[:-2] for x in devices_list])))
            device_to_model_map = {idx: models_list.index(x[:-2]) for idx, x in enumerate(devices_list)}

            ground_truths = [device_to_model_map[x] for x in ground_truths]
            predictions = [device_to_model_map[x] for x in predictions]
            ground_truths, predictions = SigNetFlow.patch_to_image(ground_truths, predictions, image_paths)

            MultinomialClassificationScores(ground_truths, predictions, one_hot=False,
                                            camera_names=devices_list).log_scores()
            VisualizationUtils.plot_confusion_matrix(ground_truths, predictions, one_hot=False,
                                                     save_to_dir=results_dir)

    @staticmethod
    def patch_to_image(ground_truths, predictions, image_paths, aggregation_method='majority_vote'):
        """
        Convert the patch level predictions to image level predictions.
        :param ground_truths:
        :param predictions:
        :param image_paths:
        :param aggregation_method: a string with values 'majority_vote'
        :return: None
        """

        from collections import Counter

        patches_per_image_dict = {}
        for patch_id, gt, pr in zip(image_paths, ground_truths, predictions):
            patch_id = Path(patch_id)
            image_id = '_'.join((patch_id.parent.name + '/' + patch_id.name).split('_')[:-1])
            if image_id not in patches_per_image_dict:
                patches_per_image_dict[image_id] = {'gt': [gt], 'pr': [pr]}
            else:
                patches_per_image_dict[image_id]['gt'].append(gt)
                patches_per_image_dict[image_id]['pr'].append(pr)

        ground_truths, predictions = [], []
        if aggregation_method == 'majority_vote':
            for idx, image_id in enumerate(patches_per_image_dict):
                ground_truths.append(Counter(patches_per_image_dict[image_id]['gt']).most_common(1)[0][0])
                predictions.append(Counter(patches_per_image_dict[image_id]['pr']).most_common(1)[0][0])
        else:
            raise ValueError('Invalid aggregation_method')

        return np.array(ground_truths), np.array(predictions)


if __name__ == '__main__':
    from utils.torchsummary import summary

    summary(SigNet.model, (3, 320, 480), logger.info)
    # SigNetFlow.extract_signatures(config_mode='train')
    # ae_predictions_train, ae_predictions_test = VisualizationUtils.visualize_ae_input_output_pairs()
