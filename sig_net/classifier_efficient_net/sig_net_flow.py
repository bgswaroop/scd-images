import json
import logging
import pickle
from collections import namedtuple
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch.cuda.amp import autocast

from configure import Configure, SigNet
from sig_net.classifier_efficient_net.data_fft import Data
from utils.evaluation_metrics import MultinomialClassificationScores
from utils.logging import log_running_time
from utils.training_utils import TrainingUtils
from utils.visualization_utils import VisualizationUtils

logger = logging.getLogger(__name__)

Stats = namedtuple('Stats', ('loss', 'acc', 'gt', 'pred', 'pred_scores'), defaults=(None, None, None, None, None))


class SigNetFlow(object):

    @staticmethod
    def train_batch(inputs, targets):

        with autocast():
            SigNet.optimizer.zero_grad()
            outputs = SigNet.model(inputs)
            loss = SigNet.criterion(outputs, targets)

        loss.backward()
        SigNet.optimizer.step()

        predictions = torch.max(outputs, dim=1).indices
        ground_truths = torch.max(targets, dim=1).indices
        accuracy = torch.mean(torch.as_tensor(ground_truths == predictions, dtype=torch.float32))

        stats = Stats(loss=loss.item(), acc=accuracy.item())
        return stats

    @staticmethod
    @torch.no_grad()
    def test_batch(inputs, targets):
        with autocast():
            outputs = SigNet.model(inputs)
            loss = SigNet.criterion(outputs, targets)

        predictions = torch.max(outputs, dim=1).indices
        prediction_scores = torch.max(outputs, dim=1).values
        ground_truths = torch.max(targets, dim=1).indices
        accuracy = torch.mean(torch.as_tensor(ground_truths == predictions, dtype=torch.float32))

        stats = Stats(loss=loss.item(), acc=accuracy.item(), gt=ground_truths.cpu().numpy(),
                      pred=predictions.cpu().numpy(), pred_scores=prediction_scores.cpu().numpy())
        return stats

    @classmethod
    @log_running_time
    def train(cls):
        # Fix the seeds for the RNGs
        torch.manual_seed(0)

        # Prepare the data
        train_loader = Data.load_data(config_file=Configure.train_data_config, config_mode='train',
                                      dataset=Configure.dataset_folder)
        test_loader = Data.load_data(config_file=Configure.test_data_config, config_mode='test',
                                     dataset=Configure.dataset_folder)

        with open(Configure.train_data_config, 'r') as f:
            dataset_dict = json.load(f)
            devices_list = list(sorted([x for x in dataset_dict['file_paths']]))
        models_list = list(sorted(set([x[:-2] for x in devices_list])))
        device_to_model_map = {idx: models_list.index(x[:-2]) for idx, x in enumerate(devices_list)}

        init_epoch, history, SigNet.model, SigNet.optimizer, SigNet.scheduler = TrainingUtils.prepare_for_training(
            Configure.signet_dir, SigNet.model, SigNet.optimizer, SigNet.scheduler)

        for epoch in range(init_epoch, SigNet.max_epochs + 1):

            # Early stopping
            stop_early, stop_early_message = TrainingUtils.early_stopping(patience=5, metric=history['val_loss'],
                                                                          delta=0.02)
            if stop_early:
                logger.info(f'Stopping the training early at epoch {epoch + 1} as the loss {stop_early_message}')
                break

            # Train
            SigNet.model.train()
            num_batches = len(train_loader)
            loss, acc = np.zeros(num_batches), np.zeros(num_batches)

            epoch_start_time = perf_counter()
            for i, (images, (labels, img_paths, img_std_dev)) in enumerate(train_loader):
                stats = cls.train_batch(images.to(Configure.device), labels.to(Configure.device))
                loss[i], acc[i] = stats.loss, stats.acc
                if i % 100 == 0:
                    logger.info(f'Running batch : {i}')

            train_acc, train_loss = np.mean(acc), np.mean(loss)
            lr = SigNet.scheduler.get_last_lr()
            SigNet.scheduler.step()
            epoch_end_time = perf_counter()

            # Validate
            logger.info(f'Running validate')
            SigNet.model.eval()
            num_batches = len(test_loader)
            loss, acc = np.zeros(num_batches), np.zeros(num_batches)
            logger.info(f'Running validate job')
            for i, (images, (labels, img_paths, img_std_dev)) in enumerate(test_loader):
                stats = cls.test_batch(images.to(Configure.device), labels.to(Configure.device),
                                       device_to_model_map=device_to_model_map)
                loss[i], acc[i] = stats.loss, stats.acc

            val_acc, val_loss = np.mean(acc), np.mean(loss)

            # Log epoch statistics
            logger.info(f'log stats')
            TrainingUtils.update_history(history, epoch, train_loss, val_loss, train_acc, val_acc, lr,
                                         Configure.signet_dir)
            logger.info(f'plot lr')
            VisualizationUtils.plot_learning_statistics(history, Configure.signet_dir)
            logger.info(f'save')
            TrainingUtils.save_model_on_epoch_end(SigNet.model.state_dict(), SigNet.optimizer.state_dict(),
                                                  SigNet.scheduler.state_dict(), history, Configure.signet_dir)

            logger.info(f"epoch : {epoch}/{SigNet.max_epochs}, "
                        f"train_loss = {train_loss:.6f}, val_loss = {val_loss:.6f}, "
                        f"train_acc = {train_acc:.3f}, val_acc = {val_acc:.3f}, "
                        f"time = {epoch_end_time - epoch_start_time:.2f} sec")

        TrainingUtils.save_best_model(pre_trained_models_dir=Configure.signet_dir,
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
        params = torch.load(pre_trained_model_path, map_location=Configure.device)
        SigNet.model.load_state_dict(params['model_state_dict'])

        if config_mode == 'train' and not images_dir:
            data_loader = Data.load_data(config_file=Configure.train_data_config, config_mode=config_mode,
                                         dataset=Configure.dataset_folder)
        elif config_mode == 'test' and not images_dir:
            data_loader = Data.load_data(config_file=Configure.test_data_config, config_mode=config_mode,
                                         dataset=Configure.dataset_folder)
        elif images_dir:
            data_loader = Data.load_data(config_file=images_dir, config_mode=config_mode)
        else:
            raise ValueError('Invalid config_mode')

        SigNet.model.eval()
        signatures = []
        for images, (labels, img_paths, img_std_dev) in data_loader:
            images = images.to(Configure.device)
            features = SigNet.model.extract_features(images).to(torch.device("cpu")).detach()
            signatures += list(zip(features, img_paths))

        logger.info(f'Number of extracted signatures: {len(signatures)}')
        return signatures

    @classmethod
    @log_running_time
    def classify(cls, aggregation_method=None, config_mode='test', pre_trained_model_path=None):
        """
        Method to extract signatures and labels
        :param aggregation_method:
        :param config_mode: string - train / test
        :param pre_trained_model_path: (optional) Pre-trained model path
        :return: list of labelled signatures
        """
        if not pre_trained_model_path:
            pre_trained_model_path = Configure.runtime_dir.joinpath('{}.pt'.format(SigNet.name))
        params = torch.load(pre_trained_model_path, map_location=Configure.device)
        SigNet.model.load_state_dict(params['model_state_dict'])
        SigNet.model.eval()

        data_loader = Data.load_data(config_file=Configure.train_data_config, config_mode=config_mode,
                                     dataset=Configure.dataset_folder)

        with open(Configure.train_data_config, 'r') as f:
            dataset_dict = json.load(f)
            devices_list = list(sorted([x for x in dataset_dict['file_paths']]))
        models_list = list(sorted(set([x[:-2] for x in devices_list])))
        device_to_model_map = {idx: models_list.index(x[:-2]) for idx, x in enumerate(devices_list)}

        num_batches = len(data_loader)
        loss, acc = np.zeros(num_batches), np.zeros(num_batches)
        ground_truths, predictions, pred_scores, std_devs = \
            [None] * num_batches, [None] * num_batches, [None] * num_batches, [None] * num_batches
        image_paths = []

        for idx, (images, (labels, img_paths, img_std_dev)) in enumerate(data_loader):
            stats = cls.test_batch(images.to(Configure.device), labels.to(Configure.device),
                                   device_to_model_map)
            loss[idx], acc[idx], ground_truths[idx], predictions[idx], pred_scores[idx], std_devs[idx] = \
                stats.loss, stats.acc, stats.gt, stats.pred, stats.pred_scores, img_std_dev
            image_paths += img_paths

        logger.info(f'Test loss: {np.mean(loss)}')
        logger.info(f'Test accuracy: {np.mean(acc)}')

        ground_truths = np.concatenate(ground_truths)
        predictions = np.concatenate(predictions)
        pred_scores = np.concatenate(pred_scores)
        std_devs = np.concatenate(std_devs)

        logger.info(f'Converting patch predictions to image predictions using {aggregation_method}')
        aggregation = SigNetFlow.patch_to_image(ground_truths=ground_truths, predictions=predictions,
                                                image_paths=image_paths, aggregation_method=aggregation_method,
                                                pred_scores=pred_scores, std_devs=std_devs)
        ground_truths, predictions = aggregation.image_ground_truths, aggregation.image_predictions

        scores = MultinomialClassificationScores(ground_truths, predictions, one_hot=False, camera_names=models_list)
        scores.log_scores()
        with open(str(Configure.signet_dir.joinpath('scores.pkl')), 'wb+') as f:
            pickle.dump(scores, f)
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

            ground_truths = [device_to_model_map[x] for x in ground_truths]
            predictions = [device_to_model_map[x] for x in predictions]
            # ground_truths, predictions = SigNetFlow.patch_to_image(ground_truths, predictions, image_paths)

            scores = MultinomialClassificationScores(ground_truths, predictions, False, models_list)
            scores.log_scores()
            with open(str(Configure.signet_dir.joinpath('scores.pkl')), 'wb+') as f:
                pickle.dump(scores, f)
            VisualizationUtils.plot_confusion_matrix(ground_truths, predictions, one_hot=False,
                                                     save_to_dir=results_dir)

    @classmethod
    @log_running_time
    @torch.no_grad()
    def predict(cls, prediction_data, pre_trained_model_path=None):
        """
        Method to extract signatures and labels
        :param prediction_data:
        :param pre_trained_model_path: (optional) Pre-trained model path
        :return: list of labelled signatures
        """
        if not pre_trained_model_path:
            pre_trained_model_path = Configure.runtime_dir.joinpath(f'{SigNet.name}.pt')
        params = torch.load(pre_trained_model_path, map_location=Configure.device)
        SigNet.model.load_state_dict(params['model_state_dict'])
        SigNet.model.eval()
        data_loader = Data.load_data(config_file=prediction_data, config_mode='test', dataset=Configure.dataset_folder)

        num_batches = len(data_loader)
        predictions, pred_scores, std_devs = [None] * num_batches, [None] * num_batches, [None] * num_batches
        image_paths = []

        for batch_id, (images, (labels, img_paths, img_std_dev)) in enumerate(data_loader):
            outputs = SigNet.model(images.to(Configure.device)).to(torch.device("cpu"))
            predictions[batch_id] = torch.max(outputs, dim=1).indices
            pred_scores[batch_id] = torch.max(outputs, dim=1).values
            std_devs[batch_id] = img_std_dev
            image_paths += img_paths

        predictions = np.concatenate(predictions)
        pred_scores = np.concatenate(pred_scores)
        std_devs = np.concatenate(std_devs)
        # _, predictions = SigNetFlow.patch_relabelling_by_aggregation(predictions, predictions, image_paths)

        return image_paths, predictions, pred_scores, std_devs

    # @staticmethod
    # def patch_relabelling_by_aggregation(ground_truths, predictions, image_paths, aggregation_method='majority_vote'):
    #     """
    #     Convert the patch level predictions to image level predictions.
    #     :param ground_truths:
    #     :param predictions:
    #     :param image_paths:
    #     :param aggregation_method: a string with values 'majority_vote'
    #     :return: None
    #     """
    #
    #     from collections import Counter
    #
    #     # Combine all the patches from an image and make a dictionary
    #     patches_per_image_dict = {}
    #     for patch_id, gt, pr in zip(image_paths, ground_truths, predictions):
    #         patch_id = Path(patch_id)
    #         image_id = '_'.join((patch_id.parent.name + '/' + patch_id.name).split('_')[:-1])
    #         if image_id not in patches_per_image_dict:
    #             patches_per_image_dict[image_id] = {'gt': [gt], 'pr': [pr]}
    #         else:
    #             patches_per_image_dict[image_id]['gt'].append(gt)
    #             patches_per_image_dict[image_id]['pr'].append(pr)
    #
    #     # Perform label aggregation, and save the result back in the dictionary
    #     if aggregation_method == 'majority_vote':
    #         majority_vote = lambda x: Counter(x).most_common(1)[0][0]
    #         for idx, image_id in enumerate(patches_per_image_dict):
    #             patches_per_image_dict[image_id]['gt'] = majority_vote(patches_per_image_dict[image_id]['gt'])
    #             patches_per_image_dict[image_id]['pr'] = majority_vote(patches_per_image_dict[image_id]['pr'])
    #     else:
    #         raise ValueError('Invalid aggregation_method')
    #
    #     # Relabel the patches with the aggregated labels
    #     for idx, (patch_id, gt, pr) in enumerate(zip(image_paths, ground_truths, predictions)):
    #         patch_id = Path(patch_id)
    #         image_id = '_'.join((patch_id.parent.name + '/' + patch_id.name).split('_')[:-1])
    #         new_label = patches_per_image_dict[image_id]
    #         ground_truths[idx] = new_label['gt']
    #         predictions[idx] = new_label['pr']
    #
    #     return np.array(ground_truths), np.array(predictions)

    @staticmethod
    def patch_to_image(ground_truths, predictions, image_paths, aggregation_method, pred_scores=None,
                       std_devs=None):
        """
        Convert the patch level predictions to image level predictions.
        :param std_devs:
        :param pred_scores:
        :param ground_truths:
        :param predictions:
        :param image_paths:
        :param aggregation_method:
        :return: None
        """

        from collections import Counter, namedtuple
        Aggregation = namedtuple('Aggregation', 'image_ground_truths, image_predictions, patch_contribution')

        if pred_scores is None:
            pred_scores = np.ones_like(predictions)
        if std_devs is None:
            std_devs = np.ones_like(predictions) * 0.5

        patches_per_image = {}
        for idx, gt, pr, pr_s, sd in zip(image_paths, ground_truths, predictions, pred_scores, std_devs):
            patch_id = Path(idx)
            image_id = '_'.join((patch_id.parent.name + '/' + patch_id.name).split('_')[:-1])
            if image_id not in patches_per_image:
                patches_per_image[image_id] = {'gt': [gt], 'pr': [pr], 'pr_s': [pr_s], 'patch_id': [idx], 'sd': [sd],
                                               'image_prediction': None}
            else:
                patches_per_image[image_id]['gt'].append(gt)
                patches_per_image[image_id]['pr'].append(pr)
                patches_per_image[image_id]['pr_s'].append(pr_s)
                patches_per_image[image_id]['patch_id'].append(idx)
                patches_per_image[image_id]['sd'].append(sd)

        ground_truths, predictions = [], []
        if aggregation_method == 'majority_vote':
            for idx, image_id in enumerate(patches_per_image):
                ground_truths.append(Counter(patches_per_image[image_id]['gt']).most_common(1)[0][0])
                predictions.append(Counter(patches_per_image[image_id]['pr']).most_common(1)[0][0])
                patches_per_image[image_id]['image_prediction'] = predictions[-1]

        elif aggregation_method == 'prediction_score_sum':
            for idx, image_id in enumerate(patches_per_image):
                ground_truths.append(Counter(patches_per_image[image_id]['gt']).most_common(1)[0][0])

                image_prediction = {}
                for pr, pr_s in zip(patches_per_image[image_id]['pr'], patches_per_image[image_id]['pr_s']):
                    if pr not in image_prediction:
                        image_prediction[pr] = pr_s
                    else:
                        image_prediction[pr] += pr_s
                predictions.append(max(image_prediction, key=lambda key: image_prediction[key]))
                patches_per_image[image_id]['image_prediction'] = predictions[-1]

        elif aggregation_method == 'log_scaled_prediction_score_sum':
            for idx, image_id in enumerate(patches_per_image):
                ground_truths.append(Counter(patches_per_image[image_id]['gt']).most_common(1)[0][0])

                image_prediction = {}
                for pr, pr_s in zip(patches_per_image[image_id]['pr'], patches_per_image[image_id]['pr_s']):
                    if pr not in image_prediction:
                        image_prediction[pr] = -1.0 / np.log(pr_s)
                    else:
                        image_prediction[pr] += -1.0 / np.log(pr_s)
                predictions.append(max(image_prediction, key=lambda key: image_prediction[key]))
                patches_per_image[image_id]['image_prediction'] = predictions[-1]

        elif aggregation_method == 'log_scaled_std_dev':
            for idx, image_id in enumerate(patches_per_image):
                ground_truths.append(Counter(patches_per_image[image_id]['gt']).most_common(1)[0][0])

                image_prediction = {}
                for pr, sd in zip(patches_per_image[image_id]['pr'], patches_per_image[image_id]['sd']):
                    if pr not in image_prediction:
                        image_prediction[pr] = -np.log(np.sum(sd))
                    else:
                        image_prediction[pr] += -np.log(np.sum(sd))
                predictions.append(max(image_prediction, key=lambda key: image_prediction[key]))
                patches_per_image[image_id]['image_prediction'] = predictions[-1]

        else:
            raise ValueError('Invalid aggregation_method')

        # Update Patch Contributions
        patch_contribution = {}
        for image_id in patches_per_image:
            image_prediction = patches_per_image[image_id]['image_prediction']
            for pr, patch_id in zip(patches_per_image[image_id]['pr'], patches_per_image[image_id]['patch_id']):
                patch_contribution[patch_id] = True if pr == image_prediction else False
        patch_contribution = [patch_contribution[x] for x in image_paths]

        return Aggregation(image_ground_truths=np.array(ground_truths),
                           image_predictions=np.array(predictions),
                           patch_contribution=np.array(patch_contribution))


if __name__ == '__main__':
    from sig_net.auto_encoder.utils import summary

    summary(SigNet.model, (3, 320, 480), logger.info)
    # SigNetFlow.extract_signatures(config_mode='train')
    # ae_predictions_train, ae_predictions_test = VisualizationUtils.visualize_ae_input_output_pairs()
