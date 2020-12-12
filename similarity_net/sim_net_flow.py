import logging
import pickle
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from configure import Configure, SimNet
from similarity_net.data import Data
from utils.evaluation_metrics import ScoreUtils, BinaryClassificationScores
from utils.logging import log_running_time
from utils.torchsummary import summary
from utils.training_utils import Utils
from utils.visualization_utils import VisualizationUtils

logger = logging.getLogger(__name__)


class SimNetFlow(object):

    @staticmethod
    def train_batch(inputs, targets, threshold=0.5):
        inputs = [x.to(Configure.device) for x in inputs]
        SimNet.optimizer.zero_grad()
        outputs = SimNet.model(inputs)
        loss = SimNet.criterion(outputs, targets)
        predictions = torch.as_tensor(outputs > threshold, dtype=torch.float32).to(Configure.device)
        accuracy = torch.mean(torch.as_tensor(targets == predictions, dtype=torch.float32))
        loss.backward()
        SimNet.optimizer.step()
        return accuracy.item(), loss.item()

    @staticmethod
    @torch.no_grad()
    def test_batch(inputs, targets, threshold=0.5, return_predictions=False):
        inputs = [x.to(Configure.device) for x in inputs]
        outputs = SimNet.model(inputs)
        loss = SimNet.criterion(outputs, targets)
        predictions = torch.as_tensor(outputs > threshold, dtype=torch.float32).to(Configure.device)
        accuracy = torch.mean(torch.as_tensor(targets == predictions, dtype=torch.float32))

        if return_predictions:
            targets = np.ndarray.flatten(targets.cpu().numpy())
            outputs = np.ndarray.flatten(outputs.cpu().numpy())
            return accuracy.item(), loss.item(), targets, outputs
        else:
            return accuracy.item(), loss.item()

    @classmethod
    @log_running_time
    def train(cls):
        # Fix the seeds for the RNGs
        torch.manual_seed(0)

        summary(SimNet.model, (1024, 1024), print_fn=logger.info)
        train_loader = Data.load_data(config_mode='train')
        test_loader = Data.load_data(config_mode='test')

        init_epoch, history, SimNet.model, SimNet.optimizer, SimNet.scheduler = \
            Utils.prepare_for_training(Configure.simnet_dir, SimNet.model, SimNet.optimizer, SimNet.scheduler)
        for epoch in range(init_epoch, SimNet.epochs + 1):
            # Train
            SimNet.model.train()
            train_acc, train_loss = 0, 0
            epoch_start_time = perf_counter()
            for sig_pairs, (sim_scores, _) in train_loader:
                acc, loss = cls.train_batch(sig_pairs, sim_scores.to(Configure.device))
                train_acc += acc
                train_loss += loss

            train_loss = train_loss / len(train_loader)
            train_acc = train_acc / len(train_loader)

            lr = SimNet.scheduler.get_last_lr()
            SimNet.scheduler.step()
            epoch_end_time = perf_counter()

            # Validate
            SimNet.model.eval()
            val_acc, val_loss = 0, 0
            for sig_pairs, (sim_scores, _) in test_loader:
                acc, loss = cls.test_batch(sig_pairs, sim_scores.to(Configure.device))
                val_acc += acc
                val_loss += loss

            val_loss = val_loss / len(test_loader)
            val_acc = val_acc / len(train_loader)

            # Log epoch statistics
            Utils.update_history(history, epoch, train_loss, val_loss, train_acc, val_acc, lr, Configure.simnet_dir)
            VisualizationUtils.plot_learning_statistics(history, Configure.simnet_dir)
            Utils.save_model_on_epoch_end(SimNet.model.state_dict(), SimNet.optimizer.state_dict(),
                                          SimNet.scheduler.state_dict(), history, Configure.simnet_dir)

            logger.info(f"epoch : {epoch}/{SimNet.epochs}, "
                        f"train_loss = {train_loss:.6f}, val_loss = {val_loss:.6f}, "
                        f"train_acc = {train_acc:.3f}, val_acc = {val_acc:.3f}, "
                        f"time = {epoch_end_time - epoch_start_time:.2f} sec")

        Utils.save_best_model(pre_trained_models_dir=Configure.simnet_dir,
                              destination_dir=Configure.runtime_dir,
                              history=history, name=SimNet.name)

    @classmethod
    @log_running_time
    def classify(cls, config_mode='test', pre_trained_model_path=None):
        """
        Method to classify signature pairs
        :param config_mode: string - train / test
        :param pre_trained_model_path: (optional) Pre-trained model path
        :return: list of similarity scores
        """
        if not pre_trained_model_path:
            pre_trained_model_path = Configure.runtime_dir.joinpath('{}.pt'.format(SimNet.name))
        params = torch.load(pre_trained_model_path)
        SimNet.model.load_state_dict(params['model_state_dict'])

        if config_mode == 'train':
            data_loader = Data.load_data(config_mode=config_mode)
            devices_list = [Path(x).name for x in sorted(Path(Configure.train_data).glob('*'))]
        elif config_mode == 'test':
            data_loader = Data.load_data(config_mode=config_mode)
            devices_list = [Path(x).name for x in sorted(Path(Configure.test_data).glob('*'))]
        else:
            raise ValueError('Invalid config_mode')

        num_batches = len(data_loader)
        loss, acc = np.zeros(num_batches), np.zeros(num_batches)
        ground_truths, similarity_scores = [None] * num_batches, [None] * num_batches
        image_paths = []

        for batch_id, (input_signatures, (target_labels, img_paths)) in enumerate(data_loader):
            loss[batch_id], acc[batch_id], ground_truths[batch_id], similarity_scores[batch_id] = cls.test_batch \
                (input_signatures, target_labels.to(Configure.device), return_predictions=True)
            image_paths += list(zip(list(img_paths[0]), list(img_paths[1])))

        logger.info(f'Test loss: {np.mean(loss)}')
        logger.info(f'Test accuracy: {np.mean(acc)}')

        ground_truths, similarity_scores = np.concatenate(ground_truths), np.concatenate(similarity_scores)
        ground_truths, similarity_scores, image_paths = \
            SimNetFlow.patch_to_image(ground_truths, similarity_scores, image_paths)

        np.save(Configure.simnet_dir.joinpath('similarity_scores.npy'), similarity_scores)
        np.save(Configure.simnet_dir.joinpath('ground_truths.npy'), ground_truths)

        VisualizationUtils.plot_roc(ground_truths, similarity_scores, Configure.simnet_dir)
        threshold = VisualizationUtils.plot_scores_with_thresholds(ground_truths, similarity_scores,
                                                                   Configure.simnet_dir)

        logger.info("Setting threshold to {} which results in the maximum F1 score.".format(threshold))
        predictions = np.where(similarity_scores > threshold, 1, 0)
        VisualizationUtils.plot_similarity_scores_distribution(similarity_scores, ground_truths, threshold,
                                                               Configure.simnet_dir)

        device_labels = [(Path(x[0]).parent.name, Path(x[1]).parent.name) for x in image_paths]
        scores = ScoreUtils(source_device_labels=device_labels, predictions=predictions, camera_names=devices_list)
        scores.log_scores()
        with open(str(Configure.simnet_dir.joinpath('scores.pkl')), 'wb+') as f:
            pickle.dump(scores, f)

        BinaryClassificationScores(ground_truths=ground_truths, predictions=predictions).log_scores()

        VisualizationUtils.plot_similarity_matrix(scores.similarity_matrix.astype(np.float),
                                                  Configure.simnet_dir)

        if Configure.compute_model_level_stats:
            logger.info('Computing model level statistics')
            models_list = list(sorted(set([x[:-2] for x in devices_list])))
            model_labels = [(Path(x[0]).parent.name[:-2], Path(x[1]).parent.name[:-2]) for x in image_paths]
            ground_truths = np.array([1.0 if x[0] == x[1] else 0.0 for x in model_labels])

            results_dir = Configure.simnet_dir.joinpath('model_level')
            results_dir.mkdir(exist_ok=True, parents=True)
            np.save(results_dir.joinpath('similarity_scores.npy'), similarity_scores)
            np.save(results_dir.joinpath('ground_truths.npy'), ground_truths)

            VisualizationUtils.plot_roc(ground_truths, similarity_scores, results_dir)
            threshold = VisualizationUtils.plot_scores_with_thresholds(ground_truths, similarity_scores, results_dir)
            logger.info("Setting threshold to {} which results in the maximum F1 score.".format(threshold))

            predictions = np.where(similarity_scores > threshold, 1, 0)
            VisualizationUtils.plot_similarity_scores_distribution(similarity_scores, ground_truths, threshold,
                                                                   results_dir)

            scores = ScoreUtils(source_device_labels=model_labels, predictions=predictions, camera_names=models_list)
            scores.log_scores()
            with open(str(Configure.simnet_dir.joinpath('scores.pkl')), 'wb+') as f:
                pickle.dump(scores, f)

            BinaryClassificationScores(ground_truths=ground_truths, predictions=predictions).log_scores()
            VisualizationUtils.plot_similarity_matrix(scores.similarity_matrix.astype(np.float), results_dir)

    @classmethod
    @log_running_time
    def classify_euclidean(cls, config_mode='test'):
        """
        Method to classify signature pairs
        :param config_mode: string - train / test
        :return: list of similarity scores
        """
        if config_mode == 'train':
            data_loader = Data.load_data(config_mode=config_mode)
            devices_list = [Path(x).name for x in sorted(Path(Configure.train_data).glob('*'))]
        elif config_mode == 'test':
            data_loader = Data.load_data(config_mode=config_mode)
            devices_list = [Path(x).name for x in sorted(Path(Configure.test_data).glob('*'))]
        else:
            raise ValueError('Invalid config_mode')

        num_batches = len(data_loader)
        similarity_scores = [None] * num_batches
        image_paths = []

        for batch_id, (input_signatures, (target_labels, img_paths)) in enumerate(data_loader):
            similarity_scores[batch_id] = -np.linalg.norm(np.array(input_signatures[0] - input_signatures[1]), axis=1)
            image_paths += list(zip(list(img_paths[0]), list(img_paths[1])))

        similarity_scores = np.concatenate(similarity_scores)
        similarity_scores = similarity_scores / min(similarity_scores)

        logger.info('Computing model level statistics')
        models_list = list(sorted(set([x[:-2] for x in devices_list])))
        model_labels = [(Path(x[0]).parent.name[:-2], Path(x[1]).parent.name[:-2]) for x in image_paths]
        ground_truths = np.array([1.0 if x[0] == x[1] else 0.0 for x in model_labels])
        ground_truths, similarity_scores, image_paths = \
            SimNetFlow.patch_to_image(ground_truths, similarity_scores, image_paths)

        results_dir = Configure.simnet_dir.joinpath('euclidean_distance')
        results_dir.mkdir(exist_ok=True, parents=True)
        np.save(results_dir.joinpath('similarity_scores.npy'), similarity_scores)
        np.save(results_dir.joinpath('ground_truths.npy'), ground_truths)

        VisualizationUtils.plot_roc(ground_truths, similarity_scores, results_dir)
        threshold = VisualizationUtils.plot_scores_with_thresholds(ground_truths, similarity_scores, results_dir)
        logger.info("Setting threshold to {} which results in the maximum F1 score.".format(threshold))

        predictions = np.where(similarity_scores > threshold, 1, 0)
        VisualizationUtils.plot_similarity_scores_distribution(similarity_scores, ground_truths, threshold, results_dir)

        scores = ScoreUtils(source_device_labels=model_labels, predictions=predictions, camera_names=models_list)
        scores.log_scores()
        BinaryClassificationScores(ground_truths=ground_truths, predictions=predictions).log_scores()
        VisualizationUtils.plot_similarity_matrix(scores.similarity_matrix.astype(np.float), results_dir)

    @classmethod
    @log_running_time
    def classify_t_tests(cls, config_mode='test'):
        """
        Method to classify signature pairs using student t-distribution of image patches from two different images.
        Note that this method works only for image patches and not for full images. Each distribution should have at
        least around 45 samples for the test to be significant, this translates to at least 10 patches per image.
        :param config_mode: string - train / test
        :return: list of similarity scores
        """
        raise NotImplementedError('In development, check future commits for completed code')
        if config_mode == 'train':
            data_loader = Data.load_data(config_mode=config_mode)
            devices_list = [Path(x).name for x in sorted(Path(Configure.train_data).glob('*'))]
        elif config_mode == 'test':
            data_loader = Data.load_data(config_mode=config_mode)
            devices_list = [Path(x).name for x in sorted(Path(Configure.test_data).glob('*'))]
        else:
            raise ValueError('Invalid config_mode')

        num_batches = len(data_loader)
        loss, acc = np.zeros(num_batches), np.zeros(num_batches)
        ground_truths, similarity_scores = [None] * num_batches, [None] * num_batches
        image_paths = []

        for batch_id, (input_signatures, (target_labels, img_paths)) in enumerate(data_loader):
            similarity_scores[batch_id] = -np.linalg.norm(np.array(input_signatures[0] - input_signatures[1]), axis=1)
            image_paths += list(zip(list(img_paths[0]), list(img_paths[1])))

        logger.info(f'Test loss: {np.mean(loss)}')
        logger.info(f'Test accuracy: {np.mean(acc)}')

        similarity_scores = np.concatenate(similarity_scores)
        similarity_scores = similarity_scores / min(similarity_scores)

        logger.info('Computing model level statistics')
        models_list = list(sorted(set([x[:-2] for x in devices_list])))
        model_labels = [(Path(x[0]).parent.name[:-2], Path(x[1]).parent.name[:-2]) for x in image_paths]
        ground_truths = np.array([1.0 if x[0] == x[1] else 0.0 for x in model_labels])

        results_dir = Configure.simnet_dir.joinpath('euclidean_distance')
        results_dir.mkdir(exist_ok=True, parents=True)
        np.save(results_dir.joinpath('similarity_scores.npy'), similarity_scores)
        np.save(results_dir.joinpath('ground_truths.npy'), ground_truths)

        VisualizationUtils.plot_roc(ground_truths, similarity_scores, results_dir)
        threshold = VisualizationUtils.plot_scores_with_thresholds(ground_truths, similarity_scores, results_dir)
        logger.info("Setting threshold to {} which results in the maximum F1 score.".format(threshold))

        predictions = np.where(similarity_scores > threshold, 1, 0)
        VisualizationUtils.plot_similarity_scores_distribution(similarity_scores, ground_truths, threshold, results_dir)

        scores = ScoreUtils(source_device_labels=model_labels, predictions=predictions, camera_names=models_list)
        scores.log_scores()
        BinaryClassificationScores(ground_truths=ground_truths, predictions=predictions).log_scores()
        VisualizationUtils.plot_similarity_matrix(scores.similarity_matrix.astype(np.float), results_dir)

    @classmethod
    def patch_to_image(cls, ground_truths, similarity_scores, image_paths, reduction_method='avg_sim_score'):
        """
        Convert the patch level predictions to image level predictions.
        :param ground_truths:
        :param similarity_scores:
        :param image_paths:
        :param reduction_method: a string with values 'avg_sim_score',
        :return: None
        """

        from collections import Counter

        patches_per_image_dict = {}
        for patch_id, gt, pr in zip(image_paths, ground_truths, similarity_scores):
            image_pair = tuple(['_'.join((Path(x).parent.name + '/' + Path(x).name).split('_')[:-1]) for x in patch_id])
            if image_pair not in patches_per_image_dict:
                patches_per_image_dict[image_pair] = {'gt': [gt], 'pr': [pr]}
            else:
                patches_per_image_dict[image_pair]['gt'].append(gt)
                patches_per_image_dict[image_pair]['pr'].append(pr)

        ground_truths, similarity_scores, image_paths = [], [], []
        if reduction_method == 'avg_sim_score':
            for idx, image_pair in enumerate(patches_per_image_dict):
                ground_truths.append(Counter(patches_per_image_dict[image_pair]['gt']).most_common(1)[0][0])
                similarity_scores.append(np.mean(patches_per_image_dict[image_pair]['pr']))
                image_paths.append(image_pair)
        else:
            raise ValueError('Invalid reduction method')

        return np.array(ground_truths), np.array(similarity_scores), image_paths


if __name__ == '__main__':
    SimNetFlow.train()
