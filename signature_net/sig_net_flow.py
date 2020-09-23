import logging
import time

import torch

from configure import Configure, SigNet
from signature_net.data_rgb import Data
from utils.logging import log_running_time
from utils.training_utils import Utils
from utils.visualization_utils import VisualizationUtils

logger = logging.getLogger(__name__)


class SigNetFlow(object):

    @staticmethod
    def train_batch(inputs, expected_outputs):
        inputs = inputs.to(Configure.device)
        SigNet.optimizer.zero_grad()
        outputs = SigNet.model(inputs)
        loss = SigNet.criterion(outputs, expected_outputs.to(Configure.device))
        loss.backward()
        SigNet.optimizer.step()

        predictions = torch.max(outputs, dim=1).indices
        target = torch.max(expected_outputs.to(Configure.device), dim=1).indices
        accuracy = torch.sum(target == predictions) / torch.tensor(target.shape[0], dtype=torch.float32)

        return loss.item(), accuracy.item()

    @staticmethod
    @torch.no_grad()
    def test_batch(inputs, expected_outputs):
        inputs = inputs.to(Configure.device)
        outputs = SigNet.model(inputs)
        loss = SigNet.criterion(outputs, expected_outputs.to(Configure.device))

        predictions = torch.max(outputs, dim=1).indices
        target = torch.max(expected_outputs.to(Configure.device), dim=1).indices
        accuracy = torch.sum(target == predictions) / torch.tensor(target.shape[0], dtype=torch.float32)

        return loss.item(), accuracy.item()

    @classmethod
    @log_running_time
    def train(cls):
        # Prepare the data
        train_loader = Data.load_data(dataset=Configure.train_data, config_mode='train')
        test_loader = Data.load_data(dataset=Configure.test_data, config_mode='test')

        init_epoch, SigNet.model, history = Utils.get_initial_epoch(model=SigNet.model,
                                                                    pre_trained_models_dir=Configure.signet_dir)
        for epoch in range(init_epoch, SigNet.epochs + 1):

            # Train
            SigNet.model.train()
            train_loss, train_acc = 0, 0
            epoch_start_time = time.perf_counter()
            for input_images, (target_labels, _) in train_loader:
                input_images = input_images.to(Configure.device)
                loss, acc = cls.train_batch(inputs=input_images, expected_outputs=target_labels)
                train_loss += loss
                train_acc += acc

            train_loss = train_loss / len(train_loader)
            train_acc = train_acc / len(train_loader)

            lr = SigNet.scheduler.get_last_lr()
            SigNet.scheduler.step()
            epoch_end_time = time.perf_counter()

            # Validate
            SigNet.model.eval()
            val_loss, val_acc = 0, 0
            for input_images, (target_labels, _) in test_loader:
                input_images = input_images.to(Configure.device)
                loss, acc = cls.test_batch(inputs=input_images, expected_outputs=target_labels)
                val_loss += loss
                val_acc += acc

            val_loss = val_loss / len(test_loader)
            val_acc = val_acc / len(test_loader)

            # Log epoch statistics
            Utils.update_history(history, epoch, train_loss, val_loss, train_acc, val_acc, lr, Configure.signet_dir)
            VisualizationUtils.plot_learning_statistics(history, Configure.signet_dir)
            Utils.save_model_on_epoch_end(SigNet.model, history, Configure.signet_dir)

            logger.info("epoch : {}/{}, train_loss = {:.6f}, val_loss = {:.6f}, time = {:.2f} sec".format(
                epoch, SigNet.epochs, train_loss, val_loss, epoch_end_time - epoch_start_time))

        Utils.save_best_model(pre_trained_models_dir=Configure.signet_dir,
                              destination_dir=Configure.runtime_dir,
                              history=history, name=SigNet.name)

    @classmethod
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
        return signatures


if __name__ == '__main__':
    from utils.torchsummary import summary
    summary(SigNet.model, (3, 320, 480), logger.info)
    # SigNetFlow.extract_signatures(config_mode='train')
    # ae_predictions_train, ae_predictions_test = VisualizationUtils.visualize_ae_input_output_pairs()
