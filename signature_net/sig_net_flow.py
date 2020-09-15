import time

import torch
from torchsummary import summary

from configure import Configure, SigNet
from signature_net.data import Data
from utils.training_utils import Utils
from utils.visualization_utils import VisualizationUtils


class SigNetFlow(object):

    @staticmethod
    def train_batch(inputs, expected_outputs):
        inputs = inputs.to(Configure.device)
        SigNet.optimizer.zero_grad()
        outputs = SigNet.model(inputs)
        loss = SigNet.criterion(outputs, expected_outputs.to(Configure.device))
        loss.backward()
        SigNet.optimizer.step()
        return loss.item()

    @staticmethod
    @torch.no_grad()
    def test_batch(inputs, expected_outputs):
        inputs = inputs.to(Configure.device)
        outputs = SigNet.model(inputs)
        loss = SigNet.criterion(outputs, expected_outputs.to(Configure.device))
        return loss.item()

    @classmethod
    def train(cls):
        # Prepare the data
        train_loader = Data.load_data(dataset=Configure.train_data, config_mode='train')
        test_loader = Data.load_data(dataset=Configure.test_data, config_mode='test')

        init_epoch, SigNet.model, history = Utils.get_initial_epoch(model=SigNet.model,
                                                                    pre_trained_models_dir=Configure.signet_dir)
        for epoch in range(init_epoch, SigNet.epochs + 1):

            # Train
            SigNet.model.train()
            train_loss = 0
            epoch_start_time = time.perf_counter()
            for input_images, (target_images, _) in train_loader:
                input_images = input_images.to(Configure.device)
                train_loss += cls.train_batch(inputs=input_images, expected_outputs=target_images)

            train_loss = train_loss / len(train_loader)
            lr = SigNet.scheduler.get_last_lr()
            SigNet.scheduler.step()
            epoch_end_time = time.perf_counter()

            # Validate
            SigNet.model.eval()
            val_loss = 0
            for input_images, (target_images, _) in test_loader:
                input_images = input_images.to(Configure.device)
                val_loss += cls.test_batch(inputs=input_images, expected_outputs=target_images)

            val_loss = val_loss / len(test_loader)

            # Log epoch statistics
            history = Utils.update_history(history=history, epoch=epoch, train_loss=train_loss, val_loss=val_loss,
                                           lr=lr, runtime_dir=Configure.signet_dir)
            VisualizationUtils.plot_learning_statistics(history, Configure.signet_dir)
            Utils.save_model_on_epoch_end(epoch, train_loss, val_loss, SigNet.model, Configure.signet_dir)

            print("epoch : {}/{}, train_loss = {:.6f}, val_loss = {:.6f}, time = {:.2f} sec".format(
                epoch, SigNet.epochs, train_loss, val_loss, epoch_end_time - epoch_start_time))

        Utils.save_best_model(pre_trained_models_dir=Configure.signet_dir,
                              destination_dir=Configure.runtime_dir,
                              history=history, name=SigNet.name)

    @classmethod
    def extract_signatures(cls, config_mode='train'):
        """
        Method to extract signatures and labels
        :param config_mode: string - train / test
        :return: list of labelled signatures
        """
        pre_trained_model_path = Configure.runtime_dir.joinpath('{}.pt'.format(SigNet.name))
        SigNet.model = torch.load(pre_trained_model_path)

        if config_mode == 'train':
            data_loader = Data.load_data(dataset=Configure.train_data, config_mode=config_mode)
        elif config_mode == 'test':
            data_loader = Data.load_data(dataset=Configure.test_data, config_mode=config_mode)
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
    summary(SigNet.model, (3, 320, 480))
    SigNetFlow.extract_signatures()
    # ae_predictions_train, ae_predictions_test = VisualizationUtils.visualize_ae_input_output_pairs()
