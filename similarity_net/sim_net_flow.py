import logging
import time

import torch

from configure import Configure, SimNet
from similarity_net.data import Data
from utils.training_utils import Utils
from utils.visualization_utils import VisualizationUtils

logger = logging.getLogger(__name__)


class SimNetFlow(object):

    @staticmethod
    def train_batch(inputs, expected_outputs, threshold=0.5):
        inputs = [x.to(Configure.device) for x in inputs]
        SimNet.optimizer.zero_grad()
        outputs = SimNet.model(inputs)
        loss = SimNet.criterion(outputs, expected_outputs.to(Configure.device))
        # fixme:
        # UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach()
        # or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).
        #   prediction = torch.tensor(outputs > threshold, dtype=torch.float32)
        prediction = torch.tensor(outputs > threshold, dtype=torch.float32)
        accuracy = sum(prediction == expected_outputs).item() / len(expected_outputs)
        loss.backward()
        SimNet.optimizer.step()
        return accuracy, loss.item()

    @staticmethod
    @torch.no_grad()
    def test_batch(inputs, expected_outputs, threshold=0.5):
        inputs = [x.to(Configure.device) for x in inputs]
        outputs = SimNet.model(inputs)
        loss = SimNet.criterion(outputs, expected_outputs.to(Configure.device))
        prediction = torch.tensor(outputs > threshold, dtype=torch.float32)
        accuracy = sum(prediction == expected_outputs).item() / len(expected_outputs)
        return accuracy, loss.item()

    @classmethod
    def train(cls):
        train_loader = Data.load_data(config_mode='train')
        test_loader = Data.load_data(config_mode='test')

        init_epoch, SimNet.model, history = Utils.get_initial_epoch(model=SimNet.model,
                                                                    pre_trained_models_dir=Configure.simnet_dir)
        for epoch in range(init_epoch, SimNet.epochs + 1):

            # Train
            SimNet.model.train()
            train_acc, train_loss = 0, 0
            epoch_start_time = time.perf_counter()
            for sig_pairs, (sim_scores, _) in train_loader:
                acc, loss = cls.train_batch(inputs=sig_pairs, expected_outputs=sim_scores)
                train_acc += acc
                train_loss += loss

            train_loss = train_loss / len(train_loader)
            train_acc = train_acc / len(train_loader)

            lr = SimNet.scheduler.get_last_lr()
            SimNet.scheduler.step()
            epoch_end_time = time.perf_counter()

            # Validate
            SimNet.model.eval()
            val_acc, val_loss = 0, 0
            for sig_pairs, (sim_scores, _) in test_loader:
                acc, loss = cls.test_batch(inputs=sig_pairs, expected_outputs=sim_scores)
                val_acc += acc
                val_loss += loss

            val_loss = val_loss / len(test_loader)
            val_acc = val_acc / len(train_loader)

            # Log epoch statistics
            Utils.update_history(history, epoch, train_loss, val_loss, train_acc, val_acc, lr, Configure.simnet_dir)
            VisualizationUtils.plot_learning_statistics(history, Configure.simnet_dir)
            Utils.save_model_on_epoch_end(SimNet.model, history, Configure.simnet_dir)

            logger.info("""epoch : {}/{}, train_loss = {:.6f}, val_loss = {:.6f}, train_acc = {:.6f}, val_acc = {:.6f}, 
            time = {:.2f} sec""".format(epoch, SimNet.epochs, train_loss, val_loss, train_acc, val_acc,
                                        epoch_end_time - epoch_start_time))

        Utils.save_best_model(pre_trained_models_dir=Configure.simnet_dir,
                              destination_dir=Configure.runtime_dir,
                              history=history, name=SimNet.name)


if __name__ == '__main__':
    from utils.torchsummary import summary
    summary(SimNet.model, (38400, 38400), logger.info)
    SimNetFlow.train()
