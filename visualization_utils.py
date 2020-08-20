import pickle
from collections import namedtuple
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from configure import Params
from data import Data
from utils import Utils

AE_Sample = namedtuple('AE_Sample', ['label', 'loss', 'input', 'output'])


class VisualizationUtils:
    def __init__(self):
        pass

    @staticmethod
    def plot_class_wise_data_distribution(dataset_dir, save_to_dir):

        # Prepare the data
        data_root = Path(dataset_dir)
        class_list = [f for f in data_root.glob("*")]
        class_wise_distribution = np.zeros(len(class_list))
        class_names = [None] * len(class_list)
        for idx, item in enumerate(class_list):
            class_wise_distribution[idx] = len([f for f in item.glob("*")])
            class_names[idx] = item.stem

        # Sort the data for better visualization
        class_names = np.array(class_names)
        sorted_indices = np.argsort(class_wise_distribution)

        # Generate the plot
        # plt.figure(figsize=(24, 24))
        plt.figure(figsize=(15, 8))
        plt.style.use('fivethirtyeight')
        plt.barh(class_names[sorted_indices], class_wise_distribution[sorted_indices])
        plt.title('Class wise distribution')
        plt.rc('ytick', labelsize=1)
        plt.xlabel('No. of training images')
        plt.tight_layout()
        plt.savefig(str(Path(save_to_dir).joinpath("data_distribution.png")))
        plt.close()

    @staticmethod
    def plot_training_history(save_to_dir, history):

        # Set up the paths for saving the plots
        acc_filename = str(Path(save_to_dir).joinpath("training_accuracy.png"))
        loss_filename = str(Path(save_to_dir).joinpath("training_loss.png"))

        # Initialize the plot data
        acc = history['accuracy']
        val_acc = history['val_accuracy']
        loss = history['loss']
        val_loss = history['val_loss']
        epochs = history['epochs']
        selected_epoch = Utils.choose_best_epoch_from_history(history)

        plt.figure()
        plt.plot(epochs, acc, 'r', label='Train')
        plt.plot(epochs, val_acc, 'b', label='Test')
        if 'val_acc2' in history:
            plt.plot(epochs, history['val_acc2'], 'g', label='Test known models unknown devices')
        if 'val_acc3' in history:
            plt.plot(epochs, history['val_acc3'], 'c', label='Test unknown models unknown devices')
        plt.axvline(x=selected_epoch, label='Selected model', linestyle='--', alpha=0.40, c='r')

        plt.title('Accuracy plot')
        plt.xlabel('epochs')
        plt.ylabel('accuracy')
        plt.legend()
        plt.savefig(acc_filename)
        plt.close()

        plt.figure()
        plt.plot(epochs, loss, 'r', label='Train')
        plt.plot(epochs, val_loss, 'b', label='Test')
        if 'val_loss2' in history:
            plt.plot(epochs, history['val_loss2'], 'g', label='Test known models unknown devices')
        if 'val_loss3' in history:
            plt.plot(epochs, history['val_loss3'], 'c', label='Test unknown models unknown devices')
        plt.axvline(x=selected_epoch, label='Selected model', linestyle='--', alpha=0.40, c='r')

        plt.title('Loss plot')
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.legend()
        plt.savefig(loss_filename)
        plt.close()

    @classmethod
    def plot_learning_rate(cls, save_to_dir, history):
        filename = str(Path(save_to_dir).joinpath("optimizer_leaning_rate.png"))

        # Initialize the plot data
        lr = history['learning_rate']
        epochs = history['epochs']

        plt.figure()
        plt.plot(epochs, lr, 'r', label='Leaning Rate')
        plt.title('Learning Rate Decay')
        plt.xlabel('epochs')
        plt.ylabel('learning rate')
        plt.legend()
        plt.savefig(filename)
        plt.close()

    @staticmethod
    def plot_learning_statistics(history, runtime_dir):
        VisualizationUtils.plot_training_history(save_to_dir=runtime_dir, history=history)
        VisualizationUtils.plot_learning_rate(save_to_dir=runtime_dir, history=history)

    @staticmethod
    def visualize_ae_input_output_pairs():
        # load the model
        params = Params()
        runtime_dir = Path(r'./runtime_dir')
        with open(runtime_dir.joinpath('ae.pt'), 'rb') as f:
            params.model = torch.load(f)

        # load the data
        train_loader = Data().load_data_for_visualization(dataset=params.train_data, config_mode='train')
        test_loader = Data().load_data_for_visualization(dataset=params.test_data, config_mode='test')

        # extract the input output pairs
        params.model.eval()
        with torch.no_grad():
            predictions_filename = runtime_dir.joinpath('ae_predictions_train.pkl')
            if predictions_filename.exists():
                with open(predictions_filename, 'rb') as f:
                    ae_predictions_train = pickle.load(f)
            else:
                ae_predictions_train = []
                for input_image, label in train_loader:
                    inputs = input_image.view(input_image.shape[0], -1).to(params.device)
                    outputs = params.model(inputs)
                    loss = params.criterion(outputs, inputs).cpu().item()

                    input_img = inputs.view(input_image.shape)[0].cpu().numpy().transpose((1, 2, 0))
                    output_img = outputs.view(input_image.shape)[0].cpu().numpy().transpose((1, 2, 0))
                    ae_predictions_train.append(AE_Sample(label, loss, input_img, output_img))

                with open(predictions_filename, 'wb+') as f:
                    pickle.dump(ae_predictions_train, f)

            predictions_filename = runtime_dir.joinpath('ae_predictions_test.pkl')
            if predictions_filename.exists():
                with open(predictions_filename, 'rb') as f:
                    ae_predictions_test = pickle.load(f)
            else:
                ae_predictions_test = []
                for input_image, label in test_loader:
                    inputs = input_image.view(input_image.shape[0], -1).to(params.device)
                    outputs = params.model(inputs)
                    loss = params.criterion(outputs, inputs).cpu().item()

                    input_img = inputs.view(input_image.shape)[0].cpu().numpy().transpose((1, 2, 0))
                    output_img = outputs.view(input_image.shape)[0].cpu().numpy().transpose((1, 2, 0))
                    ae_predictions_test.append(AE_Sample(label, loss, input_img, output_img))

                with open(predictions_filename, 'wb+') as f:
                    pickle.dump(ae_predictions_test, f)

        selected_predictions = ae_predictions_train[:10]

        # visualize the data
        from matplotlib import pyplot as plt
        for idx, ae_prediction in enumerate(selected_predictions):
            fig, ax = plt.subplots(nrows=1, ncols=2)
            ax[0].imshow(ae_prediction.input)
            ax[0].set_title('Input Image')

            ax[1].imshow(ae_prediction.output)
            ax[1].set_title('Reconstructed Output')

            # plt.savefig(runtime_dir.joinpath('{}.png'.format(idx)))
            st = fig.suptitle('Reconstruction MSE loss: {:.4f}'.format(ae_prediction.loss))
            # shift subplots down:
            st.set_y(0.90)
            fig.subplots_adjust(top=0.85)

            # plt.title()
            plt.tight_layout()
            plt.show()
            plt.close()

        return ae_predictions_train, ae_predictions_test


if __name__ == "__main__":
    utils = VisualizationUtils()
    utils.plot_class_wise_data_distribution(
        dataset_dir=None,
        save_to_dir=None
    )
