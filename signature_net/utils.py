import random

import torch
from matplotlib import pyplot as plt

from configure import Configure, SigNet
from signature_net.data_fft import Data
from utils.visualization_utils import AE_Sample


class Utils(object):

    @staticmethod
    def visualize_ae_input_output_pairs():
        # load the model
        runtime_dir = Configure.runtime_dir
        with open(runtime_dir.joinpath('{}.pt'.format(SigNet.name)), 'rb') as f:
            SigNet.model = torch.load(f, map_location=Configure.device)  # fixme: the model to read from state_dict

        train_loader = Data.load_data_for_visualization(dataset=Configure.train_data_config, config_mode='train')

        # extract the input output pairs
        SigNet.model.eval()
        with torch.no_grad():
            count = 0
            ae_predictions_train = []
            for input_image, (target_image, _) in train_loader:
                inputs = input_image.to(Configure.device)
                outputs = SigNet.model(inputs)
                loss = SigNet.criterion(outputs, target_image.to(Configure.device)).cpu().item()

                input_img = inputs.view(input_image.shape)[0].cpu().numpy().transpose((1, 2, 0))
                output_img = outputs.view(input_image.shape)[0].cpu().numpy().transpose((1, 2, 0))
                target_img = target_image.view(input_image.shape)[0].cpu().numpy().transpose((1, 2, 0))
                ae_predictions_train.append(AE_Sample(loss, input_img, output_img, target_img))
                count += 1
                if count == 10:
                    break

        save_to_dir = Configure.runtime_dir.joinpath('sample_ae_pairs')
        save_to_dir.mkdir(parents=True, exist_ok=True)

        # randomly visualize 10 sample auto-encoder pairs
        selected_predictions = random.sample(ae_predictions_train, k=10)
        for idx, ae_prediction in enumerate(selected_predictions):
            fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 4))
            ax[0].imshow(ae_prediction.input)
            ax[0].set_title('Input Spectrum')

            ax[1].imshow(ae_prediction.label)
            ax[1].set_title('Expected Output')

            ax[2].imshow(ae_prediction.output)
            ax[2].set_title('Reconstructed Output')

            plt.savefig(save_to_dir.joinpath('{}.png'.format(idx)))
            st = fig.suptitle('Reconstruction MSE loss: {:.4f}'.format(ae_prediction.loss))

            # todo: fix the height of the plot title
            # shift subplots down:
            st.set_y(0.85)
            fig.subplots_adjust(top=0.85)

            plt.tight_layout()
            # plt.show()
            plt.clf()
            plt.cla()
            plt.close()

        return ae_predictions_train

    @staticmethod
    def save_avg_fourier_images(class_wise_labels=None):
        # load the model
        save_to_dir = Configure.runtime_dir.joinpath('averaged_spectrum')
        save_to_dir.mkdir(parents=True, exist_ok=True)

        if not class_wise_labels:
            class_wise_labels = Data.compute_avg_fft_labels(dataset=Configure.train_data_config)

        for idx, label in enumerate(class_wise_labels):
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))
            img = class_wise_labels[label]
            img = img.view(img.shape).cpu().numpy().transpose((1, 2, 0))
            ax.imshow(img)
            ax.set_title(label)

            plt.savefig(save_to_dir.joinpath('{}.png'.format(label)))
            st = fig.suptitle('Averaged Spectrum')
            # shift subplots down:
            st.set_y(0.90)
            fig.subplots_adjust(top=0.85)

            # plt.title()
            plt.tight_layout()
            # plt.show()
            plt.clf()
            plt.cla()
            plt.close()
