import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from _old_strcture.configure import SigNet, Configure
from _old_strcture.miscellaneous.prepare_image_and_patch_data import level_from_hierarchical_dataset
from _old_strcture.signature_net.data_rgb import Data


def normalize_output(img):
    img = img - img.min()
    img = img / img.max()
    return img


def get_homogeneous_activations():
    # Prepare the data - Homogeneous
    temp_dir = Configure.runtime_dir.joinpath('temp')
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(exist_ok=True, parents=True)

    Configure.dataset_folder = \
        rf'/scratch/p288722/datasets/dresden/source_devices/nat_patches_128x128_200'
    Configure.test_data_config = temp_dir.joinpath(rf'test_models_fold_1.json')
    level_from_hierarchical_dataset(
        source_view=Path(rf'/data/p288722/dresden/test/18_models_from200_128x128_200/fold_1.json'),
        dest_level='models',
        dest_view=Configure.test_data_config
    )
    Configure.update()

    data_loader = Data.load_data(config_file=Configure.test_data_config,
                                 config_mode='test',
                                 dataset=Configure.dataset_folder)
    activation = {}

    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()

        return hook

    pre_trained_model_path = \
        Path(r'/scratch/p288722/runtime_data/scd_pytorch/18_models_hcal_08/fold_1/signature_net_brands.pt')
    params = torch.load(pre_trained_model_path, map_location=Configure.device)
    SigNet.model.load_state_dict(params['model_state_dict'])
    SigNet.model.eval()
    SigNet.model.conv1.register_forward_hook(get_activation('conv1'))
    for images, (labels, img_paths, img_std_dev) in data_loader:
        images = images.to(Configure.device)
        features = SigNet.model.extract_features(images).to(torch.device("cpu")).detach()
        break

    act = activation['conv1'].squeeze().to(torch.device("cpu")).detach()
    return act


def get_non_homogeneous_activations():
    # Prepare the data - Homogeneous
    temp_dir = Configure.runtime_dir.joinpath('temp')
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(exist_ok=True, parents=True)

    Configure.dataset_folder = \
        rf'/scratch/p288722/datasets/dresden/source_devices/nat_patches_non_homo_128x128_400'
    Configure.test_data_config = temp_dir.joinpath(rf'test_models_non_homo_fold_1.json')
    level_from_hierarchical_dataset(
        source_view=Path(rf'/data/p288722/dresden/test/18_models_non_homo_128x128_25/fold_1.json'),
        dest_level='models',
        dest_view=Configure.test_data_config
    )
    Configure.update()

    data_loader = Data.load_data(config_file=Configure.test_data_config,
                                 config_mode='test',
                                 dataset=Configure.dataset_folder)
    activation = {}

    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()

        return hook

    pre_trained_model_path = \
        Path(r'/scratch/p288722/runtime_data/scd_pytorch/18_models_hcal_non_homo_1/fold_1/signature_net_brands.pt')
    params = torch.load(pre_trained_model_path, map_location=Configure.device)
    SigNet.model.load_state_dict(params['model_state_dict'])
    SigNet.model.eval()
    SigNet.model.conv1.register_forward_hook(get_activation('conv1'))
    for images, (labels, img_paths, img_std_dev) in data_loader:
        images = images.to(Configure.device)
        features = SigNet.model.extract_features(images).to(torch.device("cpu")).detach()
        break

    act = activation['conv1'].squeeze().to(torch.device("cpu")).detach()
    return act


def plot_activations():

    homo_act = get_homogeneous_activations()
    non_homo_act = get_non_homogeneous_activations()

    plt.figure()
    fig, axarr = plt.subplots(nrows=2, ncols=5)
    # for ax in axarr.ravel():
    #     ax.remove()

    for idx in range(5):  # The number in the range should be less than the batch size
        axarr[0][idx].imshow(homo_act[idx][0])
        axarr[1][idx].imshow(non_homo_act[idx][0])

    plt.tight_layout()
    # plt.savefig('activations_conv1.png', bbox_inches='tight')
    plt.show()

    plt.clf()
    plt.cla()
    plt.close()


if __name__ == '__main__':
    plot_activations()
