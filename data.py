import os
from pathlib import Path

import torch
import torchvision
from PIL import Image

from configure import Params


# credits: https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
class Dataset(torch.utils.data.Dataset):
    """Characterizes a dataset for PyTorch"""

    def __init__(self, list_ids, labels, transform=None):
        """Initialization"""
        self.labels = labels
        self.list_ids = list_ids
        self.transform = transform

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.list_ids)

    def __getitem__(self, index):
        """Generates one sample of data"""
        # Select sample
        ID = self.list_ids[index]

        # Load data and get label
        # from skimage import io
        # X = io.imread(ID).transpose((2, 0, 1))  # HxWxC --> CxHxW
        X = Image.open(ID)
        # print(ID)
        y = self.labels[ID]

        if self.transform:
            X = self.transform(X)

        return X, y


class Data(object):
    def __init__(self):
        self.params = Params()

    def load_data_by_name(self, dataset_name, config_mode):
        if dataset_name == 'mnist':
            # Create a directory to download and save the data
            root_dir = Path(os.path.dirname(__file__)).joinpath('torch_dataset')
            root_dir.mkdir(parents=True, exist_ok=True)

            # Prepare the data
            if config_mode == 'train':
                dataset = torchvision.datasets.MNIST(
                    root=root_dir, train=True, download=True,
                    transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
                )
            elif config_mode == 'test':
                dataset = torchvision.datasets.MNIST(
                    root=root_dir, train=False, download=True,
                    transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
                )
            else:
                raise ValueError('Invalid mode. Possible modes are `train` and `test`.')

        elif Path(dataset_name).is_dir():
            IDs = list(Path(dataset_name).glob('*/*.jpg'))
            labels = {}
            for idx, img_path in enumerate(IDs):
                labels[img_path] = img_path.parent.name
            dataset = Dataset(IDs, labels, transform=self.params.transform)
        else:
            raise ValueError('Invalid dataset. Possible types are `mnist` or any valid dataset directory.')
        return dataset

    def load_data(self, dataset, config_mode):
        dataset = self.load_data_by_name(dataset, config_mode)
        # Prepare for processing
        if config_mode == 'train':
            return torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True, num_workers=8, pin_memory=True)
        elif config_mode == 'test':
            return torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, num_workers=8)

    def load_data_for_visualization(self, dataset, config_mode):
        dataset = self.load_data_by_name(dataset, config_mode)
        # Prepare for processing
        if config_mode == 'train':
            return torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)
        elif config_mode == 'test':
            return torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)
