import json
import logging
import random
import tarfile
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image

logger = logging.getLogger(__name__)


class PerChannelMeanSubtraction:
    """Per channel mean subtraction"""

    def __init__(self):
        pass

    def __call__(self, x):
        x = x - torch.mean(x, dim=[1, 2], keepdim=True)
        return x


# credits: https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
class Dataset(torch.utils.data.Dataset):
    """Characterizes a dataset for PyTorch"""

    # namedtuple('')

    def __init__(self, image_ids, labels, tar_file=None, transform=None):
        """Initialization"""
        self.labels = labels
        self.image_ids = image_ids
        self.transform = transform
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tar = tarfile.open(tar_file, 'r:') if tar_file else None
        logger.info(f'No. members in tar: {len(self.tar.getmembers())}')

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.image_ids)

    def __getitem__(self, index):
        """Generates one sample of data"""
        # Select sample
        image_id = self.image_ids[index]

        # Load data and get label
        try:
            X = Image.open(self.tar.extractfile(image_id)) if self.tar else Image.open(image_id)
        except Exception as e:
            logger.info(e)
            raise e

        y = self.labels[image_id]

        if self.transform:
            X = self.transform(X)

        return X, y

    def __del__(self):
        if self.tar:
            self.tar.close()


class Data(object):
    rgb_image_transform = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(),
         PerChannelMeanSubtraction()])

    @classmethod
    def prepare_torch_dataset(cls, dataset_file, balance_classes, tar_file=None):
        """
        :param tar_file:
        :param dataset_file: dataset directory or a list containing image paths
        :param balance_classes: a boolean value
        :return: a torch dataset
        """
        labels = None
        if isinstance(dataset_file, list):
            if balance_classes:
                raise NotImplementedError
            image_paths = dataset_file
        elif Path(dataset_file).is_dir():
            image_paths = list(Path(dataset_file).glob('*/*'))
            if balance_classes:
                raise NotImplementedError
        elif Path(dataset_file).suffix == '.json':
            with open(dataset_file, 'r') as f:
                image_paths_dict = json.load(f)
            image_paths, labels = [], []

            if balance_classes:
                class_wise_image_counts = []
                for device_name in image_paths_dict['file_paths']:
                    class_wise_image_counts.append(len(image_paths_dict['file_paths'][device_name]))
                num_images_per_class = int(np.mean(class_wise_image_counts))
                logger.info(f'Num of images per class: {num_images_per_class}')

            for device_name in image_paths_dict['file_paths']:
                paths = image_paths_dict['file_paths'][device_name]
                if balance_classes:
                    random.seed(0)
                    if len(paths) < num_images_per_class:
                        paths = random.choices(paths, k=num_images_per_class)
                    else:
                        paths = random.sample(paths, k=num_images_per_class)

                image_paths += paths
                labels += [device_name] * len(paths)

        logger.info(f'Total number of images: {len(image_paths)}')
        class_labels = cls.compute_one_hot_labels(dataset=image_paths, labels=labels)
        img_labels = {x: (y, str(x)) for x, y in zip(image_paths, class_labels)}
        dataset = Dataset(image_ids=image_paths, labels=img_labels, tar_file=tar_file,
                          transform=cls.rgb_image_transform)

        return dataset

    @classmethod
    def load_data(cls, dataset, config_mode, tar_file=None):
        # Prepare for processing
        if config_mode == 'train':
            dataset = cls.prepare_torch_dataset(dataset_file=dataset, balance_classes=False, tar_file=tar_file)
            return torch.utils.data.DataLoader(dataset, batch_size=2048, shuffle=True, num_workers=12)
        elif config_mode == 'test':
            dataset = cls.prepare_torch_dataset(dataset_file=dataset, balance_classes=False, tar_file=tar_file)
            return torch.utils.data.DataLoader(dataset, batch_size=2048, shuffle=False, num_workers=12)

    @classmethod
    def load_data_for_visualization(cls, dataset, config_mode):
        raise NotImplementedError()

    @classmethod
    def compute_one_hot_labels(cls, dataset, labels=None):
        """
        :param dataset: Path to a dataset directory or a list containing image paths
        :return: A one-hot encoding of the image labels
        """

        if isinstance(dataset, list):
            img_paths = dataset
            if not labels:
                labels = [Path(x).parent.name for x in img_paths]
        elif Path(dataset).is_dir():
            img_paths = list(Path(dataset).glob('*/*'))
            labels = [x.parent.name for x in img_paths]
        else:
            raise ValueError('Invalid dataset')

        devices_list = list(sorted(set(labels)))
        for idx, img_label in enumerate(labels):
            labels[idx] = torch.tensor([1 if x == img_label else 0 for x in devices_list], dtype=torch.float32)

        return labels
