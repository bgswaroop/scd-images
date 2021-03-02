import json
import logging
import pickle
import random
import warnings
from pathlib import Path

import lmdb
import numpy as np
import torch
import torchvision

warnings.filterwarnings("ignore")
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

    def __init__(self, image_ids, labels, dataset=None, transform=None):
        """Initialization"""
        self.labels = labels
        self.image_ids = image_ids
        self.transform = transform
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if dataset:
            self.camera_devices = {}
            camera_devices = Path(dataset).glob('*')
            for device_dir in camera_devices:
                self.camera_devices[device_dir.name] = {'env': lmdb.open(str(device_dir), readonly=True)}
                self.camera_devices[device_dir.name]['txn'] = self.camera_devices[device_dir.name]['env'].begin()
        else:
            self.camera_devices = None

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.image_ids)

    def __getitem__(self, index):
        """Generates one sample of data"""
        # Select sample
        image_id = str(self.image_ids[index])

        # Load data and get label
        device_name = '_'.join(str(image_id).split('_')[:-2])
        txn = self.camera_devices[device_name]['txn']
        patch = pickle.loads(txn.get(image_id.encode('ascii')))
        img = np.frombuffer(patch[0], dtype=np.uint8).reshape((128, 128, 3))
        img = np.ndarray.copy(np.array(img, dtype=np.float32)) / 255.0
        std = np.frombuffer(patch[1], dtype=np.float32).reshape((1, 3))

        # tuple: label, image_id, std
        y = self.labels[image_id], image_id, std

        if self.transform:
            img = self.transform(img)

        return img, y

    def __del__(self):
        if self.camera_devices:
            for device in self.camera_devices:
                self.camera_devices[device]['env'].close()


class Data(object):
    rgb_image_transform = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(),
         PerChannelMeanSubtraction()])

    @classmethod
    def prepare_torch_dataset(cls, config_file, balance_classes, dataset=None):
        """
        :param dataset: data set directory
        :param config_file: dataset directory or a list containing image paths
        :param balance_classes: a boolean value
        :return: a torch dataset
        """
        labels = None
        if isinstance(config_file, list):
            if balance_classes:
                raise NotImplementedError
            image_paths = config_file
        elif Path(config_file).is_dir():
            image_paths = list(Path(config_file).glob('*/*'))
            if balance_classes:
                raise NotImplementedError
        elif Path(config_file).suffix == '.json':
            with open(config_file, 'r') as f:
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
        img_labels = cls.compute_one_hot_labels(dataset=image_paths, labels=labels)
        labels_dict = {x: y for x, y in zip(image_paths, img_labels)}
        dataset = Dataset(image_ids=image_paths, labels=labels_dict, dataset=dataset,
                          transform=cls.rgb_image_transform)

        return dataset

    @classmethod
    def load_data(cls, config_file, config_mode, dataset=None):
        # Prepare for processing
        if config_mode == 'train':
            torch_dataset = cls.prepare_torch_dataset(config_file=config_file, balance_classes=False, dataset=dataset)
            return torch.utils.data.DataLoader(torch_dataset, batch_size=512, shuffle=True, num_workers=12)
        elif config_mode == 'test':
            torch_dataset = cls.prepare_torch_dataset(config_file=config_file, balance_classes=False, dataset=dataset)
            return torch.utils.data.DataLoader(torch_dataset, batch_size=512, shuffle=False, num_workers=12)

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
