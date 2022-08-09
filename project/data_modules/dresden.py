import json
import pickle
from collections import OrderedDict
from pathlib import Path
from typing import Optional, Callable

import lmdb
import numpy as np
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms

from .transforms import PerChannelMeanSubtraction
from .utils import get_train_test_split
from line_profiler_pycharm import profile


class DresdenNaturalImages(Dataset):
    def __init__(
            self,
            full_image_dataset_dir: Path,
            patches_dataset_dir: Path,
            fold: int,
            classifier_type: str,
            transform: Optional[Callable] = lambda x: x,
            train: bool = True):

        self.labels_map = OrderedDict()
        self.img_keys, self.labels = [], []
        self.device_names = []
        self.patches_dataset_dir = patches_dataset_dir
        self.full_image_dataset_dir = full_image_dataset_dir
        self.train = train
        self.fold = fold
        self.classifier_type = classifier_type
        self.transform = transform

        assert patches_dataset_dir.exists(), f'The dataset path does not exists: {patches_dataset_dir}'

    def __len__(self):
        return len(self.img_keys)

    @profile
    def __getitem__(self, item):
        image_key, device_name = self.img_keys[item]
        if not hasattr(self, 'txns'):
            self._open_lmdb()
        txn = self.txns[device_name]
        # with lmdb.open(str(self.dataset_path.joinpath(device_name)), readonly=True) as env:
        # with env.begin() as txn:
        patch = pickle.loads(txn.get(image_key))
        x = np.frombuffer(patch[0], dtype=np.uint8).reshape((128, 128, 3))  # fixme: make it flow from inputs
        # x = np.ndarray.copy(np.array(x, dtype=np.float32)) / 255.0
        x = x.astype(np.float32, copy=False) / 255.0
        # x = np.transpose(x, axes=[2, 0, 1])  # changing to channels first
        # std = np.frombuffer(patch[1], dtype=np.float32).reshape((1, 3))
        x = self.transform(x)
        y = self.labels[item]
        return x, y, image_key.decode('ascii')

    def _open_lmdb(self):
        # Open the relevant lmdb file pointers
        temp_txns = {}
        temp_envs = {}
        device_paths = self.patches_dataset_dir.glob('*')
        # device_paths = [x for x in device_paths if x.name in self.device_names]
        for device_path in device_paths:
            env = lmdb.open(str(device_path), readonly=True, lock=False)
            temp_txns[device_path.name] = env.begin()
            temp_envs[device_path.name] = env

        if not hasattr(self, 'txns'):
            self.txns = temp_txns
        else:
            # perform cleanup
            for device_path in device_paths:
                temp_envs[device_path.name].close()
            raise BlockingIOError('Transactions must be opened only once')

    def setup_brand_level_identification(self):
        if self.train:
            print('\nSetting up TRAINING DATA')
            split = get_train_test_split(self.full_image_dataset_dir, self.fold, self.classifier_type)['train']
        else:
            print('\nSetting up TEST DATA')
            split = get_train_test_split(self.full_image_dataset_dir, self.fold, self.classifier_type)['test']

        class_id = -1
        for brand_name in sorted(split):
            class_id += 1
            self.labels_map[class_id] = brand_name
            for model_name in sorted(split[brand_name]):
                for device_name in sorted(split[brand_name][model_name]):
                    img_names = split[brand_name][model_name][device_name]
                    self.img_keys.extend([(x.encode('ascii'), '_'.join(x.split('_')[:-2])) for x in img_names])
                    self.labels.extend([class_id] * len(img_names))
                    self.device_names.append(device_name)

    def setup_model_level_identification(self):
        if self.train:
            print('\nSetting up TRAINING DATA')
            split = get_train_test_split(self.full_image_dataset_dir, self.fold, self.classifier_type)['train']
        else:
            print('\nSetting up TEST DATA')
            split = get_train_test_split(self.full_image_dataset_dir, self.fold, self.classifier_type)['test']

        class_id = -1
        for brand_name in sorted(split):
            # if filter_by_brand and filter_by_brand not in brand_name:
            #     continue
            for model_name in sorted(split[brand_name]):
                class_id += 1
                self.labels_map[class_id] = model_name
                for device_name in sorted(split[brand_name][model_name]):
                    img_names = split[brand_name][model_name][device_name]
                    self.img_keys.extend([(x.encode('ascii'), '_'.join(x.split('_')[:-2])) for x in img_names])
                    self.labels.extend([class_id] * len(img_names))
                    self.device_names.append(device_name)

    def setup_device_level_identification(self):
        if self.train:
            json_file = self.splits_dir.joinpath(f'train_74_devices_fold{self.fold}.json')
        else:
            json_file = self.splits_dir.joinpath(f'test_74_devices_fold{self.fold}.json')

        with open(json_file) as f:
            split = json.load(f)

        # This scenario accounts for the 74 camera devices in the Dresden dataset
        class_id = -1
        for brand_name in sorted(split):
            for model_name in sorted(split[brand_name]):
                for device_name in sorted(split[brand_name][model_name]):
                    class_id += 1
                    self.labels_map[class_id] = device_name

                    img_names = split[brand_name][model_name][device_name]
                    self.img_keys.extend([self.patches_dataset_dir.joinpath(f'{device_name}/{x}') for x in img_names])
                    self.labels.extend([class_id] * len(img_names))


class DresdenDataModule(LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.patches_dataset_dir = args.patches_dataset_dir
        self.full_image_dataset_dir = args.full_image_dataset_dir
        self.source_dataset_dir = None
        self.batch_size = args.batch_size
        self.num_workers = args.num_processes
        self.classifier_type = args.classifier
        self.fold = args.fold

        self.train_ds = None
        self.val_ds = None
        self.test_ds = None
        self.num_classes = None
        self.num_samples = None

        self.setup()

    def prepare_data(self):
        """
        Step1:  Download the Dresden dataset - https://dl.acm.org/doi/10.1145/1774088.1774427
        Step2:  Restructure the data as shown below:
                Retain the image file names and the names of the device directories as specified in the downloaded
                dataset.

            dataset_dir
            │
            ├── device_dir1
            │   ├── img_1
            │   ├── img_2
            │   └── ...
            ├── device_dir2
            │   ├── img_1
            │   ├── img_2
            │   └── ...
            └── ...

        Step3: Extract the homogeneous crops from each image and save the patches in an .lmdb database
        (project/data_modules/utils/extract_and_save_homo_patches.py)

            dataset_dir
            │
            ├── device_dir1
            │   ├── data.mdb
            │   └── lock.mdb
            │
            ├── device_dir2
            │   ├── data.mdb
            │   └── lock.mdb
            └── ...

        Step4: Create Train / Test splits
        (project/data_modules/utils/train_test_split.py)
        """
        pass

    def setup(self, stage: Optional[str] = None):
        # called on every GPU
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                PerChannelMeanSubtraction()
            ]
        )
        self.train_ds = DresdenNaturalImages(self.full_image_dataset_dir, self.patches_dataset_dir,
                                             self.fold, self.classifier_type, transform, train=True)
        self.val_ds = DresdenNaturalImages(self.full_image_dataset_dir, self.patches_dataset_dir,
                                           self.fold, self.classifier_type, transform, train=False)

        # These methods cannot be part of the __init__ as the lmdb objects are not pickable
        if self.classifier_type == "all_brands":
            self.train_ds.setup_brand_level_identification()
            self.val_ds.setup_brand_level_identification()
        else:
            self.train_ds.setup_model_level_identification()
            self.val_ds.setup_model_level_identification()

        self.test_ds = self.val_ds  # as we are performing cross validation (due to limited size of the dataset)

        self.num_classes = len(self.train_ds.labels_map)
        print(self.train_ds.labels_map)
        self.num_samples = len(self.train_ds)

    def train_dataloader(self):
        return DataLoader(self.train_ds, self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True,
                          persistent_workers=True, prefetch_factor=10)

    def val_dataloader(self):
        return DataLoader(self.val_ds, self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True,
                          persistent_workers=True, prefetch_factor=10)

    def test_dataloader(self):
        return DataLoader(self.test_ds, self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True,
                          persistent_workers=True, prefetch_factor=10)

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        raise NotImplementedError("`predict_dataloader` must be implemented to be used with the Lightning Trainer")
