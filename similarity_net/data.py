import itertools
import logging
import random
from pathlib import Path

import torch

from configure import SimNet
from utils.logging import log_running_time

logger = logging.getLogger(__name__)


class Dataset(torch.utils.data.Dataset):
    """credits: https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
    Characterizes a dataset for PyTorch"""

    def __init__(self, signature_pairs, labels, transform=None):
        """Initialization"""
        self.labels = labels
        self.signature_pairs = signature_pairs
        self.transform = transform

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.signature_pairs)

    def __getitem__(self, index):
        """Generates one sample of data"""
        # Select sample
        X = self.signature_pairs[index]
        y = self.labels[index]
        if self.transform:
            X = self.transform(X)
        return X, y


class Data(object):
    @classmethod
    @log_running_time
    def make_pairs(cls, signatures):
        sig_pairs = list(itertools.combinations(signatures, 2))
        logger.info(f'Max num of possible sig pairs  : {len(sig_pairs)}')

        # Create all possible signature-pairs
        # def label_signature_pairs(x):
        #     sig_pair = x[0][0], x[1][0]
        #     source_image_paths = x[0][1], x[1][1]
        #     similarity_label = torch.tensor([1.0]) if Path(x[0][1]).parent == Path(x[1][1]).parent else torch.tensor(
        #         [0.0])
        #     label = (similarity_label, source_image_paths)
        #     return sig_pair, label
        #
        # labelled_data = list(map(label_signature_pairs, sig_pairs))

        # Create all possible signature-pairs
        labelled_data = [None] * len(sig_pairs)
        for idx, x in enumerate(sig_pairs):
            sig_pair = x[0][0], x[1][0]
            source_image_paths = x[0][1], x[1][1]
            similarity_label = torch.tensor([1.0]) if Path(x[0][1]).parent == Path(x[1][1]).parent \
                else torch.tensor([0.0])
            label = (similarity_label, source_image_paths)
            labelled_data[idx] = sig_pair, label

        sig_pairs = [x[0] for x in labelled_data]
        labels = [x[1] for x in labelled_data]

        # Balance the classes
        indices_1 = [i for i, x in enumerate(labels) if x[0] == 1]
        indices_0 = [i for i, x in enumerate(labels) if x[0] == 0]
        logger.info(f'Num pairs in same device       : {len(indices_1)}')
        logger.info(f'Num pairs in different device  : {len(indices_0)}')

        if SimNet.balance_classes:
            logger.info(f'Balancing signature pairs')
            num_pairs = min(len(indices_1), len(indices_0))
            indexes_1 = random.sample(indices_1, k=num_pairs)
            indexes_0 = random.sample(indices_0, k=num_pairs)

            selected_indices = random.sample(indexes_1 + indexes_0, k=num_pairs * 2)
            sig_pairs = [sig_pairs[i] for i in selected_indices]
            labels = [labels[i] for i in selected_indices]
            logger.info(f'Total number of signature pairs: {num_pairs * 2}')
            logger.info(f'Num pairs in same device       : {num_pairs}')
            logger.info(f'Num pairs in different device  : {num_pairs}')

        dataset = Dataset(signature_pairs=sig_pairs, labels=labels, transform=None)
        return dataset

    @classmethod
    @log_running_time
    def load_data(cls, config_mode):

        from signature_net.sig_net_flow import SigNetFlow
        signatures = SigNetFlow.extract_signatures(config_mode=config_mode)
        dataset = cls.make_pairs(signatures)

        # Prepare for processing
        if config_mode == 'train':
            return torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True, num_workers=4)
        elif config_mode == 'test':
            return torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False, num_workers=4)
