import json
from pathlib import Path

import torch
from torch import optim, nn

from signature_net.models import SignatureNet1
from similarity_net.models import SimilarityNet
from utils.cost_functions import CategoricalCrossEntropyLoss


class Configure(object):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_data_config = Path(rf'/data/p288722/dresden/train/18_models_128x128_5/fold_1.json')
    test_data_config = Path(rf'/data/p288722/dresden/test/18_models_128x128_5/fold_1.json')
    dataset_folder = rf'/data/p288722/dresden/source_devices/nat_patches_128x128_5'

    # data = r'D:\Data\INCIBE_dataset\source_devices'

    runtime_dir = Path(r'/scratch/p288722/runtime_data/scd_pytorch/dev')

    sig_net_name = 'signature_net'
    sim_net_name = 'similarity_net'
    signet_dir = runtime_dir.joinpath(sig_net_name)
    simnet_dir = runtime_dir.joinpath(sim_net_name)

    runtime_dir.mkdir(exist_ok=True, parents=True)
    signet_dir.mkdir(exist_ok=True, parents=True)
    simnet_dir.mkdir(exist_ok=True, parents=True)

    compute_model_level_stats = False

    @classmethod
    def update(cls):
        cls.signet_dir = cls.runtime_dir.joinpath(cls.sig_net_name)
        cls.simnet_dir = cls.runtime_dir.joinpath(cls.sim_net_name)
        cls.runtime_dir.mkdir(exist_ok=True, parents=True)
        cls.signet_dir.mkdir(exist_ok=True, parents=True)
        cls.simnet_dir.mkdir(exist_ok=True, parents=True)
        cls.runtime_dir.joinpath('signature_net_brands').mkdir(exist_ok=True, parents=True)
        cls.runtime_dir.joinpath('signature_net_Nikon').mkdir(exist_ok=True, parents=True)
        cls.runtime_dir.joinpath('signature_net_Samsung').mkdir(exist_ok=True, parents=True)
        cls.runtime_dir.joinpath('signature_net_Sony').mkdir(exist_ok=True, parents=True)

        SigNet.name = cls.sig_net_name
        SimNet.name = cls.sim_net_name


class SigNet(object):
    name = Configure.sig_net_name
    # model = AutoEncoder().to(Configure.device)
    # optimizer = optim.Adam(model.parameters(), lr=1e-3)
    # criterion = nn.MSELoss()

    # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95, last_epoch=-1)
    # epochs = 9

    with open(Configure.train_data_config, 'r') as f:
        num_classes = len(json.load(f)['file_paths'])

    is_constrained = False
    model = SignatureNet1(num_classes=num_classes, is_constrained=is_constrained).to(Configure.device)
    optimizer = optim.SGD(model.parameters(), lr=1e-1, momentum=0.80, weight_decay=0.0005)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.90, last_epoch=-1)

    criterion = CategoricalCrossEntropyLoss()
    max_epochs = 100

    # Choices: 'majority_vote', 'prediction_score_sum', 'log_scaled_prediction_score_sum', 'log_scaled_std_dev'
    patch_aggregation = 'log_scaled_std_dev'
    use_contributing_patches = False
    samples_per_class = 20_000

    @classmethod
    def update_model(cls, num_classes, is_constrained=False):
        cls.model = SignatureNet1(num_classes=num_classes, is_constrained=is_constrained).to(Configure.device)
        cls.optimizer = optim.SGD(cls.model.parameters(), lr=1e-1, momentum=0.80, weight_decay=0.0005)
        cls.scheduler = optim.lr_scheduler.ExponentialLR(cls.optimizer, gamma=0.90, last_epoch=-1)


class SimNet(object):
    name = Configure.sim_net_name
    model = SimilarityNet().to(Configure.device)
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95, last_epoch=-1)

    criterion = nn.BCELoss()

    epochs = 1

    balance_classes = True
