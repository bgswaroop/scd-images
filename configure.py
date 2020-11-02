from pathlib import Path

import torch
from torch import optim, nn
import json

from signature_net.models import SignatureNet1
from similarity_net.models import SimilarityNet
from utils.cost_functions import CategoricalCrossEntropyLoss


class Configure(object):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")

    train_data = rf'/data/p288722/dresden/train/nat_patches_18_models_128x128_1/fold_1.json'
    test_data = rf'/data/p288722/dresden/test/nat_patches_18_models_128x128_1/fold_1.json'
    data = r'D:\Data\INCIBE_dataset\source_devices'

    # runtime_dir = Path(__file__).parent.absolute().joinpath('runtime_dir_scd')
    runtime_dir = Path(r'/scratch/p288722/runtime_data/scd_pytorch/18_models_prnu')

    sig_net_name = 'signature_net'
    sim_net_name = 'similarity_net'
    signet_dir = runtime_dir.joinpath(sig_net_name)
    simnet_dir = runtime_dir.joinpath(sim_net_name)

    runtime_dir.mkdir(exist_ok=True, parents=True)
    signet_dir.mkdir(exist_ok=True, parents=True)
    simnet_dir.mkdir(exist_ok=True, parents=True)

    compute_model_level_stats = False

    @classmethod
    def create_runtime_dirs(cls):
        cls.signet_dir = cls.runtime_dir.joinpath(cls.sig_net_name)
        cls.simnet_dir = cls.runtime_dir.joinpath(cls.sim_net_name)
        cls.runtime_dir.mkdir(exist_ok=True, parents=True)
        cls.signet_dir.mkdir(exist_ok=True, parents=True)
        cls.simnet_dir.mkdir(exist_ok=True, parents=True)
        SigNet.name = cls.sig_net_name
        SimNet.name = cls.sim_net_name


class SigNet(object):
    name = Configure.sig_net_name
    # model = AutoEncoder().to(Configure.device)
    # optimizer = optim.Adam(model.parameters(), lr=1e-3)
    # criterion = nn.MSELoss()

    # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95, last_epoch=-1)
    # epochs = 9

    with open(Configure.train_data, 'r') as f:
        num_classes = len(json.load(f)['file_paths'])

    is_constrained = True
    model = SignatureNet1(num_classes=num_classes, is_constrained=is_constrained).to(Configure.device)
    optimizer = optim.SGD(model.parameters(), lr=1e-1, momentum=0.80, weight_decay=0.0005)
    criterion = CategoricalCrossEntropyLoss()

    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.90, last_epoch=-1)
    epochs = 50


class SimNet(object):
    name = Configure.sim_net_name
    model = SimilarityNet().to(Configure.device)
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.BCELoss()

    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95, last_epoch=-1)
    epochs = 1

    balance_classes = True
