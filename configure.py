from pathlib import Path

import torch
from torch import optim, nn

from signature_net.models import SignatureNet1
from similarity_net.models import SimilarityNet
from utils.cost_functions import CategoricalCrossEntropyLoss


class Configure(object):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_data = r'/data/p288722/dresden/train/nat_patches_bal_15/'
    test_data = r'/data/p288722/dresden/test/nat_patches_bal_15/'
    data = r'D:\Data\INCIBE_dataset\source_devices'

    # runtime_dir = Path(__file__).parent.absolute().joinpath('runtime_dir_scd')
    runtime_dir = Path(r'/scratch/p288722/runtime_data/scd_pytorch/dresden_nat_patches_74')

    signet_dir = runtime_dir.joinpath('signature_net')
    simnet_dir = runtime_dir.joinpath('similarity_net')

    runtime_dir.mkdir(exist_ok=True, parents=True)
    signet_dir.mkdir(exist_ok=True, parents=True)
    simnet_dir.mkdir(exist_ok=True, parents=True)

    compute_model_level_stats = True


class SigNet(object):
    name = 'signature_net'
    # model = AutoEncoder().to(Configure.device)
    # optimizer = optim.Adam(model.parameters(), lr=1e-3)
    # criterion = nn.MSELoss()

    # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95, last_epoch=-1)
    # epochs = 9

    model = SignatureNet1(num_classes=74).to(Configure.device)
    optimizer = optim.SGD(model.parameters(), lr=1e-1, momentum=0.80, weight_decay=0.0005)
    criterion = CategoricalCrossEntropyLoss()

    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.90, last_epoch=-1)
    epochs = 50


class SimNet(object):
    name = 'similarity_net'
    model = SimilarityNet().to(Configure.device)
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.BCELoss()

    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95, last_epoch=-1)
    epochs = 25

    balance_classes = True
