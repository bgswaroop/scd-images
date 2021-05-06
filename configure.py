import json
from pathlib import Path

import torch
from torch import optim, nn

from sig_net.classifier_baseline.models import SignatureNet1
from sig_net.auto_encoder.models import EfficientNetBasedAutoEncoder
from sig_net.classifier_efficient_net.models import EfficientNet
from sim_net.models import SimilarityNet
from utils.cost_functions import MSELoss, CategoricalCrossEntropyLoss


class Configure(object):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_data_config = Path(rf'/data/p288722/dresden/train/18_models_128x128_5/fold_1.json')
    test_data_config = Path(rf'/data/p288722/dresden/test/18_models_128x128_5/fold_1.json')
    dataset_folder = rf'/data/p288722/dresden/source_devices/natural'

    runtime_dir = Path(r'/scratch/p288722/runtime_data/scd_pytorch/dev')

    sig_net_name = 'signature_net'
    sim_net_name = 'similarity_net'
    signet_dir = runtime_dir.joinpath(sig_net_name)
    # simnet_dir = runtime_dir.joinpath(sim_net_name)

    runtime_dir.mkdir(exist_ok=True, parents=True)
    signet_dir.mkdir(exist_ok=True, parents=True)
    # simnet_dir.mkdir(exist_ok=True, parents=True)

    compute_model_level_stats = False

    @classmethod
    def update(cls):
        cls.signet_dir = cls.runtime_dir.joinpath(cls.sig_net_name)
        cls.simnet_dir = cls.runtime_dir.joinpath(cls.sim_net_name)
        cls.runtime_dir.mkdir(exist_ok=True, parents=True)
        cls.signet_dir.mkdir(exist_ok=True, parents=True)

        SigNet.name = cls.sig_net_name
        # SimNet.name = cls.sim_net_name


class SigNet(object):
    name = Configure.sig_net_name
    with open(Configure.train_data_config, 'r') as f:
        num_classes = len(json.load(f)['file_paths'])

    model = EfficientNetBasedAutoEncoder(version='b1', num_classes=num_classes).to(Configure.device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.90, last_epoch=-1)
    criterion = MSELoss()
    max_epochs = 100

    # model = EfficientNet(version='b1', num_classes=num_classes).to(Configure.device)
    # optimizer = optim.Adam(model.parameters(), lr=1e-3)
    # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.90, last_epoch=-1)
    # criterion = CategoricalCrossEntropyLoss()
    # max_epochs = 100


    # Choices: 'majority_vote', 'prediction_score_sum', 'log_scaled_prediction_score_sum', 'log_scaled_std_dev'
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
