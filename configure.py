import torch
import torchvision
from torch import optim, nn

from cnn_models import AutoEncoder


class Params(object):
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoEncoder().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.9, last_epoch=-1)
        self.criterion = nn.MSELoss()
        self.transform = torchvision.transforms.Compose(
            [torchvision.transforms.CenterCrop((480, 640)),
             torchvision.transforms.ToTensor()]
        )
        self.epochs = 10

        self.train_data = r'D:\Data\20200622_SCD_INCIBE_custom_Dataset'
        self.test_data = r'D:\Data\20200622_SCD_INCIBE_custom_Dataset'
