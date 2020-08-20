import torch
from torch import nn


class AutoEncoder(nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(5, 5))
        self.fc1 = nn.Linear(input_shape, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, input_shape)

    def _encoder(self, input_features):
        x = self.fc1(input_features)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        return x

    def _decoder(self, encoded_features):
        x = self.fc3(encoded_features)
        x = torch.relu(x)
        x = self.fc4(x)
        x = torch.relu(x)
        return x

    def forward(self, input_features):
        encoded_features = self._encoder(input_features)
        decoded_features = self._decoder(encoded_features)
        return decoded_features


# self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=(5, 5))
#         self.maxpool1 = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3))
#         self.maxpool2 = nn.MaxPool2d(2, 2)
#         self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3))
#         self.maxpool3 = nn.MaxPool2d(2, 2)
#         self.conv4 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3))
#
#         self.conv5 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3))
#         self.upsample6 = nn.Upsample(2, 2)
#         self.conv6 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(3, 3))
#         self.upsample7 = nn.Upsample(2, 2)
#         self.conv7 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(3, 3))
#         self.upsample8 = nn.Upsample(2, 2)
#         self.conv8 = nn.Conv2d(in_channels=8, out_channels=3, kernel_size=(5, 5))
