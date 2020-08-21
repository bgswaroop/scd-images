from torch import nn


class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=256, kernel_size=(3, 3), padding=(1, 1))
        self.conv2 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(3, 3), padding=(1, 1))
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(3, 3), padding=(1, 1))

        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=(1, 1))
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=(1, 1))
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=3, kernel_size=(3, 3), padding=(1, 1))

    @staticmethod
    def upsample(size):
        return nn.Upsample(size)

    def _encoder(self, input_features):
        x = self.conv1(input_features)
        x = nn.MaxPool2d(2, 2)(x)
        x = self.conv2(x)
        x = nn.MaxPool2d(2, 2)(x)
        x = self.conv3(x)
        return x

    def _decoder(self, encoded_features):
        x = self.conv4(encoded_features)
        x = nn.Upsample(scale_factor=2)(x)
        x = self.conv5(x)
        x = nn.Upsample(scale_factor=2)(x)
        x = self.conv6(x)
        return x

    def forward(self, input_features):
        encoded_features = self._encoder(input_features)
        decoded_features = self._decoder(encoded_features)
        return decoded_features
