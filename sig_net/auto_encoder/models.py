from math import ceil

import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torchsummary import summary
from torch.utils.checkpoint import checkpoint_sequential


class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=(3, 3), padding=(1, 1))
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(3, 3), padding=(1, 1))
        self.conv3 = nn.Conv2d(in_channels=8, out_channels=4, kernel_size=(3, 3), padding=(1, 1))
        self.conv4 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=(3, 3), padding=(1, 1))
        self.conv5 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(3, 3), padding=(1, 1))
        self.conv6 = nn.Conv2d(in_channels=8, out_channels=3, kernel_size=(3, 3), padding=(1, 1))

    @staticmethod
    def upsample(size):
        """
        :param size:
        :return:
        """
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
        x = nn.Upsample(scale_factor=2, mode='nearest')(x)
        x = self.conv5(x)
        x = nn.Upsample(scale_factor=2, mode='nearest')(x)
        x = self.conv6(x)
        return x

    def forward(self, input_features):
        encoded_features = self._encoder(input_features)
        decoded_features = self._decoder(encoded_features)
        return decoded_features

    def extract_features(self, cnn_inputs):
        encoded_features = self._encoder(cnn_inputs)
        features = torch.flatten(encoded_features, start_dim=1)
        return features


# EfficientNet implementation credits: https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/CNN_architectures/pytorch_efficientnet.py

base_model = [
    # expand_ratio, channels, repeats, stride, kernel_size
    [1, 16, 1, 1, 3],
    [6, 24, 2, 2, 3],
    [6, 40, 2, 2, 5],
    [6, 80, 3, 2, 3],
    [6, 112, 3, 1, 5],
    [6, 192, 4, 2, 5],
    [6, 320, 1, 1, 3],
]

phi_values = {
    # tuple of: (phi_value, resolution, drop_rate)
    "b0": (0, 224, 0.2),  # alpha, beta, gamma, depth = alpha ** phi
    "b1": (0.5, 240, 0.2),
    "b2": (1, 260, 0.3),
    "b3": (2, 300, 0.3),
    "b4": (3, 380, 0.4),
    "b5": (4, 456, 0.4),
    "b6": (5, 528, 0.5),
    "b7": (6, 600, 0.5),
}


class DeConvCNNBlock(nn.Module):
    def __init__(
            self, in_channels, out_channels, kernel_size, stride, padding, groups=1
    ):
        super(DeConvCNNBlock, self).__init__()
        out_padding = 1 if stride > 1 else 0
        self.cnn = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            groups=groups,
            bias=False,
            output_padding=out_padding,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.silu = nn.SiLU()  # SiLU <-> Swish

    def forward(self, x):
        return self.silu(self.bn(self.cnn(x)))


class DeConvInvertedResidualBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            expand_ratio,
            reduction=4,  # squeeze excitation
            survival_prob=0.8,  # for stochastic depth
    ):
        super(DeConvInvertedResidualBlock, self).__init__()
        self.survival_prob = survival_prob
        self.use_residual = in_channels == out_channels and stride == 1
        hidden_dim = out_channels * expand_ratio
        self.expand = in_channels != hidden_dim
        reduced_dim = int(out_channels / reduction)

        if self.expand:
            self.expand_conv = DeConvCNNBlock(
                in_channels, hidden_dim, kernel_size=3, stride=1, padding=1,
            )

        self.conv = nn.Sequential(
            DeConvCNNBlock(
                hidden_dim, hidden_dim, kernel_size, stride, padding, groups=hidden_dim,
            ),
            SqueezeExcitation(hidden_dim, reduced_dim),
            nn.ConvTranspose2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def stochastic_depth(self, x):
        if not self.training:
            return x

        binary_tensor = torch.rand(x.shape[0], 1, 1, 1, device=x.device) < self.survival_prob
        return torch.div(x, self.survival_prob) * binary_tensor

    def forward(self, inputs):
        x = self.expand_conv(inputs) if self.expand else inputs

        if self.use_residual:
            return self.stochastic_depth(self.conv(x)) + inputs
        else:
            return self.conv(x)


class CNNBlock(nn.Module):
    def __init__(
            self, in_channels, out_channels, kernel_size, stride, padding, groups=1
    ):
        super(CNNBlock, self).__init__()
        self.cnn = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.silu = nn.SiLU()  # SiLU <-> Swish

    def forward(self, x):
        return self.silu(self.bn(self.cnn(x)))


class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels, reduced_dim):
        super(SqueezeExcitation, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # C x H x W -> C x 1 x 1
            nn.Conv2d(in_channels, reduced_dim, 1),
            nn.SiLU(),
            nn.Conv2d(reduced_dim, in_channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.se(x)


class InvertedResidualBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            expand_ratio,
            reduction=4,  # squeeze excitation
            survival_prob=0.8,  # for stochastic depth
    ):
        super(InvertedResidualBlock, self).__init__()
        self.survival_prob = survival_prob
        self.use_residual = in_channels == out_channels and stride == 1
        hidden_dim = in_channels * expand_ratio
        self.expand = in_channels != hidden_dim
        reduced_dim = int(in_channels / reduction)

        if self.expand:
            self.expand_conv = CNNBlock(
                in_channels, hidden_dim, kernel_size=3, stride=1, padding=1,
            )

        self.conv = nn.Sequential(
            CNNBlock(
                hidden_dim, hidden_dim, kernel_size, stride, padding, groups=hidden_dim,
            ),
            SqueezeExcitation(hidden_dim, reduced_dim),
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def stochastic_depth(self, x):
        if not self.training:
            return x

        binary_tensor = torch.rand(x.shape[0], 1, 1, 1, device=x.device) < self.survival_prob
        return torch.div(x, self.survival_prob) * binary_tensor

    def forward(self, inputs):
        x = self.expand_conv(inputs) if self.expand else inputs

        if self.use_residual:
            return self.stochastic_depth(self.conv(x)) + inputs
        else:
            return self.conv(x)


class EfficientNetBasedAutoEncoder(nn.Module):
    def __init__(self, version, num_classes):
        super(EfficientNetBasedAutoEncoder, self).__init__()
        width_factor, depth_factor, dropout_rate = self.calculate_factors(version)
        last_channels = ceil(1280 * width_factor)
        self.pool = nn.AdaptiveAvgPool2d(1)
        encoder, decoder = self.create_features(width_factor, depth_factor, last_channels)
        self.encoder = nn.Sequential(*encoder)
        self.decoder = nn.Sequential(*decoder)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(last_channels, num_classes),
        )

    def calculate_factors(self, version, alpha=1.2, beta=1.1):
        phi, res, drop_rate = phi_values[version]
        depth_factor = alpha ** phi
        width_factor = beta ** phi
        return width_factor, depth_factor, drop_rate

    def create_features(self, width_factor, depth_factor, last_channels):
        channels = int(32 * width_factor)
        encoder = [CNNBlock(in_channels=1, out_channels=channels, kernel_size=3, stride=2, padding=1)]
        decoder = [DeConvCNNBlock(in_channels=channels, out_channels=1, kernel_size=3, stride=2, padding=1)]
        in_channels = channels

        for expand_ratio, channels, repeats, stride, kernel_size in base_model:
            out_channels = 4 * ceil(int(channels * width_factor) / 4)
            layers_repeats = ceil(repeats * depth_factor)

            for layer in range(layers_repeats):
                encoder.append(
                    InvertedResidualBlock(
                        in_channels,
                        out_channels,
                        expand_ratio=expand_ratio,
                        stride=stride if layer == 0 else 1,
                        kernel_size=kernel_size,
                        padding=kernel_size // 2,  # if k=1:pad=0, k=3:pad=1, k=5:pad=2
                    )
                )

                decoder.insert(
                    0,
                    DeConvInvertedResidualBlock(
                        out_channels,
                        in_channels,
                        expand_ratio=expand_ratio,
                        stride=stride if layer == 0 else 1,
                        kernel_size=kernel_size,
                        padding=kernel_size // 2,  # if k=1:pad=0, k=3:pad=1, k=5:pad=2
                    )
                )

                in_channels = out_channels

        encoder.append(CNNBlock(in_channels, last_channels, kernel_size=1, stride=1, padding=0))
        encoder.append(CNNBlock(last_channels, 10, kernel_size=1, stride=1, padding=0))

        decoder.insert(0, DeConvCNNBlock(last_channels, in_channels, kernel_size=1, stride=1, padding=0))
        decoder.insert(0, DeConvCNNBlock(10, last_channels, kernel_size=1, stride=1, padding=0))

        # encoder = nn.Sequential(*encoder)
        # decoder = nn.Sequential(*decoder)

        return encoder, decoder

    def forward(self, x):
        # x = self.pool(self.encoder(x))
        # return self.classifier(x.view(x.shape[0], -1))
        embedding = self.encoder(x)
        # x.shape()
        # embedding = self.pool(self.encoder(x))
        output = self.decoder(embedding)
        return output

    def extract_features(self, x):
        x = self.encoder(x)
        return nn.Flatten()(x)


if __name__ == '__main__':
    def test():
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(device)
        version = "b1"
        phi, res, drop_rate = phi_values[version]
        batch_size, num_classes = 16, 18

        # Creates a GradScaler once at the beginning of training.
        scaler = GradScaler()

        x = torch.randn((batch_size, 1, 480, 320), requires_grad=True).to(device)
        model1 = EfficientNetBasedAutoEncoder(
            version=version,
            num_classes=num_classes,
        ).to(device)
        model1.train()

        summary(model=model1, input_size=(1, 480, 320), batch_size=batch_size)

        # get the modules in the model. These modules should be in the order
        # the model should be executed
        # modules = [module for k, module in model1._modules.items()]

        # set the number of checkpoint segments
        # segments = 2

        with autocast():
            t = torch.cuda.get_device_properties(0).total_memory / (2 ** 30)
            r = torch.cuda.memory_reserved(0) / (2 ** 30)
            a = torch.cuda.memory_allocated(0) / (2 ** 30)
            f = r - a  # free inside reserved
            print(f'total: {t:.2f} GB, \t reserved: {r:.2f} GB, \t allocated: {a:.2f} GB, \t free: {f:.2f} GB')
            # now call the checkpoint API and get the output
            # out = checkpoint_sequential(modules, segments, x)

            print(model1(x).shape)  # (num_examples, num_classes)
            t = torch.cuda.get_device_properties(0).total_memory / (2 ** 30)
            r = torch.cuda.memory_reserved(0) / (2 ** 30)
            a = torch.cuda.memory_allocated(0) / (2 ** 30)
            f = r - a  # free inside reserved
            print(f'total: {t:.2f} GB, \t reserved: {r:.2f} GB, \t allocated: {a:.2f} GB, \t free: {f:.2f} GB')

    test()
