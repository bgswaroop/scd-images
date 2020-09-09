import torch
import torchvision
from torch import optim, nn

from cnn_models import AutoEncoder


class MyFFTTransform:
    """Rotate by one of the given angles."""

    def __init__(self, signal_ndim=3, normalized=True, onesided=False, *, direction):
        self.signal_ndim = 2
        self.normalized = False
        self.onesided = False
        self.direction = direction

    # credits: https://github.com/locuslab/pytorch_fft/blob/b3ac2c6fba5acde03c242f50865c964e3935e1aa/pytorch_fft/fft/fft.py#L230
    @staticmethod
    def roll_n(X, axis, n):
        f_idx = tuple(slice(None, None, None) if i != axis else slice(0, n, None)
                      for i in range(X.dim()))
        b_idx = tuple(slice(None, None, None) if i != axis else slice(n, None, None)
                      for i in range(X.dim()))
        front = X[f_idx]
        back = X[b_idx]
        return torch.cat([back, front], axis)

    # credits: https://github.com/locuslab/pytorch_fft/issues/9#issuecomment-350199179
    def fftshift(self, real, imag):
        for dim in range(1, len(real.size())):
            real = self.roll_n(real, axis=dim, n=real.size(dim) // 2)
            imag = self.roll_n(imag, axis=dim, n=imag.size(dim) // 2)
        return real, imag

    def ifftshift(self, real, imag):
        for dim in range(len(real.size()) - 1, 0, -1):
            real = self.roll_n(real, axis=dim, n=real.size(dim) // 2)
            imag = self.roll_n(imag, axis=dim, n=imag.size(dim) // 2)
        return real, imag

    def __call__(self, x):
        if self.direction == 'forward':
            # x = x * 255.0
            x = torch.Tensor.rfft(x, signal_ndim=self.signal_ndim, normalized=self.normalized, onesided=self.onesided)
            real, imag = self.fftshift(real=x[..., 0], imag=x[..., 1])
            x = torch.sqrt(torch.square(real) + torch.square(imag))  # magnitude of complex number
            x = torch.log(x)  # scale the values
            x = (x - torch.min(x)) / (torch.max(x) - torch.min(x))  # normalize the values
            return x
        elif self.direction == 'backward':
            # real, imag = self.ifftshift(real=x[..., 0], imag=x[..., 1])
            x = torch.Tensor.irfft(x, signal_ndim=self.signal_ndim, normalized=self.normalized, onesided=self.onesided)
            # x = x / 255.0
            return x
        else:
            raise ValueError(
                'Invalid value for direction, expected `forward` or `backward`, got {}'.format(str(self.direction)))


class Params(object):
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoEncoder().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.95, last_epoch=-1)
        self.criterion = nn.MSELoss()
        self.transform = torchvision.transforms.Compose(
            [torchvision.transforms.CenterCrop((480, 640)),
             torchvision.transforms.ToTensor(),
             MyFFTTransform(direction='forward')]
        )
        self.epochs = 10

        self.train_data = r'D:\Data\INCIBE_dataset\train'
        self.test_data = r'D:\Data\INCIBE_dataset\test'
