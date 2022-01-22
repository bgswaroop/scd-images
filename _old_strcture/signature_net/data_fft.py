from pathlib import Path

import torch
import torchvision
from PIL import Image


class MyFFTTransform:
    """Rotate by one of the given angles."""

    def __init__(self, signal_ndim=3, normalized=False, onesided=True, *, direction):
        self.signal_ndim = signal_ndim
        self.normalized = normalized
        self.onesided = onesided
        self.direction = direction
        self.eps = 1e-10

    # credits
    # https://github.com/locuslab/pytorch_fft/blob/b3ac2c6fba5acde03c242f50865c964e3935e1aa/pytorch_fft/fft/fft.py#L230
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

    # credits: https://github.com/locuslab/pytorch_fft/issues/9#issuecomment-350199179
    def ifftshift(self, real, imag):
        for dim in range(len(real.size()) - 1, 0, -1):
            real = self.roll_n(real, axis=dim, n=real.size(dim) // 2)
            imag = self.roll_n(imag, axis=dim, n=imag.size(dim) // 2)
        return real, imag

    def __call__(self, x):
        if self.direction == 'forward':
            x = torch.Tensor.rfft(x, signal_ndim=self.signal_ndim, normalized=self.normalized, onesided=self.onesided)
            # real, imag = self.fftshift(real=x[..., 0], imag=x[..., 1])
            real, imag = x[..., 0], x[..., 1]
            x = torch.sqrt(torch.square(real) + torch.square(imag))  # magnitude of complex number
            x = torch.log(x + self.eps)  # scale the values, adding eps for numerical stability
            x = (x - torch.min(x)) / (torch.max(x) - torch.min(x))  # normalize the values
            return x
        elif self.direction == 'backward':
            x = torch.Tensor.irfft(x, signal_ndim=self.signal_ndim, normalized=self.normalized, onesided=self.onesided)
            return x
        else:
            raise ValueError(
                'Invalid value for direction, expected `forward` or `backward`, got {}'.format(str(self.direction)))


# credits: https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
class Dataset(torch.utils.data.Dataset):
    """Characterizes a dataset for PyTorch"""

    def __init__(self, list_ids, labels, transform=None):
        """Initialization"""
        self.labels = labels
        self.list_ids = list_ids
        self.transform = transform

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.list_ids)

    def __getitem__(self, index):
        """Generates one sample of data"""
        # Select sample
        ID = self.list_ids[index]

        # Load data and get label
        X = Image.open(ID)
        y = self.labels[ID]

        if self.transform:
            X = self.transform(X)

        return X, y


class Data(object):
    spectrum_image_transform = torchvision.transforms.Compose(
        [torchvision.transforms.CenterCrop((480, 639)),
         torchvision.transforms.ToTensor(),
         MyFFTTransform(direction='forward')])

    @classmethod
    def prepare_torch_dataset(cls, image_paths):
        """
        :param image_paths: dataset directory or a list containing image paths
        :return: a torch dataset
        """
        if Path(image_paths).is_dir():
            image_paths = list(Path(image_paths).glob('*/*.jpg'))

        class_labels = cls.compute_avg_fft_labels(dataset=image_paths)
        img_labels = {}
        for img_path in image_paths:
            img_labels[img_path] = (class_labels[img_path.parent.name], str(img_path))
        dataset = Dataset(list_ids=image_paths, labels=img_labels, transform=cls.spectrum_image_transform)

        return dataset

    @classmethod
    def load_data(cls, dataset, config_mode):
        dataset = cls.prepare_torch_dataset(dataset)
        # Prepare for processing
        if config_mode == 'train':
            return torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)
        elif config_mode == 'test':
            return torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    @classmethod
    def load_data_for_visualization(cls, dataset, config_mode):
        dataset = cls.prepare_torch_dataset(dataset)
        # Prepare for processing
        if config_mode == 'train':
            return torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)
        elif config_mode == 'test':
            return torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)

    @classmethod
    def compute_avg_fft_labels(cls, dataset):
        """
        :param dataset: Path to a dataset directory or a list containing image paths
        :return: A dictionary with class_names as their keys and corresponding averaged fourier image as the value
        """

        if isinstance(dataset, list):
            labels = {class_folder: [] for class_folder in set([Path(y).parent.name for y in dataset])}
            img_paths = dataset
        elif Path(dataset).is_dir():
            labels = {class_folder: [] for class_folder in Path(dataset).glob('*')}
            img_paths = Path(dataset).glob('*/*')
        else:
            raise ValueError('Invalid dataset')

        for img_path in img_paths:
            X = Image.open(img_path)
            X = cls.spectrum_image_transform(X)
            labels[img_path.parent.name].append(X)

        for dir_path in labels:
            labels[dir_path] = torch.mean(torch.stack(labels[dir_path]), dim=0)

        return labels
