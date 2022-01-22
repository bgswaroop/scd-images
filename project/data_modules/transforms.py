import torch
from torch import Tensor
import itertools
import torchvision.transforms.functional as f


class HomogeneousCrop(torch.nn.Module):
    def __init__(self, size: int, stride: int = 64):
        """
        Return the most homogeneous crop of dimensions (size, size) based on the standard deviation of image tiles.
        The tiles are sampled from the input image based the value of the stride.
        :param size: target crop size
        :param stride: stride value for image tiling
        """
        super().__init__()
        self.size = size, size
        self.stride = stride

    @torch.no_grad()
    def forward(self, tensor: Tensor) -> Tensor:
        # extract patches
        # patches = size(num_pixels_per_patch, num_patches)
        patches = torch.nn.Unfold(self.size, dilation=1, padding=0, stride=self.stride)(tensor.unsqueeze(0))[0]

        # Compute standard deviations
        # std_devs = size(num_patches)
        std_devs = torch.std(patches, dim=0)
        min_stddev_patch_index = torch.argmin(std_devs)

        # retrieve the most homogeneous patch
        homo_patch = patches[:, min_stddev_patch_index]
        homo_patch = torch.nn.Fold(self.size, self.size)(homo_patch.unsqueeze(0).unsqueeze(2))[0]

        return homo_patch

    def __repr__(self):
        return self.__class__.__name__ + f'(size={self.size}, stride={self.stride})'


class HomogeneousCropEfficient(torch.nn.Module):
    def __init__(self, size: int, stride: int = 64):
        """
        Return the most homogeneous crop of dimensions (size, size) based on the standard deviation of image tiles.
        The tiles are sampled from the input image based the value of the stride.
        :param size: target crop size
        :param stride: stride value for image tiling
        """
        super().__init__()
        self.patch_height = size
        self.patch_width = size
        self.stride = stride

    @torch.no_grad()
    def forward(self, tensor: Tensor) -> Tensor:
        # # RGB to luminance
        # gray_img = torch.sum(torch.mul(torch.Tensor([0.2989, 0.5870, 0.1140]).reshape((3, 1, 1)), tensor), dim=0)
        # RGB to grayscale
        gray_img = torch.mean(tensor, dim=0)
        gray_img = f.pad(gray_img.unsqueeze(0), [1, 1, 0, 0])[0]  # adding a zero border to top and left sides of img

        # Compute integral image
        i1 = torch.cumsum(torch.cumsum(gray_img, dim=0), dim=1)
        i2 = torch.cumsum(torch.cumsum(gray_img ** 2, dim=0), dim=1)

        # Determine patch locations
        num_channels, img_h, img_w = tensor.shape
        h_locs = range(0, img_h - self.patch_height + 1, self.stride)
        w_locs = range(0, img_w - self.patch_width + 1, self.stride)
        tl = list(itertools.product(h_locs, w_locs))  # top-left indices
        tr = [(loc[0], loc[1] + self.patch_width) for loc in tl]  # top-right indices
        bl = [(loc[0] + self.patch_height, loc[1]) for loc in tl]  # bottom-left indices
        br = [(loc[0] + self.patch_height, loc[1]) for loc in tr]  # bottom-right indices

        # Compute standard deviations
        sum1 = torch.Tensor([i1[a] + i1[b] - i1[c] - i1[d] for (a, b, c, d) in zip(br, tl, tr, bl)])
        sum2 = torch.Tensor([i2[a] + i2[b] - i2[c] - i2[d] for (a, b, c, d) in zip(br, tl, tr, bl)])
        n = self.patch_height * self.patch_width  # num_pixels_per_patch
        std_devs = torch.sqrt((sum2 - (sum1 ** 2) / n) / n)

        # Retrieve the most homogeneous patch
        min_stddev_index = torch.argmin(std_devs)
        h_index, w_index = tl[min_stddev_index]
        homo_patch = tensor[:, h_index:h_index + self.patch_height, w_index:w_index + self.patch_width]

        return homo_patch

    def __repr__(self):
        return self.__class__.__name__ + f'(size={(self.patch_height, self.patch_width)}, stride={self.stride})'


class HomogeneousTiles(torch.nn.Module):
    def __init__(self, tile_size, img_size, stride=8):
        super().__init__()
        assert img_size % tile_size == 0, 'target_img_size must be a multiple of tile_size!'

        self.tile_size = tile_size
        self.img_size = img_size
        self.stride = stride

    @torch.no_grad()
    def forward(self, tensor: Tensor) -> Tensor:
        # RGB to luminance
        gray_img = torch.sum(torch.mul(torch.Tensor([0.2989, 0.5870, 0.1140]).reshape((3, 1, 1)), tensor), dim=0)
        gray_img = f.pad(gray_img.unsqueeze(0), [1, 1, 0, 0])[0]  # adding a zero border to top and left sides of img

        # Compute integral image
        i1 = torch.cumsum(torch.cumsum(gray_img, dim=0), dim=1)
        i2 = torch.cumsum(torch.cumsum(gray_img ** 2, dim=0), dim=1)

        # Determine patch locations
        num_channels, img_h, img_w = tensor.shape
        h_locs = range(0, img_h - self.tile_size + 1, self.stride)
        w_locs = range(0, img_w - self.tile_size + 1, self.stride)
        tl = list(itertools.product(h_locs, w_locs))  # top-left indices
        tr = [(loc[0], loc[1] + self.tile_size) for loc in tl]  # top-right indices
        bl = [(loc[0] + self.tile_size, loc[1]) for loc in tl]  # bottom-left indices
        br = [(loc[0] + self.tile_size, loc[1]) for loc in tr]  # bottom-right indices

        # Compute standard deviations
        sum1 = torch.Tensor([i1[a] + i1[b] - i1[c] - i1[d] for (a, b, c, d) in zip(br, tl, tr, bl)])
        sum2 = torch.Tensor([i2[a] + i2[b] - i2[c] - i2[d] for (a, b, c, d) in zip(br, tl, tr, bl)])
        n = self.tile_size ** 2  # num_pixels_per_patch
        std_devs = torch.sqrt((sum2 - (sum1 ** 2) / n) / n)

        # Retrieve the most homogeneous patches
        num_patches = int(self.img_size / self.tile_size) ** 2
        most_homogeneous_patch_indices = torch.argsort(std_devs)[:num_patches]  # homogeneous patch ordering
        # most_homogeneous_patch_indices = sorted(most_homogeneous_patch_indices)  # original patch ordering

        # Prepare an homogeneous image consisting of homogeneous tiles
        selected_patches = torch.zeros((1, n * 3, num_patches))  # here, (n * 3) = num_pixels_per_patch
        for idx, patch_idx in enumerate(most_homogeneous_patch_indices):
            h, w = tl[patch_idx]  # retrieve the height and width indices
            selected_patches[0, :, idx] = torch.flatten(tensor[:, h:h + self.tile_size, w:w + self.tile_size])
        homo_patch = torch.nn.Fold(
            output_size=(self.img_size, self.img_size),
            kernel_size=(self.tile_size, self.tile_size),
            stride=self.tile_size
        )(selected_patches)[0]

        return homo_patch

    def __repr__(self):
        return self.__class__.__name__ + f'(tile_size={self.tile_size}, img_size={self.img_size}, stride={self.stride})'
