from collections import namedtuple
from pathlib import Path

import cv2
import numpy as np


def get_patches(img_data, patch_dimensions):
    """
    This method extracts the upto specified number of patches per image. Note that this method can return 0 patches
    if the homogeneity criteria is not met. We extract non-overlapping patches with strides same as patch sizes.
    :param img_data: a numpy image
    :param min_std_dev: 1x3 numpy array, per channel threshold. Any patch with threshold lesser than the
    min_std_threshold will be rejected.
    :param max_std_dev: 1x3 numpy array, per channel threshold. Any patch with threshold greater than the
    max_std_threshold will be rejected.
    :param num_patches_to_extract:
    :param patch_dimensions: The size of the patch to extract, for example (128, 128)
    :return: array of extracted patches, and an empty list if no patches matched the homogeneity criteria
    """
    image_patches = {}

    patch = namedtuple('WindowSize', ['width', 'height'])(*patch_dimensions)
    stride = namedtuple('Strides', ['width_step', 'height_step'])(patch.width, patch.height)
    image = namedtuple('ImageSize', ['width', 'height'])(img_data.shape[1], img_data.shape[0])
    num_channels = 3

    # Choose the patches
    for row_idx in range(patch.height, image.height, stride.height_step):
        for col_idx in range(patch.width, image.width, stride.width_step):
            img_patch = img_data[(row_idx - patch.height):row_idx, (col_idx - patch.width):col_idx]
            std_dev = np.std(img_patch.reshape(-1, num_channels), axis=0)
            image_patches[(row_idx - patch.height, col_idx - patch.width)] = std_dev, img_patch

    return image_patches


def alpha_blend(a, fg, bg):
    """
    :param a: alpha
    :param fg: foreground color
    :param bg: background color
    :return: RGB with alpha
    """
    return ((1 - a) * fg[0] + a * bg[0],
            (1 - a) * fg[1] + a * bg[1],
            (1 - a) * fg[2] + a * bg[2])


def plot_patch_heatmap(image_path):
    """
    Plot the heat map of standard deviations of patches
    :param image_path:
    :return: None
    """
    img = cv2.imread(str(image_path))
    s = img.shape[0] % 128, img.shape[1] % 128
    img = img[:img.shape[0] - s[0]+1, :img.shape[1] - s[1]+1]

    # Print image size
    print(f'Image size: {img.size}')

    img = np.float32(img) / 255.0
    patches = get_patches(img_data=img, patch_dimensions=(128, 128))

    img_heatmap = np.copy(img) * 0.0

    # Define a color scale for the heatmap
    # BGR
    num_shades = 100
    fg = (255, 255, 255)
    bg = (000, 000, 200)
    color_scale = {}
    for i in range(int(num_shades) + 1):
        color_scale[i] = alpha_blend(i / num_shades, fg, bg)

    # Generate the heatmap
    # std_values = np.stack([patches[x][0] for x in patches])
    # max_std = np.max(std_values, axis=0)
    # min_std = np.min(std_values, axis=0)
    std_range = np.arange(start=0.000, stop=0.35, step=0.35 / (num_shades + 1))

    for patch_loc in patches:
        std_r = patches[patch_loc][0][0]  # the second 0 is for r channel
        std_g = patches[patch_loc][0][1]  # the second 0 is for r channel
        std_b = patches[patch_loc][0][2]  # the second 0 is for r channel

        for idx, value in enumerate(std_range):
            if std_r < value:
                patch_color = color_scale[idx - 1]
                img_heatmap[patch_loc[0]:patch_loc[0]+128, patch_loc[1]:patch_loc[1]+128, :] = patch_color
                cv2.imwrite(str(image_path.parent.joinpath(
                    f'patch_'
                    f'S{str(np.round((std_r + std_g + std_b), 4)).replace(".","").ljust(5, "0")}_'
                    f'R{str(np.round(std_r, 4))[2:].zfill(4)}_'
                    f'G{str(np.round(std_g, 4))[2:].zfill(4)}_'
                    f'B{str(np.round(std_b, 4))[2:].zfill(4)}.png')),
                    patches[patch_loc][1]*255.0)
                break

    # Save the heatmap
    cv2.imwrite(str(image_path.parent.joinpath('heatmap.png')), np.uint8(img_heatmap))

    heatmap_scale = np.zeros((2000, 200, 3))
    step = int(2000/100)
    for idx, value in enumerate(std_range):
        patch_color = color_scale[100-idx]
        heatmap_scale[idx*step:(idx+1)*step, :, :] = patch_color

    # Save the color bar
    cv2.imwrite(str(image_path.parent.joinpath('heatmap_scale.png')), np.uint8(heatmap_scale))


if __name__ == '__main__':
    plot_patch_heatmap(image_path=Path(r'D:\Data\scd_publication\PXL_20201228_080717979.jpg'))
