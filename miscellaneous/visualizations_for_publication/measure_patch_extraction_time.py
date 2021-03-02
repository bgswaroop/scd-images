import random
import time
from collections import namedtuple

import cv2
import numpy as np


def get_patches(img_data, max_std_dev, min_std_dev, num_patches_to_extract, patch_dimensions, patches_type):
    """
    This method extracts the upto specified number of patches per image. Note that this method can return 0 patches
    if the homogeneity criteria is not met. We extract non-overlapping patches with strides same as patch sizes.
    :param patches_type:
    :param img_data: a numpy image
    :param min_std_dev: 1x3 numpy array, per channel threshold. Any patch with threshold lesser than the
    min_std_threshold will be rejected.
    :param max_std_dev: 1x3 numpy array, per channel threshold. Any patch with threshold greater than the
    max_std_threshold will be rejected.
    :param num_patches_to_extract:
    :param patch_dimensions: The size of the patch to extract, for example (128, 128)
    :return: array of extracted patches, and an empty list if no patches matched the homogeneity criteria
    """
    homogeneous_patches = []
    non_homogeneous_patches = []

    patch = namedtuple('WindowSize', ['width', 'height'])(*patch_dimensions)
    stride = namedtuple('Strides', ['width_step', 'height_step'])(patch.width // 4, patch.height // 4)
    image = namedtuple('ImageSize', ['width', 'height'])(img_data.shape[1], img_data.shape[0])
    num_channels = 3

    # Choose the patches
    for row_idx in range(patch.height, image.height, stride.height_step):
        for col_idx in range(patch.width, image.width, stride.width_step):
            img_patch = img_data[(row_idx - patch.height):row_idx, (col_idx - patch.width):col_idx]
            std_dev = np.std(img_patch.reshape(-1, num_channels), axis=0)
            if np.prod(np.less_equal(std_dev, max_std_dev)) and \
                    np.prod(np.greater_equal(std_dev, min_std_dev)):
                homogeneous_patches.extend((std_dev, img_patch))
            else:
                non_homogeneous_patches.extend((std_dev, img_patch))

    selected_patches = homogeneous_patches
    num_homogeneous_patches = len(homogeneous_patches)
    # Filter out excess patches
    if num_homogeneous_patches > num_patches_to_extract:
        random.seed(999)
        indices = random.sample(range(num_homogeneous_patches), num_patches_to_extract)
        selected_patches = [homogeneous_patches[x] for x in indices]
    # Add additional patches
    elif num_homogeneous_patches < num_patches_to_extract:
        num_additional_patches = num_patches_to_extract - num_homogeneous_patches
        non_homogeneous_patches.sort(key=lambda x: np.mean(x[0]))
        selected_patches.extend([x[1] for x in non_homogeneous_patches[:num_additional_patches]])

    return selected_patches


if __name__ == '__main__':

    image_path = r'/data/p288722/dresden/source_devices/natural/Canon_Ixus70_1/Canon_Ixus70_1_3704.JPG'
    img = cv2.imread(str(image_path))
    img = np.float32(img) / 255.0

    num_iterations = 10

    start = time.perf_counter()
    for _ in range(num_iterations):
        patches = get_patches(img_data=img,
                              max_std_dev=np.array([0.02, 0.02, 0.02]),
                              min_std_dev=np.array([0.005, 0.005, 0.005]),
                              num_patches_to_extract=200,
                              patch_dimensions=(128, 128),
                              patches_type='homogeneous')
    end = time.perf_counter()
    time_for_single_loop_iteration = (end - start) / num_iterations

    print(time_for_single_loop_iteration)
