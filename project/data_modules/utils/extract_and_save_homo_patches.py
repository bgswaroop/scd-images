import argparse
import itertools
import pickle
import random
from collections import namedtuple
from pathlib import Path

import cv2
import lmdb
import numpy as np
import torch
import torchvision.transforms.functional as f
from skimage.feature import graycomatrix, graycoprops
from tqdm import tqdm


def integral_image_based_stddev(gray_img, patch_size, indices):
    gray_img = f.pad(gray_img.unsqueeze(0), [1, 1, 0, 0])[0]  # adding a zero border to top and left sides of img

    # Compute integral image
    i1 = torch.cumsum(torch.cumsum(gray_img, dim=0), dim=1)
    i2 = torch.cumsum(torch.cumsum(gray_img ** 2, dim=0), dim=1)

    # Compute standard deviations
    tl, tr, bl, br = indices
    sum1 = torch.Tensor([i1[a] + i1[b] - i1[c] - i1[d] for (a, b, c, d) in zip(br, tl, tr, bl)])
    sum2 = torch.Tensor([i2[a] + i2[b] - i2[c] - i2[d] for (a, b, c, d) in zip(br, tl, tr, bl)])
    n = patch_size.width * patch_size.height  # num_pixels_per_patch
    std_devs = torch.sqrt((sum2 - (sum1 ** 2) / n) / n)
    return std_devs


def integral_image_based_patch_selection(img, num_patches, patch_size, stride):
    """
    This method extracts the upto specified number of patches per image. We extract overlapping patches with
    strides equal to 1/4th of patch size using integral image for efficient computation of standard deviation of
    the overlapping patches.

    :param num_patches:
    :param stride:
    :param img: a numpy image
    :param patch_size: The size of the patch to extract, for example (128, 128)

    :return: array of extracted patches, and an empty list if no patches matched the homogeneity criteria
    """
    tensor = torch.from_numpy(img / 255.0)
    min_stddev_th = torch.from_numpy(np.array([0.005, 0.005, 0.005]))
    max_stddev_th = torch.from_numpy(np.array([0.02, 0.02, 0.02]))

    # Determine patch locations
    img_h, img_w, num_channels = tensor.shape
    h_locs = range(0, img_h - patch_size.height + 1, stride.width_step)
    w_locs = range(0, img_w - patch_size.width + 1, stride.height_step)
    tl = list(itertools.product(h_locs, w_locs))  # top-left indices
    tr = [(loc[0], loc[1] + patch_size.width) for loc in tl]  # top-right indices
    bl = [(loc[0] + patch_size.height, loc[1]) for loc in tl]  # bottom-left indices
    br = [(loc[0] + patch_size.height, loc[1]) for loc in tr]  # bottom-right indices

    stddev_r = integral_image_based_stddev(gray_img=tensor[:, :, 0], patch_size=patch_size, indices=(tl, tr, bl, br))
    stddev_g = integral_image_based_stddev(gray_img=tensor[:, :, 1], patch_size=patch_size, indices=(tl, tr, bl, br))
    stddev_b = integral_image_based_stddev(gray_img=tensor[:, :, 2], patch_size=patch_size, indices=(tl, tr, bl, br))
    stddev_rgb = torch.stack([stddev_r, stddev_g, stddev_b], dim=-1)

    homogeneous_filter = torch.prod(torch.logical_and(stddev_rgb > min_stddev_th, stddev_rgb < max_stddev_th), dim=-1)
    homogeneous_indices = torch.where(homogeneous_filter)[0]

    selected_indices = homogeneous_indices[::max(1, int(len(homogeneous_indices) / num_patches))][:50]
    if len(selected_indices) < num_patches:
        saturated_filter = torch.prod(stddev_rgb < min_stddev_th, dim=-1)
        non_homogeneous_filter = torch.logical_and(torch.logical_not(saturated_filter),
                                                   torch.logical_not(homogeneous_filter))
        saturated_indices = torch.where(saturated_filter)[0]
        non_homogeneous_indices = torch.where(non_homogeneous_filter)[0]

        additional_count = num_patches - len(selected_indices)
        unselected_indices = torch.concat([non_homogeneous_indices, saturated_indices], dim=-1)
        additional_indices = torch.tensor(
            sorted(unselected_indices, key=lambda x: torch.mean(stddev_rgb[x]))[:additional_count])
        selected_indices = torch.concat([selected_indices, additional_indices], dim=-1)

    selected_patches = [img[tl[x][0]:bl[x][0], tl[x][1]:tr[x][1]] for x in selected_indices]
    selected_scores = [stddev_rgb[x].cpu().detach().numpy() for x in selected_indices]
    return list(zip(selected_scores, selected_patches))


def get_patches(img, num_patches, patch_dims, patch_type):
    """
    This method extracts the upto specified number of patches per image. We extract overlapping patches with
    strides equal to 1/4th of patch size.

    :param img: a numpy image
    :param num_patches: Number of patches to extract
    :param patch_dims: The size of the patch to extract, for example (128, 128)
    :param patch_type: type of patch (see args for options)

    :return: array of extracted patches, and an empty list if no patches matched the homogeneity criteria
    """

    patch = namedtuple('WindowSize', ['width', 'height'])(*patch_dims)
    stride = namedtuple('Strides', ['width_step', 'height_step'])(patch.width // 4, patch.height // 4)
    image = namedtuple('ImageSize', ['width', 'height'])(img.shape[1], img.shape[0])

    if patch_type == "eff_homo_stddev":
        return integral_image_based_patch_selection(img, num_patches, patch, stride)

    # Extract all the patches
    all_patches = []
    for row_idx in range(patch.height, image.height, stride.height_step):
        for col_idx in range(patch.width, image.width, stride.width_step):
            img_patch = img[(row_idx - patch.height):row_idx, (col_idx - patch.width):col_idx]
            all_patches.append(img_patch)

    if patch_type == 'homo_stddev' or patch_type == 'non_homo_stddev':

        min_stddev_th = np.array([0.005, 0.005, 0.005]) * 255.0
        max_stddev_th = np.array([0.02, 0.02, 0.02]) * 255.0
        homogeneous_patches = []
        non_homogeneous_patches = []
        num_channels = 3

        # Categorize the patches
        for img_patch in all_patches:
            std_dev = np.std(img_patch.reshape(-1, num_channels), axis=0)
            if np.prod(np.less_equal(std_dev, max_stddev_th)) and \
                    np.prod(np.greater_equal(std_dev, min_stddev_th)):
                homogeneous_patches.append((std_dev, img_patch))
            else:
                non_homogeneous_patches.append((std_dev, img_patch))

        # Select the patches
        if patch_type == 'homo_stddev':
            selected_patches = homogeneous_patches
            # Filter out excess patches
            if len(homogeneous_patches) > num_patches:
                random.seed(999)
                indices = random.sample(range(len(homogeneous_patches)), num_patches)
                selected_patches = [homogeneous_patches[x] for x in indices]
            # Add additional patches
            elif len(homogeneous_patches) < num_patches:
                num_additional_patches = num_patches - len(homogeneous_patches)
                non_homogeneous_patches.sort(key=lambda x: np.mean(x[0]))
                selected_patches.extend(non_homogeneous_patches[:num_additional_patches])

        elif patch_type == 'non_homo_stddev':
            selected_patches = non_homogeneous_patches
            # Filter out excess patches
            if len(non_homogeneous_patches) > num_patches:
                random.seed(999)
                indices = random.sample(range(len(non_homogeneous_patches)), num_patches)
                selected_patches = [non_homogeneous_patches[x] for x in indices]
            # Add additional patches
            elif len(non_homogeneous_patches) < num_patches:
                num_additional_patches = num_patches - len(non_homogeneous_patches)
                homogeneous_patches.sort(key=lambda x: np.mean(x[0]), reverse=True)
                selected_patches.extend(homogeneous_patches[:num_additional_patches])

    elif patch_type == 'homo_glcm':
        homogeneity_scores = []

        # start = time.time()
        for img_patch in all_patches:
            img = cv2.cvtColor(img_patch, cv2.COLOR_BGR2GRAY)
            glcm = graycomatrix(img, distances=[1], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=256)
            homogeneity_scores.append(np.mean(graycoprops(glcm, prop='homogeneity')))

        saturated_indices = set(np.argwhere(np.greater(homogeneity_scores, 0.95)).ravel())
        saturated_indices = [x for x in np.argsort(homogeneity_scores) if x in saturated_indices]

        non_homogeneous_indices = np.argwhere(np.less(homogeneity_scores, 0.75)).ravel()
        non_homogeneous_indices = [x for x in np.argsort(homogeneity_scores) if x in non_homogeneous_indices]

        homogeneous_indices = sorted(
            set(range(len(homogeneity_scores))).difference(saturated_indices).difference(non_homogeneous_indices))

        selected_indices = homogeneous_indices[::max(1, int(len(homogeneous_indices) / num_patches))][:num_patches]
        if len(selected_indices) < num_patches:
            selected_indices = (selected_indices + non_homogeneous_indices)[:num_patches]
        selected_patches = [(np.array([homogeneity_scores[x]]), all_patches[x]) for x in selected_indices]

        # end = time.time()
        # print(f'Time taken for processing 1 image - {end - start} sec')
        # from matplotlib import pyplot as plt
        # from mpl_toolkits.axes_grid1 import ImageGrid
        #
        # saturated_indices = np.where(is_stddev_saturated)[0]
        # homogeneity_scores = [homogeneity_scores[x] for x in saturated_indices]
        # all_patches = [all_patches[x] for x in saturated_indices]
        #
        # sorted_indices = np.argsort(homogeneity_scores)
        # plot_indices = sorted_indices[range(0, len(sorted_indices), max(1, int(len(sorted_indices) / 100)))][:100]
        # plot_images = [all_patches[x] for x in plot_indices]
        # plot_labels = [homogeneity_scores[x] for x in plot_indices]
        #
        # fig = plt.figure(figsize=(12, 12))
        # grid = ImageGrid(fig, 111, nrows_ncols=(10, 10), axes_pad=0.3)
        # for ax, im, lb in zip(grid, plot_images, plot_labels):
        #     ax.imshow(im)
        #     ax.set_title(str(round(lb, 4)))
        #     ax.set_yticklabels([])
        #     ax.set_xticklabels([])
        #     ax.set_xticks([])
        #     ax.set_yticks([])
        # # plt.title('Distribution of GLCM Homogeneity scores across image patches')
        # plt.tight_layout()
        # plt.show()
        # plt.close()

    elif patch_type == 'random':
        # Filter out excess patches
        random.seed(999)
        indices = random.sample(range(len(all_patches)), num_patches)
        selected_patches = [all_patches[x] for x in indices]
        selected_patches = [(np.array([-1]), x) for x in selected_patches]
        # -1 is just a placeholder to indicate that no score was computed during patch selection

    else:
        raise ValueError(f'Invalid option for `patches_type`: {str(patch_type)}')

    return selected_patches


def extract_patches_from_device_dir(dir_path, num_patches, patch_dims, patch_type, dest_dir):
    """
    :param dir_path: Source directory containing images from which to extract homogeneous patches
    :param num_patches: number of patches to extract per image
    :param patch_dims: a tuple for instance, (128, 128)
    :param patch_type: the type of patch to extract (see args for available options)
    :param dest_dir: destination directory to save the extracted homogeneous patches
    :return:
    """
    print(f'Processing {dir_path}')
    image_paths = list(dir_path.glob("*.JPG"))
    estimated_img_size = np.product(patch_dims) * 3 * 8
    estimated_std_size = 120
    estimated_path_size = 177
    map_size = len(image_paths) * (estimated_img_size + estimated_std_size + estimated_path_size)
    map_size *= num_patches

    lmdb_filename = str(Path(dest_dir))

    with lmdb.open(lmdb_filename, map_size=map_size) as env:
        with env.begin(write=True) as txn:

            committed_keys = set(list(txn.cursor().iternext(values=False)))
            random.shuffle(image_paths)  # shuffling the order to enable efficient parallel extraction of patches
            for image_path in tqdm(image_paths):

                last_key = (image_path.stem + '_{}'.format(str(num_patches).zfill(3))).encode('ascii')
                if last_key in committed_keys:
                    continue

                # noinspection PyUnresolvedReferences
                img = cv2.imread(str(image_path))
                patches = get_patches(img=img,
                                      num_patches=num_patches,
                                      patch_dims=patch_dims,
                                      patch_type=patch_type)

                for patch_id, (std, patch) in enumerate(patches, 1):
                    img_name = image_path.stem + '_{}'.format(str(patch_id).zfill(3))
                    key = img_name.encode('ascii')
                    data = patch.tobytes(), std.tobytes()
                    value = pickle.dumps(data)
                    txn.put(key, value)


def extract_patches_from_hierarchical_dir(source_dataset_dir, dest_dataset_dir, num_patches, patch_dims, patch_type,
                                          device_id):
    """
    This method extracts patches from images
    :param source_dataset_dir: The source directory containing the folders for each device. In turn, each device folder
    contains images from which we need to extract homogeneous patches
    :param dest_dataset_dir: The destination dir to save homogeneous patches
    :param num_patches: Number of patches to extract from each image
    :param patch_dims: a tuple for instance (128, 128)
    :param patch_type: The type of patch to extract
    :param device_id: The index of device folder in the source_dataset_dir
    :return: None
    """

    # for device in sorted(source_dataset_dir.glob("*")):
    #     print(f'{device.name} : {len(list(device.glob("*")))}')

    source_device_dir = sorted(source_dataset_dir.glob("*"))[device_id]

    dest_device_dir = Path(dest_dataset_dir).joinpath(source_device_dir.name)
    if not dest_device_dir.exists():
        dest_device_dir.mkdir(parents=True)

    extract_patches_from_device_dir(source_device_dir, num_patches, patch_dims, patch_type, dest_device_dir)


def run_extract_and_save_patches_flow(args=None):
    """
    This method needs to be called separately for each device_id (can be set in arguments).
    This is intentionally done to allow scheduling of independent lightweight jobs.
    The possible values of device_id are determined based on the number of devices of the Dresden dataset
    :return:
    """
    args = parse_args(args)
    if args.device_id:
        extract_patches_from_hierarchical_dir(
            args.source_dataset_dir,
            args.dest_dataset_dir,
            args.num_patches,
            args.patch_dims,
            args.patch_type,
            args.device_id,
        )
    else:
        print('WARNING: Running in sequential mode. To run in parallel mode check the SLURM jobscript '
              '"misc/slurm_jobscripts/extract_homo_patches.sh"')
        for device_id in range(args.num_devices):
            extract_patches_from_hierarchical_dir(
                args.source_dataset_dir,
                args.dest_dataset_dir,
                args.num_patches,
                args.patch_dims,
                args.patch_type,
                device_id,
            )


def int_or_none(value):
    if value is None or value == 'None':
        return None
    else:
        return int(value)


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_dataset_dir', type=Path,
                        default=r'/data/p288722/datasets/dresden/source_devices/natural')
    parser.add_argument('--dest_dataset_dir', type=Path,
                        default=r'/scratch/p288722/datasets/dresden/nat_homo')
    parser.add_argument('--num_patches', type=int, default=200, choices=[5, 10, 20, 50, 100, 200, 400])
    parser.add_argument('--patch_dims', type=int, default=128, choices=[32, 64, 128, 256])
    parser.add_argument('--device_id', type=int_or_none, default=22, choices=['None', None] + list(range(74)))
    parser.add_argument('--patch_type', type=str, default='eff_homo_stddev',
                        choices=['homo_stddev', 'homo_glcm', 'random', 'non_homo_stddev', 'eff_homo_stddev'])
    # fixme: delete rest of the patch extraction methods
    if args:
        args = parser.parse_args(args)
    else:
        args = parser.parse_args()

    # Validate the arguments
    if not args.source_dataset_dir.exists():
        raise ValueError('source_dataset_dir does not exists!')

    args.patch_dims = (args.patch_dims, args.patch_dims)
    args.dest_dataset_dir = args.dest_dataset_dir.joinpath(f'patches_{args.patch_dims}_{args.num_patches}')

    if args.device_id:
        args.num_devices = len(list(args.source_dataset_dir.glob('*')))
        assert 0 <= args.device_id < args.num_devices

    return args


if __name__ == '__main__':
    run_extract_and_save_patches_flow()
