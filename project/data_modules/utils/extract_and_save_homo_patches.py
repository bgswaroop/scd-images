import argparse
import pickle
import random
from collections import namedtuple
from pathlib import Path

import cv2
import lmdb
import numpy as np
from tqdm import tqdm


def get_patches(img, num_patches, patch_dims, min_th, max_th, patch_type):
    """
    This method extracts the upto specified number of patches per image. We extract overlapping patches with
    strides equal to 1/4th of patch size.

    :param img: a numpy image
    :param num_patches: Number of patches to extract
    :param patch_dims: The size of the patch to extract, for example (128, 128)
    :param min_th: 1x3 numpy array, per channel threshold. Any patch with threshold lesser than the
    min_th will be rejected. The values indicate per-channel standard deviation.
    :param max_th: 1x3 numpy array, per channel threshold. Any patch with threshold greater than the
    max_th will be rejected. The values indicate per-channel standard deviation.
    :param patch_type: str with values 'homogeneous', 'non_homogeneous', or 'random_selection'.

    :return: array of extracted patches, and an empty list if no patches matched the homogeneity criteria
    """
    homogeneous_patches = []
    non_homogeneous_patches = []

    patch = namedtuple('WindowSize', ['width', 'height'])(*patch_dims)
    stride = namedtuple('Strides', ['width_step', 'height_step'])(patch.width // 4, patch.height // 4)
    image = namedtuple('ImageSize', ['width', 'height'])(img.shape[1], img.shape[0])
    num_channels = 3

    # Choose the patches
    for row_idx in range(patch.height, image.height, stride.height_step):
        for col_idx in range(patch.width, image.width, stride.width_step):
            img_patch = img[(row_idx - patch.height):row_idx, (col_idx - patch.width):col_idx]
            std_dev = np.std(img_patch.reshape(-1, num_channels), axis=0)
            if np.prod(np.less_equal(std_dev, max_th)) and \
                    np.prod(np.greater_equal(std_dev, min_th)):
                homogeneous_patches.append((std_dev, img_patch))
            else:
                non_homogeneous_patches.append((std_dev, img_patch))

    if patch_type == 'homogeneous':
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

    elif patch_type == 'non_homogeneous':
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

    elif patch_type == 'random_selection':
        selected_patches = homogeneous_patches + non_homogeneous_patches
        # Filter out excess patches
        random.seed(999)
        indices = random.sample(range(len(selected_patches)), num_patches)
        selected_patches = [selected_patches[x] for x in indices]

    else:
        raise ValueError(f'Invalid option for `patches_type`: {str(patch_type)}')

    return selected_patches


def extract_patches_from_device_dir(dir_path, num_patches, patch_dims, dest_dir):
    """
    :param dir_path: Source directory containing images from which to extract homogeneous patches
    :param num_patches:
    :param patch_dims: a tuple for instance, (128, 128)
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
            for image_path in tqdm(image_paths):

                last_key = (image_path.stem + '_{}'.format(str(num_patches).zfill(3))).encode('ascii')
                if last_key in committed_keys:
                    continue

                # noinspection PyUnresolvedReferences
                img = cv2.imread(str(image_path))
                img = np.float32(img) / 255.0

                patches = get_patches(img=img,
                                      num_patches=num_patches,
                                      patch_dims=patch_dims,
                                      min_th=np.array([0.005, 0.005, 0.005]),
                                      max_th=np.array([0.02, 0.02, 0.02]),
                                      patch_type='homogeneous')

                for patch_id, (std, patch) in enumerate(patches, 1):
                    img_name = image_path.stem + '_{}'.format(str(patch_id).zfill(3))
                    key = img_name.encode('ascii')
                    patch = np.uint8(patch * 255)
                    data = patch.tobytes(), std.tobytes()
                    value = pickle.dumps(data)
                    txn.put(key, value)


def extract_patches_from_hierarchical_dir(source_dataset_dir, dest_dataset_dir, num_patches, patch_dims, device_id):
    """
    This method extracts patches from images
    :param source_dataset_dir: The source directory containing the folders for each device. In turn, each device folder
    contains images from which we need to extract homogeneous patches
    :param dest_dataset_dir: The destination dir to save homogeneous patches
    :param num_patches: Number of patches to extract from each image
    :param patch_dims: a tuple for instance (128, 128)
    :param device_id: The index of device folder in the source_dataset_dir
    :return: None
    """

    source_device_dir = sorted(source_dataset_dir.glob("*"))[device_id]

    dest_dataset_dir.mkdir(parents=True, exist_ok=True)
    dest_device_dir = Path(dest_dataset_dir).joinpath(source_device_dir.name)
    dest_device_dir.mkdir(parents=True, exist_ok=True)

    extract_patches_from_device_dir(source_device_dir, num_patches, patch_dims, dest_device_dir)


def run_flow():
    """
    This method needs to be called for each device_id (can be set in arguments).
    The possible values of device_id are determined based on the number of devices of the Dresden dataset
    :return:
    """
    args = parse_args()
    extract_patches_from_hierarchical_dir(
        args.source_dataset_dir,
        args.dest_dataset_dir,
        args.num_patches,
        args.patch_dims,
        args.device_id
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_dataset_dir', type=Path,
                        default=r'/data/p288722/datasets/dresden/source_devices/natural')
    parser.add_argument('--dest_dataset_dir', type=Path, default=r'/scratch/p288722/datasets/dresden_new/nat_homo')
    parser.add_argument('--num_patches', type=int, default=400)
    parser.add_argument('--patch_dims', type=int, default=128)
    parser.add_argument('--device_id', type=int, default=0)

    args = parser.parse_args()

    # Validate the arguments
    if not args.source_dataset_dir.exists():
        raise ValueError('source_dataset_dir does not exists!')
    args.source_dataset_dir.mkdir(parents=True, exist_ok=True)

    assert args.patch_dims in {32, 64, 128, 256}
    assert args.num_patches in {5, 10, 20, 50, 100, 200, 400}

    args.patch_dims = (args.patch_dims, args.patch_dims)
    args.dest_dataset_dir = args.dest_dataset_dir.joinpath(f'patches_{args.patch_dims}_{args.num_patches}')
    args.dest_dataset_dir.mkdir(parents=True, exist_ok=True)

    num_devices = len(list(args.source_dataset_dir.glob('*')))
    assert 0 <= args.device_id < num_devices

    return args


if __name__ == '__main__':
    run_flow()
