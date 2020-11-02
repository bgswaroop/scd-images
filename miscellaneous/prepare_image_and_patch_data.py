import json
import os
import random
import shutil
from collections import namedtuple
from pathlib import Path

import numpy as np


def get_patches(img_data, std_threshold, max_num_patches, patch_size):
    """
    This method extracts the upto specified number of patches per image. Note that this method can return 0 patches
    if the homogeneity criteria is not met. We extract non-overlapping patches with strides same as patch sizes.
    :param img_data: a numpy image
    :param std_threshold: 1x3 numpy array, per channel threshold. Any patch with threshold greater than the
    std_threshold will be rejected.
    :param max_num_patches:
    :param patch_size: The size of the patch to extract, for example (128, 128)
    :return: array of extracted patches, and an empty list if no patches matched the homogeneity criteria
    """
    patches = []

    patch = namedtuple('WindowSize', ['width', 'height'])(*patch_size)
    stride = namedtuple('Strides', ['width_step', 'height_step'])(*patch_size)
    image = namedtuple('ImageSize', ['width', 'height'])(img_data.shape[1], img_data.shape[0])
    num_channels = 3

    # Choose the patches
    patch_with_least_std = None
    least_std = np.array([np.inf, np.inf, np.inf])
    for row_idx in range(patch.height, image.height, stride.height_step):
        for col_idx in range(patch.width, image.width, stride.width_step):
            cropped_img = img_data[(row_idx - patch.height):row_idx, (col_idx - patch.width):col_idx]
            patch_std = np.std(cropped_img.reshape(-1, num_channels), axis=0)
            if np.prod(np.less_equal(patch_std, std_threshold)):
                patches.append(cropped_img)
            if all(patch_std < least_std):
                least_std = patch_std
                patch_with_least_std = cropped_img

    # Filter out excess patches
    if len(patches) == 0:
        patches = [patch_with_least_std]
    if len(patches) > max_num_patches:
        random.seed(999)
        indices = random.sample(range(len(patches)), max_num_patches)
        patches = [patches[x] for x in indices]

    return patches


def extract_patches_from_images(source_images_dir, dest_patches_dir, max_num_patches=15, patch_size=(128, 128)):
    """
    This method extracts patches from images
    :param source_images_dir: The source directory containing full sized images (not patches)
    :param dest_patches_dir: The destination dir to save image patches
    :param max_num_patches:  an int
    :param patch_size: a tuple
    :return: None
    """
    import cv2

    devices = source_images_dir.glob("*")
    # Removing directory is not suited to run in parallel mode
    # if dest_patches_dir.exists():
    #     shutil.rmtree(dest_patches_dir)
    dest_patches_dir.mkdir(exist_ok=True, parents=True)

    for device in devices:
        image_paths = device.glob("*")
        destination_device_dir = dest_patches_dir.joinpath(device.name)

        # The following if-else construct makes sense on running multiple instances of this method
        if destination_device_dir.exists():
            continue
        else:
            destination_device_dir.mkdir(exist_ok=True, parents=True)

        for image_path in image_paths:
            img = cv2.imread(str(image_path))
            img = np.float32(img) / 255.0

            # img_name = image_path.stem + '_{}'.format(str(1).zfill(3)) + image_path.suffix
            # img_path = destination_device_dir.joinpath(img_name)
            # if img_path.exists():
            #     continue

            patches = get_patches(img_data=img, std_threshold=np.array([0.02, 0.02, 0.02]),
                                  max_num_patches=max_num_patches, patch_size=patch_size)
            for patch_id, patch in enumerate(patches, 1):
                img_name = image_path.stem + '_{}'.format(str(patch_id).zfill(3)) + image_path.suffix
                img_path = destination_device_dir.joinpath(img_name)
                cv2.imwrite(str(img_path), patch * 255.0)


def balance_patches(unbalanced_dir, balanced_dir):
    # Remove old directories
    if balanced_dir.exists():
        shutil.rmtree(balanced_dir)

    # Construct a hierarchical dictionary
    # device_names
    #   |-- image names
    #        |-- patch names
    patches_dictionary = {}
    for device in unbalanced_dir.glob('*'):
        patches_dictionary[device.name] = {}
        for patch in device.glob('*'):
            image_name = '_'.join(patch.name.split('_')[:-1])
            if image_name not in patches_dictionary[device.name]:
                patches_dictionary[device.name][image_name] = [patch]
            else:
                patches_dictionary[device.name][image_name].append(patch)

    # determine the minimum num of images per device
    min_num_samples = float('inf')
    for device_name in patches_dictionary:
        num_samples = len(patches_dictionary[device_name])
        if num_samples < min_num_samples:
            min_num_samples = num_samples
    print("Number of images in each class : {}".format(min_num_samples))

    # Create the directory structure along with symbolic links
    for device_name in patches_dictionary:
        # create device subdir in the destination folder
        subdir = balanced_dir.joinpath(device_name)
        subdir.mkdir(parents=True, exist_ok=True)

        # randomly select min_num_images
        images = list(patches_dictionary[device_name].keys())
        random.seed(123)  # fixed seed to produce reproducible results
        random.shuffle(images)
        images = images[:min_num_samples]

        # create symlinks
        for image in images:
            for patch_path in patches_dictionary[device_name][image]:
                symlink = subdir.joinpath(patch_path.name)
                if not symlink.exists():
                    os.symlink(src=patch_path, dst=symlink)


def regroup_devices_into_models(devices_dir, models_dir):
    # Step 0: Remove old directories
    if models_dir.exists():
        shutil.rmtree(models_dir)

    # Step 1: Create the directory structure along with symbolic links
    input_path = devices_dir.glob('*')
    for item in input_path:
        subdir = models_dir.joinpath("{}".format("_".join(item.parts[-1].split("_")[:-1])))
        Path(subdir).mkdir(parents=True, exist_ok=True)
        for img_path in item.glob("*"):
            symlink = subdir.joinpath(img_path.name)
            if not symlink.exists():
                os.symlink(src=img_path, dst=symlink)


def map_source_images_from_patches(patches_dir, source_images_dir, dest_images_dir):
    """
    Create a dataset of images based on the extracted patches.
    This is sometimes necessary as we might have to modify the patches dataset, and at a later stage perform
    comparative experiments on the whole images.
    :param patches_dir:
    :param source_images_dir:
    :param dest_images_dir:
    :return:
    """
    # Remove old directories
    if dest_images_dir.exists():
        shutil.rmtree(dest_images_dir)

    patches_dictionary = {}
    for device in patches_dir.glob('*'):
        patches_dictionary[device.name] = {}
        for patch in device.glob('*'):
            image_name = '_'.join(patch.name.split('_')[:-1])
            if image_name not in patches_dictionary[device.name]:
                patches_dictionary[device.name][image_name] = [patch]
            else:
                patches_dictionary[device.name][image_name].append(patch)

    dest_images_dir.mkdir(parents=True, exist_ok=True)
    for device in source_images_dir.glob('*'):
        subdir = dest_images_dir.joinpath(device.name)
        subdir.mkdir(exist_ok=True)

        for image_path in device.glob('*'):
            if image_path.stem in patches_dictionary[device.name]:
                symlink = subdir.joinpath(image_path.name)
                if not symlink.exists():
                    os.symlink(src=image_path, dst=symlink)


def sample_images(source_dir, destination_dir, num_images_to_sample):
    """
    This method samples only specified number of images from each class.
    :param source_dir:
    :param destination_dir:
    :param num_images_to_sample:
    :return:
    """
    destination_dir.mkdir(parents=True, exist_ok=True)
    for device_path in source_dir.glob('*'):
        # create device subdir in the destination folder
        subdir = destination_dir.joinpath(device_path.name)
        subdir.mkdir(parents=True, exist_ok=True)

        # randomly select min_num_images
        images = list(device_path.glob('*'))
        assert num_images_to_sample <= len(
            images), 'Number of images to be sampled is more than the num of images in each class'
        random.seed(123)  # fixed seed to produce reproducible results
        random.shuffle(images)
        images = images[:num_images_to_sample]

        # create symlinks
        for image_path in images:
            symlink = subdir.joinpath(image_path.name)
            if not symlink.exists():
                os.symlink(src=image_path, dst=symlink)


def split_known_unknown(source_dir, input_dir, data_type, num_patches_per_image=None):
    # keep out few devices aside to compute the accuracy on open set of devices
    unknown_models_unknown_devices = ['Sony_DSC-W170_0', 'Sony_DSC-W170_1', 'Agfa_Sensor505-x_0', 'Agfa_Sensor530s_0',
                                      'Canon_Ixus55_0', 'Panasonic_DMC-FZ50_0', 'Panasonic_DMC-FZ50_1',
                                      'Panasonic_DMC-FZ50_2']
    known_models_unknown_devices = ['Sony_DSC-T77_2', 'Sony_DSC-T77_3', 'Samsung_NV15_2', 'Samsung_L74wide_2',
                                    'Canon_Ixus70_2', 'Casio_EX-Z150_3', 'Casio_EX-Z150_4', 'Nikon_CoolPixS710_4']
    # unknown_cameras = unknown_models_unknown_devices + known_models_unknown_devices

    if data_type == 'image':
        kmkd = input_dir.parent.joinpath("{}_bal_kmkd".format(input_dir.name))
        umud = input_dir.parent.joinpath("{}_bal_umud".format(input_dir.name))
        kmud = input_dir.parent.joinpath("{}_bal_kmud".format(input_dir.name))

        for device in input_dir.glob('*'):
            if device.name in unknown_models_unknown_devices:
                subdir = umud.joinpath(device.name)
            elif device.name in known_models_unknown_devices:
                subdir = kmud.joinpath(device.name)
            else:
                subdir = kmkd.joinpath(device.name)

            subdir.mkdir(exist_ok=True, parents=True)
            for image in device.glob('*'):
                symlink = subdir.joinpath(image.name)
                source_path = source_dir.joinpath(device.name).joinpath(image.name)
                if not symlink.exists():
                    os.symlink(src=source_path, dst=symlink)

    if data_type == 'patch':
        if num_patches_per_image:
            kmkd = input_dir.parent.joinpath("{}_kmkd_{}".format(input_dir.name, num_patches_per_image))
            umud = input_dir.parent.joinpath("{}_umud_{}".format(input_dir.name, num_patches_per_image))
            kmud = input_dir.parent.joinpath("{}_kmud_{}".format(input_dir.name, num_patches_per_image))
        else:
            kmkd = input_dir.parent.joinpath("{}_kmkd".format(input_dir.name))
            umud = input_dir.parent.joinpath("{}_umud".format(input_dir.name))
            kmud = input_dir.parent.joinpath("{}_kmud".format(input_dir.name))

        patches_dictionary = {}
        for device in input_dir.glob('*'):
            patches_dictionary[device.name] = {}
            for patch in device.glob('*'):
                image_name = '_'.join(patch.name.split('_')[:-1])
                if image_name not in patches_dictionary[device.name]:
                    patches_dictionary[device.name][image_name] = [patch]
                else:
                    patches_dictionary[device.name][image_name].append(patch)

        for device in input_dir.glob('*'):
            if device.name in unknown_models_unknown_devices:
                subdir = umud.joinpath(device.name)
            elif device.name in known_models_unknown_devices:
                subdir = kmud.joinpath(device.name)
            else:
                subdir = kmkd.joinpath(device.name)

            subdir.mkdir(exist_ok=True, parents=True)
            for image_name in patches_dictionary[device.name]:
                # randomly select min_num_images

                images = patches_dictionary[device.name][image_name]
                if num_patches_per_image:
                    random.seed(123)  # fixed seed to produce reproducible results
                    random.shuffle(images)
                    images = images[:num_patches_per_image]

                for image in images:
                    symlink = subdir.joinpath(image.name)
                    source_path = source_dir.joinpath(device.name).joinpath(image.name)
                    if not symlink.exists():
                        os.symlink(src=source_path, dst=symlink)


def filter_patches(source_patches_view, dest_patches_view, num_patches=1):
    """
    Filter the specified number of patches into the destination directory.
    :param source_patches_view:
    :param dest_patches_view:
    :param num_patches:
    :return:
    """

    # Read the json source_images_view
    with open(source_patches_view, 'r') as f:
        source_patches_dict = json.load(f)

    patches_dictionary = {}
    for device_name in source_patches_dict['file_paths']:
        patches_dictionary[device_name] = {}
        for patch_path in source_patches_dict['file_paths'][device_name]:
            image_name = '_'.join(Path(patch_path).name.split('_')[:-1])
            if image_name not in patches_dictionary[device_name]:
                patches_dictionary[device_name][image_name] = [patch_path]
            else:
                patches_dictionary[device_name][image_name].append(patch_path)

    dest_patches_view.parent.mkdir(parents=True, exist_ok=True)
    labels_dictionary = {}
    for device_name in source_patches_dict['file_paths']:

        for image_name in patches_dictionary[device_name]:
            random.seed(123)  # fixed seed to produce reproducible results
            patches = patches_dictionary[device_name][image_name]
            random.shuffle(patches)
            patches = patches[:num_patches]

            if device_name not in labels_dictionary:
                labels_dictionary[device_name] = patches
            else:
                labels_dictionary[device_name] += patches

    json_dictionary = {'file_paths': labels_dictionary}
    json_string = json.dumps(json_dictionary, indent=2)

    with open(dest_patches_view, 'w+') as f:
        f.write(json_string)
        f.close()


def dresden_create_18_models(source_dir, train_dir, test_dir, keep_dest_models=True):
    """
    Create views for performing leave one out cross validation
    """
    if train_dir.exists():
        shutil.rmtree(train_dir)
    if test_dir.exists():
        shutil.rmtree(test_dir)
    train_dir.mkdir(exist_ok=True, parents=True)
    test_dir.mkdir(exist_ok=True, parents=True)

    all_devices = list(source_dir.glob('*'))
    cameras_dict = {}
    for device_path in all_devices:
        if device_path.name == 'Nikon_D70s_0' or device_path.name == 'Nikon_D70s_1':
            cam_model = 'Nikon_D70'
        else:
            cam_model = '_'.join(device_path.name.split('_')[:-1])

        if cam_model not in cameras_dict:
            cameras_dict[cam_model] = [device_path.name]
        else:
            cameras_dict[cam_model].append(device_path.name)

    min_devices = len(all_devices)
    max_devices = 1
    cam_models_to_remove = []
    for cam_model in cameras_dict:
        num_devices = len(cameras_dict[cam_model])
        if num_devices <= 1:
            cam_models_to_remove.append(cam_model)
        else:
            if num_devices < min_devices:
                min_devices = num_devices
            if num_devices > max_devices:
                max_devices = num_devices
    for cam_model in cam_models_to_remove:
        del cameras_dict[cam_model]

    for fold_id in range(max_devices):
        train_devices, test_devices = [], []
        for cam_model in cameras_dict:
            idx = fold_id % len(cameras_dict[cam_model])
            test_devices += [cameras_dict[cam_model][idx]]
            train_devices += cameras_dict[cam_model][0:idx] + cameras_dict[cam_model][idx + 1:]

        train_fp = open(train_dir.joinpath(f'fold_{fold_id + 1}.json'), 'w+')
        test_fp = open(test_dir.joinpath(f'fold_{fold_id + 1}.json'), 'w+')

        for file_pointer, devices_list in [(train_fp, train_devices), (test_fp, test_devices)]:
            labels_dictionary = {}
            for device_path in [source_dir.joinpath(x) for x in devices_list]:
                images = [str(x) for x in device_path.glob('*')]
                if device_path.name == 'Nikon_D70s_0':
                    device_name = 'Nikon_D70_2'
                elif device_path.name == 'Nikon_D70s_1':
                    device_name = 'Nikon_D70_3'
                else:
                    device_name = device_path.name

                if keep_dest_models:
                    camera_name = '_'.join(device_name.split('_')[:-1])
                else:
                    camera_name = device_name

                if camera_name not in labels_dictionary:
                    labels_dictionary[camera_name] = images
                else:
                    labels_dictionary[camera_name] += images

            json_dictionary = {'file_paths': labels_dictionary}
            json_string = json.dumps(json_dictionary, indent=2)
            file_pointer.write(json_string)
            file_pointer.close()


def dresden_filter_patches_from_images(source_images_view, source_patches_dir, dest_patches_view, num_patches=None,
                                       keep_dest_models=True):
    patches_dictionary = {}
    for device_path in source_patches_dir.glob('*'):
        if device_path.name == 'Nikon_D70s_0':
            device_name = 'Nikon_D70_2'
        elif device_path.name == 'Nikon_D70s_1':
            device_name = 'Nikon_D70_3'
        else:
            device_name = device_path.name

        if keep_dest_models:
            camera_name = '_'.join(device_name.split('_')[:-1])
        else:
            camera_name = device_name

        if camera_name not in patches_dictionary:
            patches_dictionary[camera_name] = {}
        for patch_path in device_path.glob('*'):
            image_name = '_'.join(patch_path.name.split('_')[:-1])
            if image_name not in patches_dictionary[camera_name]:
                patches_dictionary[camera_name][image_name] = [str(patch_path)]
            else:
                patches_dictionary[camera_name][image_name].append(str(patch_path))

    dest_patches_view.parent.mkdir(parents=True, exist_ok=True)

    # Read the json source_images_view
    with open(source_images_view, 'r') as f:
        source_images_dict = json.load(f)

    labels_dictionary = {}
    for camera_name in source_images_dict['file_paths']:
        for image_path in source_images_dict['file_paths'][camera_name]:
            image_path = Path(image_path)
            random.seed(123)  # fixed seed to produce reproducible results
            if image_path.stem not in patches_dictionary[camera_name]:
                continue

            patches = patches_dictionary[camera_name][image_path.stem]
            random.shuffle(patches)
            if num_patches:
                patches = patches[:num_patches]

            if camera_name not in labels_dictionary:
                labels_dictionary[camera_name] = patches
            else:
                labels_dictionary[camera_name] += patches

    json_dictionary = {'file_paths': labels_dictionary}
    json_string = json.dumps(json_dictionary, indent=2)

    with open(dest_patches_view, 'w+') as f:
        f.write(json_string)
        f.close()


if __name__ == "__main__":
    # from random import randint
    # from time import sleep
    #
    # sleep(randint(10, 100))
    #
    # extract_patches_from_images(source_images_dir=Path(rf'/data/p288722/dresden/source_devices/natural/'),
    #                             dest_patches_dir=Path(rf'/data/p288722/dresden/source_devices/nat_patches_128x128_20/'),
    #                             max_num_patches=20,
    #                             patch_size=(128, 128))
    #
    dresden_create_18_models(source_dir=Path(r'/data/p288722/dresden/source_devices/natural/'),
                             train_dir=Path(r'/data/p288722/dresden/train/nat_images_18_devices'),
                             test_dir=Path(r'/data/p288722/dresden/test/nat_images_18_devices'),
                             keep_dest_models=False)

    # fold_id = 2
    # dresden_filter_patches_from_images(
    #     source_images_view=Path(rf'/data/p288722/dresden/test/nat_images_18_devices/fold_{fold_id}.json'),
    #     source_patches_dir=Path(rf'/data/p288722/dresden/source_devices/nat_patches_128x128_20/'),
    #     dest_patches_view=Path(
    #         rf'/data/p288722/dresden/test/nat_patches_18_models_devices_128x128_20/fold_{fold_id}.json'),
    #     keep_dest_models=False
    # )

    for fold_id in range(1, 6):
        dresden_filter_patches_from_images(
            source_images_view=Path(rf'/data/p288722/dresden/train/nat_images_18_devices/fold_{fold_id}.json'),
            source_patches_dir=Path(rf'/data/p288722/dresden/source_devices/nat_patches_128x128_20/'),
            dest_patches_view=Path(rf'/data/p288722/dresden/train/nat_patches_18_models_devices_128x128_20/fold_{fold_id}.json'),
            keep_dest_models=False
        )
        dresden_filter_patches_from_images(
            source_images_view=Path(rf'/data/p288722/dresden/test/nat_images_18_devices/fold_{fold_id}.json'),
            source_patches_dir=Path(rf'/data/p288722/dresden/source_devices/nat_patches_128x128_20/'),
            dest_patches_view=Path(rf'/data/p288722/dresden/test/nat_patches_18_models_devices_128x128_20/fold_{fold_id}.json'),
            keep_dest_models=False
        )

    for num_patches in [1, 3, 10, 15]:
        for fold_id in range(1, 6):
            filter_patches(
                source_patches_view=Path(
                    rf'/data/p288722/dresden/train/nat_patches_18_models_devices_128x128_20/fold_{fold_id}.json'),
                dest_patches_view=Path(
                    rf'/data/p288722/dresden/train/nat_patches_18_models_devices_128x128_{num_patches}/fold_{fold_id}.json'),
                num_patches=num_patches)
            filter_patches(
                source_patches_view=Path(
                    rf'/data/p288722/dresden/test/nat_patches_18_models_devices_128x128_20/fold_{fold_id}.json'),
                dest_patches_view=Path(
                    rf'/data/p288722/dresden/test/nat_patches_18_models_devices_128x128_{num_patches}/fold_{fold_id}.json'),
                num_patches=num_patches)
    else:
        print(__name__)
