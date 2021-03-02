import argparse
import copy
import json
import multiprocessing
import pickle
import random
import shutil
from collections import namedtuple
from pathlib import Path

import cv2
import lmdb
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
                homogeneous_patches.append((std_dev, img_patch))
            else:
                non_homogeneous_patches.append((std_dev, img_patch))

    if patches_type == 'homogeneous':
        selected_patches = homogeneous_patches
        # Filter out excess patches
        if len(homogeneous_patches) > num_patches_to_extract:
            random.seed(999)
            indices = random.sample(range(len(homogeneous_patches)), num_patches_to_extract)
            selected_patches = [homogeneous_patches[x] for x in indices]
        # Add additional patches
        elif len(homogeneous_patches) < num_patches_to_extract:
            num_additional_patches = num_patches_to_extract - len(homogeneous_patches)
            non_homogeneous_patches.sort(key=lambda x: np.mean(x[0]))
            selected_patches.extend(non_homogeneous_patches[:num_additional_patches])

    elif patches_type == 'non_homogeneous':
        selected_patches = non_homogeneous_patches
        # Filter out excess patches
        if len(non_homogeneous_patches) > num_patches_to_extract:
            random.seed(999)
            indices = random.sample(range(len(non_homogeneous_patches)), num_patches_to_extract)
            selected_patches = [non_homogeneous_patches[x] for x in indices]
        # Add additional patches
        elif len(non_homogeneous_patches) < num_patches_to_extract:
            num_additional_patches = num_patches_to_extract - len(non_homogeneous_patches)
            homogeneous_patches.sort(key=lambda x: np.mean(x[0]), reverse=True)
            selected_patches.extend(homogeneous_patches[:num_additional_patches])

    elif patches_type == 'random_selection':
        selected_patches = homogeneous_patches + non_homogeneous_patches
        # Filter out excess patches
        random.seed(999)
        indices = random.sample(range(len(selected_patches)), num_patches_to_extract)
        selected_patches = [selected_patches[x] for x in indices]

    else:
        raise ValueError(f'Invalid option for `patches_type`: {str(patches_type)}')

    return selected_patches


PatchDataInBytes = namedtuple('PatchDataInBytes', 'img, std_dev')


def extract_patches_from_dir(device, num_patches_to_extract, patch_dimensions, dest_dir):
    image_paths = list(device.glob("*"))
    estimated_img_size = 393344
    estimated_std_size = 120
    estimated_path_size = 177
    map_size = len(image_paths) * (estimated_img_size + estimated_std_size + estimated_path_size)
    map_size *= num_patches_to_extract

    # Parameters for gamma correction
    gamma = 2.2
    inverse_gamma = 1.0 / gamma
    encoding_table = np.array([((i / 255.0) ** inverse_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    decoding_table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")

    lmdb_filename = str(Path(dest_dir).joinpath(f'{device.name}'))

    with lmdb.open(lmdb_filename, map_size=map_size) as env:
        with env.begin(write=True) as txn:

            for image_path in image_paths:

                img = cv2.imread(str(image_path))

                # # Gamma Encode 2.2
                img = cv2.LUT(img, decoding_table)

                # # Hist equalization
                # img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                # img[:, :, 2] = cv2.equalizeHist(img[:, :, 2])
                # img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

                img = np.float32(img) / 255.0

                patches = get_patches(img_data=img,
                                      max_std_dev=np.array([0.02, 0.02, 0.02]),
                                      min_std_dev=np.array([0.005, 0.005, 0.005]),
                                      num_patches_to_extract=num_patches_to_extract,
                                      patch_dimensions=patch_dimensions,
                                      patches_type='homogeneous')

                for patch_id, (std, patch) in enumerate(patches, 1):
                    img_name = image_path.stem + '_{}'.format(str(patch_id).zfill(3)) + image_path.suffix
                    key = img_name.encode('ascii')
                    patch = np.uint8(patch * 255)
                    data = patch.tobytes(), std.tobytes()
                    value = pickle.dumps(data)
                    txn.put(key, value)


def extract_patches_from_hierarchical_dir(source_images_dir, dest_dataset_folder, num_patches_to_extract,
                                          patch_dimensions=(128, 128), device_id=None):
    """
    This method extracts patches from images
    :param device_id:
    :param source_images_dir: The source directory containing full sized images (not patches)
    :param dest_dataset_folder: The destination dir to save image patches
    :param num_patches_to_extract:  an int
    :param patch_dimensions: a tuple
    :return: None
    """

    devices = list(source_images_dir.glob("*"))
    # dest_dir = dest_dataset_folder.parent.joinpath(dest_dataset_folder.stem)
    # Not suitable to run in parallel mode
    # if dest_dir.exists():
    #     shutil.rmtree(dest_dir)
    dest_dataset_folder.mkdir(parents=True, exist_ok=True)

    iterable = [(device, num_patches_to_extract, patch_dimensions, dest_dataset_folder) for device in devices]
    # with multiprocessing.Pool(len(devices)) as p:
    #     p.starmap(extract_patches_from_dir, iterable)

    if device_id:
        params = iterable[device_id]
        device_name = params[0].name
        device_dir = Path(dest_dataset_folder).joinpath(f'{device_name}')
        if device_dir.exists():
            shutil.rmtree(device_dir)
        device_dir.mkdir(parents=True)
        print(str(device_dir))
        extract_patches_from_dir(*params)
    else:
        for params in iterable:
            device_name = params[0].name
            device_dir = Path(dest_dataset_folder).joinpath(f'{device_name}')
            if not device_dir.exists():
                device_dir.mkdir(parents=True)
                print(str(device_dir))
                extract_patches_from_dir(*params)

                # with lmdb.open(str(device_dir), readonly=True) as env:
                #     with env.begin() as txn:
                #         for img_id, _ in txn.cursor():
                #             print(img_id)
                #             patch = pickle.loads(txn.get(img_id))
                #             img = np.frombuffer(patch[0], dtype=np.uint8).reshape((128, 128, 3))
                #             std = np.frombuffer(patch[1], dtype=np.float).reshape((1, 3))
                #             print(type(img), type(std))


def filter_patches(source_patches_view, dest_patches_view, num_patches_to_filter):
    """
    Filter the specified number of patches into the destination directory, from a hierarchical dataset
    :param source_patches_view:
    :param dest_patches_view:
    :param num_patches_to_filter:
    :return: None
    """

    # Read the json source_images_view
    with open(source_patches_view, 'r') as f:
        source_data = json.load(f)['file_paths']

    dest_data = copy.deepcopy(source_data)
    for brand in dest_data:
        for model in dest_data[brand]:
            for device in dest_data[brand][model]:
                for image in dest_data[brand][model][device]:
                    random.seed(123)  # fixed seed to produce reproducible results
                    dest_data[brand][model][device][image] = \
                        random.sample(dest_data[brand][model][device][image], k=num_patches_to_filter)

    json_dictionary = {'file_paths': dest_data}
    json_string = json.dumps(json_dictionary, indent=2)

    dest_patches_view.parent.mkdir(parents=True, exist_ok=True)
    with open(dest_patches_view, 'w+') as f:
        f.write(json_string)


def prepare_dresden_18_models_dataset(source_dataset_folder, dest_train_dir, dest_test_dir):
    """
    Create json views for performing leave one out cross validation
    :param source_dataset_folder: Path of dataset folder
    :param dest_train_dir:
    :param dest_test_dir:
    :return: None
    """
    # Create / Clean Up results directories
    if dest_train_dir.exists():
        shutil.rmtree(dest_train_dir)
    if dest_test_dir.exists():
        shutil.rmtree(dest_test_dir)
    dest_train_dir.mkdir(exist_ok=True, parents=True)
    dest_test_dir.mkdir(exist_ok=True, parents=True)

    # Create hierarchical structure for data
    hierarchical_patch_ids = {}

    for device_dir in Path(source_dataset_folder).glob('*'):
        with lmdb.open(str(device_dir), readonly=True) as env:
            with env.begin() as txn:
                for img_id, _ in txn.cursor():

                    img_id = str(img_id, encoding='ascii')
                    parts = Path(img_id).name.split('_')[:-1]
                    image = '_'.join(parts)
                    device = '_'.join(parts[:-1])
                    model = '_'.join(parts[:-2])
                    brand = '_'.join(parts[:1])

                    if model == 'Nikon_D70s':
                        model = 'Nikon_D70'

                    if brand not in hierarchical_patch_ids:
                        hierarchical_patch_ids[brand] = {}
                    if model not in hierarchical_patch_ids[brand]:
                        hierarchical_patch_ids[brand][model] = {}
                    if device not in hierarchical_patch_ids[brand][model]:
                        hierarchical_patch_ids[brand][model][device] = {}
                    if image not in hierarchical_patch_ids[brand][model][device]:
                        hierarchical_patch_ids[brand][model][device][image] = []

                    hierarchical_patch_ids[brand][model][device][image].append(img_id)

    # Select 18 camera models, based on num_devices_per_model
    # Also simultaneously determine the number of folds
    num_folds = -1
    empty_models_dict = {}
    for brand in list(hierarchical_patch_ids.keys()):
        for model in list(hierarchical_patch_ids[brand].keys()):
            num_devices_per_model = len(hierarchical_patch_ids[brand][model])
            if num_devices_per_model <= 1:
                hierarchical_patch_ids[brand].pop(model)
            if num_folds < num_devices_per_model:
                num_folds = num_devices_per_model
        if len(hierarchical_patch_ids[brand]) == 0:
            hierarchical_patch_ids.pop(brand)
        else:
            empty_models_dict[brand] = {x: {} for x in hierarchical_patch_ids[brand]}

    for fold_id in range(num_folds):
        train_devices, test_devices = [], []
        for brand in hierarchical_patch_ids:
            for model in hierarchical_patch_ids[brand]:
                idx = fold_id % len(hierarchical_patch_ids[brand][model])
                devices = list(hierarchical_patch_ids[brand][model].keys())
                test_devices += [devices[idx]]
                train_devices += devices[0:idx] + devices[idx + 1:]

        train_fp = open(dest_train_dir.joinpath(f'fold_{fold_id + 1}.json'), 'w+')
        test_fp = open(dest_test_dir.joinpath(f'fold_{fold_id + 1}.json'), 'w+')

        for file_pointer, devices_list in [(test_fp, test_devices), (train_fp, train_devices)]:
            hierarchical_paths = copy.deepcopy(empty_models_dict)
            for device in devices_list:
                model = '_'.join(device.split('_')[:-1])
                brand = model.split('_')[0]
                if model == 'Nikon_D70s':
                    model = 'Nikon_D70'
                hierarchical_paths[brand][model][device] = hierarchical_patch_ids[brand][model][device]

            json_dictionary = {'file_paths': hierarchical_paths}
            json_string = json.dumps(json_dictionary, indent=2)
            file_pointer.write(json_string)
            file_pointer.close()


def prepare_dresden_66_devices_dataset(source_dataset_folder, dest_train_view, dest_test_view):
    """
    The 66 devices are obtained by considering all the camera models from the Dresden 18 models dataset.
    :param source_dataset_folder: Path of dataset folder
    :param dest_train_view:
    :param dest_test_view:
    :return: None
    """
    # Create parent directories
    dest_train_view.parent.mkdir(exist_ok=True, parents=True)
    dest_test_view.parent.mkdir(exist_ok=True, parents=True)

    # Create hierarchical structure for data
    hierarchical_patch_ids = {}

    for device_dir in Path(source_dataset_folder).glob('*'):
        with lmdb.open(str(device_dir), readonly=True) as env:
            with env.begin() as txn:
                for img_id, _ in txn.cursor():

                    img_id = str(img_id, encoding='ascii')
                    parts = Path(img_id).name.split('_')[:-1]
                    image = '_'.join(parts)
                    device = '_'.join(parts[:-1])
                    model = '_'.join(parts[:-2])
                    brand = '_'.join(parts[:1])

                    if model == 'Nikon_D70s':
                        model = 'Nikon_D70'

                    if brand not in hierarchical_patch_ids:
                        hierarchical_patch_ids[brand] = {}
                    if model not in hierarchical_patch_ids[brand]:
                        hierarchical_patch_ids[brand][model] = {}
                    if device not in hierarchical_patch_ids[brand][model]:
                        hierarchical_patch_ids[brand][model][device] = {}
                    if image not in hierarchical_patch_ids[brand][model][device]:
                        hierarchical_patch_ids[brand][model][device][image] = []

                    hierarchical_patch_ids[brand][model][device][image].append(img_id)

    # Select 18 camera models, based on num_devices_per_model
    empty_models_dict = {}
    for brand in list(hierarchical_patch_ids.keys()):
        for model in list(hierarchical_patch_ids[brand].keys()):
            num_devices_per_model = len(hierarchical_patch_ids[brand][model])
            if num_devices_per_model <= 1:
                hierarchical_patch_ids[brand].pop(model)
        if len(hierarchical_patch_ids[brand]) == 0:
            hierarchical_patch_ids.pop(brand)
        else:
            empty_models_dict[brand] = {x: {} for x in hierarchical_patch_ids[brand]}

    hierarchical_patch_ids_train = copy.deepcopy(hierarchical_patch_ids)
    hierarchical_patch_ids_test = copy.deepcopy(hierarchical_patch_ids)

    for brand in list(hierarchical_patch_ids.keys()):
        for model in list(hierarchical_patch_ids[brand].keys()):
            for device in list(hierarchical_patch_ids[brand][model].keys()):
                images = list(hierarchical_patch_ids[brand][model][device].keys())
                random.shuffle(images)  # in-place random shuffle

                # 80% images per device used for train and 20% for test
                train_images = images[:int(len(images) * 0.8)]
                test_images = images[int(len(images) * 0.8):]

                hierarchical_patch_ids_train[brand][model][device] = \
                    {img: hierarchical_patch_ids[brand][model][device][img] for img in train_images}
                hierarchical_patch_ids_test[brand][model][device] = \
                    {img: hierarchical_patch_ids[brand][model][device][img] for img in test_images}

    with open(dest_train_view, 'w+') as f:
        json_dictionary = {'file_paths': hierarchical_patch_ids_train}
        json_string = json.dumps(json_dictionary, indent=2)
        f.write(json_string)

    with open(dest_test_view, 'w+') as f:
        json_dictionary = {'file_paths': hierarchical_patch_ids_test}
        json_string = json.dumps(json_dictionary, indent=2)
        f.write(json_string)


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


def restructure_dataset_to_hierarchical(source_view, dest_view):
    # Read the json source_images_view
    with open(source_view, 'r') as f:
        source_images = json.load(f)['file_paths']

    image_paths = []
    for camera in source_images:
        image_paths += source_images[camera]
    image_paths.sort()

    hierarchical_paths = {}
    for img_path in image_paths:
        parts = Path(img_path).name.split('_')[:-1]
        image = '_'.join(parts)
        device = '_'.join(parts[:-1])
        model = '_'.join(parts[:-2])
        brand = '_'.join(parts[:1])

        if model == 'Nikon_D70s':
            model = 'Nikon_D70'

        if brand not in hierarchical_paths:
            hierarchical_paths[brand] = {}
        if model not in hierarchical_paths[brand]:
            hierarchical_paths[brand][model] = {}
        if device not in hierarchical_paths[brand][model]:
            hierarchical_paths[brand][model][device] = {}
        if image not in hierarchical_paths[brand][model][device]:
            hierarchical_paths[brand][model][device][image] = []

        hierarchical_paths[brand][model][device][image].append(img_path)

    json_dictionary = {'file_paths': hierarchical_paths}
    json_string = json.dumps(json_dictionary, indent=2)

    with open(dest_view, 'w+') as f:
        f.write(json_string)
        f.close()


def generate_stats_from_hierarchical_datasets(source_view):
    # Read the json source_images_view
    with open(source_view, 'r') as f:
        data = json.load(f)['file_paths']

    # Compute statistics
    num_patches_per_image_distribution = {x: 0 for x in range(1, 201)}
    image_dataset = {x: [] for x in range(1, 201)}
    total_num_images = 0
    for brand in data:
        for model in data[brand]:
            for device in data[brand][model]:
                for image in data[brand][model][device]:
                    count = len(data[brand][model][device][image])
                    num_patches_per_image_distribution[count] += 1
                    total_num_images += 1

                    patch_path = data[brand][model][device][image][0]
                    image_path = patch_path.replace('nat_patches_128x128_200', 'natural')
                    image_path = '_'.join(image_path.split('_')[:-1]) + '.JPG'
                    image_dataset[count].append(image_path)

    with open('single_patch_images.txt', 'w+') as f:
        for item in image_dataset[1]:
            f.write(f'"{item}" ')

    csv_string = 'No. patches, No. images, Proportion\n'
    for patch_count, image_count in num_patches_per_image_distribution.items():
        csv_string += f'{patch_count}, {image_count}, {image_count / total_num_images}\n'
    with open('patches_distribution_test.csv', 'w+') as f:
        f.write(csv_string)

    from multiprocessing import Pool
    paths_list = []
    for brand in data:
        for model in data[brand]:
            for device in data[brand][model]:
                file_paths = []
                for image in data[brand][model][device]:
                    file_paths.extend(data[brand][model][device][image])
                paths_list.append((file_paths, device))

    with Pool(len(paths_list)) as p:
        p.starmap(compute_std_dev, paths_list)

    # plot the histogram of standard deviations

    # std_devs_r = np.load(rf'patch_std/std_devs_r_Canon_Ixus70_2.npy')
    # std_devs_g = np.load(rf'patch_std/std_devs_g_Canon_Ixus70_2.npy')
    # std_devs_b = np.load(rf'patch_std/std_devs_b_Canon_Ixus70_2.npy')
    #
    # for x, color in [(std_devs_r, 'r'), (std_devs_g, 'g'), (std_devs_b, 'b')]:
    #     plt.figure()
    #     n, bins, patches = plt.hist(x, 100, density=True, facecolor=color, alpha=0.75)
    #     plt.xlabel('Standard Deviation')
    #     plt.ylabel('Count')
    #     plt.title(rf'Histogram of Patch StdDev - Channel {color}')
    #     plt.xlim(-0.005, 0.025)
    #     plt.grid(True)
    #     plt.savefig(rf'{color}.png')
    #     plt.close()


def compute_std_dev(file_paths, device):
    std_devs_R, std_devs_G, std_devs_B = [], [], []
    for patch in file_paths:
        img = cv2.imread(str(patch))
        img = np.float32(img) / 255.0
        patch_std = np.std(img.reshape(-1, 3), axis=0)
        std_devs_R.append(patch_std[0])
        std_devs_G.append(patch_std[1])
        std_devs_B.append(patch_std[2])

    Path(r'patch_std').mkdir(exist_ok=True, parents=True)
    np.save(f'patch_std/std_devs_r_{device}.npy', np.array(std_devs_R))
    np.save(f'patch_std/std_devs_g_{device}.npy', np.array(std_devs_G))
    np.save(f'patch_std/std_devs_b_{device}.npy', np.array(std_devs_B))


def level_from_hierarchical_dataset(source_view, dest_level, dest_view=None):
    if isinstance(source_view, dict):
        source_images = source_view
    else:
        # Read the json source_images_view
        with open(source_view, 'r') as f:
            source_images = json.load(f)['file_paths']

    level_dict = {}
    for brand in source_images:
        key = brand
        for model in source_images[brand]:
            if dest_level == 'model':
                key = model
            for device in source_images[brand][model]:
                if dest_level == 'device':
                    key = device
                for image in source_images[brand][model][device]:
                    patch_paths = source_images[brand][model][device][image]

                    if key in level_dict:
                        level_dict[key].extend(patch_paths)
                    else:
                        level_dict[key] = patch_paths

    level_dict = {x: level_dict[x] for x in sorted(level_dict.keys())}
    if dest_view:
        json_dictionary = {'file_paths': level_dict}
        json_string = json.dumps(json_dictionary, indent=2)

        Path(dest_view).parent.mkdir(parents=True, exist_ok=True)
        with open(dest_view, 'w+') as f:
            f.write(json_string)

    return level_dict


def level_balanced_from_hierarchical_dataset(source_view, dest_level, max_patches, dest_view=None):
    if isinstance(source_view, dict):
        source_images = source_view
    else:
        # Read the json source_images_view
        with open(source_view, 'r') as f:
            source_images = json.load(f)['file_paths']

    patches_distribution = {}
    images_dict = {}

    for brand in source_images:
        num_samples_per_model = max_patches / len(source_images[brand])
        key = brand

        for model in source_images[brand]:
            if dest_level == 'model':
                num_samples_per_device = max_patches / len(source_images[brand][model])
                key = model
            else:
                num_samples_per_device = num_samples_per_model / len(source_images[brand][model])

            for device in source_images[brand][model]:
                if dest_level == 'device':
                    num_samples_per_image = round(max_patches / len(source_images[brand][model][device]))
                    key = device
                else:
                    num_samples_per_image = round(num_samples_per_device / len(source_images[brand][model][device]))

                for image in source_images[brand][model][device]:
                    patches = source_images[brand][model][device][image]

                    random.seed(123)
                    random.shuffle(patches)
                    try:
                        patch_paths = patches[:num_samples_per_image]
                    except TypeError as e:
                        print(num_patches)
                        raise e

                    if key in images_dict:
                        images_dict[key].extend(patch_paths)
                    else:
                        images_dict[key] = patch_paths

                    if len(patch_paths) in patches_distribution:
                        patches_distribution[len(patch_paths)] += 1
                    else:
                        patches_distribution[len(patch_paths)] = 1

    patches_distribution = {x: patches_distribution[x] for x in
                            sorted(patches_distribution, key=lambda x: patches_distribution[x])}
    print('Patches Distribution', patches_distribution)

    images_dict = {x: images_dict[x] for x in sorted(images_dict.keys())}
    if dest_view:
        json_dictionary = {'file_paths': images_dict}
        json_string = json.dumps(json_dictionary, indent=2)

        Path(dest_view).parent.mkdir(parents=True, exist_ok=True)
        with open(dest_view, 'w+') as f:
            f.write(json_string)

    return images_dict


def remove_dir(x):
    shutil.rmtree(x)


def clean_up_directory(dir_path):
    paths = list(dir_path.glob('*'))
    with multiprocessing.Pool(len(paths)) as p:
        p.map(func=remove_dir, iterable=paths)
    shutil.rmtree(dir_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-task_num', type=int, help='enter the task number')
    args = parser.parse_args()

    patch_size = 128
    num_patches = 20
    suffix = f'{patch_size}x{patch_size}_{num_patches}'

    print(f'running the task - {args.task_num}')

    extract_patches_from_hierarchical_dir(
        source_images_dir=Path(rf'/data/p288722/dresden/source_devices/natural/'),
        dest_dataset_folder=Path(rf'/scratch/p288722/datasets/dresden/source_devices/nat_patches_tr_gamma_dec_2.2_{suffix}/'),
        num_patches_to_extract=num_patches,
        patch_dimensions=(patch_size, patch_size),
        device_id=args.task_num
    )

    # prepare_dresden_18_models_dataset(
    #     source_dataset_folder=Path(rf'/scratch/p288722/datasets/dresden/source_devices/nat_patches_{suffix}/'),
    #     dest_train_dir=Path(rf'/data/p288722/dresden/train/18_models_from200_{suffix}'),
    #     dest_test_dir=Path(rf'/data/p288722/dresden/test/18_models_from200_{suffix}')
    # )

    # prepare_dresden_66_devices_dataset(
    #     source_dataset_folder=Path(rf'/scratch/p288722/datasets/dresden/source_devices/nat_patches_{suffix}/'),
    #     dest_train_view=Path(rf'/data/p288722/dresden/train/66_devices_from200_{suffix}.json'),
    #     dest_test_view=Path(rf'/data/p288722/dresden/test/66_devices_from200_{suffix}.json')
    # )

    # for num_patches in [1, 5, 10, 20, 40, 100, 200]:
    #     suffix_filter = f'{patch_size}x{patch_size}_{num_patches}'
    #     for i in range(1, 6):
    #         filter_patches(
    #             source_patches_view=Path(rf'/data/p288722/dresden/test/18_models_random_{suffix}/fold_{i}.json'),
    #             dest_patches_view=Path(rf'/data/p288722/dresden/test/18_models_random_{suffix_filter}/fold_{i}.json'),
    #             num_patches_to_filter=num_patches
    #         )

    # generate_stats_from_hierarchical_datasets(
    #     rf'/data/p288722/dresden/test/nat_patches_18_models_{suffix}/fold_{1}.json'
    # )

    print('Finished')
