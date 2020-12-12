import copy
import json
import multiprocessing
import os
import random
import shutil
import tarfile
from collections import namedtuple
from pathlib import Path
import tables

import cv2
import numpy as np


def get_patches(img_data, max_std_dev, min_std_dev, num_patches_to_extract, patch_dimensions):
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
                homogeneous_patches.append(img_patch)
            else:
                non_homogeneous_patches.append((std_dev, img_patch))

    # Filter out excess patches
    if len(homogeneous_patches) > num_patches_to_extract:
        random.seed(999)
        indices = random.sample(range(len(homogeneous_patches)), num_patches_to_extract)
        homogeneous_patches = [homogeneous_patches[x] for x in indices]
    # Add additional patches
    elif len(homogeneous_patches) < num_patches_to_extract:
        num_additional_patches = num_patches_to_extract - len(homogeneous_patches)
        non_homogeneous_patches.sort(key=lambda x: np.mean(x[0]))
        homogeneous_patches.extend([x[1] for x in non_homogeneous_patches[:num_additional_patches]])

    return homogeneous_patches


def extract_patches_from_dir(device, num_patches_to_extract, patch_dimensions, dest_dir):

    class PatchData(tables.IsDescription):
        path = tables.StringCol(128)
        data = tables.UInt8Col((*patch_dimensions, 3))
        std_dev = tables.Float32Col((1, 3))

    # Create a new table
    device_name = device.name.replace('-', '_').replace('.', '_')
    h5_filename = Path(dest_dir).joinpath(f'{device_name}.h5')
    with tables.open_file(h5_filename, 'w') as h5file:
        table = h5file.create_table(where='/', name=device_name, description=PatchData)
        data_row = table.row

        image_paths = device.glob("*")
        for image_path in image_paths:
            img = cv2.imread(str(image_path))
            img = np.float32(img) / 255.0

            patches = get_patches(img_data=img,
                                  max_std_dev=np.array([0.02, 0.02, 0.02]),
                                  min_std_dev=np.array([0.005, 0.005, 0.005]),
                                  num_patches_to_extract=num_patches_to_extract, patch_dimensions=patch_dimensions)
            for patch_id, patch in enumerate(patches, 1):
                img_name = image_path.stem + '_{}'.format(str(patch_id).zfill(3)) + image_path.suffix
                img_name = img_name.replace('-', '_').replace('.', '_')
                data_row['path'] = img_name
                data_row['data'] = np.uint8(patch * 255.0)
                data_row['std_dev'] = np.std(patch.reshape(-1, 3), axis=0)
                data_row.append()
        table.flush()


def extract_patches_from_hierarchical_dir(source_images_dir, dest_dataset_file, num_patches_to_extract,
                                          patch_dimensions=(128, 128)):
    """
    This method extracts patches from images
    :param source_images_dir: The source directory containing full sized images (not patches)
    :param dest_dataset_file: The destination dir to save image patches
    :param num_patches_to_extract:  an int
    :param patch_dimensions: a tuple
    :return: None
    """

    devices = list(source_images_dir.glob("*"))
    dest_dir = dest_dataset_file.parent.joinpath(dest_dataset_file.stem)
    # Not suitable to run in parallel mode
    # if dest_dir.exists():
    #     shutil.rmtree(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    iterable = [(device, num_patches_to_extract, patch_dimensions, dest_dir) for device in devices]
    # with multiprocessing.Pool(len(devices)) as p:
    #     p.starmap(extract_patches_from_dir, iterable)

    # for params in iterable:
    #     device_name = params[0].name.replace('-', '_').replace('.', '_')
    #     h5_filename = Path(dest_dir).joinpath(f'{device_name}.h5')
    #     if not h5_filename.exists():
    #         with tables.open_file(h5_filename, 'w'):
    #             pass
    #         extract_patches_from_dir(*params)

    # # Combine the datasets into a single .h5py file
    # class PatchData(tables.IsDescription):
    #     path = tables.StringCol(128)
    #     data = tables.UInt8Col((*patch_dimensions, 3))
    #     std_dev = tables.Float32Col((1, 3))
    #
    # with tables.open_file(str(dest_dataset_file), mode="w", title="dataset") as h5_dataset:
    #     table = h5_dataset.create_table(where='/', name='img_patches', description=PatchData)
    #     data_row = table.row

    for device_dataset_path in dest_dir.glob('*'):
        with tables.open_file(str(device_dataset_path), mode="r") as h5_device:
            for device in h5_device.root:
                print(device)
                for row in device.iterrows():
                    print(row['path'])
                    break
                    # data_row['path'] = row['path']
                    # data_row['data'] = row['data']
                    # data_row['std_dev'] = row['std_dev']
                    # data_row.append()
    # table.flush()

    # with tables.open_file(str(dest_dataset_file), mode="r") as h5_dataset:
    #     for row in h5_dataset.root.img_patches.iterrows():
    #         path = row['path']
    #         data = row['data']
    #         std_dev = row['std_dev']
    #         print(path, std_dev, type(data))


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


def dresden_create_18_models(source_dir, train_dir, test_dir):
    """
    Create views for performing leave one out cross validation
    """
    if train_dir.exists():
        shutil.rmtree(train_dir)
    if test_dir.exists():
        shutil.rmtree(test_dir)
    train_dir.mkdir(exist_ok=True, parents=True)
    test_dir.mkdir(exist_ok=True, parents=True)

    if source_dir.suffix == '.tar':
        all_devices = []
        with tarfile.open(source_dir, 'r:') as tar:
            for member in tar:
                if member.isfile():
                    all_devices.append(member.name)
    else:
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

        for file_pointer, devices_list in [(test_fp, test_devices), (train_fp, train_devices)]:
            hierarchical_paths = {}
            for device_path in [source_dir.joinpath(x) for x in devices_list]:
                images = [str(x) for x in device_path.glob('*')]
                for img_path in images:
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
            file_pointer.write(json_string)
            file_pointer.close()


def dresden_18_models_dataset_from_tar(source_tar_file, dest_train_dir, dest_test_dir):
    """
    Create json views for performing leave one out cross validation
    :param source_tar_file: Path of tar file containing dataset of image patches
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
    with tarfile.open(source_tar_file, 'r:') as tar:
        for member in tar:
            if Path(member.name).suffix.upper() in {'.JPG', '.PNG', '.JPEG'}:

                parts = Path(member.name).name.split('_')[:-1]
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

                hierarchical_patch_ids[brand][model][device][image].append(member.name)

    # Select 18 camera models, based on num_devices_per_model
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
    # from random import randint
    # from time import sleep
    # sleep(randint(1, 120))

    patch_size = 128
    num_patches = 5
    suffix = f'{patch_size}x{patch_size}_{num_patches}'

    # clean_up_directory(dir_path=Path(rf'/data/p288722/dresden/source_devices/nat_patches_{suffix}/'))

    extract_patches_from_hierarchical_dir(
        source_images_dir=Path(rf'/data/p288722/dresden/source_devices/natural/'),
        dest_dataset_file=Path(rf'/data/p288722/dresden/source_devices/nat_patches_{suffix}.h5/'),
        num_patches_to_extract=num_patches,
        patch_dimensions=(patch_size, patch_size)
    )

    # import os
    # for file_path in list(Path(rf'/data/p288722/dresden/source_devices/nat_patches_{suffix}').glob('*')):
    #     # os.rename(str(file_path), str(file_path.parent.joinpath(f'{file_path.stem}.h5py')))
    #     # os.rename(str(file_path), str(file_path.parent.joinpath(f'{file_path.stem}.h5')))
    #     with tables.open_file(str(file_path), 'r+') as h5_device:
    #         for device in h5_device.root:
    #             if not device.indexed:
    #                 device.cols.path.create_index()
    #             # for row in device.iterrows():
    #             #     print(row)
    #             #     print('')

    # dresden_18_models_dataset_from_tar(
    #     source_tar_file=Path(rf'/data/p288722/dresden/source_devices/nat_patches_{suffix}.tar/'),
    #     dest_train_dir=Path(rf'/data/p288722/dresden/train/18_models_{suffix}'),
    #     dest_test_dir=Path(rf'/data/p288722/dresden/test/18_models_{suffix}')
    # )

    # generate_stats_from_hierarchical_datasets(
    #     rf'/data/p288722/dresden/test/nat_patches_18_models_{suffix}/fold_{1}.json'
    # )

    # for num_patches in [60, 75]:
    #     for fold_id in range(1, 6):
    #         filter_patches(
    #             source_patches_view=Path(
    #                 rf'/data/p288722/dresden/train/nat_patches_sony_models_128x128_90/fold_{fold_id}.json'),
    #             dest_patches_view=Path(
    #                 rf'/data/p288722/dresden/train/nat_patches_sony_models_128x128_{num_patches}/fold_{fold_id}.json'),
    #             num_patches=num_patches)
    #         filter_patches(
    #             source_patches_view=Path(
    #                 rf'/data/p288722/dresden/test/nat_patches_sony_models_128x128_90/fold_{fold_id}.json'),
    #             dest_patches_view=Path(
    #                 rf'/data/p288722/dresden/test/nat_patches_sony_models_128x128_{num_patches}/fold_{fold_id}.json'),
    #             num_patches=num_patches)
    # else:
    #     print(__name__)

    print('Finished')
