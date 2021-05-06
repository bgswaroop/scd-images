import argparse
import copy
import json
import pickle
import random
import shutil
from pathlib import Path

import lmdb
import torch
import torchvision
from PIL import Image
import logging

logger = logging.getLogger(__name__)


def pre_process_device_folder(source_dir, crop_dimensions, dest_dir):
    image_paths = list(source_dir.glob("*"))
    dest_dir = Path(dest_dir).joinpath(source_dir.name)
    dest_dir.mkdir(exist_ok=True, parents=True)
    print(str(dest_dir))

    eps = 1e-10

    # Parameters for lmdb database
    estimated_img_size = 3686426
    map_size = len(image_paths) * 25 * (127 + estimated_img_size)
    lmdb_filename = str(Path(dest_dir).joinpath(f'{source_dir.name}'))

    with lmdb.open(lmdb_filename, map_size=map_size) as env:
        with env.begin(write=True) as txn:
            for idx, image_path in enumerate(image_paths):
                x = Image.open(image_path)  # pillow reads the image with the  sequence of RGB (unlike openCV)
                x = torchvision.transforms.CenterCrop(crop_dimensions)(x)
                x = torchvision.transforms.ToTensor()(x)
                x = torch.fft.rfft2(x)
                real, imag = x.real, x.imag

                x = torch.sqrt(torch.square(real) + torch.square(imag))  # magnitude of complex number
                x = torch.log(x + eps)  # scale the values, adding eps for numerical stability
                mag = (x - torch.min(x)) / (torch.max(x) - torch.min(x))  # normalize the values
                phase = torch.atan(imag / real)

                img_name = image_path.stem + image_path.suffix
                key = img_name.encode('ascii')
                data = mag.numpy().tobytes(), phase.numpy().tobytes()
                value = pickle.dumps(data)
                txn.put(key, value)


def pre_process_data(source_images_dir, dest_dataset_folder, crop_dimensions, device_id=None):
    """
    This method extracts patches from images
    :param device_id:
    :param source_images_dir: The source directory containing full sized images
    :param dest_dataset_folder: The destination dir to save pre processed images
    :param crop_dimensions: a tuple
    :return: None
    """

    devices = list(source_images_dir.glob("*"))
    # dest_dir = dest_dataset_folder.parent.joinpath(dest_dataset_folder.stem)
    # Not suitable to run in parallel mode
    # if dest_dir.exists():
    #     shutil.rmtree(dest_dir)
    dest_dataset_folder.mkdir(parents=True, exist_ok=True)

    iterable = [(device, crop_dimensions, dest_dataset_folder) for device in devices]
    # with multiprocessing.Pool(len(devices)) as p:
    #     p.starmap(extract_patches_from_dir, iterable)

    if device_id is not None:
        params = iterable[device_id]
        pre_process_device_folder(*params)
    else:
        for params in iterable:
            device_name = params[0].name
            device_dir = Path(dest_dataset_folder).joinpath(f'{device_name}')
            if not device_dir.exists():
                device_dir.mkdir(parents=True)
                print(str(device_dir))
                pre_process_device_folder(*params)

                # with lmdb.open(str(device_dir), readonly=True) as env:
                #     with env.begin() as txn:
                #         for img_id, _ in txn.cursor():
                #             print(img_id)
                #             patch = pickle.loads(txn.get(img_id))
                #             img = np.frombuffer(patch[0], dtype=np.uint8).reshape((128, 128, 3))
                #             std = np.frombuffer(patch[1], dtype=np.float).reshape((1, 3))
                #             print(type(img), type(std))


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
    hierarchical_image_ids = {}

    for device_dir in Path(source_dataset_folder).glob('*'):
        device_id = device_dir.name
        parts = device_id.split('_')
        device = '_'.join(parts)
        model = '_'.join(parts[:-1])
        brand = '_'.join(parts[:1])

        if model == 'Nikon_D70s':
            model = 'Nikon_D70'

        if brand not in hierarchical_image_ids:
            hierarchical_image_ids[brand] = {}
        if model not in hierarchical_image_ids[brand]:
            hierarchical_image_ids[brand][model] = {}
        if device not in hierarchical_image_ids[brand][model]:
            hierarchical_image_ids[brand][model][device] = [x.name for x in device_dir.glob('*')]

    # Select 18 camera models, based on num_devices_per_model
    # Also simultaneously determine the number of folds
    num_folds = -1
    empty_models_dict = {}
    for brand in list(hierarchical_image_ids.keys()):
        for model in list(hierarchical_image_ids[brand].keys()):
            num_devices_per_model = len(hierarchical_image_ids[brand][model])
            if num_devices_per_model <= 1:
                hierarchical_image_ids[brand].pop(model)
            if num_folds < num_devices_per_model:
                num_folds = num_devices_per_model
        if len(hierarchical_image_ids[brand]) == 0:
            hierarchical_image_ids.pop(brand)
        else:
            empty_models_dict[brand] = {x: {} for x in hierarchical_image_ids[brand]}

    for fold_id in range(num_folds):
        train_devices, test_devices = [], []
        for brand in hierarchical_image_ids:
            for model in hierarchical_image_ids[brand]:
                idx = fold_id % len(hierarchical_image_ids[brand][model])
                devices = list(hierarchical_image_ids[brand][model].keys())
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
                hierarchical_paths[brand][model][device] = hierarchical_image_ids[brand][model][device]

            json_dictionary = {'file_paths': hierarchical_paths}
            json_string = json.dumps(json_dictionary, indent=2)
            file_pointer.write(json_string)
            file_pointer.close()


def hierarchical_to_flat_view(source_view, dest_level, dest_view=None):
    """
    convert a hierarchical view into a flat view
    :param source_view: The source file path (or) an equivalent dictionary
    :param dest_level: This is str can take the values - 'brand', 'model', and 'device'
    :param dest_view: The destination file path
    :return: A dictionary with the flat view
    """
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

                image_paths = source_images[brand][model][device]
                if key in level_dict:
                    level_dict[key].extend(image_paths)
                else:
                    level_dict[key] = image_paths

    level_dict = {x: level_dict[x] for x in sorted(level_dict.keys())}
    if dest_view:
        json_dictionary = {'file_paths': level_dict}
        json_string = json.dumps(json_dictionary, indent=2)

        Path(dest_view).parent.mkdir(parents=True, exist_ok=True)
        with open(dest_view, 'w+') as f:
            f.write(json_string)

    return level_dict


def hierarchical_to_flat_view_with_balancing(source_view, dest_level, num_samples_per_brand=None, dest_view=None):
    """
    convert a hierarchical view into a flat view
    :param num_samples_per_brand:
    :param source_view: The source file path (or) an equivalent dictionary
    :param dest_level: This is str can take the values - 'brand', 'model', and 'device'
    :param dest_view: The destination file path
    :return: A dictionary with the flat view
    """
    if isinstance(source_view, dict):
        source_images = source_view
    else:
        # Read the json source_images_view
        with open(source_view, 'r') as f:
            source_images = json.load(f)['file_paths']

    image_distribution = {}
    images_dict = {}

    if num_samples_per_brand is None:
        num_samples_per_brand = 999_999_999_999    # start with a very high initial value
        for brand in source_images:
            for model in source_images[brand]:
                for device in source_images[brand][model]:
                    if len(source_images[brand][model][device]) < num_samples_per_brand:
                        num_samples_per_brand = len(source_images[brand][model][device])
    logger.info(f'Number of samples per brand - {num_samples_per_brand}')

    for brand in source_images:
        num_samples_per_model = num_samples_per_brand / len(source_images[brand])
        key = brand

        for model in source_images[brand]:
            if dest_level == 'model':
                num_samples_per_device = num_samples_per_brand / len(source_images[brand][model])
                key = model
            else:
                num_samples_per_device = num_samples_per_model / len(source_images[brand][model])

            for device in source_images[brand][model]:
                if dest_level == 'device':
                    key = device

                images = source_images[brand][model][device]
                random.shuffle(images)

                try:
                    image_paths = images[:round(num_samples_per_device)]
                except TypeError as e:
                    logger.error('Check the number of images and the number of images being sampled')
                    raise e

                if key in images_dict:
                    images_dict[key].extend(image_paths)
                else:
                    images_dict[key] = image_paths

                if len(image_paths) in image_distribution:
                    image_distribution[len(image_paths)] += 1
                else:
                    image_distribution[len(image_paths)] = 1

    image_distribution = {x: image_distribution[x] for x in
                          sorted(image_distribution, key=lambda x: image_distribution[x])}
    print('Image distribution', image_distribution)

    images_dict = {x: images_dict[x] for x in sorted(images_dict.keys())}
    if dest_view:
        json_dictionary = {'file_paths': images_dict}
        json_string = json.dumps(json_dictionary, indent=2)

        Path(dest_view).parent.mkdir(parents=True, exist_ok=True)
        with open(dest_view, 'w+') as f:
            f.write(json_string)

    return images_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-task_num', type=int, help='enter the task number')
    args = parser.parse_args()
    print(f'running the task - {args.task_num}')

    suffix = f'{320}_{480}'
    pre_suffix = 'fft_grey_mag'

    # pre_process_data(
    #     source_images_dir=Path(rf'/data/p288722/dresden/source_devices/natural/'),
    #     dest_dataset_folder=Path(
    #         rf'/scratch/p288722/datasets/dresden/source_devices/natural_images_{pre_suffix}_{suffix}/'),
    #     crop_dimensions=(480, 639),
    #     device_id=args.task_num
    # )

    prepare_dresden_18_models_dataset(
        source_dataset_folder=Path(rf'/data/p288722/dresden/source_devices/natural/'),
        dest_train_dir=Path(rf'/data/p288722/dresden/train/18_models_image_level'),
        dest_test_dir=Path(rf'/data/p288722/dresden/test/18_models_image_level')
    )

    print('Finished')
