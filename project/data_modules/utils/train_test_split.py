import argparse
import copy
import random
from pathlib import Path

import numpy as np
from tqdm import tqdm


def parse_args(args=None):

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_folds', type=int, default=5)
    parser.add_argument('--num_train_scenes', type=int, default=60)
    parser.add_argument('--num_train_images_per_brand', type=int, default=20_000)  # for brand-level classification
    parser.add_argument('--num_train_images_per_model', type=int, default=20_000)  # for model-level classification
    parser.add_argument('--full_image_dataset_dir', type=Path, required=True)
    parser.add_argument('--max_num_patches_per_image', type=int, default=200)

    if args:
        args = parser.parse_args(args)  # parse from custom args
    else:
        args = parser.parse_args()  # parse from command line

    # Validate the arguments
    assert args.full_image_dataset_dir.exists(), 'full_image_dataset_dir does not exists'
    return args


def parse_device_name(device_name):
    """
    Parse the name string and return the corresponding brand, model, and device names
    :param device_name: str
    :return: brand, model, device
    """
    parts = device_name.split('_')
    device = '_'.join(parts)
    model = '_'.join(parts[:-1])
    brand = '_'.join(parts[:1])

    if model == 'Nikon_D70s':
        model = 'Nikon_D70'

    return brand, model, device


def create_data_splits(args):
    """
    :param args:
    :return: folds_data
    """

    devices = sorted(args.full_image_dataset_dir.glob('*'))
    n = args.max_num_patches_per_image
    dataset_paths = {}
    dataset_struct = {}

    # Prepare the splits
    print('Preparing the splits')
    for device_dir in tqdm(devices):

        files = sorted(device_dir.glob('*'))  # image paths
        files = [x.stem + '_' + str(y).zfill(3) for x in files for y in range(1, n + 1)]  # corresponding patch names
        brand, model, device = parse_device_name(device_dir.name)

        if model not in dataset_paths:
            dataset_paths[model] = {device: files}
        else:
            dataset_paths[model][device] = files

        if brand not in dataset_struct:
            dataset_struct[brand] = {}
        if model not in dataset_struct[brand]:
            dataset_struct[brand][model] = {}

    # Consider only those models which have multiple devices
    # In order that one device can be used for test while the others are used for train
    for brand in [x for x in dataset_struct]:
        for model in [x for x in dataset_struct[brand]]:
            if len(dataset_paths[model]) < 2:
                dataset_struct[brand].pop(model)
                dataset_paths.pop(model)
        if len(dataset_struct[brand]) == 0:
            dataset_struct.pop(brand)

    # Generate splits (unbalanced data)
    folds_data = {fold: {'train': None, 'test': None} for fold in range(args.num_folds)}
    for fold in range(0, args.num_folds):
        test_data = copy.deepcopy(dataset_struct)
        train_data = copy.deepcopy(dataset_struct)

        for model, devices_per_model in dataset_paths.items():
            num_devices_per_model = len(devices_per_model)
            split_index = fold % num_devices_per_model
            brand = model.split('_')[0]
            sorted_devices = sorted([x for x in devices_per_model])
            test_device = sorted_devices[split_index]

            test_data[brand][model][test_device] = devices_per_model[test_device]
            for train_device in (sorted_devices[0:split_index] + sorted_devices[split_index + 1:]):
                train_data[brand][model][train_device] = devices_per_model[train_device]

        folds_data[fold]['test'] = test_data
        folds_data[fold]['train'] = train_data

    # print('\nCreating scene independent test set')
    # # Filter images to ensure scene-independent test set
    # Annotations = namedtuple('Annotations', 'filename, model_name, device_name, scene_id, scene_name')
    # with open(Path(__file__).parent.resolve().joinpath('dresden_dataset_annotation.csv')) as f:
    #     csv_file = csv.reader(f)
    #     csv_file.__next__()
    #     content = [x for x in csv_file]
    #     content = [
    #         Annotations(x[0][:-4], f'{x[1]}_{x[2]}', f'{x[1]}_{x[2]}_{x[3]}', (int(x[5]), int(x[7])), (x[6], x[8]))
    #         for x in content]
    #     # Correct the naming for Olympus camera models
    #     olympus = [x for x in content if 'Olympus' in x.device_name]
    #     olympus = [Annotations(x[0].replace('-', '_'),
    #                            x[1].replace('-', '_'),
    #                            x[2].replace('-', '_'), x[3], x[4]) for x in olympus]
    #     content = [x for x in content if 'Olympus' not in x.device_name] + olympus
    #
    # available_files = set([z[:-4] for _, x in dataset_paths.items() for _, y in x.items() for z in y])
    # content = [x for x in content if x.filename in available_files]
    # scenes_ids = sorted(set([x.scene_id for x in content]))
    # device_names = sorted(set([x.device_name for x in content]))
    #
    # scene_wise_counts = {}
    # for device in device_names:
    #     scene_wise_counts[device] = \
    #         [len([None for x in content if (x.scene_id == s and x.device_name == device)]) for s in scenes_ids]
    #
    # # Converting content to dictionary for easy retrieval of keys
    # content = {x.filename: x for x in content}
    # random.seed(108)
    # indices = list(range(len(scenes_ids)))
    #
    # # Create scene independent test set for all the folds
    # for fold in tqdm(range(0, args.num_folds)):
    #     train_devices = [d for (_, b) in folds_data[fold]['train'].items() for (_, m) in b.items() for d in m]
    #     test_devices = [d for (_, b) in folds_data[fold]['test'].items() for (_, m) in b.items() for d in m]
    #
    #     scene_wise_counts_train = {d: np.array(c) for d, c in scene_wise_counts.items() if d in train_devices}
    #     scene_wise_counts_test = {d: np.array(c) for d, c in scene_wise_counts.items() if d in test_devices}
    #
    #     shuffle_found = False
    #     while not shuffle_found:
    #         random.shuffle(indices)
    #         train_ids, test_ids = indices[:args.num_train_scenes], indices[args.num_train_scenes:]
    #         assert len(set(train_ids).intersection(test_ids)) == 0, 'Scene overlap found between train and test'
    #
    #         if not all([sum(c[train_ids]) for _, c in scene_wise_counts_train.items()]):
    #             continue
    #         if not all([sum(c[test_ids]) for _, c in scene_wise_counts_test.items()]):
    #             continue
    #         shuffle_found = True
    #
    #     train_scenes = set([x for (idx, x) in enumerate(scenes_ids) if idx in train_ids])
    #     test_scenes = set([x for (idx, x) in enumerate(scenes_ids) if idx in test_ids])
    #
    #     for brand, model_dict in folds_data[fold]['train'].items():
    #         for model, device_dict in model_dict.items():
    #             for device, images in device_dict.items():
    #                 images = [x for x in images if content[x[:-4]].scene_id in train_scenes]
    #                 folds_data[fold]['train'][brand][model][device] = images
    #
    #             for device, images in folds_data[fold]['test'][brand][model].items():
    #                 images = [x for x in images if content[x[:-4]].scene_id in test_scenes]
    #                 folds_data[fold]['test'][brand][model][device] = images

    # For model-level classifiers per brand (remove all other brands)
    if args.classifier_type in {"Nikon_models", "Samsung_models", "Sony_models"}:
        retain_brand = args.classifier_type.split('_')[0]
        for fold in range(0, args.num_folds):
            for brand in [x for x in folds_data[fold]['train']]:
                if retain_brand != brand:
                    folds_data[fold]['train'].pop(brand)
            for brand in [x for x in folds_data[fold]['test']]:
                if retain_brand != brand:
                    folds_data[fold]['test'].pop(brand)

    return folds_data


def balance_splits(args, unbalanced_splits):
    balanced_folds_data = unbalanced_splits
    max_ppi = args.max_num_patches_per_image
    classifier_type = args.classifier_type

    print('\nBalancing the training split')
    for fold in tqdm(range(args.num_folds)):
        for mode in ['train']:
            hierarchical_data = unbalanced_splits[fold][mode]

            for brand, model_data in hierarchical_data.items():
                if classifier_type == "all_brands":
                    num_train_imgs_per_model = int(args.num_train_images_per_brand / len(model_data))
                for model, device_data in model_data.items():

                    if classifier_type == "all_brands":
                        num_imgs_per_device = int(num_train_imgs_per_model / len(device_data))
                    else:
                        num_imgs_per_device = int(args.num_train_images_per_model / len(device_data))

                    for device, patch_data in device_data.items():
                        random.seed(108)

                        num_avail_imgs = len(patch_data) // args.max_num_patches_per_image
                        num_patches_per_img = num_imgs_per_device // num_avail_imgs
                        num_additional_patches = num_imgs_per_device - num_patches_per_img * num_avail_imgs
                        additional_image_ids = random.sample(range(num_avail_imgs), k=num_additional_patches)

                        count = np.zeros(num_avail_imgs, dtype=int)
                        count[additional_image_ids] = 1
                        count += num_patches_per_img

                        indices = set([y + idx * max_ppi for idx, x in enumerate(count) for y in range(x)])
                        patch_data = [x for idx, x in enumerate(patch_data) if idx in indices]
                        balanced_folds_data[fold][mode][brand][model][device] = patch_data

    return balanced_folds_data


def get_train_test_split(full_image_dataset_dir, fold, classifier_type):
    args = parse_args(args=['--full_image_dataset_dir', str(full_image_dataset_dir)])
    args.classifier_type = classifier_type

    splits = create_data_splits(args)
    splits = balance_splits(args, splits)
    return splits[fold]


if __name__ == '__main__':
    get_train_test_split(full_image_dataset_dir="/data2/p288722/datasets/dresden/natural",
                         fold=0, classifier_type="all_models")
