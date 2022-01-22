import argparse
import json
from pathlib import Path
import copy


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_folds', type=int, default=5)
    parser.add_argument('--dataset_dir', type=str, default=r'/data/p288722/datasets/dresden/source_devices/natural')
    parser.add_argument('--splits_dir', type=str, default=r'/data/p288722/datasets/dresden/splits/')
    args = parser.parse_args()

    # Validate the arguments
    if not Path(args.dataset_dir).exists():
        raise ValueError('dataset_dir does not exists!')
    Path(args.splits_dir).mkdir(parents=True, exist_ok=True)

    args.dataset_dir = Path(args.dataset_dir)
    args.splits_dir = Path(args.splits_dir)
    return args


def split(sequence, num_splits):
    """
    A generator that splits the sequence into num_splits
    :param sequence:
    :param num_splits:
    :return:
    """
    k, m = divmod(len(sequence), num_splits)
    return (sequence[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(num_splits))


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


def create_device_splits(args):
    """
    Idea: For all the images of each device, split the images into 5 equal parts. Here 5 denotes the number of folds.
    Fold 1 - Pick the first part for test and the remaining 4 for training
    Fold 2 - Pick the second part for test and the remaining 4 for training
    and so on...
    :param args:
    :return: write all the folds to disk as .json files and as a dictionary folds_data
    """

    devices = sorted(args.source_data_dir.glob('*'))
    dataset_paths = {}
    dataset_struct = {}

    # Prepare the splits
    for device_path in devices:
        image_paths = sorted([x.name for x in device_path.glob('*.JPG')], key=lambda x: int(x.split('_')[-1][:-4]))
        dataset_paths[device_path.name] = list(split(image_paths, args.num_folds))
        brand, model, device = parse_device_name(device_path.name)
        if brand not in dataset_struct:
            dataset_struct[brand] = {}
        if model not in dataset_struct[brand]:
            dataset_struct[brand][model] = {}
        if device not in dataset_struct[brand][model]:
            dataset_struct[brand][model][device] = []

    # Generate split files
    num_devices = len(devices)
    folds_data = {f'fold{x}': {'train': None, 'test': None} for x in range(1, args.num_folds + 1)}
    for fold in range(0, args.num_folds):
        test_data = copy.deepcopy(dataset_struct)
        train_data = copy.deepcopy(dataset_struct)

        for device_name, image_paths in dataset_paths.items():
            brand, model, device = parse_device_name(device_name)
            test_data[brand][model][device] = image_paths[fold]
            train_parts = image_paths[0:fold] + image_paths[fold + 1:]
            train_data[brand][model][device] = [x for sub_part in train_parts for x in sub_part]

        folds_data[f'fold{fold + 1}']['test'] = test_data
        folds_data[f'fold{fold + 1}']['train'] = train_data

        with open(args.splits_dir.joinpath(f'test_{num_devices}_devices_fold{fold + 1}.json'), 'w+') as f:
            json.dump(test_data, f, indent=2)
        with open(args.splits_dir.joinpath(f'train_{num_devices}_devices_fold{fold + 1}.json'), 'w+') as f:
            json.dump(train_data, f, indent=2)

    return folds_data


def run_flow():
    args = parse_args()
    create_device_splits(args)


if __name__ == '__main__':
    run_flow()
