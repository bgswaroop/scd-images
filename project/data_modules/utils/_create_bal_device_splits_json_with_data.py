import argparse
import copy
import csv
import itertools
import json
import random
from collections import Counter
from pathlib import Path

import torch
import torchvision
import torchvision.transforms.functional as f
from PIL import Image
from torch import Tensor
from torchvision import transforms
from tqdm import tqdm
from collections import namedtuple


class HomogeneousTiles(torch.nn.Module):
    def __init__(self, tile_size, img_size, stride=8):
        super().__init__()
        assert img_size % tile_size == 0, 'target_img_size must be a multiple of tile_size!'

        self.tile_size = tile_size
        self.img_size = img_size
        self.stride = stride

    @torch.no_grad()
    def forward(self, x) -> Tensor:
        # RGB to luminance
        # gray_img = torch.sum(torch.mul(torch.Tensor([0.2989, 0.5870, 0.1140]).reshape((3, 1, 1)), tensor), dim=0)

        tensor, random_seed = x

        orig_tensor = f.to_tensor(tensor)
        gray_img = f.to_tensor(f.to_grayscale(tensor))
        gray_img = gray_img.to(torch.double)  # cast is necessary to mitigate overflow errors
        gray_img = f.pad(gray_img.unsqueeze(0), [1, 1, 0, 0])[0, 0]  # adding a zero border to top and left sides of img

        # Compute integral image
        i1 = torch.cumsum(torch.cumsum(gray_img, dim=0), dim=1)
        i2 = torch.cumsum(torch.cumsum(gray_img ** 2, dim=0), dim=1)

        # Determine patch locations
        num_channels, img_h, img_w = orig_tensor.shape
        h_locs = range(0, img_h - self.tile_size + 1, self.stride)
        w_locs = range(0, img_w - self.tile_size + 1, self.stride)
        tl = list(itertools.product(h_locs, w_locs))  # top-left indices
        tr = [(loc[0], loc[1] + self.tile_size) for loc in tl]  # top-right indices
        bl = [(loc[0] + self.tile_size, loc[1]) for loc in tl]  # bottom-left indices
        br = [(loc[0] + self.tile_size, loc[1]) for loc in tr]  # bottom-right indices

        # Compute standard deviations
        sum1 = torch.Tensor([i1[a] + i1[b] - i1[c] - i1[d] for (a, b, c, d) in zip(br, tl, tr, bl)])
        sum2 = torch.Tensor([i2[a] + i2[b] - i2[c] - i2[d] for (a, b, c, d) in zip(br, tl, tr, bl)])
        n = self.tile_size ** 2  # num_pixels_per_patch
        std_devs = torch.sqrt((sum2 - (sum1 ** 2) / n) / n)

        # min_std_dev = 0.005  # to exclude completely saturated patches
        # max_std_dev = 0.02  # to exclude non-homogeneous patches
        homogeneous_patch_indices = [i for i, x in enumerate(std_devs) if 0.005 <= x <= 0.02]

        # Retrieve the homogeneous patches
        num_patches = int(self.img_size / self.tile_size) ** 2
        random.seed(random_seed)
        if len(homogeneous_patch_indices) >= num_patches:
            selected_homogeneous_patch_indices = random.sample(homogeneous_patch_indices, k=num_patches)
        else:
            selected_homogeneous_patch_indices = random.choices(homogeneous_patch_indices, k=num_patches)
        # selected_homogeneous_patch_indices = torch.argsort(std_devs)[:num_patches]  # homogeneous patch ordering
        # selected_homogeneous_patch_indices = sorted(selected_homogeneous_patch_indices)  # original patch ordering

        # Prepare an homogeneous image consisting of homogeneous tiles
        selected_patches = torch.zeros((1, n * 3, num_patches))  # here, (n * 3) = num_pixels_per_patch
        for idx, patch_idx in enumerate(selected_homogeneous_patch_indices):
            h, w = tl[patch_idx]  # retrieve the height and width indices
            selected_patches[0, :, idx] = torch.flatten(orig_tensor[:, h:h + self.tile_size, w:w + self.tile_size])
        homo_patch = torch.nn.Fold(
            output_size=(self.img_size, self.img_size),
            kernel_size=(self.tile_size, self.tile_size),
            stride=self.tile_size
        )(selected_patches)[0]

        return homo_patch

    def __repr__(self):
        return self.__class__.__name__ + f'(tile_size={self.tile_size}, img_size={self.img_size}, stride={self.stride})'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_folds', type=int, default=5)
    parser.add_argument('--num_models', type=int, default=18)
    parser.add_argument('--device_id', type=int, default=None)
    parser.add_argument('--num_train_images_per_model', type=int, default=2400)  # 80 percent train
    parser.add_argument('--num_test_images_per_model', type=int, default=600)  # 20 percent test
    parser.add_argument('--num_train_images_per_device', type=int, default=500)
    parser.add_argument('--source_data_dir', type=str,
                        default=r'/data/p288722/datasets/dresden/source_devices/natural')
    parser.add_argument('--dest_data_dir', type=str,
                        default=r'/data/p288722/datasets/dresden_new/nat_homo/tiles_384_32_bal_rand_ord')
    parser.add_argument('--splits_dir', type=str,
                        default=r'/data/p288722/datasets/dresden_new/splits/')

    # Tile parameters
    parser.add_argument('--tile_size', type=int, default=32)
    parser.add_argument('--img_size', type=int, default=384)
    parser.add_argument('--stride', type=int, default=8)

    args = parser.parse_args()

    # Validate the arguments
    if not Path(args.source_data_dir).exists():
        raise ValueError('source_data_dir does not exists!')
    Path(args.splits_dir).mkdir(parents=True, exist_ok=True)
    Path(args.dest_data_dir).mkdir(parents=True, exist_ok=True)

    args.source_data_dir = Path(args.source_data_dir)
    args.splits_dir = Path(args.splits_dir)
    args.dest_data_dir = Path(args.dest_data_dir)
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


def create_homo_tiled_dataset(args, images_per_device):
    for device_id, source_device_dir in enumerate(sorted(list(args.source_data_dir.glob('*')))):

        # To enable parallel execution for each device
        if args.device_id != None and (args.device_id != device_id):
            continue

        dest_device_dir = args.dest_data_dir.joinpath(source_device_dir.name)
        dest_device_dir.mkdir(parents=True, exist_ok=True)

        homogeneous_crop = transforms.Compose([
            # torchvision.transforms.ToTensor(),
            HomogeneousTiles(tile_size=args.tile_size, img_size=args.img_size, stride=args.stride),
            torchvision.transforms.ToPILImage(),
        ])

        print(dest_device_dir)
        for dest_filename in tqdm(sorted(images_per_device[source_device_dir.name])):
            dest_img_path = dest_device_dir.joinpath(dest_filename)
            source_img_path = list(source_device_dir.glob(f'{dest_filename[:-7]}*'))[0]
            img_id = int(dest_filename.split('_')[-1][:-4])
            if not dest_img_path.exists():
                img = Image.open(source_img_path)
                img = homogeneous_crop((img, img_id))
                img.save(str(dest_img_path))
            else:
                try:
                    Image.open(dest_img_path)
                except:
                    print(f'Overwriting bad image - {dest_img_path}')
                    img = Image.open(source_img_path)
                    img = homogeneous_crop((img, img_id))
                    img.save(str(dest_img_path))


def create_device_splits(args):
    Annotations = namedtuple('Annotations',
                             'filename, model_name, device_name, position_num, scene_name')

    with open(Path(__file__).parent.resolve().joinpath('dresden_dataset_annotation.csv')) as f:

        csv_file = csv.reader(f)
        header = csv_file.__next__()
        content = [x for x in csv_file]
        content = [Annotations(x[0], f'{x[1]}_{x[2]}', f'{x[1]}_{x[2]}_{x[3]}', (int(x[5]), int(x[7])), (x[6], x[8]))
                   for x in content]

        olympus = [x for x in content if 'Olympus' in x.device_name]
        olympus = [Annotations(x[0].replace('-', '_'),
                               x[1].replace('-', '_'),
                               x[2].replace('-', '_'), x[3], x[4]) for x in olympus]
        content = [x for x in content if 'Olympus' not in x.device_name] + olympus

    available_files = set([x.name for x in args.source_data_dir.glob(f'*/*.JPG')])
    content = [x for x in content if x.filename in available_files]
    # The Dresden dataset is not available in full !!!
    # Out of 16,961 images we only have 16,186
    # Out 83 scenes we only have 79

    scenes_ids = sorted(set([x.position_num for x in content]))
    device_names = sorted(set([x.device_name for x in content]))

    scene_wise_counts = {}
    for device in device_names:
        scene_wise_counts[device] = \
            [(len([y for y in content if y.position_num == x and y.device_name == device]), x) for x in scenes_ids]

    # with open(Path(__file__).parent.resolve().joinpath('dresden_scenes_count.csv'), 'w+') as f:
    #     f.writelines([x + ',0,0,0,0,' + ','.join([str(w[0]) for w in y]) + '\n' for x, y in scene_wise_counts.items()])

    # remove the scenes where there are no images in a device
    num_folds = args.num_folds
    scenes_per_fold = {x: {} for x in range(num_folds)}
    image_count_per_fold = {x: {} for x in range(num_folds)}

    for device in list(scene_wise_counts.keys()):
        scenes_per_device = [x for x in scene_wise_counts[device] if x[0] != 0]
        num_scenes = len(scenes_per_device)

        end_idx = 0
        for fold_id in range(num_folds):
            begin_idx, end_idx = end_idx, end_idx + round((num_scenes - end_idx) / (num_folds - fold_id))
            scenes_per_fold[fold_id][device] = [x[1] for x in scenes_per_device[begin_idx:end_idx]]
            image_count_per_fold[fold_id][device] = len([x for x in content if
                                                         x.device_name == device and
                                                         x.position_num in set(scenes_per_fold[fold_id][device])])

        print(
            f'{image_count_per_fold[0][device]}, {image_count_per_fold[1][device]}, {image_count_per_fold[2][device]},'
            f' {image_count_per_fold[3][device]}, {image_count_per_fold[4][device]}')

    # Prepare an empty dataset struct
    dataset_struct = {}
    for img in content:
        brand, model, device = parse_device_name(img.device_name)
        if brand not in dataset_struct:
            dataset_struct[brand] = {}
        if model not in dataset_struct[brand]:
            dataset_struct[brand][model] = {}
        if device not in dataset_struct[brand][model]:
            dataset_struct[brand][model][device] = []

    # Number of training samples per device - 500
    # We use this number to keep the dataset balanced
    images_per_fold = {x: {'train': {}, 'test': {}} for x in range(num_folds)}
    images_per_device = {x: set() for x in device_names}
    for f in range(num_folds):
        # populate
        images_per_fold[f]['test'] = [x for x in content if x.position_num in scenes_per_fold[f][x.device_name]]
        images_per_fold[f]['train'] = [x for x in content if x.position_num not in scenes_per_fold[f][x.device_name]]

        # prepare the dictionary
        test_dict = copy.deepcopy(dataset_struct)
        for img in images_per_fold[f]['test']:
            brand, model, device = parse_device_name(img.device_name)
            test_dict[brand][model][device].append(img.filename[:-4] + '_00.png')

        train_dict = copy.deepcopy(dataset_struct)
        for img in images_per_fold[f]['train']:
            brand, model, device = parse_device_name(img.device_name)
            train_dict[brand][model][device].append(img.filename[:-4] + '_00.png')

        # balance the training data
        for brand in train_dict:
            for model in train_dict[brand]:
                for device, images in train_dict[brand][model].items():

                    test_dict[brand][model][device] = \
                        sorted(test_dict[brand][model][device],
                               key=lambda x: (int(x.split('_')[-2]), int(x.split('_')[-1][:-4])))

                    random.seed(108)
                    num_augmented_samples = args.num_train_images_per_device - len(images)
                    index = 1
                    augmentation_images = []
                    while num_augmented_samples > len(images):
                        num_augmented_samples -= len(images)
                        augmentation_images += [x[:-7] + f'_{str(index).zfill(2)}.png' for x in images]
                        index += 1
                    rand_sel_images = random.choices(images, k=num_augmented_samples)
                    augmentation_images += [x[:-7] + f'_{str(index).zfill(2)}.png' for x in rand_sel_images]
                    train_dict[brand][model][device] = \
                        sorted(images + augmentation_images,
                               key=lambda x: (int(x.split('_')[-2]), int(x.split('_')[-1][:-4])))

                    images_per_device[device].update(test_dict[brand][model][device])
                    images_per_device[device].update(train_dict[brand][model][device])

        # num_devices = len(device_names)
        # filename = args.splits_dir.joinpath(f'train_{num_devices}_devices_fold{f + 1}.json')
        # with open(filename, 'w+') as fp:
        #     json.dump(train_dict, fp, indent=2)
        # filename = args.splits_dir.joinpath(f'test_{num_devices}_devices_fold{f + 1}.json')
        # with open(filename, 'w+') as fp:
        #     json.dump(test_dict, fp, indent=2)

    return images_per_device


def run_flow():
    args = parse_args()
    images_per_device = create_device_splits(args)
    create_homo_tiled_dataset(args, images_per_device)

    # num_samples_per_image = create_balanced_model_splits(args, splits)
    # create_homo_crop_dataset(args, num_samples_per_image)


def check_json_splits():
    def ordered(obj):
        if isinstance(obj, dict):
            return sorted((k, ordered(v)) for k, v in obj.items())
        if isinstance(obj, list):
            return sorted(ordered(x) for x in obj)
        else:
            return obj

    dir1 = sorted(Path(r'/data/p288722/datasets/dresden_new/splits').glob('*.json'))
    dir2 = sorted(Path(r'/data/p288722/datasets/dresden_new/splits1').glob('*.json'))

    same = []
    for x, y in zip(dir1, dir2):
        with open(x) as f:
            x = json.load(f)
        with open(y) as f:
            y = json.load(f)
        same.append(ordered(x) == ordered(y))
    print(same)


if __name__ == '__main__':
    run_flow()
    # create_device_splits()
