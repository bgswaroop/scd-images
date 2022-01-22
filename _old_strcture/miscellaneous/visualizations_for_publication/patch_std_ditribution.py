import pickle
from collections import namedtuple
from pathlib import Path

import lmdb
import numpy as np
from matplotlib import pyplot as plt

Dist = namedtuple('Dist', 'sat, homo, non_homo')


def get_std_dev_distribution(dataset_path):
    """
    Perhaps in the experiments section it would be interesting to show a histogram of how many saturated or
    non-homogenous patches were used at camera model level for instance. Or it can be a bar plot, showing two
    (homogenous and others) or three (homogenous, saturated, non-homogenous) bars for each camera model.
    This is to show the necessity of this step

    :param dataset_path:
    :return:
    """

    max_std_dev = np.array([0.02, 0.02, 0.02])
    min_std_dev = np.array([0.005, 0.005, 0.005])

    distribution = {}
    for device_dir in dataset_path.glob('*'):
        with lmdb.open(str(device_dir), readonly=True) as env:

            model = '_'.join(device_dir.name.split('_')[:-1])
            if model == 'Nikon_D70s':
                model = 'Nikon_D70'

            sat = 0
            homo = 0
            non_homo = 0

            with env.begin() as txn:
                for patch_id, patch in txn.cursor():
                    patch = pickle.loads(patch)
                    std = np.frombuffer(patch[1], dtype=np.float32).reshape((1, 3))
                    if np.prod(np.less_equal(std, min_std_dev)):
                        sat += 1
                    elif np.prod(np.less_equal(std, max_std_dev)):
                        homo += 1
                    else:
                        non_homo += 1

            if model in distribution:
                prev_dist = distribution[model]
                distribution[model] = Dist(sat=sat + prev_dist.sat,
                                           homo=homo + prev_dist.homo,
                                           non_homo=non_homo + prev_dist.non_homo)
            else:
                distribution[model] = Dist(sat=sat, homo=homo, non_homo=non_homo)

    return distribution


def plot_std_dev_dist(distribution):
    """
    :param distribution:
    :return:
    """

    models = sorted(distribution.keys())
    # models to ignore
    models_to_exclude = ['Agfa_DC-504', 'Agfa_DC-733s', 'Agfa_DC-830i', 'Agfa_Sensor505-x', 'Agfa_Sensor530s',
                         'Canon_Ixus55', 'Canon_PowerShotA640', 'Pentax_OptioW60']

    # Filter out those camera models that are not part of the 18 camera models
    models = list(filter(lambda x: False if x in models_to_exclude else True, models))
    distribution = {x: distribution[x] for x in sorted(models)}
    num_patches = 400

    homo = np.array([x.homo / (x.homo + x.non_homo + x.sat) for _, x in distribution.items()]) * num_patches
    non_homo = np.array([x.non_homo / (x.homo + x.non_homo + x.sat) for _, x in distribution.items()]) * num_patches
    sat = np.array([x.sat / (x.homo + x.non_homo + x.sat) for _, x in distribution.items()]) * num_patches

    ind = np.arange(start=1, stop=len(distribution) + 1, step=1)

    plt.figure(figsize=(6, 3.5), dpi=300)

    width = 0.65  # the width of the bars: can also be len(x) sequence
    p1 = plt.bar(ind, homo, width, alpha=0.8)
    p2 = plt.bar(ind, non_homo, width, bottom=homo, alpha=0.8)
    p3 = plt.bar(ind, sat, width, bottom=np.add(homo, non_homo), alpha=0.6, color='red')

    plt.xlabel('Camera models')
    plt.ylabel('Average number of patches')
    plt.title('Standard deviation distribution')
    plt.xticks(ind)
    # plt.yticks(np.arange(0, 81, 10))
    plt.legend((p1[0], p2[0], p3[0]), ('Homogeneous', 'Non-homogeneous', 'Saturated'))

    plt.tight_layout()
    plt.show()
    plt.clf()
    plt.cla()
    plt.close()


if __name__ == '__main__':

    dataset_name = 'nat_patches_128x128_400'
    std_dist_file = Path(rf'/scratch/p288722/runtime_data/scd_pytorch/dev/std_dist_{dataset_name}.pkl')
    if not std_dist_file.exists():
        std_dist = get_std_dev_distribution(
            dataset_path=Path(rf'/scratch/p288722/datasets/dresden/source_devices/{dataset_name}')
        )
        with open(std_dist_file, 'wb+') as f:
            pickle.dump(std_dist, f)

    with open(std_dist_file, 'rb') as f:
        plot_std_dev_dist(distribution=pickle.load(f))
