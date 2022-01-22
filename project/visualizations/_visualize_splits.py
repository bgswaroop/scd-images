from pathlib import Path

import matplotlib
from matplotlib import pyplot as plt
import json

if __name__ == '__main__':
    root_dir = Path(r'/home/p288722/git_code/vit-based-scd/project/data_modules/utils')
    # plt.style.use('seaborn')
    plt.figure()
    matplotlib.rc('font', size=20)
    fig, ax = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=False, figsize=(20, 10))

    for fold in [1, 2, 3, 4, 5]:
        with open(root_dir.joinpath(f'train_18_models_fold{fold}.json')) as f:
            train_data = json.load(f)
        with open(root_dir.joinpath(f'test_18_models_fold{fold}.json')) as f:
            test_data = json.load(f)

        ax[0][fold - 1].bar(range(1, 19),
                            [sum([len(z) for _, z in y.items()]) for _, x in train_data.items() for _, y in x.items()])
        ax[1][fold - 1].bar(range(1, 19),
                            [sum([len(z) for _, z in y.items()]) for _, x in test_data.items() for _, y in x.items()])

        ax[0][fold - 1].set_title(f'Fold {fold}')
        ax[1][fold - 1].set_xlabel('Camera devices')

    ax[0][0].set_ylabel('Train samples count')
    ax[1][0].set_ylabel('Test samples count')

    # plt.title('Distribution of data')
    plt.tight_layout()

    plt.show()
    plt.close()
