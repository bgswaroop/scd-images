import pickle
from pathlib import Path

import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter

from _old_strcture.utils.training_utils import Utils


def plot_learning_curves(history_dir_brands, history_dir_nikon, history_dir_samsung, history_dir_sony):
    with open(str(Path(history_dir_brands).joinpath('history.pkl')), 'rb') as f:
        history_brands = pickle.load(f)
    with open(str(Path(history_dir_nikon).joinpath('history.pkl')), 'rb') as f:
        history_nikon = pickle.load(f)
    with open(str(Path(history_dir_samsung).joinpath('history.pkl')), 'rb') as f:
        history_samsung = pickle.load(f)
    with open(str(Path(history_dir_sony).joinpath('history.pkl')), 'rb') as f:
        history_sony = pickle.load(f)

    mpl.rcParams.update(mpl.rcParamsDefault)
    start_epoch = 2
    fig, axes = plt.subplots(nrows=3, ncols=10, figsize=(13, 3.5), dpi=300)

    for ax in axes.ravel():
        ax.remove()

    grid = plt.GridSpec(3, 9)
    grid.update(wspace=0.2, hspace=0.025)

    ax_brands = fig.add_subplot(grid[:, 6:])
    ax_nikon = fig.add_subplot(grid[0, :3])
    ax_samsung = fig.add_subplot(grid[1, :3], sharex=ax_nikon)
    ax_sony = fig.add_subplot(grid[2, :3], sharex=ax_nikon)

    ax_nikon_legend = fig.add_subplot(grid[0, 3:6], frame_on=False, xticks=[], yticks=[])
    ax_samsung_legend = fig.add_subplot(grid[1, 3:6], frame_on=False, xticks=[], yticks=[])
    ax_sony_legend = fig.add_subplot(grid[2, 3:6], frame_on=False, xticks=[], yticks=[])

    ax_brands.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax_nikon.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax_samsung.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax_sony.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

    # plot Nikon
    loss = history_nikon['loss'][start_epoch:]
    val_loss = history_nikon['val_loss'][start_epoch:]
    selected_epoch = Utils.choose_best_epoch_from_history(history_nikon) + 1
    epochs = history_nikon['epochs'][start_epoch:]

    a1, = ax_nikon.plot(epochs, loss, c='#FD681C', label='Train Nikon')
    a2, = ax_nikon.plot(epochs, val_loss, c='#FAA928', label='Test Nikon')
    a3 = ax_nikon.axvline(x=selected_epoch, label='Selected Nikon model', linestyle='--', alpha=0.40, c='#FD681C')

    # plot Samsung
    loss = history_samsung['loss'][start_epoch:]
    val_loss = history_samsung['val_loss'][start_epoch:]
    selected_epoch = Utils.choose_best_epoch_from_history(history_samsung) + 1
    epochs = history_samsung['epochs'][start_epoch:]

    a4, = ax_samsung.plot(epochs, loss, c='#4C3FD4', label='Train Samsung')
    a5, = ax_samsung.plot(epochs, val_loss, c='#3F82D4', label='Test Samsung')
    a6 = ax_samsung.axvline(x=selected_epoch, label='Selected Samsung model', linestyle='--', alpha=0.40, c='#4C3FD4')
    ax_samsung.set_ylabel('loss', labelpad=10)

    # plot Sony
    loss = history_sony['loss'][start_epoch:]
    val_loss = history_sony['val_loss'][start_epoch:]
    selected_epoch = Utils.choose_best_epoch_from_history(history_sony) + 1
    epochs = history_sony['epochs'][start_epoch:]

    a7, = ax_sony.plot(epochs, loss, c='#025E00', label='Train Sony')
    a8, = ax_sony.plot(epochs, val_loss, c='#42E82C', label='Test Sony')
    a9 = ax_sony.axvline(x=selected_epoch, label='Selected Sony model', linestyle='--', alpha=0.40, c='#025E00')
    ax_sony.set_xlabel('epochs', labelpad=10)

    ax_nikon_legend.legend(
        handles=[a1, a2, a3],
        labels=['Train Nikon', 'Test Nikon', 'Selected Nikon model      '],
        loc='center left',
        borderpad=0.8
    )
    ax_samsung_legend.legend(
        handles=[a4, a5, a6],
        labels=['Train Samsung', 'Test Samsung', 'Selected Samsung model'],
        loc='center left',
        borderpad=0.8
    )
    ax_sony_legend.legend(
        handles=[a7, a8, a9],
        labels=['Train Sony', 'Test Sony', 'Selected Sony model       '],
        loc='center left',
        borderpad=0.8
    )

    # plot brands
    loss = history_brands['loss'][start_epoch:]
    val_loss = history_brands['val_loss'][start_epoch:]
    selected_epoch = Utils.choose_best_epoch_from_history(history_brands) + 1
    epochs = history_brands['epochs'][start_epoch:]

    a10, = ax_brands.plot(epochs, loss, c='#FD681C', label='Train Brands')
    a11, = ax_brands.plot(epochs, val_loss, c='#FAA928', label='Test Brands')
    a12 = ax_brands.axvline(x=selected_epoch, label='Selected Brand model', linestyle='--', alpha=0.40, c='#FD681C')
    ax_brands.set_xlabel('epochs', labelpad=10)
    ax_brands.set_ylabel('loss', labelpad=10)

    ax_brands.legend(borderpad=0.8)

    plt.tight_layout()
    plt.savefig('models_lr.png', bbox_inches='tight')
    plt.show()
    plt.clf()
    plt.cla()
    plt.close()


if __name__ == '__main__':
    base_dir = Path(r'/scratch/p288722/runtime_data/scd_pytorch/18_models_hcal_08/fold_5')
    plot_learning_curves(
        history_dir_brands=base_dir.joinpath('signature_net_brands'),
        history_dir_nikon=base_dir.joinpath('signature_net_Nikon'),
        history_dir_samsung=base_dir.joinpath('signature_net_Samsung'),
        history_dir_sony=base_dir.joinpath('signature_net_Sony')
    )
