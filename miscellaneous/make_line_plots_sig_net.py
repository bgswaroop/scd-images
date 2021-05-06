from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from sig_net.classifier_baseline.sig_net_flow import SigNetFlow


def prepare_signatures_dict(config_mode):
    signature_pairs = SigNetFlow.extract_signatures(config_mode=config_mode)
    signature_pairs = list(map(lambda x: (np.array(x[0]), Path(x[1]).parent.name), signature_pairs))
    sig_dict = {}
    for sig, device in signature_pairs:
        if device in sig_dict:
            sig_dict[device].append(sig)
        else:
            sig_dict[device] = [sig]
    return sig_dict


def make_line_plots():
    sig = prepare_signatures_dict(config_mode='train')
    # test_sig = prepare_signatures_dict(config_mode='test')

    results_dir = Path(r'D:\GitCode\auto-encoders\runtime_dir_incibe\feature_line_plots')
    for device in sig:
        plt.figure(figsize=(12, 8))
        # plt.style.use('seaborn-poster')

        axes = plt.gca()
        axes.set_ylim([-0.9, 0.9])

        # x = range(len(sig[device][0]))

        num_feat_to_plot = 500
        x = range(num_feat_to_plot)
        for feat in sig[device]:
            plt.scatter(x, feat[:num_feat_to_plot], c='r', marker='.', s=1)
            break

        plt.title(device)
        plt.xlabel('Feature dimension')
        plt.ylabel('Values')
        plt.savefig(results_dir.joinpath('one_feat_{}.png'.format(device)))
        plt.show()
        plt.close()

        break


if __name__ == '__main__':
    make_line_plots()
