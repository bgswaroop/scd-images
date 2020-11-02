import logging

from configure import Configure, SigNet, SimNet
from signature_net.sig_net_flow import SigNetFlow
from similarity_net.sim_net_flow import SimNetFlow
from utils.logging import SetupLogger, log_running_time

logger = logging.getLogger(__name__)
import argparse


@log_running_time
def run_flow(fold_id):
    """
    A high level function to run the train and evaluation flows
    :return: None
    """

    # Signature Net
    Configure.sig_net_name = f'signature_net_{fold_id}'
    Configure.train_data = rf'/data/p288722/dresden/train/nat_patches_18_models_128x128_15/fold_{fold_id}.json'
    Configure.test_data = rf'/data/p288722/dresden/test/nat_patches_18_models_128x128_15/fold_{fold_id}.json'
    Configure.create_runtime_dirs()

    SigNetFlow.train()
    SigNetFlow.classify()

    # Utils.visualize_ae_input_output_pairs()
    # Utils.save_avg_fourier_images()

    # Configure.train_data = r'/data/p288722/dresden/train/nat_patches_bal_kmkd_1/'
    # Configure.test_data = r'/data/p288722/dresden/test/nat_patches_bal_kmkd_1/'

    # Similarity Net
    # SimNetFlow.train()

    # Configure.train_data = r'/data/p288722/dresden/train/nat_patches_bal_kmkd_10/'
    # Configure.test_data = r'/data/p288722/dresden/test/nat_patches_bal_kmkd_10/'
    #
    # SimNet.balance_classes = False
    # SimNetFlow.classify()

    # SimNetFlow.classify()


@log_running_time
def run_cross_validation_flow():
    for fold_id in range(1, 6):
        # Signature Net
        Configure.train_data = rf'/data/p288722/dresden/train/nat_patches_18_models_128x128_15/fold_{fold_id}.json'
        Configure.test_data = rf'/data/p288722/dresden/test/nat_patches_18_models_128x128_15/fold_{fold_id}.json'
        Configure.sig_net_name = f'signature_net_{fold_id}'
        Configure.create_runtime_dirs()

        SigNetFlow.train()
        SigNetFlow.classify()

        # Similarity Net
        Configure.train_data = rf'/data/p288722/dresden/train/nat_patches_18_models_128x128_3/fold_{fold_id}.json'
        Configure.test_data = rf'/data/p288722/dresden/test/nat_patches_18_models_128x128_3/fold_{fold_id}.json'
        Configure.sim_net_name = f'similarity_net_{fold_id}'
        Configure.create_runtime_dirs()

        SimNet.balance_classes = True
        SimNetFlow.train()
        SimNet.balance_classes = False
        SimNetFlow.classify()


if __name__ == '__main__':
    import inspect

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-fold', type=str, help='enter fold number')
    args = parser.parse_args()
    fold_id = 1

    SetupLogger(log_file=Configure.runtime_dir.joinpath(f'scd_{fold_id}.log'))
    for item in [Configure, SigNet, SimNet]:
        logger.info(f'---------- {item.__name__} ----------')
        for name, value in inspect.getmembers(item)[26:]:
            logger.info(f'{name.ljust(20)}: {value}')
    try:
        # run_cross_validation_flow()
        run_flow(fold_id)
    except Exception as e:
        logger.error(e)
        raise e
