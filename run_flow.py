import logging

from configure import Configure, SigNet, SimNet
from signature_net.sig_net_flow import SigNetFlow
from similarity_net.sim_net_flow import SimNetFlow
from utils.logging import SetupLogger, log_running_time

logger = logging.getLogger(__name__)


@log_running_time
def run_flow():
    """
    A high level function to run the train and evaluation flows
    :return: None
    """

    # Signature Net
    # SigNetFlow.train()
    SigNetFlow.classify()
    # Utils.visualize_ae_input_output_pairs()
    # Utils.save_avg_fourier_images()

    # Configure.train_data = r'/data/p288722/dresden/train/nat_patches_bal_kmkd_1/'
    # Configure.test_data = r'/data/p288722/dresden/test/nat_patches_bal_kmkd_1/'

    # Similarity Net
    # SimNetFlow.train()
    SimNetFlow.classify()

    # SimNetFlow.classify()


if __name__ == '__main__':
    import inspect
    SetupLogger(log_file=Configure.runtime_dir.joinpath('scd.log'))
    for item in [Configure, SigNet, SimNet]:
        logger.info(f'---------- {item.__name__} ----------')
        for name, value in inspect.getmembers(item)[26:]:
            logger.info(f'{name.ljust(20)}: {value}')
    try:
        run_flow()
    except Exception as e:
        logger.error(e)
        raise e
