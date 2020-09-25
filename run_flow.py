import logging

from configure import Configure
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
    # SigNetFlow.classify()
    # Utils.visualize_ae_input_output_pairs()
    # Utils.save_avg_fourier_images()

    # Similarity Net
    SimNetFlow.train()
    SimNetFlow.classify()


if __name__ == '__main__':
    SetupLogger(log_file=Configure.runtime_dir.joinpath('scd.log'))
    try:
        run_flow()
    except Exception as e:
        logger.error(e)
        raise e
