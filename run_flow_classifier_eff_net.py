import argparse
import logging
import random

from configure import Configure, SigNet, SimNet
from miscellaneous.autoencoders.prepare_data_set import hierarchical_to_flat_view_with_balancing, \
    hierarchical_to_flat_view
from sig_net.classifier_efficient_net.sig_net_flow import SigNetFlow
from utils.logging import SetupLogger, log_running_time

logger = logging.getLogger(__name__)


@log_running_time
def run_flow():
    # Seed that will determine the sampling in the data sets
    random.seed(123)

    # Step 0: Create a temp dir to save the dataset views
    temp_dir = Configure.runtime_dir.joinpath('temp')
    temp_dir.mkdir(exist_ok=True, parents=True)
    train_view, test_view = Configure.train_data_config, Configure.test_data_config

    # Step 1: Training the auto-encoders at the models level
    target_level = 'model'
    Configure.train_data_config = temp_dir.joinpath(rf'train_{target_level}_fold_{fold_id}.json')
    hierarchical_to_flat_view_with_balancing(source_view=train_view,
                                             dest_level=target_level,
                                             dest_view=Configure.train_data_config)
    Configure.test_data_config = temp_dir.joinpath(rf'test_{target_level}_fold_{fold_id}.json')
    hierarchical_to_flat_view(source_view=test_view,
                              dest_level=target_level,
                              dest_view=Configure.test_data_config)

    SigNetFlow.train()
    SigNetFlow.classify()


if __name__ == '__main__':
    import inspect

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-fold', type=int, help='enter fold number')
    args = parser.parse_args()
    fold_id = args.fold

    SetupLogger(log_file=Configure.runtime_dir.joinpath(f'scd.log'))
    for item in [Configure, SigNet, SimNet]:
        logger.info(f'---------- {item.__name__} ----------')
        for name, value in inspect.getmembers(item)[26:]:
            logger.info(f'{name.ljust(20)}: {value}')
    try:
        run_flow()
    except Exception as e:
        logger.error(e)
        print(e)
        raise e
