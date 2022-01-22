import argparse
import logging
import shutil

from configure import Configure, SigNet, SimNet
from miscellaneous.prepare_image_and_patch_data import level_balanced_from_hierarchical_dataset, \
    level_from_hierarchical_dataset
from signature_net.sig_net_flow import SigNetFlow
from utils.logging import SetupLogger, log_running_time

logger = logging.getLogger(__name__)


@log_running_time
def run_flow_flat_device():
    """
    A high level function to run the train and evaluation flows
    :return: None
    """

    # Signature Net
    # Config
    patch_aggregation = SigNet.patch_aggregation

    # Step 0: Create a temp dir to save the dataset views
    temp_dir = Configure.runtime_dir.joinpath('temp')
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(exist_ok=True, parents=True)
    hierarchical_train_data_filepath, hierarchical_test_data_filepath = \
        Configure.train_data_config, Configure.test_data_config

    # Step 1: device level training and classification
    target_level = 'device'
    Configure.train_data_config = temp_dir.joinpath(rf'train_{target_level}.json')
    devices_dataset_train = level_balanced_from_hierarchical_dataset(source_view=hierarchical_train_data_filepath,
                                                                     dest_level=target_level,
                                                                     max_patches=SigNet.samples_per_class,
                                                                     dest_view=Configure.train_data_config)
    Configure.test_data_config = temp_dir.joinpath(rf'test_{target_level}.json')
    level_from_hierarchical_dataset(source_view=hierarchical_test_data_filepath,
                                    dest_level=target_level,
                                    dest_view=Configure.test_data_config)

    SigNet.update_model(num_classes=len(devices_dataset_train), is_constrained=False)
    Configure.sig_net_name = f'signature_net_devices'
    Configure.update()

    SigNetFlow.train()
    SigNetFlow.classify(aggregation_method=patch_aggregation)


if __name__ == '__main__':
    import inspect
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-fold', type=int, help='enter fold number')
    parser.add_argument('-num_patches', type=int, help='enter num patches')
    parser.add_argument('-patch_aggregation', type=str, help='enter patch_aggregation')
    parser.add_argument('-use_contributing_patches', type=int, help='enter use_contributing_patches')
    parser.add_argument('-patches_type', type=str, help='choose between either "random" or "non_homo"')

    args = parser.parse_args()
    num_patches = args.num_patches
    patches_type = args.patches_type

    SigNet.use_contributing_patches = bool(args.use_contributing_patches)
    SigNet.patch_aggregation = args.patch_aggregation

    # do not modify this till the experiment starts to run
    Configure.train_data_config = Path(
        rf'/data/p288722/dresden/train/66_devices_from200_128x128_{num_patches}.json')
    Configure.test_data_config = Path(
        rf'/data/p288722/dresden/test/66_devices_from200_128x128_{num_patches}.json')
    Configure.dataset_folder = \
        rf'/scratch/p288722/datasets/dresden/source_devices/nat_patches_128x128_{num_patches}'
    Configure.runtime_dir = Path(
        rf'/scratch/p288722/runtime_data/scd_pytorch/66_devices_flat_1')
    SigNet.samples_per_class = 1_500

    Configure.update()

    SetupLogger(log_file=Configure.runtime_dir.joinpath(
        f'scd_{SigNet.patch_aggregation}_use_contributing_patches_{SigNet.use_contributing_patches}.log'))
    for item in [Configure, SigNet, SimNet]:
        logger.info(f'---------- {item.__name__} ----------')
        for name, value in inspect.getmembers(item)[26:]:
            logger.info(f'{name.ljust(20)}: {value}')
    try:
        run_flow_flat_device()
    except Exception as e:
        logger.error(e)
        print(e)
        raise e
