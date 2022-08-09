from data_modules.utils.extract_and_save_homo_patches import run_extract_and_save_patches_flow
from main import run_train_flow


def run_full_flow():
    fold = 0
    patches_dataset_dir = rf'/data2/p288722/datasets/dresden/nat_homo/patches_(128, 128)_200'
    full_image_dataset_dir = rf'/data2/p288722/datasets/dresden/natural'
    runtime_dir = rf'/data2/p288722/runtime_data/scd_images/reproduce_results/fold_{fold}'
    args = ['--fold', str(fold),
            '--num_patches', '200',
            '--patches_dataset_dir', patches_dataset_dir,
            '--full_image_dataset_dir', full_image_dataset_dir,
            '--default_root_dir', runtime_dir,
            ]

    # Train the models - Here we perform sequential train of the 4 models, however, this could be done in parallel
    run_train_flow(args + ['--classifier', 'all_brands'])
    run_train_flow(args + ['--classifier', 'Nikon_models'])
    run_train_flow(args + ['--classifier', 'Samsung_models'])
    run_train_flow(args + ['--classifier', 'Sony_models'])

    # Evaluate
    # run_hierarchical_test_flow(args)


def prepare_dataset():
    args = ['--source_dataset_dir', r'/data2/p288722/datasets/dresden/natural',  # full_image_dataset_dir
            '--dest_dataset_dir', r'/data2/p288722/datasets/dresden/nat_homo',  # patches_dataset_dir
            '--num_patches', '200',
            '--patch_dims', '128',
            '--device_id', None,
            '--patch_type', 'eff_homo_stddev']
    run_extract_and_save_patches_flow(args)


if __name__ == '__main__':
    prepare_dataset()  # Need to run this only once (skip if already done in a previous iteration)
    run_full_flow()
