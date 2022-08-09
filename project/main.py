import argparse
import shutil
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping

from data_modules import DresdenDataModule
from models import MISLNet_v2_Lit


def parse_args(args=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fold', type=int, default=0, help='Enter fold number', choices=[0, 1, 2, 3, 4])
    parser.add_argument('--num_patches', type=int, default=200, help='Enter num patches',
                        choices=[5, 10, 20, 50, 100, 200, 400])
    parser.add_argument('--patches_dataset_dir', type=Path, required=True)
    parser.add_argument('--full_image_dataset_dir', type=Path, required=True)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--classifier', type=str, default="all_models",
                        choices=["all_models", "all_brands", "Nikon_models", "Samsung_models", "Sony_models"])
    parser.add_argument('--flow_type', type=str, default="train",
                        choices=['train', 'test_hierarchical'])  # fixme: can be deprecated

    parser = pl.Trainer.add_argparse_args(parser)

    if args:
        args = parser.parse_args(args)  # parse from custom args
    else:
        args = parser.parse_args()  # parse from command line

    # Validate arguments
    if not args.patches_dataset_dir.exists():
        raise ValueError('patches_dataset_dir does not exists!')
    if not args.full_image_dataset_dir.exists():
        raise ValueError('full_image_dataset_dir does not exists!')

    # Setup built-in args (PyTorch lightning)
    args.max_epochs = 45
    args.learning_rate = 0.1
    args.weight_decay = 0.0005
    args.gpus = 1
    args.num_processes = 24  # fixme: change it to 12
    Path(args.default_root_dir).mkdir(parents=True, exist_ok=True)

    # Setup trainer args
    args.deterministic = True
    args.log_every_n_steps = 50
    # args.steps_per_epoch = 2  # fixme - remove after debugging
    # args.fast_dev_run = 96
    # nohup /data/p288722/python_venv/scd-images/bin/python /home/p288722/git_code/scd_images/project/run_full_flow.py &
    # Checkpoint callbacks (cb)
    early_stop_cb = EarlyStopping(monitor="val_loss", min_delta=0.02, patience=5, verbose=True, mode="min")
    lr_monitor_cb = LearningRateMonitor(logging_interval='epoch')
    ckpt_cb1 = ModelCheckpoint(monitor='train_loss', filename='{epoch}-{train_loss:.2f}', save_last=True)
    ckpt_cb2 = ModelCheckpoint(monitor='val_loss', filename='{epoch}-{val_loss:.2f}')
    ckpt_cb3 = ModelCheckpoint(monitor='val_acc', filename='{epoch}-{val_acc:.2f}', mode='max')
    ckpt_cb4 = ModelCheckpoint(monitor='train_acc', filename='{epoch}-{train_acc:.2f}', mode='max')
    args.callbacks = [lr_monitor_cb, ckpt_cb1, ckpt_cb2, ckpt_cb3, ckpt_cb4]

    return args


def run_train_flow(args=None):
    # Init Data Module
    args = parse_args(args)
    dm = DresdenDataModule(args)

    # Update args
    args.steps = (dm.num_samples // args.batch_size) * args.max_epochs
    args.num_classes = dm.num_classes  # 18 for Dresden - Model identification

    # Init Lightning Module
    lm = MISLNet_v2_Lit(**vars(args))

    # Set up Trainer
    trainer = pl.Trainer.from_argparse_args(args)

    # Train
    trainer.fit(lm, dm)

    # Rename / Restructure the log files & overwrite previous run if exists
    if trainer.log_dir and Path(trainer.log_dir).exists():
        results_dir = Path(trainer.default_root_dir).joinpath(args.classifier)
        if results_dir.exists():
            shutil.rmtree(results_dir)
        Path(trainer.log_dir).rename(results_dir)


def run_hierarchical_test_flow(args=None):
    args = parse_args(args)
    # Set up Trainer (for testing in this case)
    trainer = pl.Trainer.from_argparse_args(args)
    args.classifier = 'all_models'
    dm = DresdenDataModule(args)

    ckpt_filepath = Path(args.default_root_dir).joinpath(f'all_brands/checkpoints/*val_acc*.ckpt')
    brand_predictions = trainer.test(dataloaders=dm, ckpt_path=str(ckpt_filepath))

    ckpt_filepath = Path(args.default_root_dir).joinpath(f'Nikon_models/checkpoints/*val_acc*.ckpt')
    nikon_predictions = trainer.test(dataloaders=dm, ckpt_path=str(ckpt_filepath))

    ckpt_filepath = Path(args.default_root_dir).joinpath(f'Samsung_models/checkpoints/*val_acc*.ckpt')
    samsung_predictions = trainer.test(dataloaders=dm, ckpt_path=str(ckpt_filepath))

    ckpt_filepath = Path(args.default_root_dir).joinpath(f'Sony_models/checkpoints/*val_acc*.ckpt')
    sony_predictions = trainer.test(dataloaders=dm, ckpt_path=str(ckpt_filepath))

    # brand_predictions * 100
    # where brand_predictions match nikon add nikon_predictions
    # where brand_predictions match samsung add samsung_predictions
    # where brand_predictions match sony add sony_predictions

    # ground_truths_array
    # predictions_array

    # compute overall accuracy and F1 score
    # use sklearn to compute confusion matrices (1 overall + 9 scenario-compression type combinations)


def run_flow():
    pl.seed_everything(42, workers=True)

    run_train_flow()
    run_hierarchical_test_flow()


if __name__ == '__main__':
    run_flow()
