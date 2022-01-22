import argparse
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping

from data_modules import DresdenDataModule
from models import MISLNet_v2_Lit


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fold', type=int, default=0, help='Enter fold number')
    parser.add_argument('--num_patches', type=int, default=200, help='Enter num patches')
    parser.add_argument('--dataset_dir', type=Path, default=None)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--classifier', type=str, default="all_models")

    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    # Validate arguments
    if not args.dataset_dir:
        args.dataset_dir = Path(r'/data/p288722/datasets/dresden_new/nat_homo/patches_128_400')
    assert args.fold in {0, 1, 2, 3, 4}
    assert args.num_patches in {5, 10, 20, 50, 100, 200, 400}
    assert args.classifier in {"all_models", "all_brands", "Nikon_models", "Samsung_models", "Sony_models"}

    # Setup built-in args
    args.max_epochs = 45
    args.learning_rate = 0.1
    args.weight_decay = 0.0005
    args.gpus = 1
    args.num_processes = 32
    # args.persistent_workers = True
    if not args.default_root_dir:
        args.default_root_dir = r'/data/p288722/runtime_data/scd_images/scene_ind_test_set'
    Path(args.default_root_dir).mkdir(parents=True, exist_ok=True)

    # Setup trainer args
    # args.accelerator = 'ddp_spawn'
    # args.plugins = DDPSpawnPlugin(find_unused_parameters=False)
    args.deterministic = True
    args.log_every_n_steps = 50
    # args.steps_per_epoch = 2

    # Checkpoint callbacks (cb)
    early_stop_cb = EarlyStopping(monitor="val_loss", min_delta=0.02, patience=5, verbose=True, mode="min")
    lr_monitor_cb = LearningRateMonitor(logging_interval='epoch')
    ckpt_cb1 = ModelCheckpoint(monitor='train_loss', filename='{epoch}-{train_loss:.2f}', save_last=True)
    ckpt_cb2 = ModelCheckpoint(monitor='val_loss', filename='{epoch}-{val_loss:.2f}')
    ckpt_cb3 = ModelCheckpoint(monitor='val_acc', filename='{epoch}-{val_acc:.2f}', mode='max')
    ckpt_cb4 = ModelCheckpoint(monitor='train_acc', filename='{epoch}-{train_acc:.2f}', mode='max')
    args.callbacks = [lr_monitor_cb, ckpt_cb1, ckpt_cb2, ckpt_cb3, ckpt_cb4]

    return args


def run_flow():
    pl.seed_everything(42, workers=True)
    args = parse_args()

    # Init Data Module
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


if __name__ == '__main__':
    run_flow()
