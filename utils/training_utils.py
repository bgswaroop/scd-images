import logging
import pickle
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


class Utils:
    def __init__(self):
        pass

    @staticmethod
    def update_history(history, epoch, train_loss, val_loss, train_acc, val_acc, lr, save_to_dir):
        history['epochs'].append(epoch)
        history['learning_rate'].append(lr)
        history['accuracy'].append(train_acc)
        history['loss'].append(train_loss)
        history['val_accuracy'].append(val_acc)
        history['val_loss'].append(val_loss)
        with open(save_to_dir.joinpath('history.pkl'), 'wb+') as f:
            pickle.dump(history, f)
        return history

    @staticmethod
    def save_model_on_epoch_end(model_state_dict, optimizer_state_dict, scheduler_state_dict, history, save_to_dir):
        train_loss = str(round(history['loss'][-1], 4)).ljust(4, '0')
        val_loss = str(round(history['val_loss'][-1], 4)).ljust(4, '0')
        if history['accuracy'][-1] and history['val_accuracy'][-1]:
            train_accuracy = str(round(history['accuracy'][-1], 4)).ljust(4, '0')
            val_accuracy = str(round(history['val_accuracy'][-1], 4)).ljust(4, '0')
        else:
            train_accuracy = None
            val_accuracy = None

        epoch = str(history['epochs'][-1]).zfill(3)
        params_to_save = {'model_state_dict': model_state_dict,
                          'optimizer_state_dict': optimizer_state_dict,
                          'scheduler_state_dict': scheduler_state_dict}
        torch.save(params_to_save, save_to_dir.joinpath('epoch{}_loss{}_valLoss{}_acc{}_valAcc{}.pt'.
                                                        format(epoch, train_loss, val_loss, train_accuracy,
                                                               val_accuracy)))

    @staticmethod
    def prepare_for_training(pre_trained_models_dir, model, optimizer, scheduler):
        initial_epoch = 1
        history = {'epochs': [], 'learning_rate': [], 'accuracy': [], 'loss': [], 'val_accuracy': [], 'val_loss': []}

        pre_trained_model_path = None
        for model_path in Path(pre_trained_models_dir).glob('epoch*.pt'):
            model_path = str(model_path)
            epoch_num = int(model_path[model_path.find("epoch") + 5:model_path.find("epoch") + 8])
            if initial_epoch <= epoch_num:
                initial_epoch = epoch_num + 1
                pre_trained_model_path = model_path

        if pre_trained_model_path:
            params = torch.load(pre_trained_model_path)
            filename = Path(pre_trained_models_dir).joinpath('history.pkl')
            if filename.exists():
                with open(str(filename), 'rb') as f:
                    history = pickle.load(f)
            model.load_state_dict(params['model_state_dict'])
            optimizer.load_state_dict(params['optimizer_state_dict'])
            scheduler.load_state_dict(params['scheduler_state_dict'])

        return initial_epoch, history, model, optimizer, scheduler

    @staticmethod
    def save_best_model(pre_trained_models_dir, destination_dir, history, name):
        import shutil

        best_epoch = Utils.choose_best_epoch_from_history(history) + 1
        pre_trained_model_path = None
        for model_path in Path(pre_trained_models_dir).glob('epoch*.pt'):
            model_path = str(model_path)
            epoch_num = int(model_path[model_path.find("epoch") + 5:model_path.find("epoch") + 8])
            if best_epoch == epoch_num:
                pre_trained_model_path = model_path
                break
        shutil.copy(pre_trained_model_path, destination_dir.joinpath('{}.pt'.format(name)))
        logger.info(f'Best trained model is found at epoch {best_epoch}, {str(pre_trained_model_path)}')

    @staticmethod
    def choose_best_epoch_from_history(history):
        import numpy as np
        num_decimals = 4

        shortlisted_epochs = range(len(history['val_loss']))
        if 'val_accuracy' in history:
            val_acc = np.round(history['val_accuracy'], decimals=num_decimals)
            shortlisted_epochs = np.where(val_acc == np.max(val_acc))[0]

        # Filter 1: Find models with least validation/test loss
        if len(shortlisted_epochs) > 1:
            val_loss = np.round(history['val_loss'], decimals=num_decimals)
            shortlisted_epochs = np.where(val_loss == np.min(val_loss[shortlisted_epochs]))[0]

        # Filter 2: Find models with least train loss
        if len(shortlisted_epochs) > 1:
            loss = np.round(history['loss'], decimals=num_decimals)
            filtered_epochs = np.where(loss == np.min(loss[shortlisted_epochs]))[0]
            shortlisted_epochs = np.array([x for x in filtered_epochs if x in shortlisted_epochs])

        # Choose the model from the beginning (to avoid models that recovered fromm over-fitting/under-fitting)
        selected_epoch = shortlisted_epochs[0]

        return selected_epoch

    @staticmethod
    def split_train_test(test_set_size, source_images_dir, train_images_dir, test_images_dir, balance_classes=False):
        """
        Split the data in a stratified manner into train and test sets based on the specified test_set_size.
        :param test_set_size:
        :param source_images_dir: 
        :param train_images_dir: 
        :param test_images_dir: 
        :param balance_classes: bool (default: False)
        :return: 
        """
        import shutil
        import random
        import os

        if not isinstance(test_set_size, float) and not 0 <= test_set_size <= 1:
            raise ValueError('Invalid test set size - must be a float in [0, 1]')
        if not isinstance(source_images_dir, Path) or not isinstance(train_images_dir, Path) or not isinstance(
                test_images_dir, Path) or not source_images_dir.exists():
            raise ValueError('Invalid directory paths!')

        # Step 0: Remove old train / test split
        if train_images_dir.exists():
            shutil.rmtree(train_images_dir)
        if test_images_dir.exists():
            shutil.rmtree(test_images_dir)

        # Step 1: Create the directory structure
        camera_devices = source_images_dir.glob('*')
        for item in camera_devices:
            for output_path in [train_images_dir, test_images_dir]:
                subdir = output_path.joinpath(item.name)
                subdir.mkdir(parents=True, exist_ok=True)

        min_num_samples = float('inf')
        if balance_classes:
            camera_devices = source_images_dir.glob('*')
            for item in camera_devices:
                num_samples = sum(1 for _ in Path(item).glob("*"))
                if num_samples < min_num_samples:
                    min_num_samples = num_samples
            logger.info("Number of images in each class : {}".format(min_num_samples))

        # Step 2: Randomly pick elements for train and test sets
        camera_devices = source_images_dir.glob('*')
        for item in camera_devices:
            img_paths = [x for x in Path(item).glob("*")]
            random.seed(123)  # fixed seed to produce reproducible results
            random.shuffle(img_paths)

            if balance_classes:
                img_paths = img_paths[:min_num_samples]

            # Create symlinks for train
            num_train_images = round(len(img_paths) * (1 - test_set_size))
            train_dir = train_images_dir.joinpath(item.name)
            for img_path in img_paths[:num_train_images]:
                symlink = train_dir.joinpath(img_path.name)
                if not symlink.exists():
                    os.symlink(src=img_path, dst=symlink)  # Need to run this in administrator/sudo mode

            # Create symlinks for test
            test_dir = test_images_dir.joinpath(item.name)
            for img_path in img_paths[num_train_images:]:
                symlink = test_dir.joinpath(img_path.name)
                if not symlink.exists():
                    os.symlink(src=img_path, dst=symlink)  # Need to run this in administrator/sudo mode


if __name__ == '__main__':
    Utils.split_train_test(test_set_size=0.2,
                           source_images_dir=Path(r'D:\Data\Dresden\source_devices\jpeg'),
                           train_images_dir=Path(r'D:\Data\Dresden\train\jpeg'),
                           test_images_dir=Path(r'D:\Data\Dresden\test\jpeg'))
