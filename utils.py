import pickle
from pathlib import Path

import torch


class Utils:
    def __init__(self):
        pass

    @staticmethod
    def update_history(history, epoch, train_loss, val_loss, lr, runtime_dir):
        history['epochs'].append(epoch)
        history['learning_rate'].append(lr)
        history['accuracy'].append(None)
        history['loss'].append(train_loss)
        history['val_accuracy'].append(None)
        history['val_loss'].append(val_loss)
        with open(runtime_dir.joinpath('history.pkl'), 'wb+') as f:
            pickle.dump(history, f)
        return history

    @staticmethod
    def save_model_on_epoch_end(epoch, train_loss, val_loss, model, runtime_dir):
        train_loss = str(round(train_loss, 4)).ljust(4, '0')
        val_loss = str(round(val_loss, 4)).ljust(4, '0')
        epoch = str(epoch).zfill(3)
        torch.save(model, runtime_dir.joinpath('epoch{}_loss{}_valLoss{}.pt'.format(epoch, train_loss, val_loss)))

    @staticmethod
    def get_initial_epoch(runtime_dir, model, history):
        initial_epoch = 1
        pre_trained_model_path = None
        for model_path in Path(runtime_dir).glob('epoch*.pt'):
            model_path = str(model_path)
            epoch_num = int(model_path[model_path.find("epoch") + 5:model_path.find("epoch") + 8])
            if initial_epoch <= epoch_num:
                initial_epoch = epoch_num + 1
                pre_trained_model_path = model_path

        if pre_trained_model_path:
            model = torch.load(pre_trained_model_path)
            filename = Path(runtime_dir).joinpath('history.pkl')
            if filename.exists():
                with open(str(filename), 'rb') as f:
                    history = pickle.load(f)

        return initial_epoch, model, history

    @staticmethod
    def save_best_model(runtime_dir, history):
        import shutil

        best_epoch = Utils.choose_best_epoch_from_history(history)
        pre_trained_model_path = None
        for model_path in Path(runtime_dir).glob('epoch*.pt'):
            model_path = str(model_path)
            epoch_num = int(model_path[model_path.find("epoch") + 5:model_path.find("epoch") + 8])
            if best_epoch == epoch_num:
                pre_trained_model_path = model_path
                break
        shutil.copy(pre_trained_model_path, runtime_dir.joinpath('ae.pt'))

    @staticmethod
    def choose_best_epoch_from_history(history):
        import numpy as np
        num_decimals = 4

        # Filter 1: Find models with least validation/test loss
        val_loss = np.round(history['val_loss'], decimals=num_decimals)
        shortlisted_epochs = np.where(val_loss == np.min(val_loss))[0]

        # Filter 2: Find models with least train loss
        if len(shortlisted_epochs) > 1:
            loss = np.round(history['loss'], decimals=num_decimals)
            filtered_epochs = np.where(loss == np.min(loss[shortlisted_epochs]))[0]
            shortlisted_epochs = np.array([x for x in filtered_epochs if x in shortlisted_epochs])

        # Choose the model from the beginning (to avoid models that recovered fromm over-fitting/under-fitting)
        selected_epoch = shortlisted_epochs[0]

        return selected_epoch
