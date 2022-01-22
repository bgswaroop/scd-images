import numpy as np
import torch
import torchmetrics
from pytorch_lightning import LightningModule
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support as macro_score
from torch import nn
from torch.optim.lr_scheduler import ExponentialLR


class MISLNet_v2(torch.nn.Module):
    def __init__(self, num_classes, is_constrained=False):
        super().__init__()
        self.is_constrained = is_constrained
        if is_constrained:
            self.conv0 = nn.Conv2d(in_channels=3, out_channels=5, kernel_size=(7, 7), padding=(3, 3), stride=(1, 1))
            self.conv1 = nn.Conv2d(in_channels=5, out_channels=96, kernel_size=(7, 7), padding=(3, 3), stride=(2, 2))
        else:
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=(7, 7), padding=(3, 3), stride=(2, 2))

        self.bn1 = nn.BatchNorm2d(num_features=96)
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=64, kernel_size=(5, 5), padding=(2, 2))
        self.bn2 = nn.BatchNorm2d(num_features=64)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(5, 5), padding=(2, 2))
        self.bn3 = nn.BatchNorm2d(num_features=64)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 1))
        self.bn4 = nn.BatchNorm2d(num_features=128)

        # the size 2048 is for inputs of size 3x128x128 fixme: make it dynamic by passing input dims to the __init__
        self.fcn5 = nn.Linear(in_features=2048, out_features=1024)
        self.dropout5 = nn.Dropout(p=0.3)
        self.fcn6 = nn.Linear(in_features=1024, out_features=200)
        self.dropout6 = nn.Dropout(p=0.3)
        self.fcn7 = nn.Linear(in_features=200, out_features=num_classes)

    def forward(self, cnn_inputs):
        x = self.extract_features(cnn_inputs)
        x = self._classify_features(x)
        return x

    def extract_features(self, cnn_inputs):

        if self.is_constrained:
            x = self.conv0(cnn_inputs)
            x = self.conv1(x)
        else:
            x = self.conv1(cnn_inputs)

        # Block 1
        x = self.bn1(x)
        x = nn.ReLU()(x)
        x = nn.MaxPool2d(2, 2)(x)

        # Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = nn.ReLU()(x)
        x = nn.MaxPool2d(2, 2)(x)

        # Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = nn.ReLU()(x)
        x = nn.MaxPool2d(2, 2)(x)

        # Block 4
        x = self.conv4(x)
        x = self.bn4(x)
        x = nn.ReLU()(x)
        x = nn.MaxPool2d(2, 2)(x)

        # Block 5
        x = torch.flatten(x, start_dim=1)
        x = self.fcn5(x)
        x = nn.Tanh()(x)
        x = self.dropout5(x)

        return x

    def _classify_features(self, features):
        # Block 6
        x = self.fcn6(features)
        x = nn.Tanh()(x)
        x = self.dropout6(x)

        # Block 7
        x = self.fcn7(x)
        # x = nn.Softmax(dim=1)(x)  # The CrossEntropy criterion also computes the SoftMax

        return x


class MISLNet_v2_Lit(LightningModule):
    def __init__(self, num_classes, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.model = MISLNet_v2(num_classes)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy()

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            params=self.model.parameters(),
            lr=self.hparams.learning_rate,
            momentum=0.9,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = {
            'scheduler': ExponentialLR(optimizer, gamma=0.9, last_epoch=-1),
            'interval': 'epoch'  # called after each training step
        }
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self.model(x)
        loss = self.criterion(y_hat, y)
        acc = self.accuracy(y_hat, y)

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y_true, img_names = batch
        y_pred = self.model(x)
        loss = self.criterion(y_pred, y_true)
        acc = self.accuracy(y_pred, y_true)

        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log('val_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return y_true, y_pred, img_names

    def validation_epoch_end(self, val_outputs):
        # patch-level predictions
        y_true = torch.hstack([x[0] for x in val_outputs]).cpu().numpy()
        y_pred = torch.argmax(torch.vstack([x[1] for x in val_outputs]), dim=1).cpu().numpy()

        # image-level predictions
        patch_names = [y for x in val_outputs for y in x[2]]
        y_true, y_pred = self._patch_to_image_level_aggregation(y_true, y_pred, patch_names)

        # Compute scores
        macro_precision, macro_recall, macro_f_score, _ = macro_score(y_true, y_pred, average='macro')
        accuracy = accuracy_score(y_true, y_pred)  # macro recall (or unweighted averaged recall) == balanced_accuracy

        cm = confusion_matrix(y_true, y_pred)
        num_x, num_y = cm.shape
        cm_str = '\n['
        for x in range(num_x):
            cm_str += '[' + ','.join([str(num) for num in cm[x]]) + '],\n'
        cm_str = cm_str[:-2] + ']\n'
        print(cm_str)

        self.log('val_macro_recall', macro_recall, on_step=False, on_epoch=True, sync_dist=True)
        self.log('val_acc_score', accuracy, on_step=False, on_epoch=True, sync_dist=True)
        self.log('val_macro_f1', macro_f_score, on_step=False, on_epoch=True, sync_dist=True)

    @staticmethod
    def _patch_to_image_level_aggregation(y_true, y_pred, patch_names):
        """
        Compute the majority vote among all the patch-level predictions to determine image-level predictions
        :param y_true: patch level ground truths
        :param y_pred: patch level predictions
        :param patch_names:
        :return: Image level predictions
        """
        image_names = {x[:-4] for x in patch_names}
        image_wise_indices = {x: [idx for idx, y in enumerate(patch_names) if x in y] for x in image_names}

        yt, yp = [], []
        for _, indices in image_wise_indices.items():
            values, counts = np.unique(y_true[indices], return_counts=True)
            ind = np.argmax(counts)
            yt.append(values[ind])
            values, counts = np.unique(y_pred[indices], return_counts=True)
            ind = np.argmax(counts)
            yp.append(values[ind])

        return yt, yp
