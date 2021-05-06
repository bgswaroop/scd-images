import torch


class CategoricalCrossEntropyLoss(object):
    def __init__(self):
        pass

    def __call__(self, predictions, target):
        sample_loss = -torch.sum(target * torch.log(predictions), dim=1)
        batch_loss = torch.mean(sample_loss)
        return batch_loss


class MSELoss(object):
    def __init__(self):
        pass

    def __call__(self, predictions, target):
        sample_loss = torch.mean(torch.square(target - predictions), dim=list(range(1, target.ndim)))
        batch_loss = torch.mean(sample_loss)
        return batch_loss
