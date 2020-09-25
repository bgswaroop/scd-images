from torch import nn
import torch


class SimilarityNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(in_features=1024, out_features=64)
        self.bn1 = nn.BatchNorm1d(num_features=64)
        self.fc2 = nn.Linear(in_features=64*3, out_features=64)
        self.bn2 = nn.BatchNorm1d(num_features=64)
        self.fc3 = nn.Linear(in_features=64, out_features=1)

    def forward(self, input_features):
        x1 = self.fc1(input_features[0])
        x1 = torch.relu(x1)
        x1 = self.bn1(x1)

        x2 = self.fc1(input_features[1])
        x2 = torch.relu(x2)
        x2 = self.bn1(x2)

        x = torch.mul(x1, x2)
        x = torch.cat([x1, x, x2], dim=1)

        x = self.fc2(x)
        x = torch.relu(x)
        x = self.bn2(x)

        x = self.fc3(x)
        x = torch.sigmoid(x)
        return x
