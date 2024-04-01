import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialTransformerNet(nn.Module):
    def __init__(self):
        super(SpatialTransformerNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=0, bias=False)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=0, bias=False)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=0, bias=False)
        self.conv5 = nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=0, bias=False)
        self.conv6 = nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=0, bias=False)

        self.fc1 = nn.Linear(32 * 4 * 4, 1024)
        self.fc2 = nn.Linear(1024, 6)

        self.init_value()

    def init_value(self):
        """
        nn.init.uniform_(self.conv1, a=0.0, b=0.0)
        nn.init.uniform_(self.conv2, a=0.0, b=0.0)
        nn.init.uniform_(self.conv3, a=0.0, b=0.0)
        nn.init.uniform_(self.conv4, a=0.0, b=0.0)
        nn.init.uniform_(self.conv5, a=0.0, b=0.0)
        nn.init.uniform_(self.conv6, a=0.0, b=0.0)
        """
        nn.init.xavier_uniform(self.conv1.weight)
        nn.init.xavier_uniform(self.conv2.weight)
        nn.init.xavier_uniform(self.conv3.weight)
        nn.init.xavier_uniform(self.conv4.weight)
        nn.init.xavier_uniform(self.conv5.weight)
        nn.init.xavier_uniform(self.conv6.weight)
        nn.init.xavier_uniform(self.fc1.weight)
        """
        self.conv1.bias.data.fill_(0.01)
        self.conv2.bias.data.fill_(0.01)
        self.conv3.bias.data.fill_(0.01)
        self.conv4.bias.data.fill_(0.01)
        self.conv5.bias.data.fill_(0.01)
        self.conv6.bias.data.fill_(0.01)
        """
        self.fc2.weight.data.zero_()
        self.fc2.bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def localization(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv5(x))
        x = F.max_pool2d(x, 2)
        # print(x.size()) # (7, 32, 4, 4)
        return x

    def stn(self, x):  # (7, 3, 256, 256)
        xs = self.localization(x)
        xs = xs.view(-1, 32 * 4 * 4)
        xs = self.fc1(xs)
        xs = self.fc2(xs)
        theta = xs.view(-1, 2, 3)

        grid = F.affine_grid(theta, torch.Size((theta.size(0), 3, 256, 256)))
        xs = F.grid_sample(x, grid)
        # print(xs.size()) # (7, 3, 256, 256)
        return xs

    def forward(self, x):
        x = self.stn(x)
        return x
