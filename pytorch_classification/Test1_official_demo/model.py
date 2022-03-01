import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):        #继承nn.Module 类
    def __init__(self):
        super(LeNet, self).__init__()     # 继承的初始化（一般都会这么做）
        self.conv1 = nn.Conv2d(3, 16, 5)  # 通道数3，16个 5*5的卷积核
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    # 输出的矩阵尺寸大小：N=(W - F + 2P) / S + 1
    # (图片大小 - 卷积核大小 + 2*边幅) / 步长 + 1
    def forward(self, x):
        x = F.relu(self.conv1(x))    # input(3, 32, 32) output(16, 28, 28)
        x = self.pool1(x)            # output(16, 14, 14)
        x = F.relu(self.conv2(x))    # output(32, 10, 10)
        x = self.pool2(x)            # output(32, 5, 5)
        x = x.view(-1, 32*5*5)       # output(32*5*5)
        x = F.relu(self.fc1(x))      # output(120)
        x = F.relu(self.fc2(x))      # output(84)
        x = self.fc3(x)              # output(10)
        return x


