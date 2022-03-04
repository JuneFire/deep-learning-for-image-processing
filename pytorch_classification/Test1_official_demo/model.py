import torch.nn as nn
import torch.nn.functional as F

'''
Pytorch Tensor 的通道排序：[batch，channel, height, weight]
'''


class LeNet(nn.Module):  # 继承nn.Module 类
    def __init__(self):
        super(LeNet, self).__init__()  # 继承的初始化（一般都会这么做）
        # 输入特征层的深度，16个卷积核，卷积核的尺度
        self.conv1 = nn.Conv2d(3, 16, 5)  # 通道数3，16个5*5的卷积核
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 5 * 5, 120)  # 三层全连接 ： 展平为一维向量 ， 120 （节点个位数）
        self.fc2 = nn.Linear(120, 84)  # 120 -> 84 个节点
        self.fc3 = nn.Linear(84, 10)  # 84 -> 10 (根据训练集的类别种类数量确定)

    # 输出的矩阵尺寸大小：N=(W - F + 2P) / S + 1
    # (图片大小 - 卷积核大小 + 2*边幅) / 步长 + 1

    def forward(self, x):
        x = F.relu(self.conv1(x))  # input(3, 32, 32) output(16, 28, 28)  ： 16 - 为channel数，即输入的卷积核个数，即深度
        x = self.pool1(x)  # output(16, 14, 14)
        x = F.relu(self.conv2(x))  # output(32, 10, 10)
        x = self.pool2(x)  # output(32, 5, 5)
        x = x.view(-1, 32 * 5 * 5)  # output(32*5*5)
        x = F.relu(self.fc1(x))  # output(120)
        x = F.relu(self.fc2(x))  # output(84)
        x = self.fc3(x)  # output(10)
        return x


import torch

input1 = torch.rand([32, 3, 32, 32])
model = LeNet()
print(model)
output = model(input1)
