import torch.nn as nn


class BasicBlockSem(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride, padding):
        super(BasicBlockSem, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        self.ca = ChannelAttention(out_planes)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)

        # Channel Attention Module
        out = self.ca(out) * out

        out = self.relu(out)

        return out


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.relu1 = nn.ReLU()

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SemBranch(nn.Module):
    """
    Generate Model Architecture
    """

    def __init__(self, scene_classes, semantic_classes=151):
        super(SemBranch, self).__init__()

        # Semantic Branch
        self.in_block_sem = nn.Sequential(
            nn.Conv2d(semantic_classes + 1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.in_block_sem_1 = BasicBlockSem(64, 128, kernel_size=3, stride=2, padding=1)
        self.in_block_sem_2 = BasicBlockSem(128, 256, kernel_size=3, stride=2, padding=1)
        self.in_block_sem_3 = BasicBlockSem(256, 512, kernel_size=3, stride=2, padding=1)

        # Semantic Scene Classification Layers
        self.dropout = nn.Dropout(0.3)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc_SEM = nn.Linear(512, scene_classes)

        # Loss
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x, sem):
        # Semantic Branch
        y = self.in_block_sem(sem)
        y2 = self.in_block_sem_1(y)
        y3 = self.in_block_sem_2(y2)
        y4 = self.in_block_sem_3(y3)

        # Semantic Classification Layer
        act_sem = self.avgpool(y4)
        act_sem = act_sem.view(act_sem.size(0), -1)
        act_sem = self.dropout(act_sem)
        act_sem = self.fc_SEM(act_sem)

        act = act_sem
        e5 = y4
        act_rgb = act_sem

        return act, e5, act_rgb, act_sem

    def loss(self, x, target):
        """
        Funtion to comput the loss
        :param x: Predictions obtained by the network
        :param target: Ground-truth scene recognition labels
        :return: Loss value
        """
        # Check inputs
        assert (x.shape[0] == target.shape[0])

        # Classification loss
        loss = self.criterion(x, target.long())

        return loss
