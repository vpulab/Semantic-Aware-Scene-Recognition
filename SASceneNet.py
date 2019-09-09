import torch
import torch.nn as nn
from torchvision.models import resnet


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

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.relu1 = nn.ReLU()

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SASceneNet(nn.Module):
    """
    Generate Model Architecture
    """

    def __init__(self, scene_classes, semantic_classes=151):
        super(SASceneNet, self).__init__()

        # Load ResNet 18 pre-trained on ImageNet
        base = resnet.resnet18(pretrained=False)

        # RGB BRANCH
        # First initial block
        self.in_block = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1, return_indices=True)
        )
        self.encoder1 = base.layer1
        self.encoder2 = base.layer2
        self.encoder3 = base.layer3
        self.encoder4 = base.layer4

        # Semantic Branch
        self.in_block_sem = nn.Sequential(
            nn.Conv2d(semantic_classes+1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.in_block_sem_1 = BasicBlockSem(64, 128, kernel_size=3, stride=2, padding=1)
        self.in_block_sem_2 = BasicBlockSem(128, 256, kernel_size=3, stride=2, padding=1)
        self.in_block_sem_3 = BasicBlockSem(256, 512, kernel_size=3, stride=2, padding=1)

        # ResNet-18
        # Semantic Scene Classification Layers
        self.fc_SEM = nn.Linear(512, scene_classes)

        # RGB Scene Classification Layers
        self.fc_RGB = nn.Linear(512, scene_classes)

        # Final Scene Classification Layers
        self.lastConvRGB1 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.lastConvRGB2 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
        )
        self.lastConvSEM1 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.lastConvSEM2 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
        )

        self.dropout = nn.Dropout(0.3)
        self.sigmoid = nn.Sigmoid()
        self.avgpool7 = nn.AvgPool2d(7, stride=1)
        self.avgpool3 = nn.AvgPool2d(3, stride=1)
        self.fc = nn.Linear(1024, scene_classes)

        # Loss
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x, sem):
        """
        Netowrk forward
        :param x: RGB Image
        :param sem: Semantic Segmentation score tensor
        :return: Scene recognition predictions
        """
        # RGB Branch
        x, pool_indices = self.in_block(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # RGB Classification Layer
        act_rgb = self.avgpool7(e4)
        act_rgb = act_rgb.view(act_rgb.size(0), -1)
        act_rgb = self.dropout(act_rgb)
        act_rgb = self.fc_RGB(act_rgb)

        # Semantic Branch
        y = self.in_block_sem(sem)
        y1 = self.in_block_sem_1(y)
        y2 = self.in_block_sem_2(y1)
        y3 = self.in_block_sem_3(y2)

        # Semantic Classification Layer
        act_sem = self.avgpool7(y3)
        act_sem = act_sem.view(act_sem.size(0), -1)
        act_sem = self.dropout(act_sem)
        act_sem = self.fc_SEM(act_sem)

        # Attention Module Layers
        e5 = self.lastConvRGB1(e4)
        e6 = self.lastConvRGB2(e5)

        y4 = self.lastConvSEM1(y3)
        y5 = self.lastConvSEM2(y4)

        # Attention Mechanism
        e7 = e6 * self.sigmoid(y5)

        # Scene Classification FC layer
        e8 = self.avgpool3(e7)
        act = e8.view(e8.size(0), -1)
        act = self.dropout(act)
        act = self.fc(act)

        return act, e7, act_rgb, act_sem

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
