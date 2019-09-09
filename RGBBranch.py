import torch.nn as nn
from torchvision.models import resnet


class RGBBranch(nn.Module):
    """
    Generate Model Architecture
    """

    def __init__(self, scene_classes=1055):
        super(RGBBranch, self).__init__()

        # Load ResNet 18 pre-trained on ImageNet
        base = resnet.resnet18(pretrained=True)

        # RGB BRANCH
        # First initial block
        self.in_block = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1, return_indices=True)
        )

        # Encoder
        self.encoder1 = base.layer1
        self.encoder2 = base.layer2
        self.encoder3 = base.layer3
        self.encoder4 = base.layer4

        # Final Scene Classification Layers
        self.dropout = nn.Dropout(0.3)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512, scene_classes)

        # Loss
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x, sem):
        """
        Netowrk forward
        :param x: RGB Image
        :return: Scene recognition predictions
        """
        # RGB Branch
        x, pool_indices = self.in_block(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Scene Classification FC layer
        act = self.avgpool(e4)
        act = act.view(act.size(0), -1)
        act = self.dropout(act)
        act = self.fc(act)

        act_rgb = act
        act_sem = act

        return act, e4, act_rgb, act_sem

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
