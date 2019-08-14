import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d

class MaskIOU(nn.Module):

    def __init__(self, num_classes, BatchNorm):
        super(MaskIOU, self).__init__()
        self.NUM_CLASSES = num_classes

        # start from (B, ?, H, W) to (B, num_classes)
        self.relu = nn.ReLU()
        # self.bn1 = BatchNorm(48)

        self.conv1 = nn.Conv2d(49, 48, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(48, 48, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(48, 48, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(48, 48, kernel_size=3, stride=2, padding=1)

        self.fc1 = nn.Linear(48*65*65, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.last_fc = nn.Linear(1024, num_classes)

        self._init_weight()

    def forward(self, x, low_level_feat):
        x = torch.max(x, dim=1)[1].float()
        x = x.unsqueeze(dim=1)# B, 1, H, W

        x = torch.cat((x, low_level_feat), dim=1)  # B, 48+1, H, W

        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        # B, 48, 65, 65
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.last_fc(x)

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)
            # elif isinstance(m, SynchronizedBatchNorm2d):
            #     m.weight.data.fill_(1)
            #     m.bias.data.zero_()
            # elif isinstance(m, nn.BatchNorm2d):
            #     m.weight.data.fill_(1)
            #     m.bias.data.zero_()

        nn.init.normal_(self.last_fc.weight, mean=0, std=0.01)
        nn.init.constant_(self.last_fc.bias, 0)


def build_maskiou(num_classes, BatchNorm):
    return MaskIOU(num_classes, BatchNorm)