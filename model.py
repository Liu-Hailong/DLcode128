import torch
import torch.nn as nn
import torch.nn.functional as F


def abn(planes, eps=1e-5, momentum=1e-3, affine=True, activation='leaky_relu', param=0):
    """################################### Basic Layers ###################################"""
    if activation is 'leaky_relu':
        return nn.Sequential(
            nn.InstanceNorm2d(planes, eps, momentum, affine),
            nn.ReLU(inplace=True) if param == 0 else nn.LeakyReLU(param, inplace=True)
        )
    elif activation is 'identity':
        return nn.InstanceNorm2d(planes, eps, momentum, affine)


class TCNN(nn.Module):
    def __init__(self):
        super(TCNN, self).__init__()
        """##################################### STN ###################################"""
        self.theta = None
        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 16, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(16, 16, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )
        self.fc_loc = nn.Sequential(
            nn.Linear(16 * 52 * 15, 256),
            nn.ReLU(True),
            nn.Linear(256, 256),
            nn.ReLU(True),
            nn.Linear(256, 3 * 2)
        )
        self.fc_loc[4].weight.data.zero_()
        self.fc_loc[4].bias.data = torch.FloatTensor([1, 0, 0, 0, 1, 0])
        """##################################### CNN ###################################"""
        self.CNN = nn.Sequential(
            # (X + 2 * Padding - kernal_size) / Stride + 1
            nn.Conv2d(1, 64, 3, 1, 1), abn(64), nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(64, 128, 3, 1, 1), abn(128), nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(128, 256, 3, 1, 1), abn(256), nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(256, 256, 3, 1, 1), abn(256), nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(256, 128, (3, 3), (2, 1), (0, 1)), abn(128),
            nn.Conv2d(128, 128, (4, 3), 1, (0, 1)), abn(128, activation='identity')
        )
        """########################## transformer encoder ##############################"""
        encoder_layer = nn.TransformerEncoderLayer(128, nhead=8, dim_feedforward=256)
        self.Transfomer = nn.Sequential(
            nn.TransformerEncoder(encoder_layer, num_layers=6),
            nn.Linear(128, 5),
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.InstanceNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 16 * 52 * 15)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        self.theta = theta  # 把 theta 拷贝出去
        grid = F.affine_grid(theta, x.size(), align_corners=True)
        x = F.grid_sample(x, grid, align_corners=True)
        return x

    def forward(self, x):
        x = self.CNN(self.stn(x))
        n, c, h, w = x.size()
        assert h == 1, "the height of conv must be 1"
        src = x.squeeze(2).permute(2, 0, 1)
        output = self.Transfomer(src)  # .permute(1, 0, 2)
        output = F.log_softmax(output, dim=2)
        return output
