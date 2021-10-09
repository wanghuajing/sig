import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
import torch


def res50(num_classes: int = 2):
    model = models.resnet50(pretrained=True)
    model.fc = nn.Linear(2048, num_classes)
    return model


def res18(num_classes: int = 2):
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(512, num_classes)
    return model


class DenseNet121(nn.Module):
    """Model modified.
    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.
    """

    def __init__(self, out_size):
        super(DenseNet121, self).__init__()
        self.densenet121 = models.densenet121(pretrained=True)
        num_ftrs = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(nn.Linear(num_ftrs, out_size), nn.Sigmoid())

    def forward(self, x):
        x = self.densenet121(x)
        return x


class DenseNet121_z(nn.Module):

    def __init__(self, classCount, isTrained=True, pool_type='average'):

        super(DenseNet121_z, self).__init__()

        self.pool_type = pool_type
        self.densenet121 = models.densenet121(pretrained=isTrained)
        kernelCount = self.densenet121.classifier.in_features
        # zhao
        self.densenet121.classifier = nn.Sequential(nn.Linear(kernelCount, classCount), nn.Sigmoid())

    def forward(self, x):
        features = self.densenet121.features(x)
        out = F.relu(features, inplace=True)
        # out = out*mask
        if self.pool_type == 'average':
            out = F.adaptive_avg_pool2d(out, (1, 1)).view(features.size(0), -1)
        elif self.pool_type == 'max':
            out = F.adaptive_max_pool2d(out, (1, 1)).view(features.size(0), -1)
        else:
            raise NotImplementedError(self.pool_type + ' : not implemented!')
        out = self.densenet121.classifier(out)
        return out


class ResBlock(nn.Module):
    def __init__(self, channel_size: int, negative_slope: float = 0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channel_size, channel_size, kernel_size=(3, 3), padding=(1, 1),
                      bias=False),
            nn.BatchNorm2d(channel_size),
            nn.LeakyReLU(negative_slope, inplace=True),
            nn.Conv2d(channel_size, channel_size, kernel_size=(3, 3), padding=(1, 1),
                      bias=False),
            nn.BatchNorm2d(channel_size)
        )

    def forward(self, x):
        return x + self.block(x)


class Sigmoid(nn.Module):
    def __init__(self, in_channels, r, n):
        super().__init__()
        self.module1 = nn.Sequential(
            nn.Conv2d(in_channels, n, kernel_size=(7, 7), padding=(3, 3)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(n, n, kernel_size=(1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(n),
        )

        resblocks = []
        for i in range(r):
            resblocks.append(ResBlock(n, 0.2))
        self.resblocks = nn.Sequential(*resblocks)

        self.module3 = nn.Sequential(
            nn.Conv2d(n, n, kernel_size=(3, 3), padding=(1, 1), bias=False),
            nn.BatchNorm2d(n)
        )
        self.module4 = nn.Conv2d(n, 3, kernel_size=(7, 7), padding=(3, 3))
        self.den121 = DenseNet121(1)

    def forward(self, x):
        out = self.module1(x)
        out_residual = out
        out = self.resblocks(out)
        out = self.module3(out)
        out = out + out_residual
        out = self.module4(out)
        out = out + x
        out = self.den121(out)
        return out


# 通过多个sigmoid叠加重新归一化图片
class sig_add(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = res50(10)
        self.sig = nn.Sigmoid()
        self.den121 = DenseNet121(1)

    def forward(self, x):
        out = self.base(x)
        x1 = (out[:, 0]).reshape([4, 1, 1, 1])
        y1 = (out[:, 1]).reshape([4, 1, 1, 1])
        x2 = (out[:, 2]).reshape([4, 1, 1, 1])
        y2 = (out[:, 3]).reshape([4, 1, 1, 1])
        x3 = (out[:, 4]).reshape([4, 1, 1, 1])
        y3 = (out[:, 5]).reshape([4, 1, 1, 1])
        x4 = (out[:, 6]).reshape([4, 1, 1, 1])
        y4 = (out[:, 7]).reshape([4, 1, 1, 1])
        x5 = (out[:, 8]).reshape([4, 1, 1, 1])
        y5 = (out[:, 9]).reshape([4, 1, 1, 1])
        x = (self.sig((x - x1) / y1) + self.sig((x - x2) / y2) + self.sig((x - x3) / y3) + self.sig(
            (x - x4) / y4) + self.sig((x - x5) / y5)) / 5
        x = self.den121(x)
        return x


class single_sig(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = res50(2)
        self.sig = nn.Sigmoid()
        self.den121 = DenseNet121(1)

    def forward(self, x):
        out = self.base(x)
        x1 = (out[:, 0]).reshape([4, 1, 1, 1])
        x2 = (out[:, 1]).reshape([4, 1, 1, 1])
        x = (x - x1) / x2
        x = self.sig(x)
        x = self.den121(x)
        return x


class test(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = res50(2)
        self.den121 = DenseNet121(1)

    def forward(self, x):
        out = self.den121(x)
        return out
