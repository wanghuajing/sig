import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
import torch
import cv2
import numpy as np


class ResNet50(nn.Module):
    def __init__(self, out_size):
        super(ResNet50, self).__init__()
        self.resnet50 = models.resnet50(pretrained=False)
        self.resnet50.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.resnet50.fc = nn.Sequential(nn.Linear(2048, out_size), nn.Sigmoid())

    def forward(self, x):
        x = self.resnet50(x)
        return x


class DenseNet121(nn.Module):

    def __init__(self, out_size):
        super(DenseNet121, self).__init__()
        self.densenet121 = models.densenet121(pretrained=True)
        num_ftrs = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(nn.Linear(num_ftrs, out_size), nn.Sigmoid())

    def forward(self, x):
        x = self.densenet121(x)
        return x


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
        self.sig = nn.Sigmoid()
        self.densenet121 = models.densenet121(pretrained=True)
        num_ftrs = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(nn.Linear(num_ftrs, 1), nn.Sigmoid())

    def forward(self, x):
        # 输出变换前的图片
        # a = x.cpu().detach().numpy()[0, :, :, :]
        # a = a.transpose(1, 2, 0)
        # a = (a * 255).astype(np.uint8)
        # cv2.imwrite('1.png', a)
        out = self.module1(x)
        out_residual = out
        out = self.resblocks(out)
        out = self.module3(out)
        out = out + out_residual
        out = self.module4(out)
        out = self.sig(out)
        out = out + x
        out = out / 2
        # 输出变换后的图片
        # a = out.cpu().detach().numpy()[0, :, :, :]
        # a = a.transpose(1, 2, 0)
        # a = (a * 255).astype(np.uint8)
        # cv2.imwrite('2.png', a)
        out = self.densenet121(out)
        return out


# 通过多个sigmoid叠加重新归一化图片
class sig_add(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = ResNet50(10)
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
        self.base = ResNet50(2)
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
        self.base = ResNet50(2)
        self.den121 = DenseNet121(1)

    def forward(self, x):
        out = self.den121(x)
        return out


class enhance_net_nopool(nn.Module):

    def __init__(self):
        super(enhance_net_nopool, self).__init__()

        self.relu = nn.ReLU(inplace=True)

        number_f = 32
        self.e_conv1 = nn.Conv2d(1, number_f, 3, 1, 1, bias=True)
        self.e_conv2 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv3 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv4 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv5 = nn.Conv2d(number_f * 2, number_f, 3, 1, 1, bias=True)
        self.e_conv6 = nn.Conv2d(number_f * 2, number_f, 3, 1, 1, bias=True)
        self.e_conv7 = nn.Conv2d(number_f * 2, 8, 3, 1, 1, bias=True)

        self.maxpool = nn.MaxPool2d(2, stride=2, return_indices=False, ceil_mode=False)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.classifier = ResNet50(1)

    def forward(self, x):
        x1 = self.relu(self.e_conv1(x))
        # p1 = self.maxpool(x1)
        x2 = self.relu(self.e_conv2(x1))
        # p2 = self.maxpool(x2)
        x3 = self.relu(self.e_conv3(x2))
        # p3 = self.maxpool(x3)
        x4 = self.relu(self.e_conv4(x3))

        x5 = self.relu(self.e_conv5(torch.cat([x3, x4], 1)))
        # x5 = self.upsample(x5)
        x6 = self.relu(self.e_conv6(torch.cat([x2, x5], 1)))

        x_r = torch.tanh(self.e_conv7(torch.cat([x1, x6], 1)))
        r1, r2, r3, r4, r5, r6, r7, r8 = torch.split(x_r, 1, dim=1)

        x = x + r1 * (torch.pow(x, 2) - x)
        x = x + r2 * (torch.pow(x, 2) - x)
        x = x + r3 * (torch.pow(x, 2) - x)
        enhance_image_1 = x + r4 * (torch.pow(x, 2) - x)
        x = enhance_image_1 + r5 * (torch.pow(enhance_image_1, 2) - enhance_image_1)
        x = x + r6 * (torch.pow(x, 2) - x)
        x = x + r7 * (torch.pow(x, 2) - x)
        enhance_image = x + r8 * (torch.pow(x, 2) - x)
        r = torch.cat([r1, r2, r3, r4, r5, r6, r7, r8], 1)
        out = self.classifier(enhance_image)
        return enhance_image_1,enhance_image,r



class get_image(nn.Module):

    def __init__(self):
        super(get_image, self).__init__()

        self.relu = nn.ReLU(inplace=True)

        number_f = 32
        self.e_conv1 = nn.Conv2d(1, number_f, 3, 1, 1, bias=True)
        self.e_conv2 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv3 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv4 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv5 = nn.Conv2d(number_f * 2, number_f, 3, 1, 1, bias=True)
        self.e_conv6 = nn.Conv2d(number_f * 2, number_f, 3, 1, 1, bias=True)
        self.e_conv7 = nn.Conv2d(number_f * 2, 8, 3, 1, 1, bias=True)

        self.maxpool = nn.MaxPool2d(2, stride=2, return_indices=False, ceil_mode=False)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.classifier = ResNet50(1)

    def forward(self, x):
        x1 = self.relu(self.e_conv1(x))
        # p1 = self.maxpool(x1)
        x2 = self.relu(self.e_conv2(x1))
        # p2 = self.maxpool(x2)
        x3 = self.relu(self.e_conv3(x2))
        # p3 = self.maxpool(x3)
        x4 = self.relu(self.e_conv4(x3))

        x5 = self.relu(self.e_conv5(torch.cat([x3, x4], 1)))
        # x5 = self.upsample(x5)
        x6 = self.relu(self.e_conv6(torch.cat([x2, x5], 1)))

        x_r = torch.tanh(self.e_conv7(torch.cat([x1, x6], 1)))
        r1, r2, r3, r4, r5, r6, r7, r8 = torch.split(x_r, 1, dim=1)

        x = x + r1 * (torch.pow(x, 2) - x)
        x = x + r2 * (torch.pow(x, 2) - x)
        x = x + r3 * (torch.pow(x, 2) - x)
        enhance_image_1 = x + r4 * (torch.pow(x, 2) - x)
        x = enhance_image_1 + r5 * (torch.pow(enhance_image_1, 2) - enhance_image_1)
        x = x + r6 * (torch.pow(x, 2) - x)
        x = x + r7 * (torch.pow(x, 2) - x)
        enhance_image = x + r8 * (torch.pow(x, 2) - x)
        r = torch.cat([r1, r2, r3, r4, r5, r6, r7, r8], 1)
        out = self.classifier(enhance_image)
        return enhance_image
