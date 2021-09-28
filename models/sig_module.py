import torch.nn as nn
from torchvision.models import resnet50, resnet18, resnet34
import torch.nn.functional as F
import torch


def res50(in_channels: int, num_classes: int = 40):
    model = resnet18(pretrained=True)
    model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model.fc = nn.Linear(512, num_classes)
    return model


# 通过多个sigmoid叠加重新归一化图片
class sig_module(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = res50(1, 20)
        self.sig = nn.Sigmoid()
        self.classification = res50(1, 2)

    def forward(self, x):
        out = self.base(x)
        img = x
        for i in range(out.shape[1] // 2):
            for j in range(out.shape[0]):
                x1 = (x[j, ...] - out[j, i * 2]) / out[j, i * 2 + 1]
                img = img + x1
        out = self.classification(img)
        return out




# if __name__ == '__main__':
#     net = sig_module()
#     loss_function = nn.MSELoss()
#     x = torch.rand(4, 1, 224, 224)
#     gt = torch.rand(1)
#     out = net(x)
#     loss = loss_function(out, gt)
#     loss.backward()
