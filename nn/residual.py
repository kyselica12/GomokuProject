
import torch.nn as nn
import torch.nn.functional as F


class ResidualLayer(nn.Module):

    def __init__(self, in_chanels=128, planes=128, stride=1):
        super(ResidualLayer, self).__init__()

        self.conv1 = nn.Conv2d(in_chanels, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_chanels, planes, kernel_size=3, stride=stride, padding=1, bias=False)

        self.bn1 = nn.BatchNorm2d(planes)
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):

        res = x

        out1 = self.bn1(self.conv1(x))
        out1 = F.relu(out1)

        out2 = self.bn2(self.conv2(out1))

        out = out2 + res
        out = F.relu(out)

        return out

if __name__ == "__main__":

    import torch

    input = torch.randn(2,128,20,20)
    print(input.shape)

    res = ResidualLayer()
    r = res(input)

    print(r.shape)

