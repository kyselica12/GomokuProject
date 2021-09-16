import torch.nn as nn
import torch.nn.functional as F


class ConvolutionalLayer(nn.Module):

    def __init__(self, board_size, in_chanels=1, planes=128, stride=1):
        super(ConvolutionalLayer, self).__init__()
        self.board_size = board_size
        self.in_chanels = in_chanels

        self.conv = nn.Conv2d(self.in_chanels, planes, kernel_size=3, stride=stride, padding=1, bias=False)

        self.bn = nn.BatchNorm2d(planes)

    def forward(self, state):
        s = state.view(-1, self.in_chanels, self.board_size, self.board_size) # batch_size x planes x board_size x board_size
        s = F.relu(self.bn(self.conv(s)))

        return s

if __name__ == "__main__":

    import torch

    input = torch.randn(2,1,20,20)
    print(input.shape)

    conv = ConvolutionalLayer(20, 1)
    r = conv(input)

    print(r.shape)


