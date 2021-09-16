import torch.nn as nn
from torch import relu


class PolicyHead(nn.Module):

    def __init__(self, in_chanels=128, board_size=20):
        super(PolicyHead, self).__init__()

        self.in_chanels = in_chanels
        self.board_size = board_size

        self.conv = nn.Conv2d(in_chanels, 2, kernel_size=1)
        self.bn = nn.BatchNorm2d(2)

        self.fc = nn.Linear(board_size*board_size*2, board_size * board_size)

        self.logsofmax = nn.LogSoftmax(dim=1)

    def forward(self, x):

        out = relu(self.bn(self.conv(x)))
        out = out.view(-1, self.board_size*self.board_size*2)

        out = self.fc(out)

        p = self.logsofmax(out).exp()

        return p

if __name__ == "__main__":
    import torch

    input = torch.randn(2,128,20,20)

    m = PolicyHead(in_chanels=128, board_size=20)

    print(input.shape)
    r = m(input)
    print(r.shape)
    print(torch.sum(r[0]))