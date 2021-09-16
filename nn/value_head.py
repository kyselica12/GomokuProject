import torch.nn as nn
import torch.nn.functional as F
from torch import tanh, relu


class ValueHead(nn.Module):

    def __init__(self, in_chanels=128, board_size=20, hidden_dim=32):
        super(ValueHead, self).__init__()

        self.board_size = board_size
        self.conv = nn.Conv2d(in_chanels, out_channels=1, kernel_size=1)
        self.bn = nn.BatchNorm2d(1)

        self.fc1 = nn.Linear(self.board_size * self.board_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out = F.relu(self.bn(self.conv(x)))
        out = out.view(-1, self.board_size * self.board_size)
        out = relu(self.fc1(out))
        out = tanh(self.fc2(out))

        return out

if __name__ == "__main__":
    import torch

    input = torch.randn(2,128,20,20)
    o = input.view(-1,400)
    m = ValueHead(in_chanels=128, board_size=20)

    print(input.shape)
    r = m(input)
    print(r.shape)
