import torch
import torch.nn as nn

from nn.residual import ResidualLayer
from nn.conv import ConvolutionalLayer
from nn.value_head import ValueHead
from nn.policy_head import PolicyHead


class BetaZero(nn.Module):

    def __init__(self, board_size=20, num_states=1, input_conv_size=128, output_conv_size=128, num_res_layers=5):
        super(BetaZero, self).__init__()

        self.conv = ConvolutionalLayer(board_size=board_size, in_chanels=num_states,stride=1, planes=input_conv_size)
        self.res_layers = [ResidualLayer(in_chanels=input_conv_size, planes=output_conv_size, stride=1) for _ in range(num_res_layers)]
        self.value_head = ValueHead(in_chanels=output_conv_size, board_size=board_size)
        self.policy_head = PolicyHead(in_chanels=output_conv_size, board_size=board_size)

    def forward(self, x):

        x = self.conv(x)

        for rl in self.res_layers:
            x = rl(x)

        value = self.value_head(x)
        policy = self.policy_head(x)

        return value, policy

    def loss(self, prediction, label):
        (v1, p1) = prediction
        (v2, p2) = label

        value_err = (v2.float() - torch.transpose(v1, 0, 1))**2
        policy_error = (p2.float() * p1.log()).sum(1)

        return (value_err - policy_error).mean()

if __name__ == "__main__":
    import torch

    input = torch.randn(2,1,20,20)

    m = BetaZero(board_size=20)

    print(input.shape)
    r = m(input)
    print(r[0].shape, r[1].shape)

