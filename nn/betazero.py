import torch
import torch.nn as nn
import dataclasses

from nn.residual import ResidualLayer
from nn.conv import ConvolutionalLayer
from nn.value_head import ValueHead
from nn.policy_head import PolicyHead


@dataclasses.dataclass
class BetaZeroConfig:
    board_size: int = 20
    num_states: int = 1
    input_cov_size: int = 128
    output_cov_size: int = 128
    num_res_layers: int = 5
    hidden_dim: int = 128
    device: str = 'cpu'


class BetaZero(nn.Module):

    def __init__(self, config: BetaZeroConfig):
        super(BetaZero, self).__init__()
        self.cofig = config
        self.conv = ConvolutionalLayer(board_size=config.board_size,
                                       in_chanels=config.num_states,
                                       stride=1,
                                       planes=config.input_cov_size)
        self.res_layers = [ResidualLayer(in_chanels=config.input_cov_size, planes=config.output_cov_size, stride=1)
                           for _ in range(config.num_res_layers)]
        self.value_head = ValueHead(in_chanels=config.output_cov_size, board_size=config.board_size, hidden_dim=config.hidden_dim)
        self.policy_head = PolicyHead(in_chanels=config.output_cov_size, board_size=config.board_size)

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

