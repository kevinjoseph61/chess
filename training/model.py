"""
AlphaZero-style chess neural network.

Architecture:
- Input: 18-plane 8x8 board representation
- Backbone: 6 residual blocks, 128 filters (small enough for browser ONNX inference)
- Policy head: outputs probability distribution over 4672 possible moves
- Value head: outputs scalar win probability in [-1, 1]

Move encoding (AlphaZero style, 8x8x73 = 4672):
  - 56 queen-type moves: 7 distances x 8 directions (N, NE, E, SE, S, SW, W, NW)
  - 8 knight moves
  - 9 underpromotions: 3 directions (left-diag, forward, right-diag) x 3 pieces (knight, bishop, rook)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_INPUT_PLANES = 18
NUM_FILTERS = 128
NUM_RES_BLOCKS = 6
NUM_POLICY_MOVES = 4672  # 8 * 8 * 73


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = F.relu(out + residual)
        return out


class AlphaZeroNet(nn.Module):
    def __init__(
        self,
        num_input_planes=NUM_INPUT_PLANES,
        num_filters=NUM_FILTERS,
        num_res_blocks=NUM_RES_BLOCKS,
    ):
        super().__init__()
        # Initial convolution
        self.conv_input = nn.Conv2d(
            num_input_planes, num_filters, 3, padding=1, bias=False
        )
        self.bn_input = nn.BatchNorm2d(num_filters)

        # Residual tower
        self.res_blocks = nn.Sequential(
            *[ResBlock(num_filters) for _ in range(num_res_blocks)]
        )

        # Policy head
        self.policy_conv = nn.Conv2d(num_filters, 32, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * 8 * 8, NUM_POLICY_MOVES)

        # Value head
        self.value_conv = nn.Conv2d(num_filters, 1, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(8 * 8, 128)
        self.value_fc2 = nn.Linear(128, 1)

    def forward(self, x):
        # Backbone
        x = F.relu(self.bn_input(self.conv_input(x)))
        x = self.res_blocks(x)

        # Policy head
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(p.size(0), -1)
        p = self.policy_fc(p)
        # Don't apply softmax here — we'll mask illegal moves and apply log_softmax in training/MCTS

        # Value head
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))

        return p, v
