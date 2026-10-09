import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, cin, cout, kernel_size=5, groups=8):
        super().__init__()
        pad = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv1d(cin, cout, kernel_size, padding=pad),
            nn.GroupNorm(groups, cout),
            nn.GELU(),
            nn.Conv1d(cout, cout, kernel_size, padding=pad),
            nn.GroupNorm(groups, cout),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class UNet1D(nn.Module):
    """1D U-Net for sparse-to-dense field reconstruction.

    Input (B, C_in, N) with N divisible by 2**len(channels); output (B, C_out, N).
    With channels (32, 64, 128, 256) and N = 4096 the encoder runs at
    4096, 2048, 1024 and 512 points and the bottleneck at 256.
    """

    def __init__(self, in_channels=3, out_channels=1, channels=(32, 64, 128, 256), kernel_size=5):
        super().__init__()
        self.down = nn.ModuleList()
        cin = in_channels
        for c in channels:
            self.down.append(ConvBlock(cin, c, kernel_size))
            cin = c
        self.pool = nn.MaxPool1d(2)
        self.bottleneck = ConvBlock(channels[-1], channels[-1], kernel_size)
        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        cin = channels[-1]
        for c in reversed(channels):
            self.up.append(nn.ConvTranspose1d(cin, c, 2, stride=2))
            self.dec.append(ConvBlock(2 * c, c, kernel_size))
            cin = c
        self.out = nn.Conv1d(channels[0], out_channels, 1)

    def forward(self, x):
        skips = []
        for block in self.down:
            x = block(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        for up, dec, skip in zip(self.up, self.dec, reversed(skips)):
            x = dec(torch.cat([up(x), skip], dim=1))
        return self.out(x)


def build_model(input_shape, output_shape=None, window=1, channels=(32, 64, 128, 256), kernel_size=5):
    """input_shape = (C_in, N); output_shape = (C_out, N), defaults to one channel."""
    out_channels = int(output_shape[0]) if output_shape is not None else 1
    return UNet1D(int(input_shape[0]), out_channels, channels, kernel_size)
