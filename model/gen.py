import torch
import torch.nn as nn


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, r=True, **kwargs):
        super(CNNBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, padding_mode='reflect', **kwargs)
            if down
            else nn.ConvTranspose2d(in_channels, out_channels, **kwargs),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU() if r else nn.Identity(),
        )

    def forward(self, x):
        return self.conv(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            CNNBlock(in_channels, out_channels, kernel_size=3, padding=1),
            CNNBlock(out_channels, out_channels, r=False, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return self.block(x) + x


class Generator(nn.Module):
    def __init__(self, in_channels=3, num_features=32, num_residual=5):
        super().__init__()
        self.initial_layer = nn.Sequential(
            nn.Conv2d(
                in_channels, num_features, kernel_size=7, stride=1, padding=3, padding_mode='reflect'
            ),
            nn.InstanceNorm2d(num_features),
            nn.ReLU(inplace=True),  # 输入输出使用同一块地址，节省空间和时间
        )
        self.down_blocks = nn.ModuleList([
            CNNBlock(num_features, num_features * 2, kernel_size=3, stride=2, padding=1),
            CNNBlock(num_features * 2, num_features * 4, kernel_size=3, stride=2, padding=1),
        ])
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(num_features * 4, num_features * 4) for _ in range(num_residual)
        ])
        self.up_blocks = nn.ModuleList([
            CNNBlock(
                num_features * 4, num_features * 2, down=False, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            CNNBlock(
                num_features * 2, num_features, down=False, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
        ])
        self.last_layer = nn.Conv2d(
            num_features, in_channels, kernel_size=7, stride=1, padding=3, padding_mode='reflect'
        )

    def forward(self, x):
        x = self.initial_layer(x)
        for layer in self.down_blocks:
            x = layer(x)
        for layer in self.residual_blocks:
            x = layer(x)
        for layer in self.up_blocks:
            x = layer(x)
        x = self.last_layer(x)
        return torch.tanh(x)


def main():
    x = torch.randn(size=(1, 3, 256, 256))
    print(f"input shape is {x.shape}")
    model = Generator(in_channels=3)
    out = model(x)
    print(f"output shape is {out.shape}")


if __name__ == "__main__":
    main()




