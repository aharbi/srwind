import torch

from torch import nn

"""
This code is adpated from the implementation in:
https://github.com/TeaPearce/Conditional_Diffusion_MNIST
"""


class DoubleConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class UNetEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetEncoder, self).__init__()

        self.model = nn.Sequential(
            DoubleConvBlock(in_channels, out_channels), nn.MaxPool2d(2)
        )

    def forward(self, x):
        return self.model(x)


class UNetDecoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetDecoder, self).__init__()

        self.model = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2),
            DoubleConvBlock(out_channels, out_channels),
            DoubleConvBlock(out_channels, out_channels),
        )

    def forward(self, x, skip):
        x = torch.cat((x, skip), 1)
        x = self.model(x)
        return x


class EmbedFC(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(EmbedFC, self).__init__()
        self.input_dim = input_dim

        self.model = nn.Sequential(
            nn.Linear(input_dim, emb_dim), nn.GELU(), nn.Linear(emb_dim, emb_dim)
        )

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        return self.model(x)


class UNet(nn.Module):
    def __init__(self, in_channels, num_features=256, embedding=False):
        super(UNet, self).__init__()

        self.in_channels = in_channels
        self.num_features = num_features
        self.embedding = embedding

        self.init_conv = DoubleConvBlock(self.in_channels, self.num_features)

        self.encoder_block_1 = UNetEncoder(self.num_features, self.num_features)
        self.encoder_block_2 = UNetEncoder(self.num_features, self.num_features * 2)

        self.bottleneck = nn.Sequential(nn.AvgPool2d(5), nn.GELU())

        if self.embedding:
            self.timeembed1 = EmbedFC(1, 2 * self.num_features)
            self.timeembed2 = EmbedFC(1, 1 * self.num_features)

        self.decoder_block_1 = nn.Sequential(
            nn.ConvTranspose2d(2 * self.num_features, 2 * self.num_features, 5, 5),
            nn.GroupNorm(8, 2 * self.num_features),
            nn.ReLU(),
        )

        self.decoder_block_2 = UNetDecoder(4 * self.num_features, self.num_features)
        self.decoder_block_3 = UNetDecoder(2 * self.num_features, self.num_features)


        if self.embedding:
            self.out = nn.Sequential(
                nn.Conv2d(2 * self.num_features, self.num_features, 3, 1, 1),
                nn.GroupNorm(8, self.num_features),
                nn.ReLU(),
                nn.Conv2d(self.num_features, int(self.in_channels / 2), 3, 1, 1),
            )
        else:
            self.out = nn.Sequential(
                nn.Conv2d(2 * self.num_features, self.num_features, 3, 1, 1),
                nn.GroupNorm(8, self.num_features),
                nn.ReLU(),
                nn.Conv2d(self.num_features, self.in_channels, 3, 1, 1),
            )

    def forward(self, x, t=None):
        x = self.init_conv(x)

        e_1 = self.encoder_block_1(x)
        e_2 = self.encoder_block_2(e_1)

        bottleneck = self.bottleneck(e_2)

        d_1 = self.decoder_block_1(bottleneck)

        if self.embedding:
            if t == None:
                raise Exception("No value of t assigned")

            temb1 = self.timeembed1(t).view(-1, self.num_features * 2, 1, 1)
            temb2 = self.timeembed2(t).view(-1, self.num_features, 1, 1)

            d_2 = self.decoder_block_2(d_1 + temb1, e_2)
            d_3 = self.decoder_block_3(d_2 + temb2, e_1)
        else:
            d_2 = self.decoder_block_2(d_1, e_2)
            d_3 = self.decoder_block_3(d_2, e_1)

        out = self.out(torch.cat((d_3, x), 1))

        return out
