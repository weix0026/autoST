"""AutoST: a 3-D U-Net over the input days, a transformer at the bottleneck,
channel and spatial attention on the skip connections, and a 2-D path for the
map layers that do not change by day. Unchanged from the study.

    x_static  (B, S, H, W)
    x_dynamic (B, T, D, H, W)
    -> logits (B, 1, H, W)
"""
import torch
import torch.nn as nn



class ChannelAttention(nn.Module):
    """CBAM-style channel attention over 3-D feature maps [B, C, T, H, W]."""

    def __init__(self, channels):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.fc = nn.Linear(channels, channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = self.avg_pool(x).squeeze(-1).squeeze(-1).squeeze(-1)
        maxp = self.max_pool(x).squeeze(-1).squeeze(-1).squeeze(-1)
        attn = self.sigmoid(self.fc(avg + maxp))
        return x * attn.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)


class SpatialAttention(nn.Module):
    """CBAM-style spatial attention over 3-D feature maps [B, C, T, H, W]."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv3d(1, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        maxp = torch.max(x, dim=1, keepdim=True)[0]
        attn = self.sigmoid(self.conv(maxp))
        return x * attn


class AutoSTTransformerBranch(nn.Module):
    """Transformer applied across the flattened spatio-temporal bottleneck tokens."""

    def __init__(self, d_model=512, nhead=8, num_layers=4):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, x):
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 3, 4, 1).reshape(B, T * H * W, C)
        x = self.encoder(x)
        x = x.reshape(B, T, H, W, C).permute(0, 4, 1, 2, 3)
        return x


def _conv_block_3d(in_c, out_c, dropout=0.3):
    return nn.Sequential(
        nn.Conv3d(in_c, out_c, 3, padding=1, bias=False),
        nn.BatchNorm3d(out_c),
        nn.ReLU(inplace=True),
        nn.Dropout3d(dropout) if dropout > 0 else nn.Identity(),
        nn.Conv3d(out_c, out_c, 3, padding=1, bias=False),
        nn.BatchNorm3d(out_c),
        nn.ReLU(inplace=True),
    )


class FireSpreadAutoSTNet(nn.Module):
    """
    AutoST-Net: 3-D UNet encoder for dynamic channels fused with a static 2-D
    pathway and a transformer bottleneck.

    Inputs
    ------
    x_static  : [B, static_channels, H, W]
    x_dynamic : [B, T, dynamic_channels, H, W]

    Output
    ------
    logits    : [B, 1, H, W]
    """

    def __init__(
        self,
        static_channels: int = 30,
        dynamic_channels: int = 18,
        out_channels: int = 1,
        t_sequence: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.t_sequence = t_sequence
        pool3d = nn.MaxPool3d((1, 2, 2))
        self.pool3d = pool3d

        # 3-D encoder
        self.enc3d_1 = _conv_block_3d(dynamic_channels, 64, dropout)
        self.enc3d_2 = _conv_block_3d(64, 128, dropout)
        self.enc3d_3 = _conv_block_3d(128, 256, dropout)
        self.enc3d_4 = _conv_block_3d(256, 512, dropout)
        self.bottleneck_pool = pool3d

        # Static pathway downsampled to bottleneck spatial size
        self.static_downsample = nn.Sequential(
            nn.Conv2d(static_channels, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 512, 3, stride=2, padding=1),
            nn.BatchNorm2d(512), nn.ReLU(),
            nn.Conv2d(512, 512, 3, stride=2, padding=1),
            nn.BatchNorm2d(512), nn.ReLU(),
        )

        self.transformer_branch = AutoSTTransformerBranch(d_model=512, nhead=8, num_layers=4)

        # Skip attentions
        self.att4 = nn.Sequential(ChannelAttention(512), SpatialAttention())
        self.att3 = nn.Sequential(ChannelAttention(256), SpatialAttention())
        self.att2 = nn.Sequential(ChannelAttention(128), SpatialAttention())
        self.att1 = nn.Sequential(ChannelAttention(64), SpatialAttention())

        # Decoder
        self.up4 = nn.ConvTranspose3d(1024, 512, (1, 2, 2), stride=(1, 2, 2))
        self.dec4 = _conv_block_3d(512 + 512, 512, dropout)
        self.up3 = nn.ConvTranspose3d(512, 256, (1, 2, 2), stride=(1, 2, 2))
        self.dec3 = _conv_block_3d(256 + 256, 256, dropout)
        self.up2 = nn.ConvTranspose3d(256, 128, (1, 2, 2), stride=(1, 2, 2))
        self.dec2 = _conv_block_3d(128 + 128, 128, dropout)
        self.up1 = nn.ConvTranspose3d(128, 64, (1, 2, 2), stride=(1, 2, 2))
        self.dec1 = _conv_block_3d(64 + 64, 64, dropout)
        self.final = nn.Conv2d(64, out_channels, 1)

        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm3d, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x_static, x_dynamic):
        x_dynamic = x_dynamic.permute(0, 2, 1, 3, 4)  # B, C, T, H, W

        e1 = self.enc3d_1(x_dynamic)
        e2 = self.enc3d_2(self.pool3d(e1))
        e3 = self.enc3d_3(self.pool3d(e2))
        e4 = self.enc3d_4(self.pool3d(e3))

        bottleneck_3d = self.bottleneck_pool(e4)
        trans_feat = self.transformer_branch(bottleneck_3d)
        bottleneck = torch.cat([bottleneck_3d, trans_feat], dim=1)  # B, 1024, T, H/16, W/16

        static_pooled = self.static_downsample(x_static)
        static_pooled = static_pooled.unsqueeze(2).repeat(1, 1, self.t_sequence, 1, 1)
        bottleneck = bottleneck + static_pooled.repeat(1, 2, 1, 1, 1)

        d = self.up4(bottleneck)
        d = torch.cat([d, self.att4(e4)], dim=1)
        d = self.dec4(d)

        d = self.up3(d)
        d = torch.cat([d, self.att3(e3)], dim=1)
        d = self.dec3(d)

        d = self.up2(d)
        d = torch.cat([d, self.att2(e2)], dim=1)
        d = self.dec2(d)

        d = self.up1(d)
        d = torch.cat([d, self.att1(e1)], dim=1)
        d = self.dec1(d)

        d = d.mean(dim=2)          # collapse T → B, 64, H, W
        out = self.final(d)
        return torch.clamp(out, -30, 30)
