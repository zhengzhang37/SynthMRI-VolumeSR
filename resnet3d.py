import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from collections import OrderedDict
import numpy as np
from torchinfo import summary
from thop import profile
import psutil
import os
import time
from torch_dwt.functional import dwt3

class MultiscaleSimAM(nn.Module):
    def __init__(self, channels=None, e_lambda=1e-4):
        super(MultiscaleSimAM, self).__init__()

        self.activation = nn.Sigmoid()
        self.e_lambda = e_lambda

        # Multi-scale convolution layers
        self.multi_scale = nn.ModuleList([
            nn.Conv3d(channels, channels // 16, kernel_size=k, padding=k//2)
            for k in [3, 5, 7]
        ])
        
        self.fc = nn.Conv3d(channels // 16 * len(self.multi_scale), channels, kernel_size=1)

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s

    @staticmethod
    def get_module_name():
        return "simam_3d"

    def forward(self, x):
        b, c, d, h, w = x.size()  # batch, channels, depth, height, width
        
        n = d * h * w - 1  # Total number of voxels minus 1

        # Compute mean across spatial dimensions
        x_mean = x.mean(dim=[2, 3, 4], keepdim=True)
        
        # Compute variance
        x_minus_mu_square = (x - x_mean).pow(2)
        normalization_term = x_minus_mu_square.sum(dim=[2, 3, 4], keepdim=True) / n + self.e_lambda
        
        # SimAM attention weights
        simam_attention = x_minus_mu_square / (4 * normalization_term) + 0.5
        simam_attention = self.activation(simam_attention)

        # Multi-scale feature extraction
        pooled = nn.AdaptiveAvgPool3d(1)(x)  # Global average pooling
        ms_features = torch.cat([conv(pooled) for conv in self.multi_scale], dim=1)

        # Fully connected layer to integrate multi-scale features
        ms_attention = self.fc(ms_features)
        ms_attention = self.activation(ms_attention)

        # Combine SimAM attention and multi-scale attention
        combined_attention = simam_attention * ms_attention
        return x * combined_attention

class DWT(nn.Module):
    def __init__(self):
        super(DWT,self).__init__()
        self.required_grad = False
    def forward(self,x):
        return dwt3(x,"haar").view(x.shape[0], -1, x.shape[2]//2, x.shape[3]//2, x.shape[4]//2)
    
class DWT_transform(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.dwt = DWT()
        self.conv1x1_low = nn.Conv3d(in_channels, out_channels, kernel_size=1, padding=0)
        self.conv1x1_high = nn.Conv3d(in_channels*7, out_channels, kernel_size=1, padding=0)
        self.in_channels = in_channels
    
    def forward(self,x):
        dwt_low_freq, dwt_high_freq = self.dwt(x)[:, :self.in_channels, :, :, :], self.dwt(x)[:, self.in_channels:, :, :,:]
        assert dwt_low_freq.ndim == 5, "5-D tensor!"
        assert dwt_high_freq.ndim == 5, "5-D tensor!"
        dwt_low_freq = self.conv1x1_low(dwt_low_freq)
        dwt_high_freq = self.conv1x1_high(dwt_high_freq)
        return dwt_low_freq, dwt_high_freq
    

class ResidualBlockWithAttention(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.ReplicationPad3d(1),
            nn.Conv3d(in_features, in_features, kernel_size=3),
            nn.GroupNorm(8, in_features),
            nn.ReLU(inplace=True),
            nn.ReplicationPad3d(1),
            nn.Conv3d(in_features, in_features, kernel_size=3),
            nn.GroupNorm(8, in_features),
        )
        self.se_block = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(in_features, in_features // 16, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_features // 16, in_features, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        residual = x
        out = self.conv_block(x)
        se_weight = self.se_block(out)
        return residual + out * se_weight

class Fusion(nn.Module):
    def __init__(self, in_features):
        super(Fusion, self).__init__()
        self.fusion_conv = nn.Conv3d(in_features * 2, in_features, kernel_size=3, padding=1)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(in_features, in_features // 16, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_features // 16, in_features, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, flair, tof):
        """
        Args:
            flair: LR FLAIR 特征 (B, C, D, H, W).
            tof: HR TOF 特征 (B, C, D, H, W).
        Returns:
            融合后的特征.
        """
        fused = self.fusion_conv(torch.cat([flair, tof], dim=1))
        attention = self.channel_attention(fused)
        return fused * attention


class ResNet3D(nn.Module):
    def __init__(self, input_nc=1, output_nc=1, in_features=64, num_residual_blocks=6):
        super(ResNet3D, self).__init__()

        # Initial convolution blocks
        self.flair_encoder = nn.Sequential(
            nn.Conv3d(input_nc, in_features, kernel_size=7, padding=3),
            nn.GroupNorm(8, in_features),
            nn.ReLU(inplace=True)
        )
        self.tof_encoder = nn.Sequential(
            nn.Conv3d(input_nc, in_features, kernel_size=7, padding=3),
            nn.GroupNorm(8, in_features),
            nn.ReLU(inplace=True)
        )

        # Fusion Module
        self.fusion = Fusion(in_features)

        # Downsampling
        self.downsampling1 = nn.Sequential(
            nn.Conv3d(in_features, in_features * 2 - 8, 3, stride=2, padding=1),
            nn.GroupNorm(4, in_features * 2 - 8),
            nn.ReLU(inplace=True),
            MultiscaleSimAM(in_features * 2 - 8),
        )
        self.dwt1 = DWT_transform(in_features, 8)

        self.downsampling2 = nn.Sequential(
            nn.Conv3d(in_features * 2, in_features * 4 - 16, 3, stride=2, padding=1),
            nn.GroupNorm(8, in_features * 4 - 16),
            nn.ReLU(inplace=True),
            MultiscaleSimAM(in_features * 4 - 16),
        )
        self.dwt2 = DWT_transform(in_features*2, 16)

        # Bottleneck
        self.bottleneck = nn.Sequential(*[ResidualBlockWithAttention(in_features * 4) for _ in range(num_residual_blocks)])

        # Upsampling
        self.upsampling1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv3d(in_features * 8 + 16, in_features * 2, 3, padding=1),
            nn.GroupNorm(8, in_features * 2),
            nn.ReLU(inplace=True),
            MultiscaleSimAM(in_features * 2),
        )

        self.upsampling2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv3d(in_features * 4 + 8, in_features * 1, 3, padding=1),
            nn.GroupNorm(8, in_features * 1),
            nn.ReLU(inplace=True),
            MultiscaleSimAM(in_features * 1)
        )

        # Final output
        self.final_conv = nn.Sequential(
            nn.Conv3d(in_features * 1, output_nc, kernel_size=7, padding=3),
            nn.Tanh()
        )

    def forward(self, x):
        lr_flair = x[:, 0, :, :, :][:, None, ...]
        hr_tof = x[:, 1, :, :, :][:, None, ...]
        flair_features = self.flair_encoder(lr_flair) # (B, C, D, H, W)
        tof_features = self.tof_encoder(hr_tof) # (B, C, D, H, W)

        # Fusion
        fused_features = self.fusion(flair_features, tof_features) # (B, C, D, H, W)
        
        d1 = self.downsampling1(fused_features) # (B, 2C, D/2, H/2, W/2)

        d1_low, d1_high = self.dwt1(fused_features) # (B, C, D/2, H/2, W/2), (B, 7C, D/2, H/2, W/2)
        d1_fused = torch.cat([d1, d1_low], dim=1) # (B, 8C, D/2, H/2, W/2)
        
        d2 = self.downsampling2(d1_fused)  # (B, 4C, D/4, H/4, W/4)
        d2_low, d2_high = self.dwt2(d1_fused) # (B, 2C, D/4, H/4, W/4), (B, 14C, D/4, H/4, W/4)
        d2_fused = torch.cat([d2, d2_low], dim=1) # (B, 16C, D/4, H/4, W/4)
        
        # Bottleneck
        d3 = self.bottleneck(d2_fused) # (B, 4C, D/4, H/4, W/4)

        # Upsampling: Concatenate or combine high-frequency features
        u2 = self.upsampling1(torch.cat([d3, d2_fused, d2_high], dim=1)) # (B, 128, D/2, H/2, W/2)
        u1 = self.upsampling2(torch.cat([u2, d1_fused, d1_high], dim=1)) # (B, 64, D, H, W)

        output = self.final_conv(u1)

        return output


class Discriminator(nn.Module):

    # The paper mentions: "For discriminator networks, we use 70 × 70 PatchGAN [22]. Let Ck denote a  4 × 4
    # Convolution-InstanceNorm-LeakyReLU layer with k filters and stride 2. After the last layer, we apply a
    # convolution to produce a 1-dimensional output. We do not use InstanceNorm for the first C64 layer. We use leaky
    # ReLUs with a slope of 0.2. The discriminator architecture is: C64-C128-C256-C512"

    def __init__(self, input_nc=1):
        super(Discriminator, self).__init__()

        model = [spectral_norm(nn.Conv3d(input_nc, 64, 4, stride=2, padding=1)),
                 nn.LeakyReLU(0.2, inplace=True)]
        model += [spectral_norm(nn.Conv3d(64, 128, kernel_size=4, stride=2, padding=1)),
                  nn.LeakyReLU(0.2, inplace=True)]
        model += [spectral_norm(nn.Conv3d(128, 256, kernel_size=4, stride=1, padding=1)),
                  nn.LeakyReLU(0.2, inplace=True)]
        model += [spectral_norm(nn.Conv3d(256, 512, kernel_size=4, stride=1, padding=1)),
                  nn.LeakyReLU(0.2, inplace=True)]
        model += [spectral_norm(nn.Conv3d(512, 1, kernel_size=4, padding=1))]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = self.model(x)
        x = F.avg_pool3d(x, x.size()[2:]).view(x.size()[0])
        return x


if __name__ == '__main__':
    # discriminator = Discriminator(input_nc=1, num_scales=3)
    generator = ResNet3D(input_nc=1, output_nc=1)
    input = torch.rand(1, 2, 96, 320, 320)
    print(generator(input).shape)
    start_time = time.time()
    flops, params = profile(generator, inputs=(input, ))
    print(f"Total FLOPs: {str(flops/1000**3)}G")
    print(f"Total Params: {str(params/1000**3)}G")
    end_time = time.time()
    print("Time taken to compute FLOPs and Params: ", end_time - start_time)
