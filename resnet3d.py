import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from collections import OrderedDict
import numpy as np
from torchinfo import summary
from thop import profile
import psutil

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        # here we are not writing self.in_features = in_features because were are not going to use in_features in any
        # other function definition other than __init__
        conv_block = [nn.ReplicationPad3d(1),    # nn.ReflectionPad2d(1),
                      spectral_norm(nn.Conv3d(in_features, in_features, 3)),
                      nn.InstanceNorm3d(in_features),
                      nn.ReLU(inplace=True),
                      nn.ReplicationPad3d(1),    # nn.ReflectionPad2d(1),
                      spectral_norm(nn.Conv3d(in_features, in_features, 3)),
                      nn.InstanceNorm3d(in_features)]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)



class ResnetGenerator3DwithSpectralNorm(nn.Module):

    # The paper mentions: "We use 6 residual blocks for 128 × 128 training images, and 9 residual blocks for
    # 256 × 256 or higher-resolution training images."

    def __init__(self, input_nc=2, output_nc=1, num_residual_blocks=9):
        super(ResnetGenerator3DwithSpectralNorm, self).__init__()

        # Initial convolution block for the input
        model = [nn.ReplicationPad3d(3),    # nn.ReflectionPad2d(3),
                 spectral_norm(nn.Conv3d(input_nc, 64, 7)),
                 nn.InstanceNorm3d(64),
                 nn.ReLU(inplace=True)]

        # Downsampling
        in_features = 64
        out_features = in_features * 2

        for i in range(2):
            model += [spectral_norm(nn.Conv3d(in_features, out_features, 3, stride=2, padding=1)),
                      nn.InstanceNorm3d(out_features),
                      nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features * 2

        # Concatenating the Residual Blocks
        for i in range(num_residual_blocks):
            model += [ResidualBlock(in_features)]

        # Upsampling
        out_features = in_features // 2
        for i in range(2):

            model += [nn.Upsample(scale_factor=2, mode='nearest'),
                      nn.ReplicationPad3d(1),   # nn.ReflectionPad2d(1),
                      spectral_norm(nn.Conv3d(in_features, out_features, kernel_size=3, stride=1, padding=0)),
                      nn.InstanceNorm3d(out_features),
                      nn.ReLU(inplace=True) ]
            in_features=out_features
            out_features=in_features//2

        # the final Output layer
        model += [nn.ReplicationPad3d(3), # nn.ReflectionPad2d(3),
                  spectral_norm(nn.Conv3d(64, output_nc, 7)),
                  nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        "Standard forward function"
        return self.model(x)

class PatchGANDiscriminatorwithSpectralNorm(nn.Module):

    # The paper mentions: "For discriminator networks, we use 70 × 70 PatchGAN [22]. Let Ck denote a  4 × 4
    # Convolution-InstanceNorm-LeakyReLU layer with k filters and stride 2. After the last layer, we apply a
    # convolution to produce a 1-dimensional output. We do not use InstanceNorm for the first C64 layer. We use leaky
    # ReLUs with a slope of 0.2. The discriminator architecture is: C64-C128-C256-C512"

    def __init__(self, input_nc=1):
        super(PatchGANDiscriminatorwithSpectralNorm, self).__init__()

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

if __name__ == "__main__":
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image_size = 128
    # x = torch.Tensor(1, 2, image_size, image_size, image_size)
    # x.to(device)
    # print("x size: {}".format(x.size()))
    # process = psutil.Process()
    # before = process.memory_info().rss / (1024 * 1024)
    # initial_mem = torch.cuda.memory_allocated()
    model = ResnetGenerator3DwithSpectralNorm(input_nc=2, output_nc=1).cuda()
    model.eval()
    input = torch.rand(1, 2, 219, 320, 290).cuda()
    with torch.no_grad():
        output = model(input)
    print(output.shape)
    # input = torch.rand(1, 2, 219, 320, 290)
    # 监控显存使用
    # with torch.no_grad():
    #     output = model(input)  # 执行推理
    # final_mem = torch.cuda.memory_allocated()

    # print(f"显存使用增加了 {(final_mem - initial_mem) / 1024**2} MB")
    # with torch.no_grad():  # 确保不计算梯度
    #     output = model(input)

    # after = process.memory_info().rss / (1024 * 1024)  # 再次获取内存使用信息
    # print(f"Memory used: {after - before} MB")
    # # output = model(input)
    # model_info = summary(model, input_size=input.size(), verbose=0)
    # print(model_info)
    # flops, params = profile(model, inputs=(input, ))
    # print(f"Total FLOPs: {str(flops/1000**3)}G")
    # print(f"Total Params: {str(params/1000**3)}G")
    # print(f"Total FLOPs: {model_info.total_flops}")
    # out = model(x)
    # print("out size: {}".format(out.size()))