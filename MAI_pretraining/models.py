import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision.models import vgg19
import math
import numpy as np


class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        vgg19_model = vgg19(pretrained=True)
        self.vgg19_54 = nn.Sequential(*list(vgg19_model.features.children())[:35])

    def forward(self, img):
        return self.vgg19_54(img)


class DenseResidualBlock(nn.Module):
    """
    The core module of paper: (Residual Dense Network for Image Super-Resolution, CVPR 18)
    """

    def __init__(self, filters, res_scale=0.2):
        super(DenseResidualBlock, self).__init__()
        self.res_scale = res_scale

        def block(in_features, non_linearity=True):
            layers = [nn.Conv2d(in_features, filters, 3, 1, 1, bias=True)]
            if non_linearity:
                layers += [nn.LeakyReLU()]
            return nn.Sequential(*layers)

        self.b1 = block(in_features=1 * filters)
        self.b2 = block(in_features=2 * filters)
        self.b3 = block(in_features=3 * filters)
        self.b4 = block(in_features=4 * filters)
        self.b5 = block(in_features=5 * filters, non_linearity=False)
        self.blocks = nn.ModuleList([self.b1, self.b2, self.b3, self.b4, self.b5])

    def forward(self, x):
        inputs = x
        for block in self.blocks:
            out = block(inputs)
            inputs = torch.cat([inputs, out], 1)
        return out.mul(self.res_scale) + x


class ResidualInResidualDenseBlock(nn.Module):
    def __init__(self, filters, res_scale=0.2):
        super(ResidualInResidualDenseBlock, self).__init__()
        self.res_scale = res_scale
        self.dense_blocks = nn.Sequential(
            DenseResidualBlock(filters), DenseResidualBlock(filters), DenseResidualBlock(filters)
        )

    def forward(self, x):
        return self.dense_blocks(x).mul(self.res_scale) + x


class encoder_layer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(encoder_layer, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, 1, 1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels, in_channels, 3, 1, 1)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(in_channels, out_channels, 2, 2, 0)
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        return x


class decoder_layer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(decoder_layer, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, 1, 1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels, in_channels, 3, 1, 1)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.ConvTranspose2d(in_channels, out_channels, 2, 2, 0)
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        return x

class F_encoding_decoding(nn.Module):
    def __init__(self, in_channels=64, inter_channels=64):
        super(F_encoding_decoding, self).__init__()

        self.enc_conv_1 = encoder_layer(in_channels, inter_channels)
        self.enc_conv_2 = encoder_layer(inter_channels, inter_channels*2)
        self.enc_conv_3 = encoder_layer(inter_channels * 2, inter_channels * 4)
        self.bottle_conv = nn.Sequential(nn.Conv2d(inter_channels * 4, inter_channels * 4, 3, 1, 1), nn.ReLU(inplace=True))

        self.dec_conv_3 = decoder_layer(inter_channels * 4, inter_channels * 2)
        self.dec_conv_2 = decoder_layer(inter_channels * 2, inter_channels)
        self.dec_conv_1 = decoder_layer(inter_channels, in_channels)

    def forward(self, x):
        x = self.enc_conv_3(self.enc_conv_2(self.enc_conv_1(x)))
        x = self.bottle_conv(x)
        x = self.dec_conv_1(self.dec_conv_2(self.dec_conv_3(x)))
        return x


class MaskedRestor(nn.Module):
    def __init__(self, channels, filters=64, num_res_blocks=16):
        super(MaskedRestor, self).__init__()

        # HF branch
        self.in_kernel_3x3 = nn.Parameter(torch.Tensor(initDCTKernel(3)))
        self.freq_conv_3x3 = F_encoding_decoding(9)
        self.out_kernel_3x3 = nn.Parameter(torch.Tensor(initIDCTKernel(3)))
        # self.inv_freq_conv_3x3 = nn.Conv2d(9, 64, 3, 1, 1)

        self.in_kernel_5x5 = nn.Parameter(torch.Tensor(initDCTKernel(5)))
        self.freq_conv_5x5 = F_encoding_decoding(25)
        self.out_kernel_5x5 = nn.Parameter(torch.Tensor(initIDCTKernel(5)))
        # self.inv_freq_conv_5x5 = nn.Conv2d(25, 64, 3, 1, 1)

        self.in_kernel_7x7 = nn.Parameter(torch.Tensor(initDCTKernel(7)))
        self.freq_conv_7x7 = F_encoding_decoding(49)
        self.out_kernel_7x7 = nn.Parameter(torch.Tensor(initIDCTKernel(7)))
        # self.inv_freq_conv_7x7 = nn.Conv2d(49, 64, 3, 1, 1)

        # First layer
        self.conv1 = nn.Conv2d(channels, filters, kernel_size=3, stride=1, padding=1)
        # Residual blocks
        self.num_res_blocks = num_res_blocks
        self.res_blocks = nn.ModuleList(ResidualInResidualDenseBlock(filters) for _ in range(num_res_blocks))
        # for _ in range(num_res_blocks):
        #     self.res_blocks.append(ResidualInResidualDenseBlock(filters))
        # Second conv layer post residual blocks
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1)
        # Final output block
        self.conv3 = nn.Sequential(
            nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(filters, channels, kernel_size=3, stride=1, padding=1),
        )

        self.conv_hf_0 = nn.Sequential(
            nn.Conv2d(3, filters, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x, m):
        h_3 = F.conv2d(input=m, weight=self.in_kernel_3x3, padding=1)
        h_3 = self.freq_conv_3x3(h_3)
        h_3 = F.conv2d(input=h_3, weight=self.out_kernel_3x3, padding=1)
        # h_3 = self.inv_freq_conv_3x3(h_3)

        h_5 = F.conv2d(input=m, weight=self.in_kernel_5x5, padding=2)
        h_5 = self.freq_conv_5x5(h_5)
        h_5 = F.conv2d(input=h_5, weight=self.out_kernel_5x5, padding=2)
        # h_5 = self.inv_freq_conv_5x5(h_5)

        h_7 = F.conv2d(input=m, weight=self.in_kernel_7x7, padding=3)
        h_7 = self.freq_conv_7x7(h_7)
        h_7 = F.conv2d(input=h_7, weight=self.out_kernel_7x7, padding=3)
        # h_7 = self.inv_freq_conv_7x7(h_7)

        # print("h_3", h_3.shape, "h_5", h_5.shape, "h_7", h_7.shape)

        h_features = torch.cat((h_3, h_5, h_7), 1)
        h_features = self.conv_hf_0(h_features)

        out0 = self.conv1(x)
        # print("out0", out0.shape, "h_features", h_features.shape)

        out0 = self.conv2(torch.add(out0, h_features))
        for i in range(self.num_res_blocks):
            if i == 0:
                out1 = self.res_blocks[i](out0)
            else:
                out1 = self.res_blocks[i](out1)
        out = self.conv3(out1)
        return out


class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()

        self.input_shape = input_shape
        in_channels, in_height, in_width = self.input_shape
        patch_h, patch_w = int(in_height / 2 ** 4), int(in_width / 2 ** 4)
        self.output_shape = (1, patch_h, patch_w)

        def discriminator_block(in_filters, out_filters, first_block=False):
            layers = []
            layers.append(nn.Conv2d(in_filters, out_filters, kernel_size=3, stride=1, padding=1))
            if not first_block:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.Conv2d(out_filters, out_filters, kernel_size=3, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        layers = []
        in_filters = in_channels
        for i, out_filters in enumerate([64, 128, 256, 512]):
            layers.extend(discriminator_block(in_filters, out_filters, first_block=(i == 0)))
            in_filters = out_filters

        layers.append(nn.Conv2d(out_filters, 1, kernel_size=3, stride=1, padding=1))

        self.model = nn.Sequential(*layers)

    def forward(self, img):
        return self.model(img)


def initDCTKernel(N):
    kernel = np.zeros((N, N, N*N))
    cnum = 0
    for i in range(N):
        for j in range(N):
            ivec = np.linspace(0.5 * math.pi / N * i, (N - 0.5) * math.pi / N * i, num=N)
            ivec = np.cos(ivec)
            jvec = np.linspace(0.5 * math.pi / N * j, (N - 0.5) * math.pi / N * j, num=N)
            jvec = np.cos(jvec)
            slice = np.outer(ivec, jvec)

            if i==0 and j==0:
                slice = slice / N
            elif i*j==0:
                slice = slice * np.sqrt(2) / N
            else:
                slice = slice * 2.0 / N

            kernel[:,:,cnum] = slice
            cnum = cnum + 1
    kernel = kernel[np.newaxis, :]
    kernel = np.transpose(kernel, (3, 0, 1, 2))
    return kernel


################################## Generate the 2D-iDCT Kernels of size NxN ##################################
def initIDCTKernel(N):
    kernel = np.zeros((N, N, N*N))
    for i_ in range(N):
        i = N - i_ - 1
        for j_ in range(N):
            j = N - j_ - 1
            ivec = np.linspace(0, (i+0.5)*math.pi/N * (N-1), num=N)
            ivec = np.cos(ivec)
            jvec = np.linspace(0, (j+0.5)*math.pi/N * (N-1), num=N)
            jvec = np.cos(jvec)
            slice = np.outer(ivec, jvec)

            ic = np.sqrt(2.0 / N) * np.ones(N)
            ic[0] = np.sqrt(1.0 / N)
            jc = np.sqrt(2.0 / N) * np.ones(N)
            jc[0] = np.sqrt(1.0 / N)
            cmatrix = np.outer(ic, jc)

            slice = slice * cmatrix
            slice = slice.reshape((1, N*N))
            slice = slice[np.newaxis, :]
            kernel[i_, j_, :] = slice / (N * N)
    kernel = kernel[np.newaxis, :]
    kernel = np.transpose(kernel, (0, 3, 1, 2))
    return kernel
