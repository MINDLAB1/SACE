import torch.nn as nn
import torch.nn.functional as F
import torch
import math
import numpy as np

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

        self.conv_1t = encoder_layer(in_channels, inter_channels)
        self.conv_2t = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1), nn.ReLU(inplace=True))
        self.conv_3t = decoder_layer(inter_channels, in_channels)

    def forward(self, x):
        x = self.conv_3t(self.conv_2t(self.conv_1t(x)))
        return x



class GeneratorRestor(nn.Module):
    def __init__(self, channels=3, filters=64):
        super(GeneratorRestor, self).__init__()

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

        self.conv_hf_0 = nn.Sequential(
            nn.Conv2d(3, filters, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, m):
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

        h_features = torch.cat((h_3, h_5, h_7), 1)
        h_features = self.conv_hf_0(h_features)

        return h_features


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
