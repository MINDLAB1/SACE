import torch.nn as nn
import numpy as np
import torch
from torchvision.models import vgg19
import ramps


class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        vgg19_model = vgg19(pretrained=True)
        self.vgg19_54 = nn.Sequential(*list(vgg19_model.features.children())[:35])

    def forward(self, img):
        return self.vgg19_54(img)


class block(nn.Module):
    def __init__(self, in_features, filters, non_linearity=True):
        super(block, self).__init__()
        self.conv_layer = nn.Conv2d(in_features, filters, 3, 1, 1, bias=True)
        self.non_linearity = non_linearity
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        x = self.conv_layer(x)
        if self.non_linearity:
            x = self.relu(x)
        return x


class DenseResidualBlock(nn.Module):
    """
    The core module of paper: (Residual Dense Network for Image Super-Resolution, CVPR 18)
    """

    def __init__(self, filters, res_scale=0.2):
        super(DenseResidualBlock, self).__init__()
        self.res_scale = res_scale

        self.b1 = block(1 * filters, filters)
        self.b2 = block(2 * filters, filters)
        self.b3 = block(3 * filters, filters)
        self.b4 = block(4 * filters, filters)
        self.b5 = block(5 * filters, filters, non_linearity=False)
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


class GeneratorRRDB(nn.Module):
    def __init__(self, channels, filters=64, num_res_blocks=16, num_upsample=1):
        super(GeneratorRRDB, self).__init__()

        # First layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(channels, filters, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1),
        )

        # Residual blocks
        self.res_blocks = nn.Sequential(*[ResidualInResidualDenseBlock(filters) for _ in range(num_res_blocks)])
        # Second conv layer post residual blocks
        self.conv2 = nn.Sequential(
            nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1),
        )
        self.conv2_1 = nn.Sequential(
            nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1),
        )
        self.conv2_2 = nn.Sequential(
            nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1),
        )
        # Upsampling layers
        upsample_layers = []
        for _ in range(num_upsample):
            upsample_layers += [
                nn.Conv2d(filters, filters * 4, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(),
                nn.PixelShuffle(upscale_factor=2),
            ]
        self.upsampling = nn.Sequential(*upsample_layers)
        # Final output block
        self.conv3 = nn.Sequential(
            nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(filters, 1, kernel_size=3, stride=1, padding=1),
        )

        ### trans

        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

        self.conv_cont_0 = nn.Sequential(
            nn.Conv2d(1, filters, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(filters, 1, kernel_size=3, stride=1, padding=1),
        )

        self.conv_cont_1 = nn.Sequential(
            nn.Conv2d(channels, filters, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1),
        )

        self.weight1 = nn.Parameter(torch.ones([1]))
        self.weight2 = nn.Parameter(torch.ones([1]))
        self.weight3 = nn.Parameter(torch.ones([1]))


    def forward(self, x, x_1, x_2, x_hf):

        # weighted input features with context info
        x_1 = self.conv_cont_0(x_1)
        x_2 = self.conv_cont_0(x_2)

        attn_context = self.cos(x_1, x_2)
        attn_context = attn_context.unsqueeze(1)
        context_feature = self.conv_cont_1(x * (1 + attn_context))


        # main protocol
        out0 = self.conv1(x)
        out1 = self.res_blocks(out0)
        out1 = out1 + self.weight1 * x_hf
        out2 = self.conv2(out1)

        out2 = out2 + self.weight2 * context_feature
        out2 = self.conv2_1(out2)

        ## features + context features + HF features
        out3 = self.weight3 * out0 + out2

        out3 = self.conv2_2(out3)

        up_sampled_features = self.upsampling(out3)

        ## predicted image + upsampled image
        predicted_7t = self.conv3(up_sampled_features)

        return predicted_7t


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



def init_model(net):
    if isinstance(net, nn.Conv2d) or isinstance(net, nn.ConvTranspose2d):
        nn.init.kaiming_normal_(net.weight.data, 0.25)
        nn.init.constant_(net.bias.data, 0)

def get_current_consistency_weight(epoch, EPOCHS):
    return np.clip(ramps.sigmoid_rampup(epoch, EPOCHS), 0.0, 1.0)
