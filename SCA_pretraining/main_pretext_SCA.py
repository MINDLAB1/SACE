"""
Super-resolution of CelebA using Generative Adversarial Networks.
The dataset can be downloaded from: https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AADIKlz8PR9zr6Y20qbkunrba/Img/img_align_celeba.zip?dl=0
(if not available there see if options are listed at http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
Instrustion on running the script:
1. Download the dataset from the provided link
2. Save the folder 'img_align_celeba' to '../../data/'
4. Run the sript using command 'python3 esrgan.py'
"""

import argparse
import os
import numpy as np
from utils import metrics
from collections import OrderedDict
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torch.autograd import Variable

from models import *
from utils import logger
from utils import loss

import torch.nn as nn
import torch

import datasets.dataset_pretext

os.makedirs("images/training_SCA", exist_ok=True)
os.makedirs("images/validation_SCA", exist_ok=True)
os.makedirs("saved_models_SCA", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--dataset_path", type=str, default="PATH_to_DATA", help="path of the dataset")
parser.add_argument("--batch_size", type=int, default=4, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=4, help="number of cpu threads to use during batch generation")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=100, help="interval between saving image samples")
parser.add_argument("--checkpoint_interval", type=int, default=2000, help="batch interval between model checkpoints")
parser.add_argument("--residual_blocks", type=int, default=23, help="number of residual blocks in the generator")
parser.add_argument("--warmup_batches", type=int, default=500, help="number of batches with pixel-wise loss only")
parser.add_argument("--lambda_adv", type=float, default=5e-3, help="adversarial loss weight")
parser.add_argument("--lambda_pixel", type=float, default=1e-2, help="pixel-wise loss weight")
opt = parser.parse_args()
print(opt)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

enc_nf = [16, 32, 32, 32]
dec_nf = [32, 32, 32, 32, 32, 16, 16]
int_downsize = 2


# Initialize generator and discriminator
generator = VxmDense(inshape=(192, 192), nb_unet_features=[enc_nf, dec_nf], int_steps=7, int_downsize=int_downsize).to(device)

discriminator = Discriminator(input_shape=[1, 192, 192]).to(device)
discriminator.apply(init_model)

feature_extractor = FeatureExtractor().to(device)

# Losses
criterion_ncc = loss.NCC().loss

criterion_GAN = torch.nn.BCEWithLogitsLoss()
criterion_content = torch.nn.L1Loss()
criterion_pixel = torch.nn.L1Loss()

weights = [1]

# prepare deformation loss
criterion_grad = loss.Grad('l2', loss_mult=int_downsize).loss

if opt.epoch != 0:
    # Load pretrained models
    generator.load_state_dict(torch.load("saved_models_SCA/generator_%d.pth" % opt.epoch))

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

dataloader = DataLoader(
    datasets.dataset_pretext.data_set(opt.dataset_path, stage='SCA', mode='train'),
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.n_cpu,)

val_dataloader = DataLoader(
    datasets.dataset_pretext.data_set(opt.dataset_path, stage='SCA', mode='val'),
    batch_size=opt.batch_size,
    shuffle=False,
    num_workers=opt.n_cpu,)


# ----------
#  Training
# ----------

log = logger.Train_Logger(os.getcwd(), "train_pretext_log")

for epoch in range(opt.epoch, opt.n_epochs):
    generator.train()
    for i, (img_c, img_s) in enumerate(dataloader):

        batches_done = epoch * len(dataloader) + i

        # Configure model input
        img_c = Variable(img_c.type(Tensor))
        img_s = Variable(img_s.type(Tensor))

        # ------------------
        #  Train Generators
        # ------------------

        valid = np.ones([img_c.size(0), 1, int(192 / 2 ** 4), int(192 / 2 ** 4)])
        valid = torch.from_numpy(valid).to(device)
        fake = np.zeros([img_c.size(0), 1, int(192 / 2 ** 4), int(192 / 2 ** 4)])
        fake = torch.from_numpy(fake).to(device)

        img_warpped, deformation_field = generator(img_c, img_s)

        loss_register = criterion_ncc(img_warpped, img_c)

        loss_deformation = criterion_grad(deformation_field)

        loss = loss_register + 0.01 * loss_deformation

        # backpropagate and optimize
        optimizer_G.zero_grad()
        loss.backward()
        optimizer_G.step()

        # --------------
        #  Log Progress
        # --------------

        print(
            "[Epoch %d/%d] [Batch %d/%d] [Deform loss: %f, register loss: %f]"
            % (
                epoch,
                opt.n_epochs,
                i,
                len(dataloader),
                loss_deformation.item(),
                loss_register.item(),
            )
        )

        if batches_done % opt.sample_interval == 0:
            img_grid = datasets.dataset_pretext.denormalize(torch.cat((img_s, img_c, img_warpped), -1))
            save_image(img_grid, "images/training_SCA/%d.png" % batches_done, nrow=1, normalize=False)
            save_image(img_grid, "images/training_SCA/%d.png" % batches_done, nrow=1, normalize=False)

        if batches_done % opt.checkpoint_interval == 0:
            # Save model checkpoints
            torch.save(generator.state_dict(), "saved_models_SCA/generator_%d.pth" % epoch)

    generator.eval()
    epoch_wrap_loss = metrics.LossAverage()
    epoch_reg_loss = metrics.LossAverage()
    epoch_val_ssim = metrics.LossAverage()
    epoch_val_psnr = metrics.LossAverage()
    os.makedirs("images/validation_SCA/%d" % epoch, exist_ok=True)
    for i, (img_c, img_s) in enumerate(val_dataloader):
        # Configure model input
        img_c = Variable(img_c.type(Tensor))
        img_s = Variable(img_s.type(Tensor))

        with torch.no_grad():
            img_warpped, deformation_field = generator(img_c, img_s)
        loss_ncc = criterion_ncc(img_warpped, img_c)
        loss_deformation = criterion_grad(deformation_field)

        epoch_wrap_loss.update(loss_ncc.item(), img_c.size(0))
        epoch_reg_loss.update(loss_deformation.item(), img_c.size(0))

        img_grid = datasets.dataset_pretext.denormalize(
            torch.cat((img_s, img_c, img_warpped), -1))
        save_image(img_grid, "images/validation_SCA/%d/%d.png" % (epoch, i), nrow=1, normalize=False)

        img_warpped = datasets.dataset_pretext.denormalize(img_warpped)
        img_c = datasets.dataset_pretext.denormalize(img_c.clone())

        img_warpped = img_warpped.cpu().numpy() * 255
        img_c = img_c.cpu().numpy() * 255

        for ii in range(img_c.shape[0]):
            img_warpped_temp = img_warpped[ii, 0, :, :]
            img_c_temp = img_c[ii, 0, :, :]
            epoch_val_ssim.update(metrics.calculate_ssim(img_warpped_temp, img_c_temp), 1)
            epoch_val_psnr.update(metrics.calculate_psnr(img_warpped_temp, img_c_temp), 1)

    val_log = OrderedDict({'Val Reg Loss': epoch_reg_loss.avg, 'Val pred Loss': epoch_wrap_loss.avg, 'Val SSIM': epoch_val_ssim.avg, 'Val PSNR': epoch_val_psnr.avg})
    log.update(epoch, val_log)
