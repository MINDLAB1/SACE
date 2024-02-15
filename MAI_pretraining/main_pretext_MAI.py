
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

import torch.nn as nn
import torch

import datasets.dataset_pretext

os.makedirs("images/training_MAI", exist_ok=True)
os.makedirs("saved_models_MAI", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--dataset_path", type=str, default="PATH_to_DATA", help="path of the dataset")
parser.add_argument("--batch_size", type=int, default=4, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")

parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=100, help="interval between saving image samples")
parser.add_argument("--checkpoint_interval", type=int, default=1000, help="batch interval between model checkpoints")
parser.add_argument("--residual_blocks", type=int, default=12, help="number of residual blocks in the generator")
parser.add_argument("--warmup_batches", type=int, default=200, help="number of batches with pixel-wise loss only")
parser.add_argument("--lambda_adv", type=float, default=5e-3, help="adversarial loss weight")
parser.add_argument("--lambda_pixel", type=float, default=1e-2, help="pixel-wise loss weight")
opt = parser.parse_args()
print(opt)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_gpus = torch.cuda.device_count()
if num_gpus > 1:
    multi_gpu = True
else:
    multi_gpu = False
###
input_shape = (192, 192)

# Initialize generator and discriminator
generator = MaskedRestor(opt.channels, filters=64, num_res_blocks=opt.residual_blocks).to(device)
discriminator = Discriminator((opt.channels, *input_shape)).to(device)
feature_extractor = FeatureExtractor().to(device)

if multi_gpu:
    generator = nn.DataParallel(generator)
    discriminator = nn.DataParallel(discriminator)
    feature_extractor = nn.DataParallel(feature_extractor)

# Set feature extractor to inference mode
feature_extractor.eval()

# Losses
criterion_GAN = torch.nn.BCEWithLogitsLoss().to(device)
criterion_content = torch.nn.L1Loss().to(device)
criterion_pixel = torch.nn.L1Loss().to(device)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

dataloader = DataLoader(
    datasets.dataset_pretext.data_set(opt.dataset_path, stage='MAI', mode='train'),
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.n_cpu,)

val_dataloader = DataLoader(
    datasets.dataset_pretext.data_set(opt.dataset_path, stage='MAI', mode='val'),
    batch_size=opt.batch_size,
    shuffle=False,
    num_workers=opt.n_cpu,)


# ----------
#  Training
# ----------

log = logger.Train_Logger(os.getcwd(), "train_log")

for epoch in range(opt.epoch, opt.n_epochs):
    generator.train()
    for i, (img_masked, img) in enumerate(dataloader):

        batches_done = epoch * len(dataloader) + i

        # Configure model input
        img_masked = Variable(img_masked.type(Tensor))
        img = Variable(img.type(Tensor))
        img_cp = img.detach().clone()
        # Adversarial ground truths
        if multi_gpu:
            valid = Variable(Tensor(np.ones((img_masked.size(0), *discriminator.module.output_shape))), requires_grad=False)
            fake = Variable(Tensor(np.zeros((img_masked.size(0), *discriminator.module.output_shape))), requires_grad=False)
        else:
            valid = Variable(Tensor(np.ones((img_masked.size(0), *discriminator.output_shape))), requires_grad=False)
            fake = Variable(Tensor(np.zeros((img_masked.size(0), *discriminator.output_shape))), requires_grad=False)

        # ------------------
        #  Train Generators
        # ------------------

        optimizer_G.zero_grad()

        # Generate a high resolution image from low resolution input
        img_restore = generator(img_masked, img)

        # Measure pixel-wise loss against ground truth
        loss_pixel = criterion_pixel(img_restore, img_cp)

        if batches_done < opt.warmup_batches:
            # Warm-up (pixel-wise loss only)
            loss_pixel.backward()
            optimizer_G.step()
            print(
                "[Epoch %d/%d] [Batch %d/%d] [G pixel: %f]"
                % (epoch, opt.n_epochs, i, len(dataloader), loss_pixel.item())
            )
            continue

        # Extract validity predictions from discriminator
        pred_real = discriminator(img).detach()
        pred_fake = discriminator(img_restore)

        # Adversarial loss (relativistic average GAN)
        loss_GAN = criterion_GAN(pred_fake - pred_real.mean(0, keepdim=True), valid)

        # Content loss
        gen_features = feature_extractor(torch.cat(3*[img_restore], 1))
        real_features = feature_extractor(torch.cat(3*[img], 1)).detach()
        loss_content = criterion_content(gen_features, real_features)

        # Total generator losssave_path
        loss_G = loss_content + opt.lambda_adv * loss_GAN + opt.lambda_pixel * loss_pixel

        loss_G.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        pred_real = discriminator(img)
        pred_fake = discriminator(img_restore.detach())

        # Adversarial loss for real and fake images (relativistic average GAN)
        loss_real = criterion_GAN(pred_real - pred_fake.mean(0, keepdim=True), valid)
        loss_fake = criterion_GAN(pred_fake - pred_real.mean(0, keepdim=True), fake)

        # Total loss
        loss_D = (loss_real + loss_fake) / 2

        loss_D.backward()
        optimizer_D.step()

        # --------------
        #  Log Progress
        # --------------

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, content: %f, adv: %f, pixel: %f]"
            % (
                epoch,
                opt.n_epochs,
                i,
                len(dataloader),
                loss_D.item(),
                loss_G.item(),
                loss_content.item(),
                loss_GAN.item(),
                loss_pixel.item(),
            )
        )

        if batches_done % opt.sample_interval == 0:
            # Save image grid with upsampled inputs and outputs
            img_grid = datasets.dataset_pretext.denormalize(torch.cat((img_masked, img_restore, img_cp), -1))
            save_image(img_grid, "images/training_MAI/%d.png" % batches_done, nrow=1, normalize=False)

        if batches_done % opt.checkpoint_interval == 0:
            # Save model checkpoints
            torch.save(generator.state_dict(), "saved_models_MAI/generator_%d.pth" % epoch)
            torch.save(discriminator.state_dict(), "saved_models_MAI/discriminator_%d.pth" %epoch)

    generator.eval()

    epoch_val_loss = metrics.LossAverage()
    epoch_val_ssim = metrics.LossAverage()
    epoch_val_psnr = metrics.LossAverage()

    for i, (img_masked, img) in enumerate(val_dataloader):
        # Configure model input
        img_masked = Variable(img_masked.type(Tensor))
        img = Variable(img.type(Tensor))
        with torch.no_grad():
            img_restore = generator(img_masked, img.clone())
        loss_pixel = criterion_pixel(img_restore, img)

        img = datasets.dataset_pretext.denormalize(img.detach())
        img_restore = datasets.dataset_pretext.denormalize(img_restore.detach())

        epoch_val_loss.update(loss_pixel.item(), img.size(0))

        img = img.cpu().numpy() * 255
        img_restore = img_restore.cpu().numpy() * 255
        for ii in range(img.shape[0]):
            img_ = img[ii, 0, :, :]
            img_restore_ = img_restore[ii, 0, :, :]
            epoch_val_ssim.update(metrics.calculate_ssim(img_, img_restore_), 1)
            epoch_val_psnr.update(metrics.calculate_psnr(img_, img_restore_), 1)

    val_log = OrderedDict({'Val Loss': epoch_val_loss.avg, 'Val SSIM': epoch_val_ssim.avg,
                           'Val PSNR': epoch_val_psnr.avg})
    log.update(epoch, val_log)
