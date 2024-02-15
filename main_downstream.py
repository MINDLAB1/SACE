import argparse
import os
from utils import metrics
from collections import OrderedDict
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from models_sp import *
from models_sr import *
from hf_models import GeneratorRestor
from utils import logger

import torch.nn as nn
import torch
import time
import datasets.dataset_downstream


os.makedirs("images/training", exist_ok=True)
os.makedirs("images/validation", exist_ok=True)
os.makedirs("images/testing", exist_ok=True)
os.makedirs("saved_models", exist_ok=True)

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
parser.add_argument("--hr_height", type=int, default=384, help="high res. image height")
parser.add_argument("--hr_width", type=int, default=384, help="high res. image width")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=100, help="interval between saving image samples")
parser.add_argument("--checkpoint_interval", type=int, default=1000, help="batch interval between model checkpoints")
parser.add_argument("--residual_blocks", type=int, default=12, help="number of residual blocks in the generator")
parser.add_argument("--warmup_batches", type=int, default=200, help="number of batches with pixel-wise loss only")
parser.add_argument("--lambda_adv", type=float, default=0.005, help="adversarial loss weight")
parser.add_argument("--lambda_pixel", type=float, default=0.01, help="pixel-wise loss weight")
opt = parser.parse_args()
print(opt)

enc_nf = [16, 32, 32, 32]
dec_nf = [32, 32, 32, 32, 32, 16, 16]
int_downsize = 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_gpus = torch.cuda.device_count()
if num_gpus > 1:
    multi_gpu = True
else:
    multi_gpu = False

print("GPU num: ", num_gpus)

hr_shape = (opt.hr_height, opt.hr_width)
lr_shape = (192, 192)

# Initialize generator and discriminator
deformation_extractor = VxmDense(inshape=(192, 192),
                                 nb_unet_features=[enc_nf, dec_nf],
                                 int_steps=7,
                                 int_downsize=int_downsize).to(device)

deformation_extractor.load_state_dict(torch.load("saved_models_SCA/generator.pth"))

hf_feature_extractor = GeneratorRestor(filters=64).to(device)
model_dict = hf_feature_extractor.state_dict()
pretrained_dict = torch.load("saved_models_MAI/generator.pth")
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
model_dict.update(pretrained_dict)
hf_feature_extractor.load_state_dict(model_dict)


generator = GeneratorRRDB(opt.channels, filters=64, num_res_blocks=opt.residual_blocks).to(device)
generator.apply(init_model)
discriminator = Discriminator(input_shape=(opt.channels, *hr_shape)).to(device)
discriminator.apply(init_model)


feature_extractor = FeatureExtractor().to(device)

if multi_gpu:
    deformation_extractor = nn.DataParallel(deformation_extractor)
    hf_feature_extractor = nn.DataParallel(hf_feature_extractor)
    generator = nn.DataParallel(generator)
    discriminator = nn.DataParallel(discriminator)
    feature_extractor = nn.DataParallel(feature_extractor)

# Set feature extractor to inference mode
feature_extractor.eval()
hf_feature_extractor.eval()
deformation_extractor.eval()

# Losses
criterion_GAN = torch.nn.BCEWithLogitsLoss()
criterion_content = torch.nn.L1Loss()
criterion_pixel = torch.nn.L1Loss()

if opt.epoch != 0:
    # Load pretrained models
    generator.load_state_dict(torch.load("saved_models/generator_%d.pth" % opt.epoch))
    discriminator.load_state_dict(torch.load("saved_models/discriminator_%d.pth" % opt.epoch))
# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

train_dataset = datasets.dataset_downstream.data_set(opt.dataset_path, mode='train')
dataloader = DataLoader(
    train_dataset,
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.n_cpu,)

val_dataset = datasets.dataset_downstream.data_set(opt.dataset_path, mode='val')
val_dataloader = DataLoader(
    val_dataset,
    batch_size=opt.batch_size,
    shuffle=False,
    num_workers=opt.n_cpu,)

test_dataset = datasets.dataset_downstream.data_set(opt.dataset_path, mode='test')
test_dataloader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=opt.n_cpu,)

print("train", train_dataset.img_num, "val", val_dataset.img_num)


def deformation_extraction(deformation_extractor, img_self, img_n):
    x = torch.cat((img_self, img_n), 1)
    x = deformation_extractor.module.unet_model(x)

    # transform into flow field
    flow_field = deformation_extractor.module.flow(x)

    # resize flow for integration
    pos_flow = flow_field
    if deformation_extractor.module.resize:
        pos_flow = deformation_extractor.module.resize(pos_flow)

    # integrate to produce diffeomorphic warp
    if deformation_extractor.module.integrate:
        pos_flow = deformation_extractor.module.integrate(pos_flow)

        # resize to final resolution
        if deformation_extractor.module.fullsize:
            pos_flow = deformation_extractor.module.fullsize(pos_flow)
    x = deformation_extractor.module.transformer(img_n, pos_flow)
    return x.detach()


# ----------
#  Training
# ----------

log_train = logger.Train_Logger(os.getcwd(), "train_log")
log_val = logger.Train_Logger(os.getcwd(), "val_log")
log_test = logger.Train_Logger(os.getcwd(), "test_log")

for epoch in range(opt.epoch, opt.n_epochs):
    generator.train()
    # deformation_extractor.eval()

    epoch_D_loss = metrics.LossAverage()
    epoch_Total_loss = metrics.LossAverage()
    epoch_content_loss = metrics.LossAverage()
    epoch_G_loss = metrics.LossAverage()
    epoch_pixel_loss = metrics.LossAverage()

    for i, imgs in enumerate(dataloader):

        batches_done = epoch * len(dataloader) + i

        # Configure model input
        imgs_lr = imgs["lr"]
        imgs_lr_m1 = imgs["lr-1"]
        imgs_lr_p1 = imgs["lr+1"]
        mask = imgs["mask"]

        imgs_lr, imgs_lr_p1, imgs_lr_m1 = imgs_lr.float().to(device), imgs_lr_p1.float().to(
            device), imgs_lr_m1.float().to(device)
        mask = mask.float().to(device)

        imgs_hr = imgs["hr"]
        imgs_hr = imgs_hr.float().to(device)

        # Adversarial ground truths
        valid = np.ones([imgs_lr.size(0), 1, int(hr_shape[0] / 2 ** 4), int(hr_shape[1] / 2 ** 4)])
        valid = torch.from_numpy(valid).to(device)
        fake = np.zeros([imgs_lr.size(0), 1, int(hr_shape[0] / 2 ** 4), int(hr_shape[1] / 2 ** 4)])
        fake = torch.from_numpy(fake).to(device)

        # ------------------
        #  Train Generators
        # ------------------

        optimizer_G.zero_grad()
        # start = time.time()
        # Generate a high resolution image from low resolution input
        with torch.no_grad():
            refined_slices = deformation_extraction(deformation_extractor, torch.cat([imgs_lr] * 2, 0),
                                                    torch.cat((imgs_lr_m1, imgs_lr_p1), 0))
            refined_last_slice, refined_next_slice = refined_slices[:imgs_lr.shape[0], ...], refined_slices[
                                                                                             imgs_lr.shape[0]:, ...]
            hf_features = hf_feature_extractor(imgs_lr.detach().clone())
        gen_hr = generator(imgs_lr, refined_last_slice.detach(), refined_next_slice.detach(), hf_features)
        # end = time.time()
        # print("time", end - start)
        # Measure pixel-wise loss against ground truth
        loss_pixel = criterion_pixel(gen_hr, imgs_hr)

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
        pred_real = discriminator(imgs_hr).detach()
        pred_fake = discriminator(gen_hr)

        # Adversarial loss (relativistic average GAN)
        loss_GAN = criterion_GAN(pred_fake - pred_real.mean(0, keepdim=True), valid)

        # Content loss
        gen_features_global = feature_extractor(torch.cat(3 * [gen_hr], 1))
        real_features_global = feature_extractor(torch.cat(3 * [imgs_hr], 1)).detach()
        gen_features_local = feature_extractor(torch.cat(3 * [gen_hr * mask], 1))
        real_features_local = feature_extractor(torch.cat(3 * [imgs_hr * mask], 1)).detach()

        weight = get_current_consistency_weight(epoch, 100)

        loss_content = (1 - weight) * criterion_content(gen_features_global,
                                                        real_features_global) + \
                       weight * criterion_content(gen_features_local, real_features_local)

        # Total generator losssave_path
        loss_G = loss_content + opt.lambda_adv * loss_GAN + opt.lambda_pixel * loss_pixel

        loss_G.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        pred_real = discriminator(imgs_hr)
        pred_fake = discriminator(gen_hr.detach())

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

        epoch_D_loss.update(loss_D.item(), imgs_lr.shape[0])
        epoch_Total_loss.update(loss_G.item(), imgs_lr.shape[0])
        epoch_content_loss.update(loss_content.item(), imgs_lr.shape[0])
        epoch_G_loss.update(loss_GAN.item(), imgs_lr.shape[0])
        epoch_pixel_loss.update(loss_pixel.item(), imgs_lr.shape[0])

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
            imgs_lr = nn.functional.interpolate(imgs_lr, scale_factor=2)
            img_grid = datasets.dataset_downstream.denormalize(torch.cat((imgs_lr, imgs_hr, gen_hr), -1))
            save_image(img_grid, "images/training/%d.png" % batches_done, nrow=1, normalize=False)

    torch.save(generator.state_dict(), "saved_models/generator_%d.pth" % epoch)
    torch.save(discriminator.state_dict(), "saved_models/discriminator_%d.pth" % epoch)

    temp_log_train = OrderedDict({'Discriminator loss': epoch_D_loss.avg, 'Total loss': epoch_Total_loss.avg,
                                  'Content loss': epoch_content_loss.avg,
                                  'GAN loss': epoch_G_loss.avg, 'Pixel loss': epoch_pixel_loss.avg})
    log_train.update(epoch, temp_log_train)

    generator.eval()
    epoch_val_loss = metrics.LossAverage()
    epoch_val_ssim = metrics.LossAverage()
    epoch_val_psnr = metrics.LossAverage()

    os.makedirs("images/validation/%d" % epoch, exist_ok=True)

    with torch.no_grad():
        for i, imgs in enumerate(val_dataloader):
            # Configure model input
            imgs_lr = imgs["lr"]
            imgs_lr_m1 = imgs["lr-1"]
            imgs_lr_p1 = imgs["lr+1"]

            imgs_lr, imgs_lr_p1, imgs_lr_m1 = imgs_lr.float().to(device), imgs_lr_p1.float().to(
                device), imgs_lr_m1.float().to(device)

            imgs_hr = imgs["hr"]
            imgs_hr = imgs_hr.float().to(device)

            with torch.no_grad():
                refined_slices = deformation_extraction(deformation_extractor, torch.cat([imgs_lr] * 2, 0),
                                                        torch.cat((imgs_lr_m1, imgs_lr_p1), 0))
                refined_last_slice, refined_next_slice = refined_slices[:imgs_lr.shape[0], ...], refined_slices[
                                                                                                 imgs_lr.shape[0]:, ...]
                hf_features = hf_feature_extractor(imgs_lr.detach().clone())
                gen_sr = generator(imgs_lr, refined_last_slice.detach(), refined_next_slice.detach(), hf_features)
            loss_pixel = criterion_pixel(gen_sr, imgs_hr)

            epoch_val_loss.update(loss_pixel.item(), imgs_hr.size(0))

            imgs_lr = nn.functional.interpolate(imgs_lr, scale_factor=2)
            img_grid = datasets.dataset_downstream.denormalize(
                torch.cat((imgs_lr, imgs_hr, gen_sr), -1))
            save_image(img_grid, "images/validation/%d/%d.png" % (epoch, i), nrow=1, normalize=False)

            imgs_hr = datasets.dataset_downstream.denormalize(imgs_hr.detach())
            imgs_sr = datasets.dataset_downstream.denormalize(gen_sr.detach())

            imgs_sr = imgs_sr.cpu().numpy() * 255
            imgs_hr = imgs_hr.cpu().numpy() * 255
            for ii in range(imgs_sr.shape[0]):
                imgs_sr_ = imgs_sr[ii, 0, :, :]
                imgs_hr_ = imgs_hr[ii, 0, :, :]
                epoch_val_ssim.update(metrics.ssim(imgs_sr_, imgs_hr_), 1)
                epoch_val_psnr.update(metrics.calculate_psnr(imgs_sr_, imgs_hr_), 1)
        val_log = OrderedDict({'Val Loss': epoch_val_loss.avg, 'Val SSIM': epoch_val_ssim.avg,
                               'Val PSNR': epoch_val_psnr.avg})
        log_val.update(epoch, val_log)

    epoch_test_loss = metrics.LossAverage()
    epoch_test_ssim = metrics.LossAverage()
    epoch_test_psnr = metrics.LossAverage()

    os.makedirs("images/testing/%d" % epoch, exist_ok=True)

    with torch.no_grad():
        for i, imgs in enumerate(test_dataloader):
            # Configure model input
            imgs_lr = imgs["lr"]
            imgs_lr_m1 = imgs["lr-1"]
            imgs_lr_p1 = imgs["lr+1"]
            name = imgs["name"]
            imgs_lr, imgs_lr_p1, imgs_lr_m1 = imgs_lr.float().to(device), imgs_lr_p1.float().to(
                device), imgs_lr_m1.float().to(device)

            imgs_hr = imgs["hr"]
            imgs_hr = imgs_hr.float().to(device)

            with torch.no_grad():
                refined_slices = deformation_extraction(deformation_extractor, torch.cat([imgs_lr] * 2, 0),
                                                        torch.cat((imgs_lr_m1, imgs_lr_p1), 0))
                refined_last_slice, refined_next_slice = refined_slices[:imgs_lr.shape[0], ...], refined_slices[
                                                                                                 imgs_lr.shape[0]:, ...]
                hf_features = hf_feature_extractor(imgs_lr.detach().clone())
                imgs_sr = generator(imgs_lr, refined_last_slice.detach(), refined_next_slice.detach(), hf_features)
            loss_pixel = criterion_pixel(imgs_sr, imgs_hr)

            epoch_test_loss.update(loss_pixel.item(), imgs_hr.size(0))

            imgs_LR = nn.functional.interpolate(imgs_lr, scale_factor=2)
            imgs_LR = datasets.dataset_downstream.denormalize(imgs_LR.detach())
            imgs_SR = datasets.dataset_downstream.denormalize(imgs_sr.detach())
            imgs_HR = datasets.dataset_downstream.denormalize(imgs_hr.detach())

            save_image(imgs_LR, "images/testing/%d/" % epoch + name[0][:-4] + "_LR.png", nrow=1, normalize=False)
            save_image(imgs_SR, "images/testing/%d/" % epoch + name[0][:-4] + "_SR.png", nrow=1, normalize=False)
            save_image(imgs_HR, "images/testing/%d/" % epoch + name[0][:-4] + "_HR.png", nrow=1, normalize=False)

            imgs_SR = imgs_SR.cpu().numpy() * 255
            imgs_HR = imgs_HR.cpu().numpy() * 255

            for ii in range(imgs_sr.shape[0]):
                imgs_sr_ = imgs_SR[ii, 0, :, :]
                imgs_hr_ = imgs_HR[ii, 0, :, :]
                epoch_test_ssim.update(metrics.ssim(imgs_sr_, imgs_hr_), 1)
                epoch_test_psnr.update(metrics.calculate_psnr(imgs_sr_, imgs_hr_), 1)
        val_log = OrderedDict({'Test Loss': epoch_test_loss.avg, 'Test SSIM': epoch_val_ssim.avg,
                               'Test PSNR': epoch_test_psnr.avg})
        log_test.update(epoch, val_log)