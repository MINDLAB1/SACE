import os
from torch.utils.data import Dataset as dataset_torch
import numpy as np
import random
from PIL import Image, ImageOps
import torch
import torchvision.transforms as transforms

mean = np.array([0.5,])
std = np.array([0.5,])


def denormalize(tensors):
    """ Denormalizes image tensors using mean and std """
    for c in range(1):
        tensors[:, c].mul_(std[c]).add_(mean[c])
    return torch.clamp(tensors, 0, 255)


def _make_image_namelist(dir):
    labeled_img_path = []
    labeled_namelist = []

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if fname.endswith('png'):
                item_path = os.path.join(root, fname)
                labeled_namelist.append(fname)
                labeled_img_path.append(item_path)

    return labeled_namelist, labeled_img_path


class data_set(dataset_torch):
    def __init__(self, root, mode='train'):
        self.root = root
        assert mode in ['train', 'val', 'testing']
        self.mode = mode
        self.img_names, self.img_paths = _make_image_namelist(os.path.join(self.root, self.mode+'_3t'))

        self.epi = 0
        self.img_num = len(self.img_names)

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        path_img_3t = self.img_paths[index]
        case_name = self.img_names[index]

        img_3t = Image.open(path_img_3t)
        img_3t = self.transform(img_3t)

        case_name_cp = case_name[:-4]
        case_name_split = case_name_cp.split('_')
        slice_num = case_name_split[len(case_name_split) - 1]
        patient_name = case_name_cp.replace(slice_num, '')
        slice_num = int(slice_num)
        path_img_3t_p1 = path_img_3t.replace(case_name, patient_name + str(slice_num + 1).zfill(3) + '.png')
        path_img_3t_m1 = path_img_3t.replace(case_name, patient_name + str(slice_num - 1).zfill(3) + '.png')

        if os.path.exists(path_img_3t_m1):
            img_3t_m1 = Image.open(path_img_3t_m1)
            img_3t_m1 = self.transform(img_3t_m1)
        else:
            img_3t_m1 = img_3t

        if os.path.exists(path_img_3t_p1):
            img_3t_p1 = Image.open(path_img_3t_p1)
            img_3t_p1 = self.transform(img_3t_p1)
        else:
            img_3t_p1 = img_3t

        path_img_7t = path_img_3t.replace('3t', '7t')
        img_7t = Image.open(path_img_7t)
        img_7t = img_7t.resize([384, 384])
        img_7t = self.transform(img_7t)

        if os.path.exists(path_img_3t.replace('3t', 'mask')):
            mask = Image.open(path_img_3t.replace('3t', 'mask'))
            mask = mask.resize([384, 384], Image.NEAREST)
            mask = np.array(mask, np.float)
            mask = mask / 52 - 1
            mask = torch.from_numpy(mask)
            mask = torch.unsqueeze(mask, 0)
            mask = mask.type(torch.FloatTensor)
        else:
            mask = torch.zeros_like(img_7t)

        imgs = {"lr-1": img_3t_m1, "lr": img_3t, "lr+1": img_3t_p1, "hr": img_7t, "mask": mask, "name": case_name}
        return imgs
