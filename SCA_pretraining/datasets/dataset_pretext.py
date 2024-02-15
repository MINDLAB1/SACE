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


def _make_image_namelist(dir, mode):
    img_path = []
    namelist = []

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if fname.endswith('png'):
                namelist.append(fname)
                item_path = os.path.join(root, fname)
                img_path.append(item_path)

    if mode == 'training':
        _, data_namelist = select_nth_fifth_images_for_validation(namelist, 5)
        _, data_img_path = select_nth_fifth_images_for_validation(img_path, 5)
    else:
        data_namelist, _ = select_nth_fifth_images_for_validation(namelist, 5)
        data_img_path, _ = select_nth_fifth_images_for_validation(img_path, 5)

    return data_namelist, data_img_path


def select_nth_fifth_images_for_validation(image_list, n):
    if n < 1 or n > 5:
        raise ValueError("n must be between 1 and 5")
    unique_3d_images = set('_'.join(image.split('_')[:-1]) for image in image_list)
    sorted_unique_3d_images = sorted(unique_3d_images)
    fifth = len(sorted_unique_3d_images) // 5
    start_index = (n - 1) * fifth
    end_index = start_index + fifth if n < 5 else len(sorted_unique_3d_images)
    selected_images = set(sorted_unique_3d_images[start_index:end_index])
    remaining_images = set()
    for i in range(1, 6):
        if i != n:
            start = (i - 1) * fifth
            end = start + fifth if i < 5 else len(sorted_unique_3d_images)
            remaining_images.update(sorted_unique_3d_images[start:end])
    selected_filtered_images = [image for image in image_list if '_'.join(image.split('_')[:-1]) in selected_images]
    remaining_filtered_images = [image for image in image_list if '_'.join(image.split('_')[:-1]) in remaining_images]

    return selected_filtered_images, remaining_filtered_images


class data_set(dataset_torch):
    def __init__(self, root, stage='MAI', mode='train'):
        self.root = root
        assert mode in ['train', 'val']
        assert stage in ['MAI', 'SCA']
        self.mode = mode
        self.stage = stage
        self.img_names, self.img_paths = _make_image_namelist(os.path.join(self.root), mode)

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

        if self.stage == 'SCA':
            case_name_cp = case_name[:-4]
            case_name_split = case_name_cp.split('_')
            slice_num = case_name_split[len(case_name_split) - 1]
            patient_name = case_name_cp.replace(slice_num, '')
            slice_num = int(slice_num)

            path_img_3t_p1 = path_img_3t.replace(case_name, patient_name + str(slice_num + 1).zfill(3) + '.png')
            path_img_3t_m1 = path_img_3t.replace(case_name, patient_name + str(slice_num - 1).zfill(3) + '.png')

            if os.path.exists(path_img_3t_m1):
                img_3t_c = Image.open(path_img_3t_m1)
            else:
                img_3t_c = Image.open(path_img_3t_p1)

            img_3t_c = self.transform(img_3t_c)
            return img_3t, img_3t_c
        else:
            img_3t_masked = img_3t.clone()
            l_x = l_y = r_x = r_y = np.int_(0)
            i = 0
            while (torch.sum(img_3t_masked[:, l_x: r_x, l_y: r_y] > -1) / (96 * 96)) < 0.6 and i < 50:
                center_x = np.random.randint(48, 144, 1)[0]
                center_y = np.random.randint(48, 144, 1)[0]
                l_x = np.int_(center_x - 48)
                r_x = np.int_(center_x + 48)
                l_y = np.int_(center_y - 48)
                r_y = np.int_(center_y + 48)
                i += 1
            if i == 50:
                l_x = 96 - 48
                r_x = 96 + 48
                l_y = 96 - 48
                r_y = 96 + 48
            img_3t_masked[:, l_x: r_x, l_y: r_y] = -1.0
            return img_3t_masked, img_3t
