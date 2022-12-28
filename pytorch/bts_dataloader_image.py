# Copyright (C) 2019 Jin Han Lee
#
# This file is a part of BTS.
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.utils.data.distributed
from torchvision import transforms
from PIL import Image
import os
import random
from distributed_sampler_no_evenly_divisible import *
import cv2
from LFDepth.pytorch.func_pfm import *


def _is_pil_image(img):
    return isinstance(img, Image.Image)


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


def preprocessing_transforms(mode):
    return transforms.Compose([
        ToTensor(mode=mode)
    ])


class BtsDataLoader(object):
    def __init__(self, args, mode):
        if mode == 'train':
            self.training_samples = DataLoadPreprocess(args, mode, transform=preprocessing_transforms(mode))
            if args.distributed:
                self.train_sampler = torch.utils.data.distributed.DistributedSampler(self.training_samples)
            else:
                self.train_sampler = None

            self.data = DataLoader(self.training_samples, args.batch_size,
                                   shuffle=(self.train_sampler is None))

        elif mode == 'online_eval':
            self.testing_samples = DataLoadPreprocess(args, mode, transform=preprocessing_transforms(mode))
            if args.distributed:
                # self.eval_sampler = torch.utils.data.distributed.DistributedSampler(self.testing_samples, shuffle=False)
                self.eval_sampler = DistributedSamplerNoEvenlyDivisible(self.testing_samples, shuffle=False)
            else:
                self.eval_sampler = None
            self.data = DataLoader(self.testing_samples, 1,
                                   shuffle=True,
                                   num_workers=1,
                                   pin_memory=True,
                                   sampler=self.eval_sampler)

        elif mode == 'test':
            self.testing_samples = DataLoadPreprocess(args, mode, transform=preprocessing_transforms(mode))
            self.data = DataLoader(self.testing_samples, 1, shuffle=False, num_workers=1)

        else:
            print('mode should be one of \'train, test, online_eval\'. Got {}'.format(mode))


class DataLoadPreprocess(Dataset):
    def __init__(self, args, mode, transform=None, is_for_online_eval=False):
        self.args = args
        # if mode == 'online_eval':
        #     with open(args.filenames_file_eval, 'r') as f:
        #         self.filenames = f.readlines()
        # else:
        #     with open(args.filenames_file, 'r') as f:
        #         self.filenames = f.readlines()
        self.sence = ['antinous' , 'dishes', 'kitchen', 'museum', 'pillows', 'rosemary', 'tomb', 'town',
                 'boardgames', 'greek', 'medieval2', 'pens', 'platonic', 'table', 'tower', 'vinyl',
                      'backgammon', 'boxes', 'cotton', 'dino', 'dots', 'pyramids', 'sideboard', 'stripes']

        self.test_sence = ['backgammon', 'boxes', 'cotton', 'dino', 'dots', 'pyramids', 'sideboard', 'stripes']

        self.data_path = '/home/yzm/flow_sence_dataset/additional2/'
        self.test_path = '/home/yzm/flow_sence_dataset/additional2/'
        self.mode = mode
        self.transform = transform
        self.to_tensor = ToTensor
        self.is_for_online_eval = is_for_online_eval
        back_name = ['040', '036', '037', '038', '039', '041', '042', '043', '044', '004', '013', '022', '031',
                     '049', '058', '067', '076', '008', '016', '024', '032', '048', '056', '064', '072', '000',
                     '010', '020', '030', '050', '060', '070', '080']
        self.img_list = []
        self.depth_gt = []
        img = []
        test_img_list = []
        self.test_img = []
        self.test_depth_gt = []
        for j in range(len(self.sence)):
            depth_path = os.path.join(self.data_path, self.sence[j], 'gt_disp_lowres.pfm')
            for i in range(len(back_name)):
                image_path = os.path.join(self.data_path, self.sence[j], 'input_Cam'+back_name[i]+'.png')
                img.append(image_path)
            self.img_list.append(img)
            img = []
            depth_gt = read_pfm(depth_path)
            self.depth_gt.append(depth_gt)
        for k in range(len(self.test_sence)):
            test_depth_path = os.path.join(self.test_path, self.test_sence[k], 'gt_disp_lowres.pfm')
            for i in range(len(back_name)):
                img_test_path = os.path.join(self.test_path, self.test_sence[k], 'input_Cam' + back_name[i] + '.png')
                test_img_list.append(img_test_path)
            self.test_img.append(test_img_list)
            test_img_list = []
            test_depth_gt = read_pfm(test_depth_path)
            self.test_depth_gt.append(test_depth_gt)


        # self.img_list = np.concatenate(self.img_list, axis=2)

    def __getitem__(self, idx):
        # sample_path = self.filenames[idx]
        # focal = float(sample_path.split()[2])
        if self.mode == 'train' or self.mode == 'online_eval':
            data = []
            data_ = []
            depth_gt = self.depth_gt[idx]
            depth_gt = Image.fromarray(depth_gt)
            depth_gt = np.expand_dims(depth_gt, axis=0)
            for i in range(len(self.img_list[idx])):
                image = Image.open(self.img_list[idx][i])
                image = np.asarray(image, dtype=np.float32) / 255.0
                data.append(image)
                sample = {'image': data[i], 'depth': depth_gt}
                sample = self.transform(sample)
                data_.append(sample['image'])
            data_ = np.concatenate(data_, axis=0)
            sample = {'image': data_, 'depth': depth_gt}
        else:
            data = []
            data_ = []
            depth_gt = self.test_depth_gt[idx]
            depth_gt = Image.fromarray(depth_gt)
            depth_gt = np.expand_dims(depth_gt, axis=0)
            for i in range(len(self.test_img[idx])):
                image = Image.open(self.test_img[idx][i])
                image = np.asarray(image, dtype=np.float32) / 255.0
                data.append(image)
                sample = {'image': data[i], 'depth': depth_gt}
                sample = self.transform(sample)
                data_.append(sample['image'])
            data_ = np.concatenate(data_, axis=0)



            sample = {'image': data_, 'image_path': self.test_img[idx][0], 'depth':depth_gt}



        return sample

    def rotate_image(self, image, angle, flag=Image.BILINEAR):
        result = image.rotate(angle, resample=flag)
        return result

    def random_crop(self, img, depth, height, width):
        assert img.shape[0] >= height
        assert img.shape[1] >= width
        assert img.shape[0] == depth.shape[0]
        assert img.shape[1] == depth.shape[1]
        x = random.randint(0, img.shape[1] - width)
        y = random.randint(0, img.shape[0] - height)
        img = img[y:y + height, x:x + width, :]
        depth = depth[y:y + height, x:x + width, :]
        return img, depth

    def train_preprocess(self, image, depth_gt):
        # Random flipping
        do_flip = random.random()
        if do_flip > 0.5:
            image = (image[:, ::-1, :]).copy()
            depth_gt = (depth_gt[:, ::-1, :]).copy()

        # Random gamma, brightness, color augmentation
        do_augment = random.random()
        if do_augment > 0.5:
            image = self.augment_image(image)

        return image, depth_gt

    def augment_image(self, image):
        # gamma augmentation
        gamma = random.uniform(0.9, 1.1)
        image_aug = image ** gamma

        # brightness augmentation
        if self.args.dataset == 'nyu':
            brightness = random.uniform(0.75, 1.25)
        else:
            brightness = random.uniform(0.9, 1.1)
        image_aug = image_aug * brightness

        # color augmentation
        colors = np.random.uniform(0.9, 1.1, size=3)
        white = np.ones((image.shape[0], image.shape[1]))
        color_image = np.stack([white * colors[i] for i in range(3)], axis=2)
        image_aug *= color_image
        image_aug = np.clip(image_aug, 0, 1)

        return image_aug

    def __len__(self):
        if self.mode == 'train' or self.mode == 'online_eval':
            return len(self.img_list)
        else:
            return len(self.test_img)


class ToTensor(object):
    def __init__(self, mode):
        self.mode = mode
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __call__(self, sample):
        image = sample['image']
        image = self.to_tensor(image)
        image = self.normalize(image)

        # if self.mode == 'test':
        #     return {'image': image}

        depth = sample['depth']
        if self.mode == 'train':
            depth = self.to_tensor(depth)
            return {'image': image, 'depth': depth}
        else:
            # has_valid_depth = sample['has_valid_depth']
            return {'image': image, 'depth': depth}

    def to_tensor(self, pic):
        if not (_is_pil_image(pic) or _is_numpy_image(pic)):
            raise TypeError(
                'pic should be PIL Image or ndarray. Got {}'.format(type(pic)))

        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            return img

        # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)

        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.float()
        else:
            return img