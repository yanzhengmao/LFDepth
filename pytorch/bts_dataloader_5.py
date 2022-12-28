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

##epi  dataloader


import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.utils.data.distributed
from torchvision import transforms
import imageio
from PIL import Image
import os
import random
from distributed_sampler_no_evenly_divisible import *
import cv2
from func_pfm import *
import torch.nn.functional as F
from PIL import Image
import skimage

def _is_pil_image(img):
    return isinstance(img, Image.Image)


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


def preprocessing_transforms(mode):
    return transforms.Compose([
        ToTensor(mode=mode)
    ])


class BtsDataLoader(object):
    def __init__(self, args, mode, input_size, scene_name):
        if mode == 'train':
            # self.training_samples = DataLoadPreprocess(args, mode, input_size, preprocessing_transforms(mode))
            self.training_samples = DataLoadPreprocess(args, mode, input_size, '')
            if args.distributed:
                self.train_sampler = torch.utils.data.distributed.DistributedSampler(self.training_samples)
            else:
                self.train_sampler = None

            self.data = DataLoader(self.training_samples, args.batch_size,
                                   shuffle=(self.train_sampler is None),
                                   num_workers=args.num_threads,
                                   pin_memory=True,
                                   sampler=self.train_sampler)



        elif mode == 'test':
            self.testing_samples = DataLoadPreprocess(args, mode, input_size, scene_name, transform=preprocessing_transforms(mode))
            # self.data = DataLoader(self.testing_samples, 984, shuffle=False)
            
            if args.distributed:
                self.eval_sampler = torch.utils.data.distributed.DistributedSampler(self.testing_samples, shuffle=False)
                # self.eval_sampler = DistributedSamplerNoEvenlyDivisible(self.testing_samples, shuffle=False)
            else:
                self.eval_sampler = None
            self.data = DataLoader(self.testing_samples, 984,
                                   shuffle=False,
                                   num_workers=1,
                                   pin_memory=True)

        else:
            print('mode should be one of \'train, test, online_eval\'. Got {}'.format(mode))


class DataLoadPreprocess(Dataset):
    def __init__(self, args, mode, input_size, scene_name, transform=None, is_for_online_eval=False):
        self.args = args
        self.mode = mode
        self.input_size = input_size
        self.label_size = input_size
        self.Setting02_AngualrViews = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
        self.data_path = '/home/yzm/flow_sence_dataset/additional/'
        self.disp_offset = 4
        self.disp_precition = 0.035
        self.num_of_classes = 229
        self.test_sence = ['backgammon/epi', 'boxes/epi', 'cotton/epi', 'dino/epi', 'dots/epi', 'pyramids/epi', 'sideboard/epi', 'stripes/epi']
        # self.scene_name = ['backgammon', 'boxes', 'cotton', 'dino', 'dots', 'pyramids', 'sideboard', 'stripes']
        self.transform = transforms.Compose([transforms.ToTensor()])
        if self.mode == 'train':
            self.sence = ['antinous/epi' , 'dishes/epi', 'kitchen/epi', 'museum/epi', 'pillows/epi', 'rosemary/epi', 'tomb/epi', 'town/epi',
                 'boardgames/epi', 'greek/epi', 'medieval2/epi', 'pens/epi', 'platonic/epi', 'table/epi', 'tower/epi', 'vinyl/epi', 'tower/epi',
                 'pens/epi', 'platonic/epi', 'pillows/epi', 'tomb/epi']
            self.traindata_all, self.traindata_label = self.load_LFdata(self.sence)
        else:
            self.valdata_all, self.valdata_label = self.load_LFdata_test(scene_name)

    def __getitem__(self, idx):

        if self.mode == 'train':


            epi_patch_0   = self.traindata_all[idx][0]
            epi_patch_90  = self.traindata_all[idx][1]
            epi_patch_45  = self.traindata_all[idx][2]
            epi_patch_135 = self.traindata_all[idx][3]
            epi_label     = self.traindata_label[idx]

            (epi_patch_0, epi_patch_90, epi_patch_45, epi_patch_135) = self.data_augmentation_for_train(
                epi_patch_0, epi_patch_90, epi_patch_45, epi_patch_135)

            (epi_patch_0, epi_patch_90,
             epi_patch_45, epi_patch_135) = self.__gaussian_noise(epi_patch_0, epi_patch_90, epi_patch_45, epi_patch_135, add_prob=0.1, var_range=[0.00002, 0.0002], retain_center=True)

            (epi_patch_0, epi_patch_90,
             epi_patch_45, epi_patch_135,
             epi_label) = self.flip_data(epi_patch_0,epi_patch_90,epi_patch_45,epi_patch_135,epi_label)


            tra_annotations, tra_hot_annotation = self.gt_preprocess(epi_label, train_stage='train')
            tra_annotations = torch.from_numpy(tra_annotations)
            tra_hot_annotation = torch.from_numpy(tra_hot_annotation)
            epi_patch_0   = torch.from_numpy(epi_patch_0).permute(2, 0, 1)
            epi_patch_90  = torch.from_numpy(epi_patch_90).permute(2, 0, 1)
            epi_patch_45  = torch.from_numpy(epi_patch_45).permute(2, 0, 1)
            epi_patch_135 = torch.from_numpy(epi_patch_135).permute(2, 0, 1)
            # traindata_label_batchNxN = traindata_label_batchNxN[15:-15,15:-15,:]

            sample = {'traindata_batch_90d': epi_patch_90, 'traindata_batch_0d': epi_patch_0,
                      'traindata_batch_45d': epi_patch_45,
                      'traindata_batch_m45d': epi_patch_135, 'tra_annotations': tra_annotations, 'tra_hot_annotation' : tra_hot_annotation}

        elif self.mode == 'test':

            R = 0.299  ### 0,1,2,3 = R, G, B, Gray // 0.299 0.587 0.114
            G = 0.587
            B = 0.114

            epi_patch_0 = self.valdata_all[idx][0] / 255.0
            epi_patch_90 = self.valdata_all[idx][1] / 255.0
            epi_patch_45 = self.valdata_all[idx][2] / 255.0
            epi_patch_135 = self.valdata_all[idx][3] / 255.0
            epi_label = self.valdata_label[idx]

            epi_patch_0 = R * epi_patch_0[:,:,0] + G * epi_patch_0[:,:,1] + B * epi_patch_0[:,:,2]
            epi_patch_90 = R * epi_patch_90[:, :, 0] + G * epi_patch_90[:, :, 1] + B * epi_patch_90[:, :, 2]
            epi_patch_45 = R * epi_patch_45[:, :, 0] + G * epi_patch_45[:, :, 1] + B * epi_patch_45[:, :, 2]
            epi_patch_135 = R * epi_patch_135[:, :, 0] + G * epi_patch_135[:, :, 1] + B * epi_patch_135[:, :, 2]

            tra_annotations = self.gt_preprocess(epi_label, train_stage='test')
            tra_annotations = torch.from_numpy(tra_annotations)
            epi_patch_0 = torch.from_numpy(epi_patch_0).unsqueeze(0)
            epi_patch_90 = torch.from_numpy(epi_patch_90).unsqueeze(0)
            epi_patch_45 = torch.from_numpy(epi_patch_45).unsqueeze(0)
            epi_patch_135 = torch.from_numpy(epi_patch_135).unsqueeze(0)

            sample = {'traindata_batch_90d': epi_patch_90, 'traindata_batch_0d': epi_patch_0,
                      'traindata_batch_45d': epi_patch_45,
                      'traindata_batch_m45d': epi_patch_135, 'tra_annotations': tra_annotations}



        return sample

    def gt_preprocess(self, annotation, train_stage="train"):
        # float to class
        annotation = (annotation+self.disp_offset)/self.disp_precition

        ann_round = np.round(annotation).astype(np.int16)

        if train_stage != "train":
            ann_round = np.array(ann_round, dtype='float32')
            return ann_round

        ann_floor = np.floor(annotation).astype(np.int16)
        ann_ceil  =  np.ceil(annotation).astype(np.int16)

        floor_prob = (1-(annotation-ann_floor))
        ceil_prob  = (1-floor_prob)
        #tf.zeros()
        two_hot_labels = np.zeros(list(annotation.shape)+[self.num_of_classes], dtype='float32') # float32
        old_shape = two_hot_labels.shape
        #print((ann_floor==ann_floor).shape)
        two_hot_labels = two_hot_labels.reshape(-1,two_hot_labels.shape[-1])
        ann_floor = ann_floor.reshape(-1)  ; ann_ceil = ann_ceil.reshape(-1)
        floor_prob = floor_prob.reshape(-1); ceil_prob = ceil_prob.reshape(-1)

        #start = time.time()
        for i in range(two_hot_labels.shape[0]):
            two_hot_labels[i][ann_floor[i]] = floor_prob[i]
            two_hot_labels[i][ann_ceil[i]] += ceil_prob[i] # += !!!!!!!!! cos ann_floor may == ann_ceil

        two_hot_labels = two_hot_labels.reshape(old_shape)
        #end = time.time()
        #print("{:.2f} s".format(end - start))

        #two_hot_labels[:,:,ann_floor] = 1#floor_prob
        # maybe overlap, when floor == ceil, so use += instead of =
        #two_hot_labels[ann_ceil]  += ceil_prob
        ann_round = np.array(ann_round, dtype='float32')

        return ann_round, two_hot_labels


    def load_LFdata(self, dir_LFimages):

        input_imgs = []
        input_labels = []

        root_path = '/data/Dataset/additional2'
        label_name = 'valid_disp_map'
        epi_list = ['epi_0_patches', 'epi_90_patches', 'epi_45_patches', 'epi_135_patches']
        for dir_LFimage in dir_LFimages:
            print("loading " + dir_LFimage + " ...")
            path = os.path.join(root_path, dir_LFimage)
            imgs = [self.load_npy(path, p_name) for p_name in epi_list]

            lables = self.load_npy(("{}/{}".format(root_path, dir_LFimage)), label_name)

            # (4,512,512,9,13,3) or (4,768,768,9,13,3) to (512*512,4,9,13,3) or (768*768,4,9,13,3)
            # 3. TEMP_TEMP 4#
            # imgs = np.expand_dims(imgs, axis=-1)

            imgs = np.array(imgs).transpose((1, 2, 0, 3, 4, 5))
            imgs = imgs.reshape((-1,4,9,21,3))
            lables = lables.reshape(-1,)
            # imgs = imgs * 255

            # substract mean
            '''
            if self.mean_type != MEAN_TYPE.NO_MEAN.value:
                imgs = self.subtract_mean(imgs, self.mean_type)
            else:
                print("--- mean_type: NO MEAN ---")
            '''
            input_imgs.append(imgs)
            input_labels.append(lables)

        if len(input_imgs) > 0:
            input_imgs = np.concatenate(input_imgs, axis=0)
        else:
            input_imgs = None
        if len(input_labels) > 0:
            input_labels = np.concatenate(input_labels, axis=0)
        else:
            input_labels = None
        return input_imgs, input_labels

    def load_LFdata_test(self, dir_LFimage):

        input_imgs = []
        input_labels = []

        root_path = '/data/Dataset/additional2'
        label_name = 'valid_disp_map'
        epi_list = ['epi_0_patches', 'epi_90_patches', 'epi_45_patches', 'epi_135_patches']
        dir_LFimage += '/epi'
        print("loading " + dir_LFimage + " ...")
        path = os.path.join(root_path, dir_LFimage)
        imgs = [self.load_npy(path, p_name) for p_name in epi_list]
        lables = self.load_npy(("{}/{}".format(root_path, dir_LFimage)), label_name)
        # (4,512,512,9,13,3) or (4,768,768,9,13,3) to (512*512,4,9,13,3) or (768*768,4,9,13,3)
        # 3. TEMP_TEMP 4#
        # imgs = np.expand_dims(imgs, axis=-1)
        imgs = np.array(imgs).transpose((1, 2, 0, 3, 4, 5))
        imgs = imgs.reshape((-1,4,9,21,3))
        lables = lables.reshape(-1,)
        # imgs = imgs * 255
        # substract mean
        '''
        if self.mean_type != MEAN_TYPE.NO_MEAN.value:
            imgs = self.subtract_mean(imgs, self.mean_type)
        else:
            print("--- mean_type: NO MEAN ---")
        '''
        input_imgs.append(imgs)
        input_labels.append(lables)
        if len(input_imgs) > 0:
            input_imgs = np.concatenate(input_imgs, axis=0)
        else:
            input_imgs = None
        if len(input_labels) > 0:
            input_labels = np.concatenate(input_labels, axis=0)
        else:
            input_labels = None
        return input_imgs, input_labels

    def load_npy(self, f_dir, f_name_split):
        # f_name_split doesn't include file extension, i.e. .npy
        data = np.load("{}/{}.npy".format(f_dir, f_name_split))

        return data

    def rotate_image(self, image, angle, flag=Image.BILINEAR):
        result = image.rotate(angle, resample=flag)
        return result

    def data_augmentation_for_train(self, epi_0, epi_90, epi_45, epi_135):
        """
            For Data augmentation
            (rotation, transpose and gamma)

        """
        rand_3color = 0.05 + np.random.rand(3)
        rand_3color = rand_3color / np.sum(rand_3color)
        R = rand_3color[0]
        G = rand_3color[1]
        B = rand_3color[2]

        epi_0 = np.expand_dims((R * epi_0[:,:,0] + G * epi_0[:,:,1] + B * epi_0[:,:,2]),axis=-1)
        epi_90 = np.expand_dims((R * epi_90[:, :, 0] + G * epi_90[:, :, 1] + B * epi_90[:, :, 2]), axis=-1)
        epi_45 = np.expand_dims((R * epi_45[:, :, 0] + G * epi_45[:, :, 1] + B * epi_45[:, :, 2]), axis=-1)
        epi_135 = np.expand_dims((R * epi_135[:, :, 0] + G * epi_135[:, :, 1] + B * epi_135[:, :, 2]), axis=-1)


        gray_rand = 0.4 * np.random.rand() + 0.8

        epi_0 = pow(epi_0, gray_rand)
        epi_90 = pow(epi_90, gray_rand)
        epi_45 = pow(epi_45, gray_rand)
        epi_135 = pow(epi_135, gray_rand)

        epi_0 = epi_0 / 255.0
        epi_90 = epi_90 / 255.0
        epi_45 = epi_45 / 255.0
        epi_135 = epi_135 / 255.0


        return epi_0, epi_90, epi_45, epi_135

    def flip_data(self, epi_0, epi_90, epi_45, epi_135, lbs, flip_prob=0.5):
        epi_0_copy = epi_0.copy()
        epi_90_copy = epi_90.copy()
        epi_45_copy = epi_45.copy()
        epi_135_copy = epi_135.copy()

        lbs_copy = lbs.copy()
        hei_dim = -3
        #flip_imgs = np.flip(imgs, axis=wid_dim)

        # flip
        flip_mask = np.random.choice([True, False], p=[flip_prob, 1-flip_prob])
        flip_epi_0 = np.flip(epi_0_copy[flip_mask], axis=hei_dim)
        flip_epi_90 = np.flip(epi_90_copy[flip_mask], axis=hei_dim)
        flip_epi_45 = np.flip(epi_45_copy[flip_mask], axis=hei_dim)
        flip_epi_135 = np.flip(epi_135_copy[flip_mask], axis=hei_dim)

        if flip_mask == True:
            lbs_copy = -lbs_copy

        # assert
        epi_0_copy[flip_mask] = flip_epi_0
        epi_90_copy[flip_mask] = flip_epi_90
        epi_45_copy[flip_mask] = flip_epi_45
        epi_135_copy[flip_mask] = flip_epi_135
        # lbs_copy[flip_mask] = flip_lbs

        #imgs = np.concatenate((imgs, flip_imgs), axis=0) # vstack
        '''
        if lbs is not None:
            flip_lbs = -lbs
            #lbs = np.concatenate((lbs, flip_lbs), axis=0)
        '''


        return epi_0_copy,epi_90_copy,epi_45_copy,epi_135_copy, lbs_copy

    def __gaussian_noise(self,  epi_0, epi_90, epi_45, epi_135, add_prob=0.1, var_range=[0, 0.1], retain_center=False):
        # X_imgs must on range [0, 1], floating
        epi_0_copy = epi_0.copy()
        epi_90_copy = epi_90.copy()
        epi_45_copy = epi_45.copy()
        epi_135_copy = epi_135.copy()# .astype(np.float32)
        row, col, _ = epi_0_copy.shape


        # Add Salt noise
        if np.random.choice([0, 1], 1, p=[1 - add_prob, add_prob]):
            var = random.uniform(var_range[0], var_range[1])
            # X_imgs_copy must on range [0, 1] if float, return float gs_img on range [0, 1] if X_imgs_copy is uint
            epi_0_copy = skimage.util.random_noise(epi_0_copy, mode='gaussian', seed=None, clip=True,
                                                       var=var)
            epi_90_copy = skimage.util.random_noise(epi_90_copy, mode='gaussian', seed=None, clip=True,
                                                   var=var)
            epi_45_copy = skimage.util.random_noise(epi_45_copy, mode='gaussian', seed=None, clip=True,
                                                   var=var)
            epi_135_copy = skimage.util.random_noise(epi_135_copy, mode='gaussian', seed=None, clip=True,
                                                   var=var)
        if retain_center:
            epi_0_copy[row // 2, col // 2] = epi_0[row // 2, col // 2]
            epi_90_copy[row // 2, col // 2] = epi_90[row // 2, col // 2]
            epi_45_copy[row // 2, col // 2] = epi_45[row // 2, col // 2]
            epi_135_copy[row // 2, col // 2] = epi_135[row // 2, col // 2]

        return epi_0_copy,epi_90_copy,epi_45_copy,epi_135_copy

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
        if self.mode == 'train':
            return len(self.traindata_all)
        elif self.mode == 'online_eval':
            return len(self.traindata_all)
        else:
            return len(self.valdata_all)


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