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
import imageio
from PIL import Image
import os
import random
from distributed_sampler_no_evenly_divisible import *
# import cv2
from func_pfm import *
import torch.nn.functional as F


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
                                   shuffle=(self.train_sampler is None),
                                   num_workers=args.num_threads,
                                   pin_memory=True,
                                   sampler=self.train_sampler)

        elif mode == 'online_eval':
            self.testing_samples = DataLoadPreprocess(args, mode, transform=preprocessing_transforms(mode))
            if args.distributed:
                # self.eval_sampler = torch.utils.data.distributed.DistributedSampler(self.testing_samples, shuffle=False)
                self.eval_sampler = DistributedSamplerNoEvenlyDivisible(self.testing_samples, shuffle=False)
            else:
                self.eval_sampler = None
            self.data = DataLoader(self.testing_samples, 1,
                                   shuffle=False,
                                   num_workers=1,
                                   pin_memory=True)

        elif mode == 'test':
            self.testing_samples = DataLoadPreprocess(args, mode, transform=preprocessing_transforms(mode))
            self.data = DataLoader(self.testing_samples, 1, shuffle=False, num_workers=1)

        else:
            print('mode should be one of \'train, test, online_eval\'. Got {}'.format(mode))


class DataLoadPreprocess(Dataset):
    def __init__(self, args, mode, transform=None, is_for_online_eval=False):
        self.args = args
        self.mode = mode
        self.input_size = args.input_height
        self.label_size = self.input_size
        self.Setting02_AngualrViews = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
        self.data_path = '/data/Dataset/full_data/additional'
        self.valdata_list = []
        self.val_label_list = []
        self.sence = ['antinous' , 'dishes', 'kitchen', 'museum', 'pillows', 'rosemary', 'tomb', 'town',
             'boardgames', 'greek', 'medieval2', 'pens', 'platonic', 'table', 'tower', 'vinyl']
        self.test_sence = ['backgammon', 'boxes', 'cotton', 'dino', 'dots', 'pyramids', 'sideboard', 'stripes']
        if self.mode == 'train':
            self.traindata_all, self.traindata_label = self.load_LFdata(self.sence)
            self.traindata_90d, self.traindata_0d, self.traindata_45d, self.traindata_m45d, _ = self.generate_traindata512(self.traindata_all, self.traindata_label, self.Setting02_AngualrViews)
            bool_mask_img4 = imageio.imread('/data/yanzhengmao/projects/LFDepth/pytorch/dataset/additional_invalid_area/kitchen/input_Cam040_invalid_ver2.png')
            bool_mask_img6 = imageio.imread('/data/yanzhengmao/projects/LFDepth/pytorch/dataset/additional_invalid_area/museum/input_Cam040_invalid_ver2.png')
            bool_mask_img15 = imageio.imread('/data/yanzhengmao/projects/LFDepth/pytorch/dataset/additional_invalid_area/vinyl/input_Cam040_invalid_ver2.png')

            self.boolmask_img4 = 1.0 * bool_mask_img4[:, :, 3] > 0
            self.boolmask_img6 = 1.0 * bool_mask_img6[:, :, 3] > 0
            self.boolmask_img15 = 1.0 * bool_mask_img15[:, :, 3] > 0
        elif self.mode == 'online_eval':
            self.valdata_all, self.valdata_label = self.load_LFdata(self.test_sence)
            self.valdata_90d, self.valdata_0d, self.valdata_45d, self.valdata_m45d, _ = self.generate_traindata512(self.valdata_all, self.valdata_label, self.Setting02_AngualrViews)
            

    def __getitem__(self, idx):

        if self.mode == 'train':
            (traindata_batch_90d, traindata_batch_0d,
             traindata_batch_45d, traindata_batch_m45d, 
             traindata_label_batchNxN)= self.generate_traindata_for_train(idx, self.traindata_all,self.traindata_label,
                                                                     self.input_size, self.label_size, 
                                                                     self.Setting02_AngualrViews,
                                                                     self.boolmask_img4, self.boolmask_img6,self.boolmask_img15)  

            (traindata_batch_90d, traindata_batch_0d,
             traindata_batch_45d,traindata_batch_m45d, 
             traindata_label_batchNxN) =  self.data_augmentation_for_train(traindata_batch_90d, 
                                                                      traindata_batch_0d,
                                                                      traindata_batch_45d,
                                                                      traindata_batch_m45d, 
                                                                      traindata_label_batchNxN) 

            train_data = np.concatenate([traindata_batch_90d, traindata_batch_0d, traindata_batch_45d, traindata_batch_m45d], axis=-1)
            train_data = torch.from_numpy(train_data).permute(2,0,1)
            train_label=torch.from_numpy(traindata_label_batchNxN)


            sample = {'image': train_data, 'depth': train_label}

        elif self.mode == 'online_eval':
            valdata_list = []
            for idx_ in range(0, 512, 128):
                for idy_ in range(0, 512, 128):
                    (valdata_batch_90d, valdata_batch_0d,
                    valdata_batch_45d,valdata_batch_m45d, 
                    valdata_label_batchNxN) = self.generate_valdata_for_eval(idx, self.valdata_all,
                                                                            self.valdata_label,
                                                                            self.input_size,
                                                                            self.label_size,
                                                                            idx_, idy_,
                                                                            self.Setting02_AngualrViews)
                    valdata = np.concatenate([valdata_batch_90d, valdata_batch_0d, valdata_batch_45d, valdata_batch_m45d], axis=-1)
                    valdata_list.append(valdata)
                    
            # valdata, val_label = self.generate_traindata_for_train(idx, self.valdata_all, self.valdata_label,
            #                                                           self.input_size, self.label_size,
            #                                                           self.Setting02_AngualrViews)


            valdata = np.stack(valdata_list, 0)
            # val_label = np.stack(val_label_list, 0)
            # valdata = torch.from_numpy(valdata.copy()).permute(2,3,0,1).view(-1, self.input_size, self.input_size)
            valdata = torch.from_numpy(valdata.copy()).permute(0,3,1,2)

            sample = {'image': valdata, 'depth': self.valdata_label[idx],  'valdata_label': self.valdata_label[idx], 'scene':self.test_sence[idx]

                      }


        return sample

    
    def generate_traindata_for_train(self, idx, traindata_all,traindata_label,input_size,label_size,Setting02_AngualrViews,boolmask_img4,boolmask_img6,boolmask_img15):
    
        """
        input: traindata_all   (16x512x512x9x9x3) uint8
                traindata_label (16x512x512x9x9)   float32
                input_size 23~   int
                label_size 1~    int
                batch_size 16    int
                Setting02_AngualrViews [0,1,2,3,4,5,6,7,8] for 9x9 
                boolmask_img4 (512x512)  bool // reflection mask for images[4]
                boolmask_img6 (512x512)  bool // reflection mask for images[6]
                boolmask_img15 (512x512) bool // reflection mask for images[15]
        Generate traindata using LF image and disparity map
        by randomly chosen variables.
        1.  gray image: random R,G,B --> R*img_R + G*img_G + B*imgB 
        2.  patch-wise learning: random x,y  --> LFimage[x:x+size1,y:y+size2]
        3.  scale augmentation: scale 1,2,3  --> ex> LFimage[x:x+2*size1:2,y:y+2*size2:2]
        
        
        output: traindata_batch_90d   (batch_size x input_size x input_size x len(Setting02_AngualrViews)) float32        
                traindata_batch_0d    (batch_size x input_size x input_size x len(Setting02_AngualrViews)) float32  
                traindata_batch_45d   (batch_size x input_size x input_size x len(Setting02_AngualrViews)) float32
                traindata_batch_m45d  (batch_size x input_size x input_size x len(Setting02_AngualrViews)) float32
                traindata_batch_label (batch_size x label_size x label_size )                   float32
        """
        
        
    
        """ initialize image_stack & label """ 
        traindata_batch_90d=np.zeros((input_size,input_size,len(Setting02_AngualrViews)),dtype=np.float32)
        traindata_batch_0d=np.zeros((input_size,input_size,len(Setting02_AngualrViews)),dtype=np.float32)
        traindata_batch_45d=np.zeros((input_size,input_size,len(Setting02_AngualrViews)),dtype=np.float32)
        traindata_batch_m45d=np.zeros((input_size,input_size,len(Setting02_AngualrViews)),dtype=np.float32)        
        
        traindata_batch_label=np.zeros((label_size,label_size))
        
        
        
        
        """ inital variable """
        start1=Setting02_AngualrViews[0]
        end1=Setting02_AngualrViews[-1]    
        crop_half1=int(0.5*(input_size-label_size))
        
        
    
        
        """ Generate image stacks"""
        sum_diff=0
        valid=0

        # while( sum_diff<0.01*input_size*input_size  or  valid<1 ): 
            
        """//Variable for gray conversion//"""
        rand_3color=0.05+np.random.rand(3)
        rand_3color=rand_3color/np.sum(rand_3color) 
        R=rand_3color[0]
        G=rand_3color[1]
        B=rand_3color[2]
        
        
        
        """
            We use totally 16 LF images,(0 to 15) 
            Since some images(4,6,15) have a reflection region, 
            We decrease frequency of occurrence for them. 
            Details in our epinet paper.
        """
        
        image_id=idx



        """
            //Shift augmentation for 7x7, 5x5 viewpoints,.. //
            Details in our epinet paper.
        """
        if(len(Setting02_AngualrViews)==7):
            ix_rd = np.random.randint(0,3)-1
            iy_rd = np.random.randint(0,3)-1
        if(len(Setting02_AngualrViews)==9):
            ix_rd = 0
            iy_rd = 0
            

        
        scale = 1
            
        idx_start = np.random.randint(0,512-scale*input_size)
        idy_start = np.random.randint(0,512-scale*input_size)    
        valid=1           
        """
            boolmask: reflection masks for images(4,6,15)
        """
        # if(image_id==4 or 6 or 15):
        #     if(image_id==4):
        #         a_tmp=boolmask_img4
        #     if(image_id==6):
        #         a_tmp=boolmask_img6    
        #     if(image_id==15):
        #         a_tmp=boolmask_img15                            
        #         if( np.sum(a_tmp[idx_start+scale*crop_half1: idx_start+scale*crop_half1+scale*label_size:scale,
        #                         idy_start+scale*crop_half1: idy_start+scale*crop_half1+scale*label_size:scale])>0
        #             or np.sum(a_tmp[idx_start: idx_start+scale*input_size:scale, 
        #                             idy_start: idy_start+scale*input_size:scale])>0 ):
        #             valid=0
                
        if(valid>0):      
            seq0to8=np.array(Setting02_AngualrViews)+ix_rd    
            seq8to0=np.array(Setting02_AngualrViews[::-1])+iy_rd
            
            image_center=(1/255)*np.squeeze(R*traindata_all[image_id, idx_start: idx_start+scale*input_size:scale, idy_start: idy_start+scale*input_size:scale, 4+ix_rd, 4+iy_rd,0].astype('float32')+
                                            G*traindata_all[image_id, idx_start: idx_start+scale*input_size:scale, idy_start: idy_start+scale*input_size:scale, 4+ix_rd, 4+iy_rd,1].astype('float32')+
                                            B*traindata_all[image_id, idx_start: idx_start+scale*input_size:scale, idy_start: idy_start+scale*input_size:scale, 4+ix_rd, 4+iy_rd,2].astype('float32'))
            sum_diff=np.sum(np.abs(image_center-np.squeeze(image_center[int(0.5*input_size),int(0.5*input_size)])))

            '''
            Four image stacks are selected from LF full(512x512) images.
            gray-scaled, cropped and scaled  
            
            traindata_batch_0d  <-- RGBtoGray( traindata_all[random_index, scaled_input_size, scaled_input_size, 4(center),    0to8    ] )
            traindata_batch_90d   <-- RGBtoGray( traindata_all[random_index, scaled_input_size, scaled_input_size, 8to0,       4(center) ] )
            traindata_batch_45d  <-- RGBtoGray( traindata_all[random_index, scaled_input_size, scaled_input_size, 8to0,         0to8    ] )
            traindata_batch_m45d <-- RGBtoGray( traindata_all[random_index, scaled_input_size, scaled_input_size, 0to8,         0to8    ] )      
            '''
            traindata_batch_0d[:,:,:]=np.squeeze(R*traindata_all[image_id:image_id+1, idx_start: idx_start+scale*input_size:scale, idy_start: idy_start+scale*input_size:scale, 4+ix_rd, seq0to8.tolist(),0].astype('float32')+
                                                    G*traindata_all[image_id:image_id+1, idx_start: idx_start+scale*input_size:scale, idy_start: idy_start+scale*input_size:scale, 4+ix_rd, seq0to8.tolist(),1].astype('float32')+
                                                    B*traindata_all[image_id:image_id+1, idx_start: idx_start+scale*input_size:scale, idy_start: idy_start+scale*input_size:scale, 4+ix_rd, seq0to8.tolist(),2].astype('float32'))
            
            traindata_batch_90d[:,:,:]=np.squeeze(R*traindata_all[image_id:image_id+1, idx_start: idx_start+scale*input_size:scale, idy_start: idy_start+scale*input_size:scale, seq8to0.tolist(), 4+iy_rd,0].astype('float32')+
                                                    G*traindata_all[image_id:image_id+1, idx_start: idx_start+scale*input_size:scale, idy_start: idy_start+scale*input_size:scale, seq8to0.tolist(), 4+iy_rd,1].astype('float32')+
                                                    B*traindata_all[image_id:image_id+1, idx_start: idx_start+scale*input_size:scale, idy_start: idy_start+scale*input_size:scale, seq8to0.tolist(), 4+iy_rd,2].astype('float32'))
            for kkk in range(start1,end1+1):
                
                traindata_batch_45d[:,:,kkk-start1]=np.squeeze(R*traindata_all[image_id:image_id+1, idx_start: idx_start+scale*input_size:scale, idy_start: idy_start+scale*input_size:scale, (8)-kkk+ix_rd, kkk+iy_rd,0].astype('float32')+
                                                                G*traindata_all[image_id:image_id+1, idx_start: idx_start+scale*input_size:scale, idy_start: idy_start+scale*input_size:scale, (8)-kkk+ix_rd, kkk+iy_rd,1].astype('float32')+
                                                                B*traindata_all[image_id:image_id+1, idx_start: idx_start+scale*input_size:scale, idy_start: idy_start+scale*input_size:scale, (8)-kkk+ix_rd, kkk+iy_rd,2].astype('float32'))
                            
                traindata_batch_m45d[:,:,kkk-start1]=np.squeeze(R*traindata_all[image_id:image_id+1, idx_start: idx_start+scale*input_size:scale, idy_start: idy_start+scale*input_size:scale, kkk+ix_rd, kkk+iy_rd,0].astype('float32')+
                                                                    G*traindata_all[image_id:image_id+1, idx_start: idx_start+scale*input_size:scale, idy_start: idy_start+scale*input_size:scale, kkk+ix_rd, kkk+iy_rd,1].astype('float32')+
                                                                    B*traindata_all[image_id:image_id+1, idx_start: idx_start+scale*input_size:scale, idy_start: idy_start+scale*input_size:scale, kkk+ix_rd, kkk+iy_rd,2].astype('float32'))
            '''
            traindata_batch_label  <-- scale_factor*traindata_label[random_index, scaled_label_size, scaled_label_size] 
            '''                
            if(len(traindata_label.shape)==5):
                traindata_batch_label[:,:]=(1.0/scale)*traindata_label[image_id, idx_start+scale*crop_half1: idx_start+scale*crop_half1+scale*label_size:scale,
                                                                            idy_start+scale*crop_half1: idy_start+scale*crop_half1+scale*label_size:scale,4+ix_rd,4+iy_rd]
            else:
                traindata_batch_label[:,:]=(1.0/scale)*traindata_label[image_id, idx_start+scale*crop_half1: idx_start+scale*crop_half1+scale*label_size:scale,
                                                                                idy_start+scale*crop_half1: idy_start+scale*crop_half1+scale*label_size:scale]
                                    
        traindata_batch_90d=np.float32((1/255)*traindata_batch_90d)
        traindata_batch_0d =np.float32((1/255)*traindata_batch_0d)
        traindata_batch_45d=np.float32((1/255)*traindata_batch_45d)
        traindata_batch_m45d=np.float32((1/255)*traindata_batch_m45d)
        
        return traindata_batch_90d,traindata_batch_0d,traindata_batch_45d,traindata_batch_m45d, traindata_batch_label  #,usage_check 

    def generate_valdata_for_eval(self, idx, traindata_all,traindata_label,input_size,label_size,x, y, Setting02_AngualrViews):
    
        """
        input: traindata_all   (16x512x512x9x9x3) uint8
                traindata_label (16x512x512x9x9)   float32
                input_size 23~   int
                label_size 1~    int
                batch_size 16    int
                Setting02_AngualrViews [0,1,2,3,4,5,6,7,8] for 9x9 
                boolmask_img4 (512x512)  bool // reflection mask for images[4]
                boolmask_img6 (512x512)  bool // reflection mask for images[6]
                boolmask_img15 (512x512) bool // reflection mask for images[15]
        Generate traindata using LF image and disparity map
        by randomly chosen variables.
        1.  gray image: random R,G,B --> R*img_R + G*img_G + B*imgB 
        2.  patch-wise learning: random x,y  --> LFimage[x:x+size1,y:y+size2]
        3.  scale augmentation: scale 1,2,3  --> ex> LFimage[x:x+2*size1:2,y:y+2*size2:2]
        
        
        output: traindata_batch_90d   (batch_size x input_size x input_size x len(Setting02_AngualrViews)) float32        
                traindata_batch_0d    (batch_size x input_size x input_size x len(Setting02_AngualrViews)) float32  
                traindata_batch_45d   (batch_size x input_size x input_size x len(Setting02_AngualrViews)) float32
                traindata_batch_m45d  (batch_size x input_size x input_size x len(Setting02_AngualrViews)) float32
                traindata_batch_label (batch_size x label_size x label_size )                   float32
        """
        
        
    
        """ initialize image_stack & label """ 
        traindata_batch_90d=np.zeros((input_size,input_size,len(Setting02_AngualrViews)),dtype=np.float32)
        traindata_batch_0d=np.zeros((input_size,input_size,len(Setting02_AngualrViews)),dtype=np.float32)
        traindata_batch_45d=np.zeros((input_size,input_size,len(Setting02_AngualrViews)),dtype=np.float32)
        traindata_batch_m45d=np.zeros((input_size,input_size,len(Setting02_AngualrViews)),dtype=np.float32)        
        
        traindata_batch_label=np.zeros((label_size,label_size))
        
        
        
        
        """ inital variable """
        start1=Setting02_AngualrViews[0]
        end1=Setting02_AngualrViews[-1]    
        
        
    
        
        """ Generate image stacks"""
        sum_diff=0
        valid=0

            
        """//Variable for gray conversion//"""
        rand_3color=0.05+np.random.rand(3)
        rand_3color=rand_3color/np.sum(rand_3color) 
        R=rand_3color[0]
        G=rand_3color[1]
        B=rand_3color[2]
        
        
        
        """
            We use totally 16 LF images,(0 to 15) 
            Since some images(4,6,15) have a reflection region, 
            We decrease frequency of occurrence for them. 
            Details in our epinet paper.
        """
        



        """
            //Shift augmentation for 7x7, 5x5 viewpoints,.. //
            Details in our epinet paper.
        """
        if(len(Setting02_AngualrViews)==7):
            ix_rd = np.random.randint(0,3)-1
            iy_rd = np.random.randint(0,3)-1
        if(len(Setting02_AngualrViews)==9):
            ix_rd = 0
            iy_rd = 0
            

        
        scale = 1
            
        idx_start = x
        idy_start = y   
        valid=1           
        image_id = idx
        if(valid>0):      
            seq0to8=np.array(Setting02_AngualrViews)+ix_rd    
            seq8to0=np.array(Setting02_AngualrViews[::-1])+iy_rd
            
            '''
            Four image stacks are selected from LF full(512x512) images.
            gray-scaled, cropped and scaled  
            
            traindata_batch_0d  <-- RGBtoGray( traindata_all[random_index, scaled_input_size, scaled_input_size, 4(center),    0to8    ] )
            traindata_batch_90d   <-- RGBtoGray( traindata_all[random_index, scaled_input_size, scaled_input_size, 8to0,       4(center) ] )
            traindata_batch_45d  <-- RGBtoGray( traindata_all[random_index, scaled_input_size, scaled_input_size, 8to0,         0to8    ] )
            traindata_batch_m45d <-- RGBtoGray( traindata_all[random_index, scaled_input_size, scaled_input_size, 0to8,         0to8    ] )      
            '''
            traindata_batch_0d[:,:,:]=np.squeeze(R*traindata_all[image_id:image_id+1, idx_start: idx_start+scale*input_size:scale, idy_start: idy_start+scale*input_size:scale, 4+ix_rd, seq0to8.tolist(),0].astype('float32')+
                                                    G*traindata_all[image_id:image_id+1, idx_start: idx_start+scale*input_size:scale, idy_start: idy_start+scale*input_size:scale, 4+ix_rd, seq0to8.tolist(),1].astype('float32')+
                                                    B*traindata_all[image_id:image_id+1, idx_start: idx_start+scale*input_size:scale, idy_start: idy_start+scale*input_size:scale, 4+ix_rd, seq0to8.tolist(),2].astype('float32'))
            
            traindata_batch_90d[:,:,:]=np.squeeze(R*traindata_all[image_id:image_id+1, idx_start: idx_start+scale*input_size:scale, idy_start: idy_start+scale*input_size:scale, seq8to0.tolist(), 4+iy_rd,0].astype('float32')+
                                                    G*traindata_all[image_id:image_id+1, idx_start: idx_start+scale*input_size:scale, idy_start: idy_start+scale*input_size:scale, seq8to0.tolist(), 4+iy_rd,1].astype('float32')+
                                                    B*traindata_all[image_id:image_id+1, idx_start: idx_start+scale*input_size:scale, idy_start: idy_start+scale*input_size:scale, seq8to0.tolist(), 4+iy_rd,2].astype('float32'))
            for kkk in range(start1,end1+1):
                
                traindata_batch_45d[:,:,kkk-start1]=np.squeeze(R*traindata_all[image_id:image_id+1, idx_start: idx_start+scale*input_size:scale, idy_start: idy_start+scale*input_size:scale, (8)-kkk+ix_rd, kkk+iy_rd,0].astype('float32')+
                                                                G*traindata_all[image_id:image_id+1, idx_start: idx_start+scale*input_size:scale, idy_start: idy_start+scale*input_size:scale, (8)-kkk+ix_rd, kkk+iy_rd,1].astype('float32')+
                                                                B*traindata_all[image_id:image_id+1, idx_start: idx_start+scale*input_size:scale, idy_start: idy_start+scale*input_size:scale, (8)-kkk+ix_rd, kkk+iy_rd,2].astype('float32'))
                            
                traindata_batch_m45d[:,:,kkk-start1]=np.squeeze(R*traindata_all[image_id:image_id+1, idx_start: idx_start+scale*input_size:scale, idy_start: idy_start+scale*input_size:scale, kkk+ix_rd, kkk+iy_rd,0].astype('float32')+
                                                                    G*traindata_all[image_id:image_id+1, idx_start: idx_start+scale*input_size:scale, idy_start: idy_start+scale*input_size:scale, kkk+ix_rd, kkk+iy_rd,1].astype('float32')+
                                                                    B*traindata_all[image_id:image_id+1, idx_start: idx_start+scale*input_size:scale, idy_start: idy_start+scale*input_size:scale, kkk+ix_rd, kkk+iy_rd,2].astype('float32'))
            '''
            traindata_batch_label  <-- scale_factor*traindata_label[random_index, scaled_label_size, scaled_label_size] 
            '''                
            if(len(traindata_label.shape)==5):
                traindata_batch_label[:,:]=(1.0/scale)*traindata_label[image_id, idx_start: idx_start+scale*label_size:scale,
                                                                            idy_start: idy_start+scale*label_size:scale,4+ix_rd,4+iy_rd]
            else:
                traindata_batch_label[:,:]=(1.0/scale)*traindata_label[image_id, idx_start: idx_start+scale*label_size:scale,
                                                                                idy_start: idy_start+scale*label_size:scale]
                                    
        traindata_batch_90d=np.float32((1/255)*traindata_batch_90d)
        traindata_batch_0d =np.float32((1/255)*traindata_batch_0d)
        traindata_batch_45d=np.float32((1/255)*traindata_batch_45d)
        traindata_batch_m45d=np.float32((1/255)*traindata_batch_m45d)
        
        return traindata_batch_90d,traindata_batch_0d,traindata_batch_45d,traindata_batch_m45d, traindata_batch_label
    
    def data_augmentation_for_train(self, traindata_batch_90d, traindata_batch_0d,
                                    traindata_batch_45d,traindata_batch_m45d, 
                                    traindata_label_batchNxN):
        """  
            For Data augmentation 
            (rotation, transpose and gamma)
            
        """ 
    
        gray_rand=0.4*np.random.rand()+0.8
        
        traindata_batch_90d[:,:,:]=pow(traindata_batch_90d[:,:,:],gray_rand)
        traindata_batch_0d[:,:,:]=pow(traindata_batch_0d[:,:,:],gray_rand)
        traindata_batch_45d[:,:,:]=pow(traindata_batch_45d[:,:,:],gray_rand)
        traindata_batch_m45d[:,:,:]=pow(traindata_batch_m45d[:,:,:],gray_rand)               

        rotation_or_transp_rand=np.random.randint(0,5)    

        if rotation_or_transp_rand==4: 

            traindata_batch_90d_tmp6=np.copy(np.transpose(np.squeeze(traindata_batch_90d[:,:,:]),(1, 0, 2)) )   
            traindata_batch_0d_tmp6=np.copy(np.transpose(np.squeeze(traindata_batch_0d[:,:,:]),(1, 0, 2)) ) 
            traindata_batch_45d_tmp6=np.copy(np.transpose(np.squeeze(traindata_batch_45d[:,:,:]),(1, 0, 2)) )
            traindata_batch_m45d_tmp6=np.copy(np.transpose(np.squeeze(traindata_batch_m45d[:,:,:]),(1, 0, 2)) )

            traindata_batch_0d[:,:,:]=np.copy(traindata_batch_90d_tmp6[:,:,::-1])
            traindata_batch_90d[:,:,:]=np.copy(traindata_batch_0d_tmp6[:,:,::-1])
            traindata_batch_45d[:,:,:]=np.copy(traindata_batch_45d_tmp6[:,:,::-1])
            traindata_batch_m45d[:,:,:]=np.copy(traindata_batch_m45d_tmp6)#[:,:,::-1])
            traindata_label_batchNxN[:,:]=np.copy(np.transpose(traindata_label_batchNxN[:,:],(1, 0))) 
    
    
        if(rotation_or_transp_rand==1): # 90도

            traindata_batch_90d_tmp3=np.copy(np.rot90(traindata_batch_90d[:,:,:],1,(0,1)))
            traindata_batch_0d_tmp3=np.copy(np.rot90(traindata_batch_0d[:,:,:],1,(0,1)))
            traindata_batch_45d_tmp3=np.copy(np.rot90(traindata_batch_45d[:,:,:],1,(0,1)))
            traindata_batch_m45d_tmp3=np.copy(np.rot90(traindata_batch_m45d[:,:,:],1,(0,1)))

            traindata_batch_90d[:,:,:]=traindata_batch_0d_tmp3   
            traindata_batch_45d[:,:,:]=traindata_batch_m45d_tmp3
            traindata_batch_0d[:,:,:]=traindata_batch_90d_tmp3[:,:,::-1] 
            traindata_batch_m45d[:,:,:]=traindata_batch_45d_tmp3[:,:,::-1] 
            
            traindata_label_batchNxN[:,:]=np.copy(np.rot90(traindata_label_batchNxN[:,:],1,(0,1))) 

        if(rotation_or_transp_rand==2): # 180도

            traindata_batch_90d_tmp4=np.copy(np.rot90(traindata_batch_90d[:,:,:],2,(0,1)))
            traindata_batch_0d_tmp4=np.copy(np.rot90(traindata_batch_0d[:,:,:],2,(0,1)))
            traindata_batch_45d_tmp4=np.copy(np.rot90(traindata_batch_45d[:,:,:],2,(0,1)))
            traindata_batch_m45d_tmp4=np.copy(np.rot90(traindata_batch_m45d[:,:,:],2,(0,1)))

            traindata_batch_90d[:,:,:]=traindata_batch_90d_tmp4[:,:,::-1]
            traindata_batch_0d[:,:,:]=traindata_batch_0d_tmp4[:,:,::-1] 
            traindata_batch_45d[:,:,:]=traindata_batch_45d_tmp4[:,:,::-1] 
            traindata_batch_m45d[:,:,:]=traindata_batch_m45d_tmp4[:,:,::-1] 
            
            traindata_label_batchNxN[:,:]=np.copy(np.rot90(traindata_label_batchNxN[:,:],2,(0,1)))
            
        if(rotation_or_transp_rand==3): # 270도

            traindata_batch_90d_tmp5=np.copy(np.rot90(traindata_batch_90d[:,:,:],3,(0,1)))
            traindata_batch_0d_tmp5=np.copy(np.rot90(traindata_batch_0d[:,:,:],3,(0,1)))
            traindata_batch_45d_tmp5=np.copy(np.rot90(traindata_batch_45d[:,:,:],3,(0,1)))
            traindata_batch_m45d_tmp5=np.copy(np.rot90(traindata_batch_m45d[:,:,:],3,(0,1)))

            traindata_batch_90d[:,:,:]=traindata_batch_0d_tmp5[:,:,::-1]
            traindata_batch_0d[:,:,:]=traindata_batch_90d_tmp5
            traindata_batch_45d[:,:,:]=traindata_batch_m45d_tmp5[:,:,::-1]
            traindata_batch_m45d[:,:,:]=traindata_batch_45d_tmp5
            

            traindata_label_batchNxN[:,:]=np.copy(np.rot90(traindata_label_batchNxN[:,:],3,(0,1)))  



        return traindata_batch_90d, traindata_batch_0d,traindata_batch_45d,traindata_batch_m45d, traindata_label_batchNxN


    def generate_traindata512(self, traindata_all,traindata_label,Setting02_AngualrViews):
        """   
        Generate validation or test set( = full size(512x512) LF images) 
        
        input: traindata_all   (16x512x512x9x9x3) uint8
                traindata_label (16x512x512x9x9)   float32
                Setting02_AngualrViews [0,1,2,3,4,5,6,7,8] for 9x9            
        
        
        output: traindata_batch_90d   (batch_size x 512 x 512 x len(Setting02_AngualrViews)) float32        
                traindata_batch_0d    (batch_size x 512 x 512 x len(Setting02_AngualrViews)) float32  
                traindata_batch_45d   (batch_size x 512 x 512 x len(Setting02_AngualrViews)) float32
                traindata_batch_m45d  (batch_size x 512 x 512 x len(Setting02_AngualrViews)) float32
                traindata_label_batchNxN (batch_size x 512 x 512 )               float32            
        """
    #        else:
        input_size=512
        label_size=512
        traindata_batch_90d=np.zeros((len(traindata_all),input_size,input_size,len(Setting02_AngualrViews)),dtype=np.float32)
        traindata_batch_0d=np.zeros((len(traindata_all),input_size,input_size,len(Setting02_AngualrViews)),dtype=np.float32)
        traindata_batch_45d=np.zeros((len(traindata_all),input_size,input_size,len(Setting02_AngualrViews)),dtype=np.float32)
        traindata_batch_m45d=np.zeros((len(traindata_all),input_size,input_size,len(Setting02_AngualrViews)),dtype=np.float32)        
        
        traindata_label_batchNxN=np.zeros((len(traindata_all),label_size,label_size))
        
        """ inital setting """
        ### sz = (16, 27, 9, 512, 512) 

        crop_half1=int(0.5*(input_size-label_size))
        start1=Setting02_AngualrViews[0]
        end1=Setting02_AngualrViews[-1]
    #        starttime=time.process_time() 0.375초 정도 걸림. i5 기준
        for ii in range(0,len(traindata_all)):
            
            R = 0.299 ### 0,1,2,3 = R, G, B, Gray // 0.299 0.587 0.114
            G = 0.587
            B = 0.114

            image_id = ii

            ix_rd = 0
            iy_rd = 0
            idx_start = 0
            idy_start = 0

            seq0to8=np.array(Setting02_AngualrViews)+ix_rd
            seq8to0=np.array(Setting02_AngualrViews[::-1])+iy_rd

            traindata_batch_0d[ii,:,:,:]=np.squeeze(R*traindata_all[image_id:image_id+1, idx_start: idx_start+input_size, idy_start: idy_start+input_size, 4+ix_rd, seq0to8,0].astype('float32')+
                                                    G*traindata_all[image_id:image_id+1, idx_start: idx_start+input_size, idy_start: idy_start+input_size, 4+ix_rd, seq0to8,1].astype('float32')+
                                                    B*traindata_all[image_id:image_id+1, idx_start: idx_start+input_size, idy_start: idy_start+input_size, 4+ix_rd, seq0to8,2].astype('float32'))
            
            traindata_batch_90d[ii,:,:,:]=np.squeeze(R*traindata_all[image_id:image_id+1, idx_start: idx_start+input_size,idy_start: idy_start+input_size, seq8to0, 4+iy_rd,0].astype('float32')+
                                                    G*traindata_all[image_id:image_id+1, idx_start: idx_start+input_size,idy_start: idy_start+input_size, seq8to0, 4+iy_rd,1].astype('float32')+
                                                    B*traindata_all[image_id:image_id+1, idx_start: idx_start+input_size,idy_start: idy_start+input_size, seq8to0, 4+iy_rd,2].astype('float32'))
            for kkk in range(start1,end1+1):
                
                traindata_batch_45d[ii,:,:,kkk-start1]=np.squeeze(R*traindata_all[image_id:image_id+1, idx_start: idx_start+input_size,idy_start: idy_start+input_size, (8)-kkk+ix_rd, kkk+iy_rd,0].astype('float32')+
                                                                G*traindata_all[image_id:image_id+1, idx_start: idx_start+input_size,idy_start: idy_start+input_size, (8)-kkk+ix_rd, kkk+iy_rd,1].astype('float32')+
                                                                B*traindata_all[image_id:image_id+1, idx_start: idx_start+input_size,idy_start: idy_start+input_size, (8)-kkk+ix_rd, kkk+iy_rd,2].astype('float32'))
                            
                traindata_batch_m45d[ii,:,:,kkk-start1]=np.squeeze(R*traindata_all[image_id:image_id+1, idx_start: idx_start+input_size,idy_start: idy_start+input_size, kkk+ix_rd, kkk+iy_rd,0].astype('float32')+
                                                                    G*traindata_all[image_id:image_id+1, idx_start: idx_start+input_size,idy_start: idy_start+input_size, kkk+ix_rd, kkk+iy_rd,1].astype('float32')+
                                                                    B*traindata_all[image_id:image_id+1, idx_start: idx_start+input_size,idy_start: idy_start+input_size, kkk+ix_rd, kkk+iy_rd,2].astype('float32'))
                if(len(traindata_all)>=12 and traindata_label.shape[-1]==9):                                
                    traindata_label_batchNxN[ii,:,:]=traindata_label[image_id,idx_start+crop_half1: idx_start+crop_half1+label_size,idy_start+crop_half1: idy_start+crop_half1+label_size, 4+ix_rd, 4+iy_rd]
                elif(len(traindata_label.shape)==5):
                    traindata_label_batchNxN[ii,:,:]=traindata_label[image_id ,idx_start+crop_half1: idx_start+crop_half1+label_size,idy_start+crop_half1: idy_start+crop_half1+label_size,0,0]
                else:
                    traindata_label_batchNxN[ii,:,:]=traindata_label[image_id ,idx_start+crop_half1: idx_start+crop_half1+label_size,idy_start+crop_half1: idy_start+crop_half1+label_size]

        traindata_batch_90d=np.float32((1/255)*traindata_batch_90d)
        traindata_batch_0d =np.float32((1/255)*traindata_batch_0d)
        traindata_batch_45d=np.float32((1/255)*traindata_batch_45d)
        traindata_batch_m45d=np.float32((1/255)*traindata_batch_m45d)

        traindata_batch_90d=np.minimum(np.maximum(traindata_batch_90d,0),1)
        traindata_batch_0d=np.minimum(np.maximum(traindata_batch_0d,0),1)
        traindata_batch_45d=np.minimum(np.maximum(traindata_batch_45d,0),1)
        traindata_batch_m45d=np.minimum(np.maximum(traindata_batch_m45d,0),1)

        return traindata_batch_90d,traindata_batch_0d,traindata_batch_45d,traindata_batch_m45d, traindata_label_batchNxN

    
    def load_LFdata(self, dir_LFimages):
        traindata_all = np.zeros((len(dir_LFimages), 512, 512, 9, 9, 3), np.uint8)
        traindata_label = np.zeros((len(dir_LFimages), 512, 512), np.float32)

        image_id = 0

        for dir_LFimage in dir_LFimages:
            print(dir_LFimage)
            for i in range(81):
                try:
                    tmp = np.float32(imageio.imread(
                        '/data/Dataset/full_data/additional/' + dir_LFimage + '/input_Cam0%.2d.png' % i))  # load LF images(9x9)
                except:
                    print('/data/Dataset/full_data/additional/' + dir_LFimage + '/input_Cam0%.2d.png..does not exist' % i)
                traindata_all[image_id, :, :, i // 9, i - 9 * (i // 9), :] = tmp
                del tmp
            try:

                tmp = np.float32(read_pfm(
                    '/data/Dataset/full_data/additional/' + dir_LFimage + '/gt_disp_lowres.pfm'))  # load LF disparity map
            except:
                print('/data/Dataset/full_data/additional/' + dir_LFimage + '/gt_disp_lowres.pfm..does not exist' % i)
            traindata_label[image_id, :, :] = tmp
            del tmp
            image_id = image_id + 1
        return traindata_all, traindata_label


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
        if self.mode == 'train':
            return len(self.traindata_all) 
        elif self.mode == 'online_eval':
            return len(self.valdata_all)
        else:
            return len(self.valdata_list)


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