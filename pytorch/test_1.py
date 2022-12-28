# Copyright (C) 2019 Jin Han Lee

# This file is a part of BTS.
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>

from __future__ import absolute_import, division, print_function

import os
import argparse
import time
from tkinter.tix import Tree
import numpy as np
import cv2
import sys
from bts_patch import *
from PIL import  Image
import torch
import torch.nn as nn
from torch.autograd import Variable
from bts_dataloader_patch import *
from evalfunctions7x7 import *
import errno
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.colors as colors
import matplotlib.cm as cmx
from torchvision.transforms import ToPILImage


# from bts_dataloader import *


def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield arg


parser = argparse.ArgumentParser(description='BTS PyTorch implementation.', fromfile_prefix_chars='@')
parser.convert_arg_line_to_args = convert_arg_line_to_args

parser.add_argument('--model_name', type=str, help='model_Binocular name', default='bts_patch')
parser.add_argument('--encoder', type=str, help='type of encoder, vgg or desenet121_bts or densenet161_bts',
                    default='densenet161_bts')
parser.add_argument('--data_path', type=str, help='path to the data', default='./dataset/kitti_dataset/')
parser.add_argument('--filenames_file', type=str, help='path to the filenames text file',
                    default='../train_test_inputs/kitti_test.txt')
parser.add_argument('--input_height', type=int, help='input height', default=32)
parser.add_argument('--input_width', type=int, help='input width', default=32)
parser.add_argument('--input_channel',             type=int,   help='input channel', default=9)
parser.add_argument('--conv_depth',                  type=int,   help='convolutional blocks for second layer', default=7)
parser.add_argument('--filt_num',                    type=int,   default=70)
parser.add_argument('--max_depth', type=float, help='maximum depth in estimation', default=4)
parser.add_argument('--checkpoint_path', type=str, help='path to a specific checkpoint to load',
                    default='/home/yzm/project/TransDepth-main/pytorch/model_flow_sence_disp_patch/bts_eigen_v2_pytorch_att_rank_3/vis_att_bts_eigen_v2_pytorch_att/HCI_patch32_25000_14.44091')
parser.add_argument('--dataset', type=str, help='dataset to train on, make3d or nyudepthv2', default='')
parser.add_argument('--do_kb_crop', help='if set, crop input images as kitti benchmark images', action='store_true',
                    default=False)
parser.add_argument('--save_lpg', help='if set, save outputs from lpg layers', action='store_true', default=True)
parser.add_argument('--bts_size', type=int, help='initial num_filters in bts', default=512)
parser.add_argument('--att_rank', type=int, help='initial rank in attention structure', default=5)

if sys.argv.__len__() == 2:
    arg_filename_with_prefix = '@' + sys.argv[1]
    args = parser.parse_args([arg_filename_with_prefix])
else:
    args = parser.parse_args()

model_dir = os.path.dirname(args.checkpoint_path)
sys.path.append(model_dir)

for key, val in vars(__import__(args.model_name)).items():
    if key.startswith('__') and key.endswith('__'):
        continue
    vars()[key] = val


# def get_num_lines(file_path):
#     f = open(file_path, 'r')
#     lines = f.readlines()
#     f.close()
#     return len(lines)

def compute_errors(gt, pred):


    train_diff = np.abs(pred - gt)
    bad_pixel_7 = (train_diff >= 0.07)

    badpix_7 = 100 * np.average(bad_pixel_7)

    return badpix_7


def test(params):
    """Test function."""
    model = BtsModel(params=args)

    checkpoint = torch.load(args.checkpoint_path, map_location='cuda:0')
    model.load_state_dict(checkpoint['model_flow_sence_disp_patch'])
    model.eval()
    model.cuda()

    # num_test_samples = get_num_lines(args.filenames_file)
    #
    # with open(args.filenames_file) as f:
    #     lines = f.readlines()
    # print(lines[0])

    print('now testing {} files with {}'.format(8, args.checkpoint_path))

    pred_depths = []
    gt_depths = []
    scene_name = []

    start_time = time.time()
    i = 0
    lines = []
    args.mode = 'test'
    scene_name = ['backgammon', 'boxes', 'cotton', 'dino', 'dots', 'pyramids', 'sideboard', 'stripes']
    for scene in scene_name:
        dataloader = BtsDataLoader(args, 'test', 32, scene)

        merge_image = np.zeros([512, 512]).astype(float)

        with torch.no_grad():
            avg_score = torch.zeros(1)
            for step, sample in enumerate(tqdm(dataloader.data)):
                valdata_batch_90d = sample['valdata_batch_90d'].cuda().permute(0, 3, 1, 2)
                valdata_batch_0d = sample['valdata_batch_0d'].cuda().permute(0, 3, 1, 2)
                valdata_batch_45d = sample['valdata_batch_45d'].cuda().permute(0, 3, 1, 2)
                valdata_batch_m45d = sample['valdata_batch_m45d'].cuda().permute(0, 3, 1, 2)
                # scenename = str(sample['scene_name'][0])
                valdata_label_patch = sample['valdata_label_patch'].cuda()

                img = torch.cat([valdata_batch_90d, valdata_batch_0d, valdata_batch_45d, valdata_batch_m45d], dim=1)
                valdata_lable = sample['valdata_label'].cuda()
                #
                _, _, _, _, pred_depth = model(img, args.att_rank)
                valdata_label_patch = valdata_label_patch.cpu().numpy().squeeze()
                pred_depth = pred_depth.cpu().numpy().squeeze()

                error_score = compute_errors(pred_depth, valdata_label_patch)

                # pred_depth = pred_depth[:, :, 16:-16, 16:-16]
                #
                # pred_depth = pred_depth.cpu().numpy().squeeze()
                idx = (step %31) * 16
                idy = (step //31) * 16
                # if idx != 512-16 or idy!=512-16:
                merge_image[idx:idx+32, idy:idy+32] = pred_depth.copy()

        valdata_lable = valdata_lable.cpu().numpy().squeeze()
        # plt.imsave(filename_pred_png, (pred_depth))

        pred_depth = merge_image
        scene_name = scene
        #
        i = i + 1
        # error_score = compute_errors(pred_depth, valdata_lable)
        error_score = 100 - get_scores_file_by_name_clear(pred_depth, scene)
        # avg_score += error_score
        print("\n-----------------scene {}---------------".format(scene))
        print("{:>7}".format('badpix_7'))
        print({'badpix_7': error_score})
        # print("\n-----------------avg score {}".format((avg_score / 8).numpy()[0]))
        elapsed_time = time.time() - start_time
        print('Elapesed time: %s' % str(elapsed_time))
        print('Done.')
        # save_name = 'result_' + args.model_name
        save_name = 'vis_att_' + args.model_name
        print('Saving result pngs..')
        if not os.path.exists(os.path.dirname(save_name)):
            try:
                os.mkdir(save_name)
                os.mkdir(save_name + '/raw')
                os.mkdir(save_name + '/cmap')
                os.mkdir(save_name + '/rgb')
                os.mkdir(save_name + '/gt')
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise
        if args.dataset == 'kitti':
            filename_pred_png = save_name + '/raw/' + scene_name + '.png'
            filename_cmap_png = save_name + '/cmap/' + scene_name + '.png'
            filename_image_png = save_name + '/rgb/' + scene_name + '.png'
        else:
            filename_pred_png = save_name + '/raw/' + scene_name + '.png'
            filename_cmap_png = save_name + '/cmap/' + scene_name + '.png'
            filename_gt_png = save_name + '/gt/' + scene_name + '.png'


        # rgb_path = os.path.join(args.data_path, './' + lines[s].split()[0])
        # image = cv2.imread(rgb_path)
        if args.dataset == 'kitti' or args.dataset == 'kitti_benchmark':
            pred_depth_scaled = pred_depth * 256.0
        else:
            pred_depth_scaled = pred_depth
        pred_depth_scaled = pred_depth_scaled.astype(np.uint16)
        cv2.imwrite(filename_pred_png, pred_depth_scaled, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        if args.save_lpg:
            # cv2.imwrite(filename_image_png, image)
            plt.imsave(filename_cmap_png, (pred_depth))
            plt.imsave(filename_gt_png, (valdata_lable))
    return


if __name__ == '__main__':
    test(args)


