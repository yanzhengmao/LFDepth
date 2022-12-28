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
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
from tensorboardX import SummaryWriter
import time
import argparse
import datetime
import sys
import os

import torch
import torch.nn as nn
import torch.nn.utils as utils
import errno
import cv2

import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
from evalfunctions7x7 import *
# from tensorboardX import SummaryWriter

import matplotlib
import matplotlib.cm
# import threading
from tqdm import tqdm

from bts import BtsModel

from bts_dataloader_patch import *


withwandb = False
try:
    import wandb
except ImportError:
    withwandb = False
    print('WandB disabled')

def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield arg


parser = argparse.ArgumentParser(description='BTS PyTorch implementation.', fromfile_prefix_chars='@')
parser.convert_arg_line_to_args = convert_arg_line_to_args

parser.add_argument('--mode',                      type=str,   help='train or test', default='train')
parser.add_argument('--model_name',                type=str,   help='model_Binocular name', default='bts_eigen_v2')
parser.add_argument('--encoder',                   type=str,   help='type of encoder, desenet121_bts, densenet161_bts, '
                                                                    'resnet101_bts, resnet50_bts, resnext50_bts or resnext101_bts',
                                                               default='densenet161_bts')
# Dataset
parser.add_argument('--dataset',                   type=str,   help='dataset to train on, kitti or nyu', default='')
parser.add_argument('--data_path',                 type=str,   help='path to the data', default='./dataset/kitti_dataset/')
parser.add_argument('--gt_path',                   type=str,   help='path to the groundtruth data', default='./dataset/kitti_dataset/')
parser.add_argument('--filenames_file',            type=str,   help='path to the filenames text file', default = '../train_test_inputs/new_eigen_train_files_with_gt.txt' )
parser.add_argument('--input_height',              type=int,   help='input height', default=256)
parser.add_argument('--input_width',               type=int,   help='input width',  default=256)
parser.add_argument('--input_channel',             type=int,   help='input channel', default=64)
parser.add_argument('--angRes',                    type=int,   help='input of angle size', default=9)
parser.add_argument('--conv_depth',                  type=int,   help='convolutional blocks for second layer', default=7)
parser.add_argument('--filt_num',                    type=int,   default=70)
parser.add_argument('--max_depth',                 type=float, help='maximum depth in estimation', default=4)

# Log and save
parser.add_argument('--log_directory',             type=str,   help='directory to save checkpoints and summaries', default='/model_flow_sence_disp_patch')
parser.add_argument('--checkpoint_path',           type=str,   help='path to a checkpoint to load', default='')
parser.add_argument('--log_freq',                  type=int,   help='Logging frequency in global steps', default=500)
parser.add_argument('--save_freq',                 type=int,   help='Checkpoint saving frequency in global steps', default=500)

# Training
parser.add_argument('--fix_first_conv_blocks',                 help='if set, will fix the first two conv blocks', action='store_true')
parser.add_argument('--fix_first_conv_block',                  help='if set, will fix the first conv block', action='store_true')
parser.add_argument('--bn_no_track_stats',                     help='if set, will not track running stats in batch norm layers', action='store_true')
parser.add_argument('--weight_decay',              type=float, help='weight decay factor for optimization', default=1e-2)
parser.add_argument('--bts_size',                  type=int,   help='initial num_filters in bts', default=512)
parser.add_argument('--retrain',                               help='if used with checkpoint_path, will restart training from step zero', action='store_true', default=True)
parser.add_argument('--adam_eps',                  type=float, help='epsilon in Adam optimizer', default=1e-6)
parser.add_argument('--batch_size',                type=int,   help='batch size', default=8)
parser.add_argument('--num_epochs',                type=int,   help='number of epochs', default=50000)
parser.add_argument('--learning_rate',             type=float, help='initial learning rate', default=1e-4)
parser.add_argument('--end_learning_rate',         type=float, help='end learning rate', default=-1)
parser.add_argument('--variance_focus',            type=float, help='lambda in paper: [0, 1], higher value more focus on minimizing variance of error', default=0.85)
parser.add_argument('--att_rank',                  type=int,   help='initial rank in attention structure', default=3)
# Preprocessing
parser.add_argument('--do_random_rotate',                      help='if set, will perform random rotation for augmentation', action='store_true')
parser.add_argument('--degree',                    type=float, help='random rotation maximum degree', default=2.5)
parser.add_argument('--do_kb_crop',                            help='if set, crop input images as kitti benchmark images', action='store_true')
parser.add_argument('--use_right',                             help='if set, will randomly use right images when train on KITTI', action='store_true')

# Multi-gpu training
parser.add_argument('--num_threads',               type=int,   help='number of threads to use for data loading', default=1)
parser.add_argument('--world_size',                type=int,   help='number of nodes for distributed training', default=1)
parser.add_argument('--rank',                      type=int,   help='node rank for distributed training', default=0)
parser.add_argument('--dist_url',                  type=str,   help='url used to set up distributed training', default='tcp://127.0.0.1:1236')
parser.add_argument('--dist_backend',              type=str,   help='distributed backend', default='nccl')
parser.add_argument('--gpu',                       type=int,   help='GPU id to use.', default='0')
parser.add_argument('--multiprocessing_distributed',           help='Use multi-processing distributed training to launch '
                                                                    'N processes per node, which has N GPUs. This is the '
                                                                    'fastest way to use PyTorch for either single node or '
                                                                    'multi node data parallel training', action='store_true',default=False)
# Online eval
parser.add_argument('--do_online_eval',                        help='if set, perform online eval in every eval_freq steps', action='store_true', default=True)
parser.add_argument('--data_path_eval',            type=str,   help='path to the data for online evaluation', required=False)
parser.add_argument('--gt_path_eval',              type=str,   help='path to the groundtruth data for online evaluation', required=False)
parser.add_argument('--filenames_file_eval',       type=str,   help='path to the filenames text file for online evaluation', required=False)
parser.add_argument('--min_depth_eval',            type=float, help='minimum depth for evaluation', default=-4)
parser.add_argument('--max_depth_eval',            type=float, help='maximum depth for evaluation', default=4)
parser.add_argument('--eigen_crop',                            help='if set, crops according to Eigen NIPS14', action='store_true')
parser.add_argument('--garg_crop',                             help='if set, crops accoroptimizerding to Garg  ECCV16', action='store_true')
parser.add_argument('--eval_freq',                 type=int,   help='Online evaluation frequency in global steps', default=500)
parser.add_argument('--eval_summary_directory',    type=str,   help='output directory for eval summary,'
                                                                    'if empty outputs to checkpoint folder', default='')
parser.add_argument('--exp', type=str, default='noname', help='experiment name')
parser.add_argument('--save_lpg', help='if set, save outputs from lpg layers', action='store_true', default=True)
if sys.argv.__len__() == 2:
    arg_filename_with_prefix = '@' + sys.argv[1]
    args = parser.parse_args([arg_filename_with_prefix])
else:
    args = parser.parse_args()

if args.mode == 'train' and not args.checkpoint_path:
    from bts import *

if args.mode == 'train' and args.checkpoint_path:
    model_dir = os.path.dirname(args.checkpoint_path)
    model_name = 'lf_trans'
    # model_name = args.model_name + '_rank_'+str(args.att_rank)

    import sys
    sys.path.append(model_dir)
    for key, val in vars(__import__(model_name)).items():
        if key.startswith('__') and key.endswith('__'):
            continue
        vars()[key] = val


# eval_metrics = ['silog', 'abs_rel', 'log10', 'rms', 'sq_rel', 'log_rms', 'd1', 'd2', 'd3']
eval_metrics = ['badpix_7']

def compute_errors(gt, pred):


    train_diff = np.abs(pred - gt)
    bad_pixel_7 = (train_diff >= 0.07)

    badpix_7 = 100 - (100 * np.average(bad_pixel_7))

    return badpix_7


def block_print():
    sys.stdout = open(os.devnull, 'w')

def enable_print():
    sys.stdout = sys.__stdout__


def get_num_lines(file_path):
    f = open(file_path, 'r')
    lines = f.readlines()
    f.close()
    return len(lines)


def colorize(value, vmin=None, vmax=None, cmap='Greys'):
    value = value.cpu().numpy()[:, :, :]
    value = np.log10(value)

    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax

    if vmin != vmax:
        value = (value - vmin) / (vmax - vmin)
        value = value*0.

    cmapper = matplotlib.cm.get_cmap(cmap)
    value = cmapper(value, bytes=True)

    img = value[:, :, :3]

    return img.transpose((2, 0, 1))


def normalize_result(value, vmin=None, vmax=None):
    value = value.cpu().numpy()[0, :, :]

    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax

    if vmin != vmax:
        value = (value - vmin) / (vmax - vmin)
    else:
        value = value * 0.

    return np.expand_dims(value, 0)


def set_misc(model):
    if args.bn_no_track_stats:
        print("Disabling tracking running stats in batch norm layers")
        model.apply(bn_init_as_tf)

    if args.fix_first_conv_blocks:
        if 'resne' in args.encoder:
            fixing_layers = ['base_model.conv1', 'base_model.layer1.0', 'base_model.layer1.1', '.bn']
        else:
            fixing_layers = ['conv0', 'denseblock1.denselayer1', 'denseblock1.denselayer2', 'norm']
        print("Fixing first two conv blocks")
    elif args.fix_first_conv_block:
        if 'resne' in args.encoder:
            fixing_layers = ['base_model.conv1', 'base_model.layer1.0', '.bn']
        else:
            fixing_layers = ['conv0', 'denseblock1.denselayer1', 'norm']
        print("Fixing first conv block")
    else:
        if 'resne' in args.encoder:
            fixing_layers = ['base_model.conv1', '.bn']
        else:
            fixing_layers = ['conv0', 'norm']
        print("Fixing first conv layer")

    for name, child in model.named_children():
        if not 'encoder' in name:
            continue
        for name2, parameters in child.named_parameters():
            # print(name, name2)
            if any(x in name2 for x in fixing_layers):
                parameters.requires_grad = False


def online_eval(model, dataloader_eval, gpu, ngpus, global_step, args):
    eval_measures = torch.zeros(10).cuda(device=gpu)
    for _, eval_sample_batched in enumerate(tqdm(dataloader_eval.data)):
        with torch.no_grad():
            valdata_batch_90d = eval_sample_batched['traindata_batch_90d'].cuda().permute(0, 3, 1, 2)
            valdata_batch_0d = eval_sample_batched['traindata_batch_0d'].cuda().permute(0, 3, 1, 2)
            valdata_batch_45d = eval_sample_batched['traindata_batch_45d'].cuda().permute(0, 3, 1, 2)
            valdata_batch_m45d = eval_sample_batched['traindata_batch_m45d'].cuda().permute(0, 3, 1, 2)
            label = eval_sample_batched['traindata_label'].cuda()
            # label = label[:, 11:-11, 11:-11]
            #
            # pad = [16, 16, 16, 16]
            # valdata_batch_90d = F.pad(valdata_batch_90d, pad)
            # valdata_batch_0d = F.pad(valdata_batch_0d, pad)
            # valdata_batch_45d = F.pad(valdata_batch_45d, pad)
            # valdata_batch_m45d = F.pad(valdata_batch_m45d, pad)

            img = torch.cat([valdata_batch_0d, valdata_batch_90d, valdata_batch_45d, valdata_batch_m45d], dim=1)

            scenename = str(eval_sample_batched['scene_name'][0])


            # _, _, _, _, pred_depth = model([valdata_batch_90d, valdata_batch_0d, valdata_batch_45d, valdata_batch_m45d], args.att_rank)

            _, _, _, _,  pred_depth = model(img, args.att_rank)
            # pred_depth = pred_depth[:, :, 16:-16, 16:-16]

            pred_depth = pred_depth.cpu().numpy().squeeze()
            # label = label.cpu().numpy().squeeze()

        # pred_depth[pred_depth < args.min_depth_eval] = args.min_depth_eval
        # pred_depth[pred_depth > args.max_depth_eval] = args.max_depth_eval
        # pred_depth[np.isinf(pred_depth)] = args.max_depth_eval
        # pred_depth[np.isnan(pred_depth)] = args.min_depth_eval
        # measures = compute_errors(pred_depth, label)

        measures = 100.0 - get_scores_file_by_name_clear(pred_depth, scenename)

        eval_measures[:1] += torch.tensor(measures).cuda(device=gpu)
        eval_measures[1] += 1

    eval_measures_cpu = eval_measures.cpu()
    cnt = eval_measures_cpu[1].item()
    eval_measures_cpu /= cnt
    print('\nComputing errors for {} eval samples'.format(int(cnt)))

    print("{:>7}".format('val_badpix_7'))
    print('{:7.3f}'.format(eval_measures_cpu[0]))

    return eval_measures_cpu



def online_test(model, gpu, ngpus, global_step, args):

    start_time = time.time()
    i = 0

    args.mode = 'test'
    scene_name = ['backgammon', 'boxes', 'cotton', 'dino', 'dots', 'pyramids', 'sideboard', 'stripes']
    avg_score = 0
    for scene in scene_name:
        dataloader = BtsDataLoader(args, 'test', 256, scene)

        merge_image = np.zeros([512, 512]).astype(float)

        with torch.no_grad():

            for step, sample in enumerate(tqdm(dataloader.data)):
                valdata = sample['valdata'].cuda().view(1, -1, args.input_height, args.input_width)
                val_patch_label = sample['val_patch_label']

                # scenename = str(sample['scene_name'][0])

                pred_depth = model(valdata)
                # valdata_label_patch = valdata_label_patch.cpu().numpy().squeeze()
                pred_depth = pred_depth.cpu().numpy().squeeze()

                error_score = compute_errors(pred_depth, val_patch_label)

                # pred_depth = pred_depth[:, :, 16:-16, 16:-16]
                #
                # pred_depth = pred_depth.cpu().numpy().squeeze()
                idx = (step // 2) * 256
                idy = (step % 2) * 256
                # if idx != 512-16 or idy!=512-16:
                merge_image[idx:idx + 256, idy:idy + 256] = pred_depth.copy()

        valdata_label = sample['valdata_label']
        valdata_label = valdata_label.cpu().numpy().squeeze()
        # plt.imsave(filename_pred_png, (pred_depth))

        pred_depth = merge_image
        scene_name = scene
        #
        i = i + 1
        # error_score = compute_errors(pred_depth, valdata_lable)
        error_score = 100 - get_scores_file_by_name_clear(pred_depth, scene)
        avg_score += error_score
        print("\n-----------------scene {}---------------".format(scene))
        print("{:>7}".format('badpix_7'))
        print({'badpix_7': error_score})
        # print("\n-----------------avg score {}".format((avg_score / 8).numpy()[0]))
        elapsed_time = time.time() - start_time
        print('Elapesed time: %s' % str(elapsed_time))
        print('Done.')
        save_name = 'result'
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
            plt.imsave(filename_gt_png, (valdata_label))
    return avg_score / 8

def main_worker(gpu, ngpus_per_node, args):
    if withwandb:
        wandb.init(project="pga", group=args.dataset, name='transformer')
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    model = BtsModel(args)
    model.train()
    # model.decoder.apply(weights_init_xavier)
    # set_misc(model)
    ### test model_Binocular paremet size
    num_params = sum([np.prod(p.size()) for p in model.parameters()])
    print("Total number of parameters: {}".format(num_params))

    num_params_update = sum([np.prod(p.shape) for p in model.parameters() if p.requires_grad])
    print("Total number of learning parameters: {}".format(num_params_update))

    model.cuda()

    if args.distributed:
        print("Model Initialized on GPU: {}".format(args.gpu))
    else:
        print("Model Initialized")

    global_step = 0

    # Training parameters
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, eps=args.adam_eps)
    model_just_loaded = False
    if args.checkpoint_path != '':
        if os.path.isfile(args.checkpoint_path):
            print("Loading checkpoint '{}'".format(args.checkpoint_path))
            if args.gpu is None:
                checkpoint = torch.load(args.checkpoint_path)
            else:
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.checkpoint_path, map_location=loc)

            global_step = checkpoint['global_step']
            model.load_state_dict(checkpoint['model_flow_sence_disp_patch'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            # try:
            #     best_eval_measures_higher_better = checkpoint['best_eval_measures_higher_better'].cpu()
            #     best_eval_measures_lower_better = checkpoint['best_eval_measures_lower_better'].cpu()
            #     best_eval_steps = checkpoint['best_eval_steps']
            # except KeyError:
            #     print("Could not load values for online evaluation")


            print("Loaded checkpoint '{}' (global_step {})".format(args.checkpoint_path, checkpoint['global_step']))
        else:
            print("No checkpoint found at '{}'".format(args.checkpoint_path))
        model_just_loaded = True

    if args.retrain:
        global_step = 0

    cudnn.benchmark = True

    dataloader = BtsDataLoader(args, 'train', 256, '')
    # dataloader_eval = BtsDataLoader(args, 'online_eval', 512)
    # dataloader_test = BtsDataLoader(args, 'test', 64)


    silog_criterion = silog_loss(variance_focus=args.variance_focus)

    start_time = time.time()
    duration = 0

    num_log_images = args.batch_size
    end_learning_rate = args.end_learning_rate if args.end_learning_rate != -1 else 0.1 * args.learning_rate

    var_sum = [var.sum() for var in model.parameters() if var.requires_grad]
    var_cnt = len(var_sum)
    var_sum = np.sum(var_sum)

    print("Initial variables' sum: {:.3f}, avg: {:.3f}".format(var_sum, var_sum/var_cnt))

    steps_per_epoch = len(dataloader.data)
    num_total_steps = args.num_epochs * steps_per_epoch
    epoch = global_step // steps_per_epoch

    best_result = 0.0
    avg_err = 0.0
    test_measures = 0
    while epoch < args.num_epochs:

        if args.distributed:
            dataloader.train_sampler.set_epoch(epoch)

        for step, sample_batched in enumerate(dataloader.data):
            optimizer.zero_grad()
            before_op_time = time.time()
            writer = SummaryWriter('./log')
            train_data = sample_batched['train_data'].cuda().view(args.batch_size, -1, args.input_height, args.input_width)
            train_label = sample_batched['train_label'].cuda().unsqueeze(1)

            depth_est = model(train_data)
            # depth_est = depth_est.detach().cpu().numpy()
            # train_label = train_label.detach().cpu().numpy()


            # depth_est = depth_est[:, :, 11:-11, 11:-11]
            if args.dataset == 'nyu':
                mask = train_label > 0.1
            else:
                mask = train_label > - 4.0

            loss = silog_criterion.forward(depth_est, train_label, mask.to(torch.bool))

            writer.add_scalar('loss', loss, global_step)
            writer.close()
            loss.backward()
            for param_group in optimizer.param_groups:
                current_lr = param_group['lr']
            if global_step % 10000 == 0 and global_step != 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] * 0.8
            # for param_group in optimizer.param_groups:
            #     current_lr = (args.learning_rate - end_learning_rate) * (1 - global_step / num_total_steps) ** 0.9 + end_learning_rate
            #     param_group['lr'] = current_lr

            optimizer.step()
            save_name = '/home/yzm/project/TransDepth-main/pytorch/model_flow_sence_disp_patch/bts_eigen_v2_pytorch_att_rank_3/vis_att_bts_eigen_v2_pytorch_att'
            with torch.no_grad():
                # from PIL import Image
                # train_data = train_data.cpu().numpy()
                depth_est = depth_est.cpu().numpy().squeeze()
                train_label = train_label.cpu().numpy().squeeze()
                # print(train_data.shape)
                # train_data = Image.fromarray(train_data[:,:,4,4,:,:].squeeze()* 255.0)
                # train_data.convert("RGB").save("E:/project/TransDepth-main/result/traindata.jpg")
                # train_label = Image.fromarray(train_label)
                # depth_est = Image.fromarray(depth_est)
                # train_label.convert("RGB").save("E:/project/TransDepth-main/result/train_label.jpg")
                # depth_est.convert("RGB").save("E:/project/TransDepth-main/result/pred_disp.jpg")

                err = compute_errors(depth_est, train_label)
                # writer = SummaryWriter('./log')
                #
                # writer.add_scalar('loss', loss, global_step)
                # for i in range(len(depth_est)):
                #     filename_cmap_png = save_name + '/cmap/' + str(i) + '.png'
                #     plt.imsave(filename_cmap_png, (depth_est[i]))
                #     filename_rgb_png = save_name + '/rgb/' + str(i) + '.png'
                #     plt.imsave(filename_rgb_png, (traindata_label[i]))
                avg_err += err
            if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
                print('[epoch][s/s_per_e/gs]: [{}][{}/{}/{}], lr: {:.12f}, loss: {:.12f}, train_score:{:.3f}'.format(epoch, step, steps_per_epoch, global_step, current_lr, loss, err))
                if np.isnan(loss.cpu().item()):
                    print('NaN in loss occurred. Aborting training.')
                    return -1

            duration += time.time() - before_op_time
            if global_step and global_step % args.log_freq == 0 and not model_just_loaded:
                var_sum = [var.sum() for var in model.parameters() if var.requires_grad]
                var_cnt = len(var_sum)
                var_sum = np.sum(var_sum)
                examples_per_sec = args.batch_size / duration * args.log_freq
                duration = 0
                time_sofar = (time.time() - start_time) / 3600
                training_time_left = (num_total_steps / global_step - 1.0) * time_sofar
                if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
                    print("{}".format(args.model_name))
                print_string = 'GPU: {} | examples/s: {:4.2f} | loss: {:.5f} | var sum: {:.3f} avg: {:.3f} | time elapsed: {:.2f}h | time left: {:.2f}h'
                print(print_string.format(args.gpu, examples_per_sec, loss, var_sum.item(), var_sum.item()/var_cnt, time_sofar, training_time_left))



            if not args.do_online_eval and global_step and global_step % args.save_freq == 0:
                if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
                    checkpoint = {'global_step': global_step,
                                  'model_flow_sence_disp_patch': model.state_dict(),
                                  'optimizer': optimizer.state_dict()}
                    torch.save(checkpoint, args.log_directory + '/' + args.model_name + '/model_flow_sence_disp_patch-{}'.format(global_step))


            if args.do_online_eval and global_step and global_step % args.eval_freq == 0 and not model_just_loaded:
                time.sleep(0.1)
                model.eval()
                print('avg_err : {}'.format(avg_err / args.eval_freq))
                avg_err = 0
                # eval_measures = online_eval(model, dataloader_eval, gpu, ngpus_per_node,global_step, args)
                test_measures = online_test(model, gpu, ngpus_per_node,global_step, args)
                # if test_measures[0] > best_result:
                #     best_result = test_measures[0]
                # model_save_name = '/model_flow_sence_disp_patch_{}_{}_{:.5f}'.format(global_step, eval_metrics[0],
                #                                                                      test_measures[0])
                model_save_name = '/HCI_patch32_AG_{}_{:.5f}'.format(global_step, test_measures)
                checkpoint = {'global_step': global_step,
                              'model_flow_sence_disp_patch': model.state_dict(),
                              'optimizer': optimizer.state_dict()
                              }
                torch.save(checkpoint, save_name + model_save_name)
                    # eval_summary_writer.flush()
                model.train()
                block_print()
                set_misc(model)
                enable_print()

            model_just_loaded = False
            global_step += 1

        epoch += 1


def main():

    if args.mode != 'train':
        print('bts_main.py is only for training. Use bts_test.py instead.')
        return -1

    model_filename = args.model_name + '.py'
    args.model_name=args.model_name + '_rank_'+str(args.att_rank)
    # command = 'mkdir -p ' + args.log_directory + '/' + args.model_name
    # os.system(command)

    # args_out_path = args.log_directory + '/' + args.model_name + '/' + sys.argv[1]
    # command = 'cp ' + sys.argv[1] + ' ' + args_out_path
    # os.system(command)

    if args.checkpoint_path == '':
        model_out_path = args.log_directory + '/' + args.model_name + '/' + model_filename
        # command = 'cp bts.py ' + model_out_path
        # os.system(command)
        # aux_out_path = args.log_directory + '/' + args.model_name + '/.'
        # command = 'cp bts_main.py ' + aux_out_path
        # os.system(command)
        # command = 'cp bts_dataloader.py ' + aux_out_path
        # os.system(command)
    else:
        loaded_model_dir = os.path.dirname(args.checkpoint_path)
        loaded_model_name = os.path.basename(loaded_model_dir)
        loaded_model_filename = loaded_model_name + '.py'

        model_out_path = args.log_directory + '/' + args.model_name + '/' + model_filename
        # command = 'cp ' + loaded_model_dir + '/' + loaded_model_filename + ' ' + model_out_path
        # os.system(command)

    torch.cuda.empty_cache()
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if ngpus_per_node > 1 and not args.multiprocessing_distributed:
        print("This machine has more than 1 gpu. Please specify --multiprocessing_distributed, or set \'CUDA_VISIBLE_DEVICES=0\'")
        return -1

    if args.do_online_eval:
        print("You have specified --do_online_eval.")
        print("This will evaluate the model_Binocular every eval_freq {} steps and save best models for individual eval metrics."
              .format(args.eval_freq))

    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        main_worker(args.gpu, ngpus_per_node, args)


if __name__ == '__main__':

    main()