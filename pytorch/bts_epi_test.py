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
# os.environ['OPENBLAS_NUM_THREADS'] = '1'
# os.environ['OMP_NUM_THREADS'] = '1'
import time
import argparse
import datetime
# from numpy import *
import sys
import os
import torch
import torch.nn as nn
import torch.nn.utils as utils

import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
from evalfunctions7x7 import *
from tensorboardX import SummaryWriter
# from test_utils import *
import matplotlib
import matplotlib.cm
from tqdm import tqdm
from epi import BtsModel, test, pad_2_full_size, label2disp, prob2disp
from lr_scheduler import build_scheduler
from bts_dataloader_5 import *



withwandb = False
# try:
#     import wandb
# except ImportError:
#     withwandb = False
#     print('WandB disabled')

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
parser.add_argument('--input_height',              type=int,   help='input height', default=2)
parser.add_argument('--input_width',               type=int,   help='input width',  default=10)
parser.add_argument('--input_channel',             type=int,   help='input channel', default=3)
parser.add_argument('--conv_depth',                  type=int,   help='convolutional blocks for second layer', default=7)
parser.add_argument('--filt_num',                    type=int,   default=70)
parser.add_argument('--max_depth',                 type=float, help='maximum depth in estimation', default=4)

# Log and save
parser.add_argument('--stage',                     type=int, default=1)
parser.add_argument('--log_directory',             type=str,   help='directory to save checkpoints and summaries', default='./epi_1')
parser.add_argument('--checkpoint_path',           type=str,   help='path to a checkpoint to load', default='')
parser.add_argument('--log_freq',                  type=int,   help='Logging frequency in global steps', default=500)
parser.add_argument('--save_freq',                 type=int,   help='Checkpoint saving frequency in global steps', default=500)

# Training
parser.add_argument('--fix_first_conv_blocks',                 help='if set, will fix the first two conv blocks', action='store_true')
parser.add_argument('--fix_first_conv_block',                  help='if set, will fix the first conv block', action='store_true')
parser.add_argument('--bn_no_track_stats',                     help='if set, will not track running stats in batch norm layers', action='store_true')
parser.add_argument('--weight_decay',              type=float, help='weight decay factor for optimization', default=1e-2)
parser.add_argument('--bts_size',                  type=int,   help='initial num_filters in bts', default=512)
parser.add_argument('--retrain',                               help='if used with checkpoint_path, will restart training from step zero', action='store_true', default=False)
parser.add_argument('--adam_eps',                  type=float, help='epsilon in Adam optimizer', default=1e-6)
parser.add_argument('--batch_size',                type=int,   help='batch size', default=64)
parser.add_argument('--mask_ratio',                type=float,   help='mask ratio', default=0.7)
parser.add_argument('--num_epochs',                type=int,   help='number of epochs', default=50000)
parser.add_argument('--learning_rate',             type=float, help='initial learning rate', default=0.0001)
parser.add_argument('--end_learning_rate',         type=float, help='end learning rate', default=-1)
parser.add_argument('--warmup_epochs',             type=float, help='epochs of warm up')
parser.add_argument('--warmup_lr',                 type=float, help='learning rate of lr')
parser.add_argument('--min_lr',                    type=float, help='end learning rate', default=-1)
parser.add_argument('--lr_scheduler',              type=str, help='name of lr scheduler')
parser.add_argument('--warmup_prefix',             type=bool)
parser.add_argument('--decay_rate',                type=float)
parser.add_argument('--gamma',                     type=bool)
parser.add_argument('--milestones',                type=list)
parser.add_argument('--multi_steps',               type=list)
parser.add_argument('--decay_epochs',              type=int)
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
parser.add_argument('--dist_url',                  type=str,   help='url used to set up distributed training', default='tcp://127.0.0.1:1234')
parser.add_argument('--dist_backend',              type=str,   help='distributed backend', default='nccl')
parser.add_argument('--gpu',                       type=int,   help='GPU id to use.', default='0')
parser.add_argument('--multiprocessing_distributed',           help='Use multi-processing distributed training to launch '
                                                                    'N processes per node, which has N GPUs. This is the '
                                                                    'fastest way to use PyTorch for either single node or '
                                                                    'multi node data parallel training', action='store_true',)
# Online eval
parser.add_argument('--do_online_eval',                        help='if set, perform online eval in every eval_freq steps', action='store_true', default=True)
parser.add_argument('--data_path_eval',            type=str,   help='path to the data for online evaluation', required=False)
parser.add_argument('--gt_path_eval',              type=str,   help='path to the groundtruth data for online evaluation', required=False)
parser.add_argument('--filenames_file_eval',       type=str,   help='path to the filenames text file for online evaluation', required=False)
parser.add_argument('--min_depth_eval',            type=float, help='minimum depth for evaluation', default=-4)
parser.add_argument('--max_depth_eval',            type=float, help='maximum depth for evaluation', default=4)
parser.add_argument('--eigen_crop',                            help='if set, crops according to Eigen NIPS14', action='store_true')
parser.add_argument('--garg_crop',                             help='if set, crops accoroptimizerding to Garg  ECCV16', action='store_true')
parser.add_argument('--eval_freq',                 type=int,   help='Online evaluation frequency in global steps', default=2500)
parser.add_argument('--eval_summary_directory',    type=str,   help='output directory for eval summary,'
                                                                    'if empty outputs to checkpoint folder', default='')
parser.add_argument('--exp', type=str, default='noname', help='experiment name')
if sys.argv.__len__() == 2:
    arg_filename_with_prefix = '@' + sys.argv[1]
    args = parser.parse_args([arg_filename_with_prefix])
else:
    args = parser.parse_args()

if args.mode == 'train' and not args.checkpoint_path:
    from epi import *

elif args.mode == 'train' and args.checkpoint_path:
    model_dir = os.path.dirname(args.checkpoint_path)
    # model_name = 'bts_nyu_v2_pytorch_att'
    # model_name = 'bts_hcinew_epi_v2_pytorch_att'
    model_name = 'epi'
    import sys
    sys.path.append(model_dir)
    for key, val in vars(__import__(model_name)).items():
        if key.startswith('__') and key.endswith('__'):
            continue
        vars()[key] = val


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





def main_worker(gpu, ngpus_per_node, args):
    # torch.backends.cudnn.enabled = False
    # if withwandb:
    #     wandb.init(project="pga", group=args.dataset, name='transformer')
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank)
    
    model = BtsModel(args)
    print(model)
    model.train()
    # model.decoder.apply(weights_init_xavier)
    set_misc(model)
    ## test model_Binocular paremet size
    num_params = sum([np.prod(p.size()) for p in model.parameters()])
    print("Total number of parameters: {}".format(num_params))

    num_params_update = sum([np.prod(p.shape) for p in model.parameters() if p.requires_grad])
    print("Total number of learning parameters: {}".format(num_params_update))


    if args.distributed:
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            args.batch_size = int(args.batch_size / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        else:
            model.cuda()
            model = DataParallelModel(model, find_unused_parameters=True)
    else:
        model = torch.nn.DataParallel(model)
        model.cuda()

    if args.distributed:
        print("Model Initialized on GPU: {}".format(args.gpu))
    else:
        print("Model Initialized")

    global_step = 0

    # Training parameters
    # optimizer = torch.optim.Adam([{'params': model.parameters()}], lr=args.learning_rate)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # lr_scheduler = build_scheduler(args, optimizer, len(dataloader.data))
    # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.milestones, args.gamma)

    model_just_loaded = False
    if args.checkpoint_path != '':
        if os.path.isfile(args.checkpoint_path):
            print("Loading checkpoint '{}'".format(args.checkpoint_path))
            if args.gpu is None:
                checkpoint = torch.load(args.checkpoint_path)
            else:
                # loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
            global_step = checkpoint['global_step']
            model.load_state_dict(checkpoint['epi'], strict=False)
            # optimizer.load_state_dict(checkpoint['optimizer'])
            # lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

            print("Loaded checkpoint '{}' (global_step {})".format(args.checkpoint_path, checkpoint['global_step']))
        else:
            print("No checkpoint found at '{}'".format(args.checkpoint_path))
        model_just_loaded = True

    if args.retrain:
        global_step = 0

    cudnn.benchmark = True


    var_sum = [var.sum().item() for var in model.parameters() if var.requires_grad]
    # for name, var in enumerate(model.parameters()):
    #     if var.requires_grad:
    #         print('name', var)
    var_cnt = len(var_sum)
    var_sum = np.sum(var_sum)

    print("Initial variables' sum: {:.3f}, avg: {:.3f}".format(var_sum, var_sum/var_cnt))


    # epi_loss = get_loss()
    epi_accuracy = get_accuracy()


    # writer = SummaryWriter('./log')
    start = time.time()



    model.eval()
    label_acc, scores_acc = online_test(model,args, epi_accuracy)
save_name = '/data/yanzhengmao/projects/LFDepth/pytorch/epi_1/'
def online_test(model,args, epi_accuracy):
    pad_2_full_size_ = pad_2_full_size()
    label2disp_ = label2disp()
    prob2disp_ = prob2disp()
    c_pre_list, lb_pre_list, raw_net_output, logits_out = [], [], [], []
    max_of_4_list = []
    infer_list = []
    start = time.time()
    img_hei, img_wid, _ = 512, 512, 3
    bp007 = BadPix(0.07)
    with_gt = True
    save_result = False

    scene_name = ['backgammon', 'boxes', 'cotton', 'dino', 'dots', 'pyramids', 'sideboard','stripes']
    valid_hei, valid_wid = 492, 492
    scores = 0
    disp_fuse_scores = 0
    max_of_4_scores = 0
    ava_label_acc = 0
    if args.rank == 0:
        for scene in scene_name:
            print(scene)
            dataloader_test = BtsDataLoader(args, 'test', 128, scene)
            c_pre_list, lb_pre_list, raw_net_output, logits_out = [], [], [], []
            max_of_4_list = []
            infer_list = []
            bp007=BadPix(0.07)
            for step, eval_sample_batched in enumerate(tqdm(dataloader_test.data)):
                with torch.no_grad():
                    valdata_batch_0d =   eval_sample_batched['traindata_batch_0d'].cuda().float()
                    valdata_batch_90d =  eval_sample_batched['traindata_batch_90d'].cuda().float()
                    valdata_batch_45d =  eval_sample_batched['traindata_batch_45d'].cuda().float()
                    valdata_batch_m45d = eval_sample_batched['traindata_batch_m45d'].cuda().float()

                    tra_annotations = eval_sample_batched['tra_annotations'].cuda()
                    # scene_name = eval_sample_batched['scene_name']
                    # print(scene_name[0][0])
                    # print(type(scene_name[0][0]))

                    img = torch.stack([valdata_batch_0d, valdata_batch_90d, valdata_batch_45d, valdata_batch_m45d], dim=1)

                    output = model(img, args.att_rank)
                    # output, pred_list, epi_4_smax, logits_smax, max_of_4_smax = model(img, args.att_rank)


                    label_pre, correct_pre, accuracy = epi_accuracy(output, tra_annotations)
                    # new_out = [label_pre, logits_smax, max_of_4_smax,correct_pre, epi_4_smax]
                    lb_pre_list.append(label_pre)
                    # raw_net_output.append(logits_smax)
                    # max_of_4_list.append(max_of_4_smax)

                    if with_gt:
                        c_pre_list.append(correct_pre)
                    # if save_result:
                    #     infer_list.append(epi_4_smax)
            lb_pre_list = torch.stack(lb_pre_list, dim=0).reshape((valid_hei, valid_wid))
            # raw_net_output = torch.stack(raw_net_output, dim = 0).float().reshape((valid_hei, valid_wid, 229))
            # max_of_4_list = torch.stack(max_of_4_list, dim=0).float().reshape(raw_net_output.shape)
            lb_pre_list = pad_2_full_size_(lb_pre_list, img_hei, img_wid, prob_distribution=False)
            # raw_net_output = pad_2_full_size_(raw_net_output, img_hei, img_wid, prob_distribution=True).cuda()
            # max_of_4_list = pad_2_full_size_(max_of_4_list, img_hei, img_wid, prob_distribution=True).cuda()
            disp_pre = label2disp_(lb_pre_list)
            # disp_fuse, confidence_fuse = prob2disp_(raw_net_output, ret_confidence=True)
            # max_of_4_fuse, max_of_4_conf = prob2disp_(max_of_4_list, ret_confidence=True)
            end = time.time()
            run_time = end - start
            kv_dict = {}
            # if save_result:
            #     img_dir = ('{}/{}').format(raw_dir, imgs_list[idx])
            #     os.mkdir(img_dir)
            print(scene)
            # EVAL_ROOT = '/home/yzm/project/TransDepth-main/evaluation_toolkit/data'
            if with_gt:
                # category = my_misc.infer_scene_category(scene)
                # scene = my_misc.get_scene(scene, category, data_path=EVAL_ROOT)
                c_pre_list = torch.stack(c_pre_list, dim=0).reshape((valid_hei, valid_wid))
                # lb_err_pre = c_pre_list.astype(np.uint8)
                lb_acc = torch.mean(c_pre_list)
                ava_label_acc += lb_acc
                # disp_gt = data_reader.load_file_func(data_reader,
                #                                      ('{}/{}/{}').format(img_root, imgs_list[idx], VAL_TEST_PATH),
                #                                      DISP_GT_PATH, data_reader.lb_dtype)
                # disp_err_pre, value_acc, vacc_no_mask = self.error_acc(disp_pre, disp_gt)
                # disp_err_fuse, fuse_acc, facc_no_mask = self.error_acc(disp_fuse, disp_gt)
                # max_of_4_err, max_of_4_acc, max_of_4_no_mask = self.error_acc(max_of_4_fuse, disp_gt)
                # pre_err = 100 - eval_tools.compute_scores(scene, [bp007], disp_pre)[bp007.get_id()]['value']
                # disp_fuse = disp_fuse.detach().cpu().numpy()
                # max_of_4_fuse = max_of_4_fuse.detach().cpu().numpy().squeeze()
                # fuse_err = 100 - eval_tools.compute_scores(scene, [bp007], disp_fuse)[bp007.get_id()]['value']
                # m4_err = 100 - eval_tools.compute_scores(scene, [bp007], max_of_4_fuse)[bp007.get_id()]['value']
                disp_pre = disp_pre.detach().cpu().numpy().squeeze()
                # disp_fuse = disp_fuse.detach().cpu().numpy().squeeze()
                # max_of_4_fuse = max_of_4_fuse.detach().cpu().numpy().squeeze()
                pre_err = 100 - get_scores_file_by_name_clear(disp_pre, scene)
                # fuse_err = 100 - get_scores_file_by_name_clear(disp_fuse, scene)
                # m4_err = 100 - get_scores_file_by_name_clear(max_of_4_fuse, scene)
                scores += pre_err
                # disp_fuse_scores += fuse_err
                # max_of_4_scores += m4_err

                # fuse_err = 100 - get_scores_file_by_name_clear(disp_fuse, scene)
                # m4_err = 100 - get_scores_file_by_name_clear(max_of_4_fuse, scene)
                print(('Label accuracy: {:.4f}').format(lb_acc))
                print(('Value({}) accuracy: {:.3f}').format(0.07, pre_err))
                # print(('Fuse neighbour: {:.3f}').format(fuse_err))
                # print(('max_of_4: {:.3f}').format(m4_err))
                print(('{:.2f} s').format(run_time))
                print('')
                filename_cmap_png = save_name +  str(scene) + '.png'
                plt.imsave(filename_cmap_png, (disp_pre))


        print('average_score: {:5f}'.format((scores / 8)))
        # print('average_disp_fuse_score: {:5f}'.format((disp_fuse_scores/8)))
        # print('average_max_of_4_score: {:5f}'.format((max_of_4_scores/8)))
        print('average_label_accuracy: {:5f}'.format((ava_label_acc / 8)))
    return ava_label_acc / 8, scores / 8



def main():

    if args.mode != 'train':
        print('bts_main.py is only for training. Use bts_test.py instead.')
        return -1

    model_filename = args.model_name + '.py'
    args.model_name=args.model_name + '_rank_'+str(args.att_rank)
    command = 'mkdir -p ' + args.log_directory + '/' + args.model_name
    os.system(command)

    args_out_path = args.log_directory + '/' + args.model_name + '/' + sys.argv[1]
    command = 'cp ' + sys.argv[1] + ' ' + args_out_path
    os.system(command)

    if args.checkpoint_path == '':
        model_out_path = args.log_directory + '/' + args.model_name + '/' + model_filename
        command = 'cp bts.py ' + model_out_path
        os.system(command)
        aux_out_path = args.log_directory + '/' + args.model_name + '/.'
        command = 'cp bts_main.py ' + aux_out_path
        os.system(command)
        command = 'cp bts_dataloader.py ' + aux_out_path
        os.system(command)
    else:
        loaded_model_dir = os.path.dirname(args.checkpoint_path)
        loaded_model_name = os.path.basename(loaded_model_dir)
        loaded_model_filename = loaded_model_name + '.py'

        model_out_path = args.log_directory + '/' + args.model_name + '/' + model_filename
        command = 'cp ' + loaded_model_dir + '/' + loaded_model_filename + ' ' + model_out_path
        os.system(command)

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