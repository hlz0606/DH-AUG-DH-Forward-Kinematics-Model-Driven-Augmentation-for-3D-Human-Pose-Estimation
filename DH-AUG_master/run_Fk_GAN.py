from __future__ import print_function, absolute_import, division

import datetime
import os
import os.path as path

import numpy as np
import torch
import torch.nn as nn

from function_baseline.model_pos_preparation import model_pos_preparation
from function_aug.config import get_parse_args
from function_aug.data_preparation import data_preparation
from function_aug.dataloader_update import dataloader_update
from function_aug.model_pos_eval import evaluate_posenet
from function_aug.model_pos_train import train_posenet
from utils.gan_utils import Sample_from_Pool
from utils.log import Logger
from utils.utils import save_ckpt, Summary, get_scheduler
from torch.utils.data import DataLoader

from pylab import *
import loguru

import zipfile
import h5py
from glob import glob
from shutil import rmtree
import torch.nn.functional as F
import torch.optim as optim
import os
import sys
import errno
import cdflib

import copy

import matplotlib
#matplotlib.use("Qt5Agg")
#matplotlib.use('Agg')
from function_aug import config
from models_Fk_GAN import special_operate
#import models_Fk_GAN.myMainWindow

import datetime
import random
from models_Fk_GAN.model_fk_gan_train import traditional_solutions_FK_generator
from models_Fk_GAN import myMainWindow
from models_Fk_GAN.forward_kinematics_DH_model import Forward_Kinematics_DH_Model
from models_Fk_GAN.model_fk_gan_train import my_get_poseFk_model, GAN_solutions_FK_generator, \
        video_mode_my_get_poseFk_model
from models_Fk_GAN.video_mode_operate import video_mode_fk_data_preparation, video_mode_train_posenet, \
        video_mode_evaluate_posenet, video_mode_dataloader_update, GAN_dataSet_video_mode_train_posenet, \
        video_receptive_field
from utils.loss import mpjpe, p_mpjpe, compute_PCK, compute_AUC
from models_Fk_GAN.video_GAN_fun import video_mode_GAN_solutions_FK_generator


def single_frame_mode_main(args, FK_DH_Class, data_dict, train_subjects):
    print('==> Using settings {}'.format(args))
    device = torch.device("cuda")

    print('==> Loading dataset...')
    data_dict = data_dict

    print("==> Creating PoseNet model...")
    model_pos = model_pos_preparation(args, data_dict['dataset'], device)
    model_pos_eval = model_pos_preparation(args, data_dict['dataset'], device)  #

    posenet_optimizer = torch.optim.Adam(model_pos.parameters(), lr=args.lr_p)

    posenet_lr_scheduler = get_scheduler(posenet_optimizer, policy='lambda', nepoch_fix=0,
                                         nepoch=args.epochs)

    poseFk_dict = my_get_poseFk_model(args, data_dict['dataset'], FK_DH_Class)

    criterion = nn.MSELoss(reduction='mean').to(device)

    args.checkpoint = path.join(args.checkpoint, args.posenet_name, args.keypoints,
                              datetime.datetime.now().isoformat() + '_' + args.note)
    os.makedirs(args.checkpoint, exist_ok=True)
    os.mkdir(path.join(args.checkpoint, 'tmp'))
    print('==> Making checkpoint dir: {}'.format(args.checkpoint))

    logger = Logger(os.path.join(args.checkpoint, 'log.txt'), args)
    logger.record_args(str(model_pos))

    logger.set_names(['epoch', 'lr', 'error_h36m_p1', 'error_h36m_p2',
                          'error_3dhp_p1', 'error_3dhp_p2', 'PCK', 'AUC'])

    #########################################################
    summary = Summary(args.checkpoint)
    writer = summary.create_summary()

    start_epoch = 0
    dhpp1_best = 9999
    s911p1_best = 9999
    lr_now = 0
    h36m_p1 = 0
    h36m_p2 = 0
    dhp_p1 = 0
    dhp_p2 = 0
    PCK = 0
    AUC = 0

    print('args.pretrain： {} args.set_demo_mode：{} '.format(args.pretrain, args.set_demo_mode))
    if args.pretrain == True:
        pass

    else:
        for now_epoch in range(start_epoch, args.epochs + args.additional_train_epoch):

            if args.data_enhancement_method == 'GAN' or args.data_enhancement_method == 'normal':
                dataloader_update(args=args, data_dict=data_dict, device=device)

            if args.data_enhancement_method == 'GAN':
                if args.GAN_whether_use_preAngle == True:  #
                    print('==> GAN use_preAngle')
                GAN_solutions_FK_generator(args, poseFk_dict, data_dict, model_pos, summary, writer, train_subjects)

            elif args.data_enhancement_method == 'normal':
                traditional_solutions_FK_generator(args, FK_DH_Class, data_dict, train_subjects)

            elif args.data_enhancement_method == 'NO_enhance':
                pass

            else:
                raise ('args.data_enhancement_method error')

            if (summary.epoch > args.warmup and args.data_enhancement_method == 'GAN')\
                    or (args.data_enhancement_method == 'normal'):

                train_posenet(model_pos, data_dict['train_fake2d3d_loader'], posenet_optimizer,
                              criterion, device, args)

                h36m_p1, h36m_p2, dhp_p1, dhp_p2, PCK, AUC = evaluate_posenet(args, data_dict, model_pos,
                                                                          model_pos_eval, device,
                                                                          summary, writer, tag='_fake',
                                                                          get_pck_auc=True)
                logger.append([summary.epoch, 0, h36m_p1, h36m_p2, dhp_p1, dhp_p2, PCK, AUC])


                train_posenet(model_pos, data_dict['train_det2d3d_loader'], posenet_optimizer,
                              criterion, device, args)

                h36m_p1, h36m_p2, dhp_p1, dhp_p2, PCK, AUC = evaluate_posenet(args, data_dict, model_pos,
                                                                    model_pos_eval, device,
                                                                    summary, writer, tag='_real',
                                                                    get_pck_auc=True)
            elif args.data_enhancement_method=='NO_enhance':
                train_posenet(model_pos, data_dict['train_det2d3d_loader'], posenet_optimizer,
                              criterion, device, args)

                h36m_p1, h36m_p2, dhp_p1, dhp_p2, PCK, AUC = evaluate_posenet(args, data_dict, model_pos,
                                                                          model_pos_eval, device,
                                                                          summary, writer, tag='_real',
                                                                          get_pck_auc=True)

            ########################
            if now_epoch < args.epochs:
                posenet_lr_scheduler.step()
                lr_now = posenet_optimizer.param_groups[0]['lr']
                print('\nEpoch: %d | LR: %.8f' % (summary.epoch, lr_now))
            else:
                for param_group in posenet_optimizer.param_groups:
                    param_group['lr'] *= args.additional_LR_decay

                lr_now = posenet_optimizer.param_groups[0]['lr']
                print('\nEpoch: %d | LR: %.8f' % (summary.epoch, lr_now))

            logger.append([summary.epoch, lr_now, h36m_p1, h36m_p2, dhp_p1, dhp_p2, PCK, AUC])

            if dhpp1_best is None or dhpp1_best > dhp_p1:
                dhpp1_best = dhp_p1
                logger.record_args("==> Saving checkpoint at epoch '{}', with dhp_p1 {}".format(summary.epoch, dhpp1_best))
                save_ckpt({'epoch': summary.epoch, 'model_pos': model_pos.state_dict()}, args.checkpoint, suffix='best_dhp_p1')

            if s911p1_best is None or s911p1_best > h36m_p1:
                s911p1_best = h36m_p1
                logger.record_args("==> Saving checkpoint at epoch '{}', with s911p1 {}".format(summary.epoch, s911p1_best))
                save_ckpt({'epoch': summary.epoch, 'model_pos': model_pos.state_dict()}, args.checkpoint, suffix='best_h36m_p1')

            summary.summary_epoch_update()

    writer.close()
    logger.close()




def vedio_multi_frame_mode_main(args, FK_DH_Class, data_dict, train_subjects, video_frame_num):
    print('==> Using settings {}'.format(args))
    device = torch.device("cuda")

    print('==> Loading dataset...')
    data_dict = data_dict

    print("==> Creating PoseNet model...")
    model_pos = model_pos_preparation(args, data_dict['dataset'], device)
    model_pos_eval = model_pos_preparation(args, data_dict['dataset'], device, flag='test')

    posenet_optimizer = torch.optim.Adam(model_pos.parameters(), lr=args.lr_p)

    posenet_lr_scheduler = get_scheduler(posenet_optimizer, policy='lambda', nepoch_fix=0,
                                         nepoch=args.epochs)

    poseFk_dict = video_mode_my_get_poseFk_model(args, data_dict['dataset'], FK_DH_Class, video_frame_num)

    criterion = mpjpe

    args.checkpoint = path.join(args.checkpoint, args.posenet_name, args.keypoints,
                              datetime.datetime.now().isoformat() + '_' + args.note)
    os.makedirs(args.checkpoint, exist_ok=True)
    os.mkdir(path.join(args.checkpoint, 'tmp'))
    print('==> Making checkpoint dir: {}'.format(args.checkpoint))

    logger = Logger(os.path.join(args.checkpoint, 'log.txt'), args)
    logger.record_args(str(model_pos))
    logger.set_names(['epoch', 'lr', 'error_h36m_p1', 'error_h36m_p2',
                      'error_3dhp_p1', 'error_3dhp_p2', 'PCK', 'AUC'])

    summary = Summary(args.checkpoint)
    writer = summary.create_summary()

    start_epoch = 0
    dhpp1_best = None
    s911p1_best = None
    lr_now = 0
    h36m_p1 = 0
    h36m_p2 = 0
    dhp_p1 = 0
    dhp_p2 = 0
    PCK = 0
    AUC = 0

    #####################################################################

    if args.pretrain == True:
        h36m_p1, h36m_p2, dhp_p1, dhp_p2, PCK, AUC = video_mode_evaluate_posenet(args, data_dict, model_pos,
                                                                        model_pos_eval, device,
                                                                        summary, writer, tag='_real',
                                                                        get_pck_auc=True)

    else:
        for now_epoch in range(start_epoch, args.epochs +
                                            args.warmup + args.single_dis_warmup_epoch +
                                            args.additional_train_epoch):

            if args.data_enhancement_method == 'GAN':

                video_mode_dataloader_update(args=args, data_dict=data_dict, device=device)

                video_mode_GAN_solutions_FK_generator(args, poseFk_dict, data_dict, model_pos,
                                                      summary, writer, train_subjects)

            else:
                print('==> ')

            if (summary.epoch > (args.warmup + args.single_dis_warmup_epoch) and
                args.data_enhancement_method == 'GAN')\
                   or (args.data_enhancement_method == 'normal'):

                if args.data_enhancement_method == 'GAN':
                    GAN_dataSet_video_mode_train_posenet(model_pos, data_dict['train_fake2d3d_loader'], posenet_optimizer,
                                             criterion, device, args)
                    h36m_p1, h36m_p2, dhp_p1, dhp_p2, PCK, AUC = video_mode_evaluate_posenet(args, data_dict, model_pos,
                                                                        model_pos_eval, device,
                                                                        summary, writer, tag='_fake',
                                                                        get_pck_auc=True)

                    logger.append([summary.epoch, 0, h36m_p1, h36m_p2, dhp_p1, dhp_p2, PCK, AUC])

                video_mode_train_posenet(model_pos, data_dict['train_det2d3d_loader'], posenet_optimizer,
                            criterion, device, args)
                h36m_p1, h36m_p2, dhp_p1, dhp_p2, PCK, AUC = video_mode_evaluate_posenet(args, data_dict, model_pos,
                                                                    model_pos_eval, device,
                                                                    summary, writer, tag='_real',
                                                                    get_pck_auc=True)


                if args.data_enhancement_method == 'GAN':
                    if now_epoch < args.epochs + args.warmup + args.single_dis_warmup_epoch:
                        posenet_lr_scheduler.step()
                        lr_now = posenet_optimizer.param_groups[0]['lr']
                        print('\nEpoch: %d | LR: %.8f' % (summary.epoch, lr_now))
                    else:
                        for param_group in posenet_optimizer.param_groups:
                            param_group['lr'] *= args.additional_LR_decay
                        lr_now = posenet_optimizer.param_groups[0]['lr']
                        print('\nEpoch: %d | LR: %.8f' % (summary.epoch, lr_now))
                else:
                    if now_epoch < args.epochs:
                        posenet_lr_scheduler.step()
                        lr_now = posenet_optimizer.param_groups[0]['lr']
                        print('\nEpoch: %d | LR: %.8f' % (summary.epoch, lr_now))
                    else:
                        for param_group in posenet_optimizer.param_groups:
                            param_group['lr'] *= args.additional_LR_decay

                        lr_now = posenet_optimizer.param_groups[0]['lr']
                        print('\nEpoch: %d | LR: %.8f' % (summary.epoch, lr_now))

            logger.append([summary.epoch, lr_now, h36m_p1, h36m_p2, dhp_p1, dhp_p2, PCK, AUC])

            if dhpp1_best is None or dhpp1_best > dhp_p1:
                dhpp1_best = dhp_p1
                logger.record_args("==> Saving checkpoint at epoch '{}', with dhp_p1 {}".format(summary.epoch, dhpp1_best))
                save_ckpt({'epoch': summary.epoch, 'model_pos': model_pos.state_dict()}, args.checkpoint, suffix='best_dhp_p1')

            if s911p1_best is None or s911p1_best > h36m_p1:
                s911p1_best = h36m_p1
                logger.record_args("==> Saving checkpoint at epoch '{}', with s911p1 {}".format(summary.epoch, s911p1_best))
                save_ckpt({'epoch': summary.epoch, 'model_pos': model_pos.state_dict()}, args.checkpoint, suffix='best_h36m_p1')

            summary.summary_epoch_update()

        writer.close()
        logger.close()



if __name__ == '__main__':

    args = get_parse_args()

    whole_subjects = ['S1', 'S5', 'S6', 'S7', 'S8', 'S9', 'S11']
    train_subjects = []
    if args.s1only:
        train_subjects = ['S1']
    elif args.s1s5only:
        train_subjects = ['S1', 'S5']
    else:
        train_subjects = ['S1', 'S5', 'S6', 'S7', 'S8']

    if args.s1only == True and args.s1s5only == True:
        raise KeyError(' args.s1only and args.s1s5only both set true')

    if train_subjects == []:
        raise 'train_subjects   error'

    test_subjects = ['S9', 'S11']

    ################# GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    print('==> GPU: {0}'.format(args.gpu_id))
    ########################################

    seed = args.random_seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  #
    random.seed(seed)  # Python random module.
    os.environ['PYTHONHASHSEED'] = str(seed) #############
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    ##########################################


    data_dict = {}
    video_frame_num = 0
    if args.single_or_multi_train_mode == 'single':
        data_dict = special_operate.fk_data_preparation(args, r'.')

    elif args.single_or_multi_train_mode == 'multi':
        data_dict, video_frame_num = video_mode_fk_data_preparation(args, r'.')

    else:
        raise "args.single_or_multi_train_mode"

    if data_dict == {}:
        raise 'data_dict'

    dataset = copy.deepcopy(data_dict['dataset'])
    pose_2d_pix = copy.deepcopy(data_dict['keypoints'])

    FK_DH_Class = Forward_Kinematics_DH_Model(args, train_subjects, dataset)

    FK_DH_Class.get_dataSet_3d_and_2d_pose(dataset, pose_2d_pix)

    FK_DH_Class.get_bone_len_from_dataSet()

    FK_DH_Class.get_root_3d_pos_from_dataSet()

    show_FK_point_3d = FK_DH_Class.init_Fk_DH_angle()

    if args.single_or_multi_train_mode == 'single':
        single_frame_mode_main(args, FK_DH_Class, data_dict, train_subjects)

    elif args.single_or_multi_train_mode == 'multi':
        vedio_multi_frame_mode_main(args, FK_DH_Class, data_dict, train_subjects, video_frame_num)

    else:
        raise "args.single_or_multi_train_mode error"



