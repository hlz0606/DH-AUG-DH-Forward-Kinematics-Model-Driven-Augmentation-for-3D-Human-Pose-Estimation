import numpy as np
import torch
import torch.nn as nn

from torch.autograd import Variable
from torch.utils.data import DataLoader

from common.camera import project_to_2d
from common.data_loader import PoseDataSet
from progress.bar import Bar
from utils.gan_utils import get_discriminator_accuracy
from utils.loss import diff_range_loss, rectifiedL2loss
from utils.utils import AverageMeter, set_grad

import time

from common.h36m_dataset import H36M_32_To_16_Table

from common.h36m_dataset import h36m_cameras_extrinsic_params, h36m_cameras_intrinsic_params
from common.camera import *

import copy

from models_baseline.mlp.linear_model import init_weights
from utils.utils import get_scheduler

from models_Fk_GAN.Fk_generator import Fk_Generator, Video_Fk_Generator
from models_Fk_GAN.Fk_discriminator import calc_gradient_penalty, Fk_3D_Discriminator, Fk_2D_Discriminator, \
        Video_motion_Fk_3D_Discriminator, Video_motion_Fk_2D_Discriminator


from models_Fk_GAN.special_operate import my_visual_3D_pos, my_draw_loss_picture, my_visual_2D_pos, \
        my_visual_GAN_video, my_visual_3D_pos_for_parer, my_visual_2D_pos_for_paper
import random
import torch.backends.cudnn as cudnn

from models_Fk_GAN.video_mode_operate import video_receptive_field
from models_Fk_GAN.model_fk_gan_train import train_Fk_discriminator

from torch.utils.data import Dataset
from functools import reduce

import os


class video_mode_PoseDataSet(Dataset):
    def __init__(self, poses_3d, poses_2d, actions, cams, receptive_field):

        assert poses_3d is not None

        self.receptive_field = receptive_field
        self.used_3D_ID = int((receptive_field - 1) / 2)

        self._poses_3d = np.concatenate(poses_3d)
        self._poses_2d = np.concatenate(poses_2d)

        self._actions = reduce(lambda x, y: x + y, actions)    #
        self._cams = np.concatenate(cams)    #

        assert self._poses_3d.shape[0] == self._poses_2d.shape[0]
        assert self._poses_3d.shape[0] == self._cams.shape[0]
        print('Generating {} poses...'.format(len(self._actions)))

    def __getitem__(self, video_index):
        out_pose_3d = self._poses_3d[video_index][self.used_3D_ID]
        out_pose_2d = self._poses_2d[video_index]
        out_action = self._actions[video_index]
        out_cam = self._cams[video_index]

        out_pose_3d = torch.from_numpy(out_pose_3d).float()
        out_pose_2d = torch.from_numpy(out_pose_2d).float()

        return out_cam, out_pose_3d, out_pose_2d

    def __len__(self):
        return len(self._actions)


def video_mode_GAN_solutions_FK_generator(args, poseFk_dict, data_dict, model_pos, summary, writer, train_subjects):

    filter_widths = [int(x) for x in args.architecture.split(',')]
    receptive_field = video_receptive_field(filter_widths)

    device = torch.device("cuda")
    batch_time = AverageMeter()
    data_time = AverageMeter()

    model_G = poseFk_dict['model_G']
    model_d3d = poseFk_dict['model_d3d']
    model_d2d = poseFk_dict['model_d2d']
    model_motion_d3d = poseFk_dict['model_motion_d3d']
    model_motion_d2d = poseFk_dict['model_motion_d2d']

    g_optimizer = poseFk_dict['optimizer_G']
    d3d_optimizer = poseFk_dict['optimizer_d3d']
    d2d_optimizer = poseFk_dict['optimizer_d2d']
    motion_d3d_optimizer = poseFk_dict['optimizer_motion_d3d']
    motion_d2d_optimizer = poseFk_dict['optimizer_motion_d2d']

    torch.set_grad_enabled(True)
    model_G.train()
    model_d3d.train()
    model_d2d.train()
    model_motion_d3d.train()
    model_motion_d2d.train()
    model_pos.train()
    end = time.time()

    tmp_3d_pose_buffer_list = []
    tmp_2d_pose_buffer_list = []
    tmp_camparam_buffer_list = []
    Wasserstein_D_3D = 0
    D_cost_3D = 0
    Wasserstein_D_2D = 0
    D_cost_2D = 0
    G_cost = 0
    flip_Wasserstein_D_3D = 0
    flip_D_cost_3D = 0

    video_motion_Wasserstein_D_3D = 0
    video_motion_D_cost_3D = 0
    video_motion_Wasserstein_D_2D = 0
    video_motion_D_cost_2D = 0
    flip_video_motion_Wasserstein_D_3D = 0
    flip_video_motion_D_cost_3D = 0
    flip_video_motion_Wasserstein_D_2D = 0
    flip_video_motion_D_cost_2D = 0

    visual_real_Pos_v = np.array([])
    visual_fake_Pos = np.array([])
    visual_real_3d_flip = np.array([])
    visual_fake_3d_flip = np.array([])
    visual_inputs_2d = np.array([])
    visual_show_16key_2D_pix_norm = np.array([])
    visual_real_2d_flip = np.array([])
    visual_fake_2d_flip = np.array([])

    adv_loss_3d = 0
    adv_loss_2d = 0
    adv_loss_motion_3d = 0
    adv_loss_motion_2d = 0
    flip_adv_loss_3d = 0
    flip_adv_loss_2d = 0
    flip_adv_loss_motion_3d = 0
    flip_adv_loss_motion_2d = 0

    one = torch.tensor(1, dtype=torch.float)   # 1
    mone = one * -1   # -1

    one = one.to(device)
    mone = mone.to(device)

    bar = Bar('Train pose gan', max=data_dict['target_GAN_loader'].num_batches)

    for i, (cam_param, inputs_3d, inputs_2d) in enumerate(data_dict['target_GAN_loader'].next_epoch()):

        if inputs_3d.shape[0] < args.batch_size:
            continue

        data_time.update(time.time() - end)

        inputs_3d = torch.from_numpy(inputs_3d.astype('float32'))
        cam_param = torch.from_numpy(cam_param.astype('float32'))
        inputs_2d = torch.from_numpy(inputs_2d.astype('float32'))

        inputs_3d, cam_param = inputs_3d.to(device), cam_param.to(device)
        inputs_2d = inputs_2d.to(device)

        model_G.GAN_generator_get_bone_length(inputs_3d)

        inputs_3d = inputs_3d.view(-1, 16, 3)    #
        inputs_3d = inputs_3d.to(device)

        cam_param = torch.unsqueeze(cam_param, 1).repeat(1, receptive_field, 1)
        real_cam_R = cam_param[:, :, 9:13]    #
        real_cam_T = cam_param[:, :, 13:16]

        real_pos_3d_world = \
            video_GAN_torch_camera_to_world(inputs_3d, R=real_cam_R, t=real_cam_T)  #

        real_pos_3d_world[:, :, :] = real_pos_3d_world[:, :, :] - real_pos_3d_world[:, :1, :]

        set_grad([model_d3d], True)
        set_grad([model_d2d], True)
        set_grad([model_motion_d3d], True)
        set_grad([model_motion_d2d], True)
        set_grad([model_G], False)
        set_grad([model_pos], False)

        real_Pos_v = torch.autograd.Variable(real_pos_3d_world)
        real_Pos_v = real_Pos_v.to(device)

        noise = torch.randn(args.batch_size, 128)
        noise = noise.to(device)
        noisev = torch.autograd.Variable(noise)
        fake_Pos = torch.autograd.Variable(model_G(noisev).data)
        fake_Pos = fake_Pos.to(device)
        fake_Pos = fake_Pos.view(-1, 16, 3)
        fake_3D_root = torch.clone(fake_Pos[:, :1, :])
        fake_Pos = fake_Pos[:, :, :] - fake_Pos[:, :1, :]

        fake_Pos = fake_Pos.view(-1, 16 * 3)
        real_Pos_v = real_Pos_v.view(-1, 16 * 3)

        visual_real_Pos_v = copy.deepcopy(real_Pos_v.detach().cpu().numpy())
        visual_fake_Pos = copy.deepcopy(fake_Pos.detach().cpu().numpy())

        Wasserstein_D_3D, D_cost_3D = train_Fk_discriminator(model_d3d, torch.clone(real_Pos_v),
                                                    torch.clone(fake_Pos), summary,
                                                    writer, writer_name='Fk_d3d',
                                                    optimizerD=d3d_optimizer, args=args, one=one, mone=mone)

        if summary.epoch >= args.single_dis_warmup_epoch:
            video_motion_Wasserstein_D_3D, video_motion_D_cost_3D = \
                    train_Fk_discriminator(model_motion_d3d, torch.clone(real_Pos_v), torch.clone(fake_Pos), summary,
                                              writer, writer_name='motion_Fk_d3d', optimizerD=motion_d3d_optimizer,
                                           args=args, one=one, mone=mone, dis_mode='motion')

        if args.GAN_video_playback_input == True:
            real_Pos_v = real_Pos_v.view(-1, receptive_field, 16 * 3)
            fake_Pos = fake_Pos.view(-1, receptive_field, 16 * 3)
            back_real_Pos_v = torch.clone(torch.flip(real_Pos_v, dims=[1]))
            back_fake_Pos = torch.clone(torch.flip(fake_Pos, dims=[1]))

            if summary.epoch >= args.single_dis_warmup_epoch:
                back_video_motion_Wasserstein_D_3D, back_video_motion_D_cost_3D = \
                    train_Fk_discriminator(model_motion_d3d, torch.clone(back_real_Pos_v),
                                           torch.clone(back_fake_Pos), summary,
                                           writer, writer_name='back_motion_Fk_d3d', optimizerD=motion_d3d_optimizer,
                                           args=args, one=one, mone=mone, dis_mode='motion')
                video_motion_Wasserstein_D_3D = (video_motion_Wasserstein_D_3D + back_video_motion_Wasserstein_D_3D) / 2
                video_motion_D_cost_3D = (video_motion_D_cost_3D + back_video_motion_D_cost_3D) / 2

        fake_Pos = fake_Pos.view(-1, 16, 3)
        real_Pos_v = real_Pos_v.view(-1, 16, 3)

        if args.flip_GAN_model_input == True:
            joints_left = [4, 5, 6, 10, 11, 12]
            joints_right = [1, 2, 3, 13, 14, 15]
            out_left = [4, 5, 6, 10, 11, 12]
            out_right = [1, 2, 3, 13, 14, 15]

            real_3d_flip = real_Pos_v.detach().clone()
            real_3d_flip[:, :, 0] *= -1
            real_3d_flip[:, out_left + out_right, :] = real_3d_flip[:, out_right + out_left, :]

            fake_3d_flip = fake_Pos.detach().clone()
            fake_3d_flip[:, :, 0] *= -1
            fake_3d_flip[:, out_left + out_right, :] = fake_3d_flip[:, out_right + out_left, :]

            visual_real_3d_flip = copy.deepcopy(real_3d_flip.detach().cpu().numpy())
            visual_fake_3d_flip = copy.deepcopy(fake_3d_flip.detach().cpu().numpy())

            flip_Wasserstein_D_3D, flip_D_cost_3D = \
                    train_Fk_discriminator(model_d3d, torch.clone(real_3d_flip),
                                            torch.clone(fake_3d_flip), summary,writer, writer_name='Fk_d3d',
                                            optimizerD=d3d_optimizer, args=args, one=one, mone=mone)

            if summary.epoch >= args.single_dis_warmup_epoch:
                flip_video_motion_Wasserstein_D_3D, flip_video_motion_D_cost_3D = \
                    train_Fk_discriminator(model_motion_d3d, torch.clone(real_3d_flip),
                                           torch.clone(fake_3d_flip), summary,
                                           writer, writer_name='motion_Fk_d3d', optimizerD=motion_d3d_optimizer,
                                           args=args, one=one, mone=mone, dis_mode='motion')

            if args.GAN_video_playback_input == True:
                real_3d_flip = real_3d_flip.view(-1, receptive_field, 16 * 3)
                fake_3d_flip = fake_3d_flip.view(-1, receptive_field, 16 * 3)
                back_real_3d_flip = torch.clone(torch.flip(real_3d_flip, dims=[1]))
                back_fake_3d_flip = torch.clone(torch.flip(fake_3d_flip, dims=[1]))

                if summary.epoch >= args.single_dis_warmup_epoch:
                    back_flip_video_motion_Wasserstein_D_3D, back_flip_video_motion_D_cost_3D = \
                        train_Fk_discriminator(model_motion_d3d, torch.clone(back_real_3d_flip),
                                               torch.clone(back_fake_3d_flip), summary,
                                               writer, writer_name='back_flip_motion_Fk_d3d',
                                               optimizerD=motion_d3d_optimizer,
                                               args=args, one=one, mone=mone, dis_mode='motion')
                    flip_video_motion_Wasserstein_D_3D = \
                            (flip_video_motion_Wasserstein_D_3D + back_flip_video_motion_Wasserstein_D_3D) / 2
                    flip_video_motion_D_cost_3D = (flip_video_motion_D_cost_3D + back_flip_video_motion_D_cost_3D) / 2

            fake_3d_flip = fake_3d_flip.view(-1, 16 * 3)
            real_3d_flip = real_3d_flip.view(-1, 16 * 3)

            Wasserstein_D_3D = (Wasserstein_D_3D + flip_Wasserstein_D_3D) / 2
            D_cost_3D = (D_cost_3D + flip_D_cost_3D) / 2
            video_motion_Wasserstein_D_3D = (video_motion_Wasserstein_D_3D + flip_video_motion_Wasserstein_D_3D) / 2
            video_motion_D_cost_3D = (video_motion_D_cost_3D + flip_video_motion_D_cost_3D) / 2

        train_subjects_id = np.random.randint(0, len(train_subjects))
        choice_subject = train_subjects[train_subjects_id]

        cam_id = np.random.randint(0, 4)

        cam_R = np.array(h36m_cameras_extrinsic_params[choice_subject][cam_id]['orientation']).reshape(1, 4)
        cam_t = np.array(
            h36m_cameras_extrinsic_params[choice_subject][cam_id]['translation']).reshape(1, 3) / 1000.0
        res_w = float(h36m_cameras_intrinsic_params[cam_id]['res_w'])
        res_h = float(h36m_cameras_intrinsic_params[cam_id]['res_h'])

        f = np.array(h36m_cameras_intrinsic_params[cam_id]['focal_length']) / res_w * 2.0
        c = normalize_screen_coordinates(np.array(h36m_cameras_intrinsic_params[cam_id]['center']),
                                         w=res_w, h=res_h).astype('float32')
        k = np.array(h36m_cameras_intrinsic_params[cam_id]['radial_distortion'])
        p = np.array(h36m_cameras_intrinsic_params[cam_id]['tangential_distortion'])
        cam_para_temp = np.zeros((args.batch_size, 9))
        cam_para_temp[:, :2] = f
        cam_para_temp[:, 2:4] = c
        cam_para_temp[:, 4:7] = k
        cam_para_temp[:, 7:] = p

        cam_R = torch.tensor(cam_R, dtype=torch.float32).to(device)
        cam_t = torch.tensor(cam_t, dtype=torch.float32).to(device)
        fake_Pos = fake_Pos.view(-1, 16, 3)
        fake_Pos = fake_Pos[:, :, :] + fake_3D_root
        cam_para_temp = (torch.tensor(cam_para_temp, dtype=torch.float32).view(-1, 9)).to(device)
        res_w = torch.tensor(res_w, dtype=torch.float32).to(device)
        res_h = torch.tensor(res_h, dtype=torch.float32).to(device)

        pos_3d_cam = GAN_torch_world_to_camera(fake_Pos, R=torch.clone(cam_R), t=torch.clone(cam_t))

        cam_para_temp = torch.unsqueeze(cam_para_temp, dim=1).repeat(1, receptive_field, 1)
        cam_para_temp = cam_para_temp.view(-1, cam_para_temp.shape[-1])

        show_16key_2D_picture = project_to_2d(pos_3d_cam, cam_para_temp)

        show_16key_2D_pix_norm = show_16key_2D_picture

        show_16key_2D_pix_norm = show_16key_2D_pix_norm.to(device)

        visual_inputs_2d = copy.deepcopy(inputs_2d.detach().cpu().numpy())
        visual_show_16key_2D_pix_norm = copy.deepcopy(show_16key_2D_pix_norm.detach().cpu().numpy())

        Wasserstein_D_2D, D_cost_2D = \
            train_Fk_discriminator(model_d2d, inputs_2d, show_16key_2D_pix_norm, summary,
                                    writer, writer_name='d2d', optimizerD=d2d_optimizer,
                                    args=args, one=one, mone=mone)

        if summary.epoch >= args.single_dis_warmup_epoch:
            video_motion_Wasserstein_D_2D, video_motion_D_cost_2D = \
                train_Fk_discriminator(model_motion_d2d, torch.clone(inputs_2d),
                                        torch.clone(show_16key_2D_pix_norm), summary,
                                        writer, writer_name='motion_d2d', optimizerD=motion_d2d_optimizer,
                                        args=args, one=one, mone=mone)


        if args.GAN_video_playback_input == True:
            inputs_2d = inputs_2d.view(-1, receptive_field, 16 * 2)
            show_16key_2D_pix_norm = show_16key_2D_pix_norm.view(-1, receptive_field, 16 * 2)
            back_inputs_2d = torch.clone(torch.flip(inputs_2d, dims=[1]))
            back_show_16key_2D_pix_norm = torch.clone(torch.flip(show_16key_2D_pix_norm, dims=[1]))

            if summary.epoch >= args.single_dis_warmup_epoch:
                back_video_motion_Wasserstein_D_2D, back_video_motion_D_cost_2D = \
                    train_Fk_discriminator(model_motion_d2d, torch.clone(back_inputs_2d),
                                           torch.clone(back_show_16key_2D_pix_norm), summary,
                                           writer, writer_name='back_motion_d2d', optimizerD=motion_d2d_optimizer,
                                           args=args, one=one, mone=mone)
                video_motion_Wasserstein_D_2D = \
                        (video_motion_Wasserstein_D_2D + back_video_motion_Wasserstein_D_2D) / 2
                video_motion_D_cost_2D = (video_motion_D_cost_2D + back_video_motion_D_cost_2D)/2

        if args.flip_GAN_model_input == True:
            joints_left = [4, 5, 6, 10, 11, 12]
            joints_right = [1, 2, 3, 13, 14, 15]
            out_left = [4, 5, 6, 10, 11, 12]
            out_right = [1, 2, 3, 13, 14, 15]

            inputs_2d = inputs_2d.view(-1, 16, 2)
            show_16key_2D_pix_norm = show_16key_2D_pix_norm.view(-1, 16, 2)

            real_2d_flip = inputs_2d.detach().clone()
            real_2d_flip[:, :, 0] *= -1
            real_2d_flip[:, out_left + out_right, :] = real_2d_flip[:, out_right + out_left, :]

            fake_2d_flip = show_16key_2D_pix_norm.detach().clone()
            fake_2d_flip[:, :, 0] *= -1
            fake_2d_flip[:, out_left + out_right, :] = fake_2d_flip[:, out_right + out_left, :]

            visual_real_2d_flip = copy.deepcopy(real_2d_flip.detach().cpu().numpy())
            visual_fake_2d_flip = copy.deepcopy(fake_2d_flip.detach().cpu().numpy())

            real_2d_flip = real_2d_flip.view(-1, receptive_field, 16 * 2)
            fake_2d_flip = fake_2d_flip.view(-1, receptive_field, 16 * 2)

            flip_Wasserstein_D_2D, flip_D_cost_2D = \
                train_Fk_discriminator(model_d2d, real_2d_flip, fake_2d_flip, summary,
                                       writer, writer_name='d2d', optimizerD=d2d_optimizer,
                                       args=args, one=one, mone=mone)

            if summary.epoch >= args.single_dis_warmup_epoch:
                flip_video_motion_Wasserstein_D_2D, flip_video_motion_D_cost_2D = \
                    train_Fk_discriminator(model_motion_d2d, real_2d_flip, fake_2d_flip, summary,
                                                writer, writer_name='d2d', optimizerD=motion_d2d_optimizer,
                                                args=args, one=one, mone=mone)

            if args.GAN_video_playback_input == True:
                real_2d_flip = real_2d_flip.view(-1, receptive_field, 16 * 2)
                fake_2d_flip = fake_2d_flip.view(-1, receptive_field, 16 * 2)
                back_real_2d_flip = torch.clone(torch.flip(real_2d_flip, dims=[1]))
                back_fake_2d_flip = torch.clone(torch.flip(fake_2d_flip, dims=[1]))

                if summary.epoch >= args.single_dis_warmup_epoch:
                    back_flip_video_motion_Wasserstein_D_2D, back_flip_video_motion_D_cost_2D = \
                        train_Fk_discriminator(model_motion_d2d, torch.clone(back_real_2d_flip),
                                               torch.clone(back_fake_2d_flip), summary,
                                               writer, writer_name='back_flip_motion_d2d',
                                               optimizerD=motion_d2d_optimizer,
                                               args=args, one=one, mone=mone)
                    flip_video_motion_Wasserstein_D_2D = \
                        (flip_video_motion_Wasserstein_D_2D + back_flip_video_motion_Wasserstein_D_2D) / 2
                    flip_video_motion_D_cost_2D = (flip_video_motion_D_cost_2D + back_flip_video_motion_D_cost_2D) / 2

            Wasserstein_D_2D = (Wasserstein_D_2D + flip_Wasserstein_D_2D) / 2
            D_cost_2D = (D_cost_2D + flip_D_cost_2D) / 2
            video_motion_Wasserstein_D_2D = (video_motion_Wasserstein_D_2D + flip_video_motion_Wasserstein_D_2D) / 2
            video_motion_D_cost_2D = (video_motion_D_cost_2D + flip_video_motion_D_cost_2D) / 2

        summary.summary_train_discrim_update()

        if summary.train_iter_num % 5 == 4:
            set_grad([model_d3d], False)
            set_grad([model_d2d], False)
            set_grad([model_motion_d3d], False)
            set_grad([model_motion_d2d], False)
            set_grad([model_G], True)
            set_grad([model_pos], False)

            model_G.zero_grad()

            noise = torch.randn(args.batch_size, 128)
            noise = noise.to(device)

            noisev = torch.autograd.Variable(noise)

            fake_Pos_3d_world = model_G(noisev)
            fake_Pos_3d_world = (fake_Pos_3d_world.view(-1, 16, 3)).to(device)

            fake_pos_3d_cam = GAN_torch_world_to_camera(fake_Pos_3d_world, R=cam_R, t=cam_t)

            fake_pos_2d_norm = project_to_2d(fake_pos_3d_cam, cam_para_temp)
            fake_pos_2d_norm = fake_pos_2d_norm.to(device)

            fake_Pos_3d_world = fake_Pos_3d_world.view(-1, 16, 3)
            fake_Pos_3d_world = fake_Pos_3d_world[:, :, :] - fake_Pos_3d_world[:, :1, :]
            adv_loss_3d = model_d3d(fake_Pos_3d_world)
            adv_loss_3d = adv_loss_3d.mean()
            adv_loss_3d = adv_loss_3d.to(device)

            adv_loss_2d = model_d2d(fake_pos_2d_norm)
            adv_loss_2d = adv_loss_2d.mean()
            adv_loss_2d = adv_loss_2d.to(device)

            if summary.epoch >= args.single_dis_warmup_epoch:
                adv_loss_motion_3d = model_motion_d3d(torch.clone(fake_Pos_3d_world))
                adv_loss_motion_3d = adv_loss_motion_3d.mean()
                adv_loss_motion_3d = adv_loss_motion_3d.to(device)

            if summary.epoch >= args.single_dis_warmup_epoch:
                adv_loss_motion_2d = model_motion_d2d(fake_pos_2d_norm)
                adv_loss_motion_2d = adv_loss_motion_2d.mean()
                adv_loss_motion_2d = adv_loss_motion_2d.to(device)

            if args.GAN_video_playback_input == True:

                fake_Pos_3d_world = fake_Pos_3d_world.view(-1, receptive_field, 16 * 2)
                fake_pos_2d_norm = fake_pos_2d_norm.view(-1, receptive_field, 16 * 2)
                back_fake_Pos_3d_world = torch.clone(torch.flip(fake_Pos_3d_world, dims=[1]))
                back_fake_pos_2d_norm = torch.clone(torch.flip(fake_pos_2d_norm, dims=[1]))

                if summary.epoch >= args.single_dis_warmup_epoch:  #
                    back_adv_loss_motion_3d = model_motion_d3d(torch.clone(back_fake_Pos_3d_world))
                    back_adv_loss_motion_3d = back_adv_loss_motion_3d.mean()
                    back_adv_loss_motion_3d = back_adv_loss_motion_3d.to(device)
                    adv_loss_motion_3d = (adv_loss_motion_3d + back_adv_loss_motion_3d) / 2

                if summary.epoch >= args.single_dis_warmup_epoch:
                    back_adv_loss_motion_2d = model_motion_d2d(back_fake_pos_2d_norm)
                    back_adv_loss_motion_2d = back_adv_loss_motion_2d.mean()
                    back_adv_loss_motion_2d = back_adv_loss_motion_2d.to(device)
                    adv_loss_motion_2d = (adv_loss_motion_2d + back_adv_loss_motion_2d) / 2

            if args.flip_GAN_model_input == True:
                joints_left = [4, 5, 6, 10, 11, 12]
                joints_right = [1, 2, 3, 13, 14, 15]
                out_left = [4, 5, 6, 10, 11, 12]
                out_right = [1, 2, 3, 13, 14, 15]

                fake_Pos_3d_world = fake_Pos_3d_world.view(-1, 16, 3)
                fake_pos_2d_norm = fake_pos_2d_norm.view(-1, 16, 2)

                fake_3d_flip = fake_Pos_3d_world.detach().clone()
                fake_3d_flip[:, :, 0] *= -1
                fake_3d_flip[:, out_left + out_right, :] = fake_3d_flip[:, out_right + out_left, :]

                fake_2d_flip = fake_pos_2d_norm.detach().clone()
                fake_2d_flip[:, :, 0] *= -1
                fake_2d_flip[:, out_left + out_right, :] = fake_2d_flip[:, out_right + out_left, :]

                flip_adv_loss_3d = model_d3d(fake_3d_flip)
                flip_adv_loss_3d = flip_adv_loss_3d.mean()
                flip_adv_loss_3d = flip_adv_loss_3d.to(device)

                flip_adv_loss_2d = model_d2d(fake_2d_flip)
                flip_adv_loss_2d = flip_adv_loss_2d.mean()
                flip_adv_loss_2d = flip_adv_loss_2d.to(device)

                if summary.epoch >= args.single_dis_warmup_epoch:
                    flip_adv_loss_motion_3d = model_motion_d3d(torch.clone(torch.clone(fake_3d_flip)))
                    flip_adv_loss_motion_3d = flip_adv_loss_motion_3d.mean()
                    flip_adv_loss_motion_3d = flip_adv_loss_motion_3d.to(device)

                if summary.epoch >= args.single_dis_warmup_epoch:
                    flip_adv_loss_motion_2d = model_motion_d2d(fake_2d_flip)
                    flip_adv_loss_motion_2d = flip_adv_loss_motion_2d.mean()
                    flip_adv_loss_motion_2d = flip_adv_loss_motion_2d.to(device)

                if args.GAN_video_playback_input == True:

                    fake_3d_flip = fake_3d_flip.view(-1, receptive_field, 16 * 2)
                    fake_2d_flip = fake_2d_flip.view(-1, receptive_field, 16 * 2)
                    back_fake_3d_flip = torch.clone(torch.flip(fake_3d_flip, dims=[1]))
                    back_fake_2d_flip = torch.clone(torch.flip(fake_2d_flip, dims=[1]))

                    if summary.epoch >= args.single_dis_warmup_epoch:
                        back_flip_adv_loss_motion_3d = model_motion_d3d(torch.clone(back_fake_3d_flip))
                        back_flip_adv_loss_motion_3d = back_flip_adv_loss_motion_3d.mean()
                        back_flip_adv_loss_motion_3d = back_flip_adv_loss_motion_3d.to(device)
                        flip_adv_loss_motion_3d = (flip_adv_loss_motion_3d + back_flip_adv_loss_motion_3d) / 2

                    if summary.epoch >= args.single_dis_warmup_epoch:
                        back_flip_adv_loss_motion_2d = model_motion_d2d(back_fake_2d_flip)
                        back_flip_adv_loss_motion_2d = back_flip_adv_loss_motion_2d.mean()
                        back_flip_adv_loss_motion_2d = back_flip_adv_loss_motion_2d.to(device)
                        flip_adv_loss_motion_2d = (flip_adv_loss_motion_2d + back_flip_adv_loss_motion_2d) / 2

                adv_loss_3d = (adv_loss_3d + flip_adv_loss_3d) / 2
                adv_loss_2d = (adv_loss_2d + flip_adv_loss_2d) / 2
                adv_loss_motion_3d = (adv_loss_motion_3d + flip_adv_loss_motion_3d) / 2
                adv_loss_motion_2d = (adv_loss_motion_2d + flip_adv_loss_motion_2d) / 2


            if summary.epoch >= args.single_dis_warmup_epoch:
                gen_loss = adv_loss_3d * args.GAN_3d_loss_weight + \
                           adv_loss_2d * args.GAN_2d_loss_weight + \
                           adv_loss_motion_3d * args.GAN_3d_motion_loss_weight + \
                           adv_loss_motion_2d * args.GAN_2d_motion_loss_weight
            else:
                gen_loss = adv_loss_3d * args.GAN_3d_loss_weight + \
                           adv_loss_2d * args.GAN_2d_loss_weight

            gen_loss.backward(mone)    ##
            G_cost = -gen_loss

            g_optimizer.step()

            summary.summary_train_fakepose_iter_num_update()

        pos_3d_cam = pos_3d_cam.view(args.batch_size, receptive_field, 16, 3)
        show_16key_2D_pix_norm = show_16key_2D_pix_norm.view(args.batch_size, receptive_field, 16, 2)
        cam_para_temp = cam_para_temp.view(args.batch_size, receptive_field, 9)

        tmp_3d_pose_buffer_list.append(pos_3d_cam.detach().cpu().numpy())    # [675, 16, 3]
        tmp_2d_pose_buffer_list.append(show_16key_2D_pix_norm.detach().cpu().numpy())    # [675, 16, 2]
        tmp_camparam_buffer_list.append(cam_para_temp.detach().cpu().numpy())    # [675, 9]

        summary.summary_train_iter_num_update()

        batch_time.update(time.time() - end)
        end = time.time()

        bar.suffix = '({batch}/{size}) Data: {data:.6f}s | Batch: {bt:.3f}s | Total: {ttl:} | ETA: {eta:} ' \
            .format(batch=i + 1, size=data_dict['target_GAN_loader'].num_batches,
                    data=data_time.avg, bt=batch_time.avg,
                    ttl=bar.elapsed_td, eta=bar.eta_td)
        bar.next()


    bar.finish()

    train_fake2d3d_loader = DataLoader(video_mode_PoseDataSet(tmp_3d_pose_buffer_list, tmp_2d_pose_buffer_list,
                                                   [['none'] * len(np.concatenate(tmp_camparam_buffer_list))],
                                                   tmp_camparam_buffer_list, receptive_field),
                                       batch_size=args.batch_size,
                                       shuffle=True, num_workers=args.num_workers, pin_memory=True)

    data_dict['train_fake2d3d_loader'] = train_fake2d3d_loader

    os.mkdir(os.path.join(args.checkpoint, 'tmp', 'video_' + str(summary.epoch)))

    visual_real_Pos_v = visual_real_Pos_v.reshape(-1, receptive_field, 16, 3)
    visual_inputs_2d = visual_inputs_2d.reshape(-1, receptive_field, 16, 2)
    visual_fake_Pos = visual_fake_Pos.reshape(-1, receptive_field, 16, 3)
    visual_show_16key_2D_pix_norm = visual_show_16key_2D_pix_norm.reshape(-1, receptive_field, 16, 2)

    my_visual_GAN_video(args, visual_real_Pos_v, visual_inputs_2d, 'real_video',
                                receptive_field, summary.epoch)
    my_visual_GAN_video(args, visual_fake_Pos, visual_show_16key_2D_pix_norm, 'fake_video',
                                receptive_field, summary.epoch)

    return