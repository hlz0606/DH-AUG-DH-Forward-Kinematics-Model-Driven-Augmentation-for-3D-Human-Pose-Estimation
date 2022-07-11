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


from models_Fk_GAN.video_mode_operate import video_receptive_field

import os


def traditional_solutions_FK_generator(args, FK_DH_Class, data_dict, train_subjects):
    device = torch.device("cuda")
    batch_time = AverageMeter()
    data_time = AverageMeter()

    end = time.time()

    tmp_3d_pose_buffer_list = []
    tmp_2d_pose_buffer_list = []
    tmp_camparam_buffer_list = []

    new_generator_3d_pos_world, _, _, _, _ = FK_DH_Class.handler_but_generater()    #(10000, 32, 3)

    new_generator_3d_16pos_world = new_generator_3d_pos_world[:, H36M_32_To_16_Table, :]

    for subject in train_subjects:

        for cam_id in range(4):

            cam_R = np.array(h36m_cameras_extrinsic_params[subject][cam_id]['orientation']).reshape(1, 4)
            cam_t = np.array(h36m_cameras_extrinsic_params[subject][cam_id]['translation']).reshape(1, 3) / 1000.0
            res_w = h36m_cameras_intrinsic_params[cam_id]['res_w']
            res_h = h36m_cameras_intrinsic_params[cam_id]['res_h']

            f = np.array(h36m_cameras_intrinsic_params[cam_id]['focal_length']) / res_w * 2.0
            c = normalize_screen_coordinates(np.array(h36m_cameras_intrinsic_params[cam_id]['center']), w=res_w,
                                             h=res_h).astype('float32')
            k = np.array(h36m_cameras_intrinsic_params[cam_id]['radial_distortion'])
            p = np.array(h36m_cameras_intrinsic_params[cam_id]['tangential_distortion'])
            cam_para_temp = np.zeros((9))
            cam_para_temp[:2] = f
            cam_para_temp[2:4] = c
            cam_para_temp[4:7] = k
            cam_para_temp[7:] = p

            pos_3d_cam = world_to_camera(new_generator_3d_16pos_world, R=cam_R, t=cam_t)

            show_16key_2D_picture = wrap(project_to_2d, True, pos_3d_cam, cam_para_temp)

            show_16key_2D_pix = image_coordinates(show_16key_2D_picture, w=res_w, h=res_h)

            show_16key_2D_pix_norm = normalize_screen_coordinates(show_16key_2D_pix, w=res_w, h=res_h)

            cam_temp_no_use = np.zeros((show_16key_2D_pix_norm.shape[0], 1))

            tmp_3d_pose_buffer_list.append(pos_3d_cam)
            tmp_2d_pose_buffer_list.append(show_16key_2D_pix_norm)
            tmp_camparam_buffer_list.append(cam_temp_no_use)

    train_fake2d3d_loader = DataLoader(PoseDataSet(tmp_3d_pose_buffer_list, tmp_2d_pose_buffer_list,
                                                   [['none'] * len(np.concatenate(tmp_camparam_buffer_list))],
                                                   tmp_camparam_buffer_list),
                                       batch_size=args.batch_size,
                                       shuffle=True, num_workers=args.num_workers, pin_memory=True)

    data_dict['train_fake2d3d_loader'] = train_fake2d3d_loader
    return



def my_get_poseFk_model(args, dataset, FK_DH_Class):

    print("==> Creating model...")
    device = torch.device("cuda")
    num_joints = dataset.skeleton().num_joints()

    model_G = Fk_Generator(FK_DH_Class, args, device, INPUT_VEC_DIM=128).to(device)
    print("==> Total parameters: {:.2f}M".format(sum(p.numel() for p in model_G.parameters()) / 1000000.0))

    model_d3d = Fk_3D_Discriminator(device, args).to(device)        #Pos3dDiscriminator(num_joints).to(device)
    print("==> Total parameters: {:.2f}M".format(sum(p.numel() for p in model_d3d.parameters()) / 1000000.0))

    model_d2d = Fk_2D_Discriminator(args, num_joints).to(device)
    print("==> Total parameters: {:.2f}M".format(sum(p.numel() for p in model_d2d.parameters()) / 1000000.0))

    now_learnning_rate = 1e-4
    g_optimizer = \
        torch.optim.Adam(model_G.parameters(), lr=now_learnning_rate, betas=(0.5, 0.9))
    d3d_optimizer = \
        torch.optim.Adam(model_d3d.parameters(), lr=now_learnning_rate, betas=(0.5, 0.9))
    d2d_optimizer = \
        torch.optim.Adam(model_d2d.parameters(), lr=now_learnning_rate, betas=(0.5, 0.9))

    return {
        'model_G': model_G,
        'model_d3d': model_d3d,
        'model_d2d': model_d2d,
        'optimizer_G': g_optimizer,
        'optimizer_d3d': d3d_optimizer,
        'optimizer_d2d': d2d_optimizer,

    }


def video_mode_my_get_poseFk_model(args, dataset, FK_DH_Class, video_frame_num):

    device = torch.device("cuda")
    num_joints = dataset.skeleton().num_joints()

    model_G = Video_Fk_Generator(video_frame_num, FK_DH_Class, args, device, INPUT_VEC_DIM=128).to(device)

    model_d3d = Fk_3D_Discriminator(device, args).to(device)        #Pos3dDiscriminator(num_joints).to(device)

    model_d2d = Fk_2D_Discriminator(args).to(device)

    model_motion_d3d = Video_motion_Fk_3D_Discriminator(device, args, video_frame_num).to(device)
    print("==> Total parameters: {:.2f}M".format(sum(p.numel() for p in model_motion_d3d.parameters()) / 1000000.0))

    model_motion_d2d = Video_motion_Fk_2D_Discriminator(device, args, video_frame_num).to(device)
    print("==> Total parameters: {:.2f}M".format(sum(p.numel() for p in model_motion_d2d.parameters()) / 1000000.0))

    now_learnning_rate = 1e-4
    g_optimizer = \
        torch.optim.Adam(model_G.parameters(), lr=now_learnning_rate, betas=(0.5, 0.9))
    d3d_optimizer = \
        torch.optim.Adam(model_d3d.parameters(), lr=now_learnning_rate, betas=(0.5, 0.9))  #
    d2d_optimizer = \
        torch.optim.Adam(model_d2d.parameters(), lr=now_learnning_rate, betas=(0.5, 0.9))  #
    motion_d3d_optimizer = \
        torch.optim.Adam(model_motion_d3d.parameters(), lr=now_learnning_rate, betas=(0.5, 0.9))  #
    motion_d2d_optimizer = \
        torch.optim.Adam(model_motion_d2d.parameters(), lr=now_learnning_rate, betas=(0.5, 0.9))  #

    return {
        'model_G': model_G,
        'model_d3d': model_d3d,
        'model_d2d': model_d2d,
        'model_motion_d3d': model_motion_d3d,
        'model_motion_d2d': model_motion_d2d,

        'optimizer_G': g_optimizer,
        'optimizer_d3d': d3d_optimizer,
        'optimizer_d2d': d2d_optimizer,
        'optimizer_motion_d3d': motion_d3d_optimizer,
        'optimizer_motion_d2d': motion_d2d_optimizer,

    }



def train_Fk_discriminator(model_dis, data_real, data_fake, summary,
                           writer, writer_name, optimizerD, args,
                           one, mone, dis_mode='single'):    #

    device = torch.device("cuda")
    model_dis.zero_grad()

    optimizerD.zero_grad()

    data_real = data_real.to(device)
    data_fake = data_fake.to(device)

    D_real = model_dis(data_real)

    D_real = D_real.mean()

    D_real.backward(mone)
    ############################

    D_fake = model_dis(data_fake)

    D_fake = D_fake.mean()

    if summary.train_discrim_iter_num % 400 == 0:
        print('\niter_d_sum：{0}   {1}  D_fake: {2}'.format(summary.train_discrim_iter_num, writer_name, D_fake))

    D_fake.backward(one)

    real_used_num = 1
    if args.single_or_multi_train_mode == 'multi' and dis_mode != 'motion':
        filter_widths = [int(x) for x in args.architecture.split(',')]
        receptive_field = video_receptive_field(filter_widths)
        real_used_num = receptive_field
    else:
        real_used_num = 1

    gradient_penalty = calc_gradient_penalty(model_dis, data_real.data, data_fake.data,
                                                 args.batch_size * real_used_num, args.GAN_LAMBDA, device)

    gradient_penalty.backward()

    D_cost = D_fake - D_real + gradient_penalty
    Wasserstein_D = D_real - D_fake
    if summary.train_iter_num % 400 == 0:
       print('iter_d_sum：{0}   {1}   Wasserstein_D: {2}'.format(summary.train_iter_num, writer_name, Wasserstein_D))

    optimizerD.step()

    writer.add_scalar('train_G_iter_PoseFk/{}_D_real'.format(writer_name), D_real, summary.train_iter_num)
    writer.add_scalar('train_G_iter_PoseFk/{}_D_fake'.format(writer_name), D_fake, summary.train_iter_num)
    writer.add_scalar('train_G_iter_PoseFk/{}_Wasserstein_D'.format(writer_name),
                      Wasserstein_D.item(), summary.train_iter_num)

    return Wasserstein_D, D_cost





def GAN_solutions_FK_generator(args, poseFk_dict, data_dict, model_pos, summary, writer, train_subjects):

    device = torch.device("cuda")
    batch_time = AverageMeter()
    data_time = AverageMeter()

    model_G = poseFk_dict['model_G']
    model_d3d = poseFk_dict['model_d3d']
    model_d2d = poseFk_dict['model_d2d']

    g_optimizer = poseFk_dict['optimizer_G']
    d3d_optimizer = poseFk_dict['optimizer_d3d']
    d2d_optimizer = poseFk_dict['optimizer_d2d']

    model_G.train()
    model_d3d.train()
    model_d2d.train()
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

    one = torch.tensor(1, dtype=torch.float32)   # 1
    mone = one * -1   # -1

    one = one.to(device)
    mone = mone.to(device)

    bar = Bar('Train pose gan', max=len(data_dict['train_gt2d3d_loader']))

    for i, ((inputs_3d, _, _, cam_param), target_d2d, target_d3d) in enumerate(
            zip(data_dict['train_gt2d3d_loader'], data_dict['target_2d_loader'], data_dict['target_3d_loader'])):

        if inputs_3d.shape[0] < args.batch_size:
            continue

        data_time.update(time.time() - end)

        inputs_3d, cam_param = inputs_3d.to(device), cam_param.to(device)
        target_d2d = target_d2d.to(device)

        model_G.GAN_generator_get_bone_length(inputs_3d)

        inputs_3d = inputs_3d.view(-1, 16, 3)
        inputs_3d = inputs_3d.to(device)

        real_cam_R = cam_param[:, 9:13]
        real_cam_T = cam_param[:, 13:16]

        real_pos_3d_world = \
            GAN_torch_camera_to_world_batch(inputs_3d, R=real_cam_R, t=real_cam_T)

        real_pos_3d_world[:, :, :] = real_pos_3d_world[:, :, :] - real_pos_3d_world[:, :1, :]

        set_grad([model_d3d], True)
        set_grad([model_d2d], True)
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

        Wasserstein_D_3D, D_cost_3D = train_Fk_discriminator(model_d3d, torch.clone(real_Pos_v),
                                                             torch.clone(fake_Pos), summary,
                                                      writer, writer_name='Fk_d3d',
                                                      optimizerD=d3d_optimizer, args=args, one=one, mone=mone)

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

            flip_Wasserstein_D_3D, flip_D_cost_3D = \
                    train_Fk_discriminator(model_d3d, torch.clone(real_3d_flip),
                                                                torch.clone(fake_3d_flip), summary,
                                                                 writer, writer_name='Fk_d3d',
                                                                 optimizerD=d3d_optimizer, args=args, one=one,
                                                                 mone=mone)

            Wasserstein_D_3D = (Wasserstein_D_3D + flip_Wasserstein_D_3D) / 2
            D_cost_3D = (D_cost_3D + flip_D_cost_3D) / 2


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

        show_16key_2D_picture = project_to_2d(pos_3d_cam, cam_para_temp)

        show_16key_2D_pix_norm = show_16key_2D_picture

        show_16key_2D_pix_norm = show_16key_2D_pix_norm.to(device)

        Wasserstein_D_2D, D_cost_2D = train_Fk_discriminator(model_d2d, torch.clone(target_d2d),
                                                             torch.clone(show_16key_2D_pix_norm), summary,
                                                        writer, writer_name='d2d',
                                                        optimizerD=d2d_optimizer, args=args, one=one, mone=mone)

        if args.flip_GAN_model_input == True:
            joints_left = [4, 5, 6, 10, 11, 12]
            joints_right = [1, 2, 3, 13, 14, 15]
            out_left = [4, 5, 6, 10, 11, 12]
            out_right = [1, 2, 3, 13, 14, 15]

            real_2d_flip = target_d2d.detach().clone()
            real_2d_flip[:, :, 0] *= -1
            real_2d_flip[:, out_left + out_right, :] = real_2d_flip[:, out_right + out_left, :]

            fake_2d_flip = show_16key_2D_pix_norm.detach().clone()
            fake_2d_flip[:, :, 0] *= -1
            fake_2d_flip[:, out_left + out_right, :] = fake_2d_flip[:, out_right + out_left, :]

            flip_Wasserstein_D_2D, flip_D_cost_2D = train_Fk_discriminator(model_d2d,
                                                                torch.clone(real_2d_flip), torch.clone(fake_2d_flip),
                                                                summary,
                                                                 writer, writer_name='d2d',
                                                                 optimizerD=d2d_optimizer, args=args, one=one,
                                                                 mone=mone)

            Wasserstein_D_2D = (Wasserstein_D_2D + flip_Wasserstein_D_2D) / 2
            D_cost_2D = (D_cost_2D + flip_D_cost_2D) / 2

        summary.summary_train_discrim_update()

        #########################################################################################

        if summary.train_iter_num % 5 == 4:
            set_grad([model_d3d], False)
            set_grad([model_d2d], False)
            set_grad([model_G], True)
            set_grad([model_pos], False)

            model_G.zero_grad()

            g_optimizer.zero_grad()

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

            if args.flip_GAN_model_input == True:
                joints_left = [4, 5, 6, 10, 11, 12]
                joints_right = [1, 2, 3, 13, 14, 15]
                out_left = [4, 5, 6, 10, 11, 12]
                out_right = [1, 2, 3, 13, 14, 15]

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

                adv_loss_3d = (adv_loss_3d + flip_adv_loss_3d) / 2
                adv_loss_2d = (adv_loss_2d + flip_adv_loss_2d) / 2

            gen_loss = adv_loss_3d * args.GAN_3d_loss_weight + adv_loss_2d * args.GAN_2d_loss_weight
            if summary.epoch > args.warmup:
                gen_loss = adv_loss_3d * args.GAN_3d_loss_weight + adv_loss_2d * args.GAN_2d_loss_weight
            else:
                gen_loss = adv_loss_3d * args.GAN_3d_loss_weight + adv_loss_2d * args.GAN_2d_loss_weight

            gen_loss.backward(mone)
            G_cost = -gen_loss
            g_optimizer.step()

            summary.summary_train_fakepose_iter_num_update()


        tmp_3d_pose_buffer_list.append(pos_3d_cam.detach().cpu().numpy())
        tmp_2d_pose_buffer_list.append(show_16key_2D_pix_norm.detach().cpu().numpy())
        tmp_camparam_buffer_list.append(cam_para_temp.detach().cpu().numpy())

        summary.summary_train_iter_num_update()

        batch_time.update(time.time() - end)
        end = time.time()

        bar.suffix = '({batch}/{size}) Data: {data:.6f}s | Batch: {bt:.3f}s | Total: {ttl:} | ETA: {eta:} ' \
            .format(batch=i + 1, size=len(data_dict['train_gt2d3d_loader']), data=data_time.avg, bt=batch_time.avg,
                    ttl=bar.elapsed_td, eta=bar.eta_td)
        bar.next()

    bar.finish()

    # buffer loader will be used to save fake pose pair
    train_fake2d3d_loader = DataLoader(PoseDataSet(tmp_3d_pose_buffer_list, tmp_2d_pose_buffer_list,
                                                   [['none'] * len(np.concatenate(tmp_camparam_buffer_list))],
                                                   tmp_camparam_buffer_list),
                                       batch_size=args.batch_size,
                                       shuffle=True, num_workers=args.num_workers, pin_memory=True)

    data_dict['train_fake2d3d_loader'] = train_fake2d3d_loader

    return