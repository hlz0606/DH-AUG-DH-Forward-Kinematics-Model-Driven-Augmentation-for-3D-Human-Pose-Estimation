import numpy as np
import copy
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec  # 使用 GridSpec 自定义子图位置
from matplotlib.pyplot import MultipleLocator
import torch
#from __future__ import print_function, absolute_import, division

import os.path as path
from torch.utils.data import DataLoader
from utils.data_utils import fetch, read_3d_data, create_2d_data

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import os

import time
from progress.bar import Bar
from utils.utils import AverageMeter, set_grad

from models_Fk_GAN.special_operate import my_visual_2D_pos, my_visual_3D_pos
from utils.loss import mpjpe, p_mpjpe, compute_PCK, compute_AUC
from itertools import zip_longest
from common.camera import project_to_2d
from function_aug.dataloader_update import random_bl_aug

from utils.gan_utils import get_bone_unit_vecbypose3d, get_pose3dbyBoneVec


class GAN_video_ChunkedGenerator:

    def __init__(self, batch_size, cameras, poses_3d, poses_2d,
                 chunk_length, pad=0, causal_shift=0,
                 shuffle=True, random_seed=1234,
                 augment=False, kps_left=None, kps_right=None, joints_left=None, joints_right=None,
                 endless=False):
        assert poses_3d is None or len(poses_3d) == len(poses_2d), (len(poses_3d), len(poses_2d))
        assert cameras is None or len(cameras) == len(poses_2d)

        pairs = []  # (seq_idx, start_frame, end_frame, flip) tuples
        for i in range(len(poses_2d)):
            assert poses_3d is None or poses_3d[i].shape[0] == poses_3d[i].shape[0]

            n_chunks = (poses_2d[i].shape[0] + chunk_length - 1) // chunk_length

            offset = (n_chunks * chunk_length - poses_2d[i].shape[0]) // 2

            bounds = np.arange(n_chunks + 1) * chunk_length - offset

            augment_vector = np.full(len(bounds - 1), False, dtype=bool)

            pairs += zip(np.repeat(i, len(bounds - 1)), bounds[:-1], bounds[1:], augment_vector)

            if augment:
                pairs += zip(np.repeat(i, len(bounds - 1)), bounds[:-1], bounds[1:], ~augment_vector)

        if cameras is not None:
            self.batch_cam = np.empty((batch_size, cameras[0].shape[-1]))

        if poses_3d is not None:
            self.batch_3d = np.empty((batch_size, chunk_length + 2 * pad, poses_3d[0].shape[-2], poses_3d[0].shape[-1]))

        self.batch_2d = np.empty((batch_size, chunk_length + 2 * pad, poses_2d[0].shape[-2], poses_2d[0].shape[-1]))

        self.num_batches = (len(pairs) + batch_size - 1) // batch_size

        self.batch_size = batch_size
        self.random = np.random.RandomState(random_seed)
        self.pairs = pairs
        self.shuffle = shuffle
        self.pad = pad
        self.causal_shift = causal_shift
        self.endless = endless
        self.state = None

        self.cameras = cameras
        self.poses_3d = poses_3d
        self.poses_2d = poses_2d

        self.augment = augment
        self.kps_left = kps_left
        self.kps_right = kps_right
        self.joints_left = joints_left
        self.joints_right = joints_right

    def num_frames(self):  #
        return self.num_batches * self.batch_size

    def random_state(self):
        return self.random

    def set_random_state(self, random):  #
        self.random = random

    def augment_enabled(self):  #
        return self.augment

    def next_pairs(self):  #

        if self.state is None:
            if self.shuffle:
                pairs = self.random.permutation(self.pairs)
            else:
                pairs = self.pairs
            return 0, pairs
        else:
            return self.state

    def next_epoch(self):
        enabled = True

        while enabled:
            start_idx, pairs = self.next_pairs()

            for b_i in range(start_idx, self.num_batches):

                chunks = pairs[b_i * self.batch_size: (b_i + 1) * self.batch_size]

                for i, (seq_i, start_3d, end_3d, flip) in enumerate(chunks):
                    start_2d = start_3d - self.pad - self.causal_shift

                    end_2d = end_3d + self.pad - self.causal_shift

                    seq_2d = self.poses_2d[seq_i]

                    low_2d = max(start_2d, 0)
                    high_2d = min(end_2d, seq_2d.shape[0])

                    pad_left_2d = low_2d - start_2d

                    pad_right_2d = end_2d - high_2d

                    if pad_left_2d != 0 or pad_right_2d != 0:
                        self.batch_2d[i] = np.pad(seq_2d[low_2d:high_2d], ((pad_left_2d, pad_right_2d), (0, 0), (0, 0)),
                                                  'edge')
                    else:  #
                        self.batch_2d[i] = seq_2d[low_2d:high_2d]

                    if flip:
                        self.batch_2d[i, :, :, 0] *= -1
                        self.batch_2d[i, :, self.kps_left + self.kps_right] = self.batch_2d[i, :,
                                                                              self.kps_right + self.kps_left]

                    if self.poses_3d is not None:
                        seq_3d = self.poses_3d[seq_i]

                        low_3d = low_2d   #
                        high_3d = high_2d

                        pad_left_3d = pad_left_2d  #low_3d - start_3d
                        pad_right_3d = pad_right_2d  #end_3d - high_3d

                        if pad_left_3d != 0 or pad_right_3d != 0:
                            self.batch_3d[i] = np.pad(seq_3d[low_3d:high_3d],
                                                      ((pad_left_3d, pad_right_3d), (0, 0), (0, 0)), 'edge')
                        else:
                            self.batch_3d[i] = seq_3d[low_3d:high_3d]

                        if flip:
                            self.batch_3d[i, :, :, 0] *= -1

                            self.batch_3d[i, :, self.joints_left + self.joints_right] = \
                                self.batch_3d[i, :, self.joints_right + self.joints_left]

                    if self.cameras is not None:
                        self.batch_cam[i] = self.cameras[seq_i]
                        if flip:
                            self.batch_cam[i, 2] *= -1
                            self.batch_cam[i, 7] *= -1

                if self.endless:
                    self.state = (b_i + 1, pairs)
                if self.poses_3d is None and self.cameras is None:
                    yield None, None, self.batch_2d[:len(chunks)]
                elif self.poses_3d is not None and self.cameras is None:
                    yield None, self.batch_3d[:len(chunks)], self.batch_2d[:len(chunks)]
                elif self.poses_3d is None:
                    yield self.batch_cam[:len(chunks)], None, self.batch_2d[:len(chunks)]
                else:
                    yield self.batch_cam[:len(chunks)], self.batch_3d[:len(chunks)], self.batch_2d[:len(chunks)]

            if self.endless:
                self.state = None
            else:
                enabled = False


class ChunkedGenerator:

    def __init__(self, batch_size, cameras, poses_3d, poses_2d,
                 chunk_length, pad=0, causal_shift=0,
                 shuffle=True, random_seed=1234,
                 augment=False, kps_left=None, kps_right=None, joints_left=None, joints_right=None,
                 endless=False):
        assert poses_3d is None or len(poses_3d) == len(poses_2d), (len(poses_3d), len(poses_2d))
        assert cameras is None or len(cameras) == len(poses_2d)

        pairs = []
        for i in range(len(poses_2d)):
            assert poses_3d is None or poses_3d[i].shape[0] == poses_3d[i].shape[0]

            n_chunks = (poses_2d[i].shape[0] + chunk_length - 1) // chunk_length

            offset = (n_chunks * chunk_length - poses_2d[i].shape[0]) // 2

            bounds = np.arange(n_chunks + 1) * chunk_length - offset

            augment_vector = np.full(len(bounds - 1), False, dtype=bool)

            pairs += zip(np.repeat(i, len(bounds - 1)), bounds[:-1], bounds[1:], augment_vector)

            if augment:
                pairs += zip(np.repeat(i, len(bounds - 1)), bounds[:-1], bounds[1:], ~augment_vector)

        if cameras is not None:
            self.batch_cam = np.empty((batch_size, cameras[0].shape[-1]))

        if poses_3d is not None:
            self.batch_3d = np.empty((batch_size, chunk_length, poses_3d[0].shape[-2], poses_3d[0].shape[-1]))

        self.batch_2d = np.empty((batch_size, chunk_length + 2 * pad, poses_2d[0].shape[-2], poses_2d[0].shape[-1]))

        self.num_batches = (len(pairs) + batch_size - 1) // batch_size

        self.batch_size = batch_size
        self.random = np.random.RandomState(random_seed)
        self.pairs = pairs
        self.shuffle = shuffle
        self.pad = pad
        self.causal_shift = causal_shift
        self.endless = endless
        self.state = None

        self.cameras = cameras
        self.poses_3d = poses_3d
        self.poses_2d = poses_2d

        self.augment = augment
        self.kps_left = kps_left
        self.kps_right = kps_right
        self.joints_left = joints_left
        self.joints_right = joints_right

    def num_frames(self):  #
        return self.num_batches * self.batch_size

    def random_state(self):
        return self.random

    def set_random_state(self, random):
        self.random = random

    def augment_enabled(self):  #
        return self.augment

    def next_pairs(self):  #

        if self.state is None:
            if self.shuffle:
                pairs = self.random.permutation(self.pairs)
            else:
                pairs = self.pairs
            return 0, pairs
        else:
            return self.state

    def next_epoch(self):
        enabled = True

        while enabled:
            start_idx, pairs = self.next_pairs()

            for b_i in range(start_idx, self.num_batches):
                chunks = pairs[b_i * self.batch_size: (b_i + 1) * self.batch_size]

                for i, (seq_i, start_3d, end_3d, flip) in enumerate(chunks):
                    start_2d = start_3d - self.pad - self.causal_shift

                    end_2d = end_3d + self.pad - self.causal_shift

                    seq_2d = self.poses_2d[seq_i]

                    low_2d = max(start_2d, 0)
                    high_2d = min(end_2d, seq_2d.shape[0])

                    pad_left_2d = low_2d - start_2d

                    pad_right_2d = end_2d - high_2d

                    if pad_left_2d != 0 or pad_right_2d != 0:
                        self.batch_2d[i] = np.pad(seq_2d[low_2d:high_2d], ((pad_left_2d, pad_right_2d), (0, 0), (0, 0)),
                                                  'edge')
                    else:
                        self.batch_2d[i] = seq_2d[low_2d:high_2d]


                    if flip:
                        self.batch_2d[i, :, :, 0] *= -1
                        self.batch_2d[i, :, self.kps_left + self.kps_right] = self.batch_2d[i, :,
                                                                              self.kps_right + self.kps_left]

                    if self.poses_3d is not None:
                        seq_3d = self.poses_3d[seq_i]

                        low_3d = max(start_3d, 0)
                        high_3d = min(end_3d, seq_3d.shape[0])

                        pad_left_3d = low_3d - start_3d
                        pad_right_3d = end_3d - high_3d

                        if pad_left_3d != 0 or pad_right_3d != 0:
                            self.batch_3d[i] = np.pad(seq_3d[low_3d:high_3d],
                                                      ((pad_left_3d, pad_right_3d), (0, 0), (0, 0)), 'edge')
                        else:
                            self.batch_3d[i] = seq_3d[low_3d:high_3d]

                        if flip:
                            self.batch_3d[i, :, :, 0] *= -1
                            self.batch_3d[i, :, self.joints_left + self.joints_right] = \
                                self.batch_3d[i, :, self.joints_right + self.joints_left]

                    if self.cameras is not None:
                        self.batch_cam[i] = self.cameras[seq_i]
                        if flip:
                            self.batch_cam[i, 2] *= -1
                            self.batch_cam[i, 7] *= -1

                if self.endless:
                    self.state = (b_i + 1, pairs)
                if self.poses_3d is None and self.cameras is None:
                    yield None, None, self.batch_2d[:len(chunks)]
                elif self.poses_3d is not None and self.cameras is None:
                    yield None, self.batch_3d[:len(chunks)], self.batch_2d[:len(chunks)]
                elif self.poses_3d is None:
                    yield self.batch_cam[:len(chunks)], None, self.batch_2d[:len(chunks)]
                else:
                    yield self.batch_cam[:len(chunks)], self.batch_3d[:len(chunks)], self.batch_2d[:len(chunks)]

            if self.endless:
                self.state = None
            else:
                enabled = False


class UnchunkedGenerator:

    def __init__(self, cameras, poses_3d, poses_2d, pad=0, causal_shift=0,
                 augment=False, kps_left=None, kps_right=None, joints_left=None, joints_right=None):
        assert poses_3d is None or len(poses_3d) == len(poses_2d)
        assert cameras is None or len(cameras) == len(poses_2d)

        self.augment = augment
        self.kps_left = kps_left
        self.kps_right = kps_right
        self.joints_left = joints_left
        self.joints_right = joints_right

        self.pad = pad
        self.causal_shift = causal_shift
        self.cameras = [] if cameras is None else cameras
        self.poses_3d = [] if poses_3d is None else poses_3d
        self.poses_2d = poses_2d

    def num_frames(self):
        count = 0
        for p in self.poses_2d:
            count += p.shape[0]
        return count

    def augment_enabled(self):
        return self.augment

    def set_augment(self, augment):
        self.augment = augment

    def next_epoch(self):
        for seq_cam, seq_3d, seq_2d in zip_longest(self.cameras, self.poses_3d, self.poses_2d):

            batch_cam = None if seq_cam is None else np.expand_dims(seq_cam, axis=0)
            batch_3d = None if seq_3d is None else np.expand_dims(seq_3d, axis=0)
            batch_2d = np.expand_dims(np.pad(seq_2d,
                                             ((self.pad + self.causal_shift, self.pad - self.causal_shift), (0, 0),
                                              (0, 0)),
                                             'edge'), axis=0)
            if self.augment:
                if batch_cam is not None:
                    batch_cam = np.concatenate((batch_cam, batch_cam), axis=0)
                    batch_cam[1, 2] *= -1
                    batch_cam[1, 7] *= -1

                if batch_3d is not None:
                    batch_3d = np.concatenate((batch_3d, batch_3d), axis=0)
                    batch_3d[1, :, :, 0] *= -1
                    batch_3d[1, :, self.joints_left + self.joints_right] = batch_3d[1, :,
                                                                           self.joints_right + self.joints_left]

                batch_2d = np.concatenate((batch_2d, batch_2d), axis=0)
                batch_2d[1, :, :, 0] *= -1
                batch_2d[1, :, self.kps_left + self.kps_right] = batch_2d[1, :, self.kps_right + self.kps_left]

            yield batch_cam, batch_3d, batch_2d




def video_receptive_field(pad):
    frames = 1
    for f in pad:
        frames *= f
    return frames

def video_mode_fk_data_preparation(args, file_path_prefix):
    dataset_path = path.join(file_path_prefix, 'data', 'data_3d_' + args.dataset + '.npz')

    if args.dataset == 'h36m':
        from common.h36m_dataset import Human36mDataset, TEST_SUBJECTS
        dataset = Human36mDataset(dataset_path)
        if args.s1only:
            subjects_train = ['S1']    #
        elif args.s1s5only:
            subjects_train = ['S1', 'S5']  #
        else:
            subjects_train = ['S1', 'S5', 'S6', 'S7', 'S8']
        subjects_test = TEST_SUBJECTS

        if args.s1only == True and args.s1s5only == True:
            raise KeyError(' args.s1only and args.s1s5only both set true')
    else:
        raise KeyError('Invalid dataset')

    dataset = read_3d_data(dataset)

    keypoints = create_2d_data(path.join(file_path_prefix, 'data', 'data_2d_' + args.dataset + '_' + args.keypoints + '.npz'), dataset)

    action_filter = None if args.actions == '*' else args.actions.split(',')
    if action_filter is not None:
        action_filter = map(lambda x: dataset.define_actions(x)[0], action_filter)
        print('==> Selected actions: {}'.format(action_filter))

    stride = args.downsample

    poses_train, poses_train_2d, actions_train, cams_train = fetch(subjects_train, dataset, keypoints,
                                                                   args, 'train', action_filter,
                                                                   stride, whether_need_cam_external=True)
    poses_valid, poses_valid_2d, actions_valid, cams_valid = fetch(subjects_test, dataset, keypoints,
                                                                   args, 'test', action_filter,
                                                                   stride, whether_need_cam_external=True)

    joints_left = [4, 5, 6, 10, 11, 12]
    joints_right = [1, 2, 3, 13, 14, 15]
    out_left = [4, 5, 6, 10, 11, 12]
    out_right = [1, 2, 3, 13, 14, 15]

    filter_widths = [int(x) for x in args.architecture.split(',')]  # 默认为default='3,3,3'   可修改的
    receptive_field = video_receptive_field(filter_widths)

    pad = (receptive_field - 1) // 2


    train_det2d3d_loader = \
        ChunkedGenerator(args.batch_size//1, cams_train, poses_train, poses_train_2d, chunk_length=1,
                        pad=pad, causal_shift=0, shuffle=True, augment=False,
                        kps_left=out_left, kps_right=out_right, joints_left=joints_left, joints_right=joints_right)

    valid_loader = \
        UnchunkedGenerator(copy.deepcopy(cams_valid), copy.deepcopy(poses_valid), copy.deepcopy(poses_valid_2d),
                                    pad=pad, causal_shift=0, augment=False,
                                    kps_left=out_left, kps_right=out_right,
                                    joints_left=joints_left, joints_right=joints_right)

    if args.posenet_name != 'mulit_farme_videopose':    # !=
        valid_loader =  \
            ChunkedGenerator(args.batch_size // 1, copy.deepcopy(cams_valid), copy.deepcopy(poses_valid),
                         copy.deepcopy(poses_valid_2d), chunk_length=1,
                         pad=pad, causal_shift=0, shuffle=False, augment=False,
                         kps_left=out_left, kps_right=out_right, joints_left=joints_left, joints_right=joints_right)


    target_GAN_loader = \
        GAN_video_ChunkedGenerator(args.batch_size//1, copy.deepcopy(cams_train), copy.deepcopy(poses_train),
                         copy.deepcopy(poses_train_2d), chunk_length=1,
                        pad=pad, causal_shift=0, shuffle=True, augment=False,
                        kps_left=out_left, kps_right=out_right, joints_left=joints_left, joints_right=joints_right)


    mpi3d_npz = np.load(file_path_prefix + '/data_extra/test_set/test_3dhp.npz')    # this is the 2929 version
    video_mpi_inf_3d = []
    video_mpi_inf_2d = []
    cam_temp = []
    for frame_id in [(0, 603), (603, 1143), (1143, 1648),
                     (1648, 2201), (2201, 2477), (2477, 2929)]:
        video_mpi_inf_3d.append(mpi3d_npz['pose3d'][frame_id[0]: frame_id[1]])
        video_mpi_inf_2d.append(mpi3d_npz['pose2d'][frame_id[0]: frame_id[1]])
        cam_temp.append(np.array([0.00]))

    mpi3d_loader =  \
        UnchunkedGenerator(copy.deepcopy(cam_temp), copy.deepcopy(video_mpi_inf_3d), copy.deepcopy(video_mpi_inf_2d),
                                    pad=pad, causal_shift=0, augment=False,
                                    kps_left=out_left, kps_right=out_right,
                                    joints_left=joints_left, joints_right=joints_right)
    if args.posenet_name != 'mulit_farme_videopose':    # !=
        mpi3d_loader =  \
            ChunkedGenerator(args.batch_size // 1, copy.deepcopy(cam_temp), copy.deepcopy(video_mpi_inf_3d),
                         copy.deepcopy(video_mpi_inf_2d), chunk_length=1,
                         pad=pad, causal_shift=0, shuffle=False, augment=False,
                         kps_left=out_left, kps_right=out_right, joints_left=joints_left, joints_right=joints_right)


    return {
        'dataset': dataset,
        'train_det2d3d_loader': train_det2d3d_loader,
        'target_GAN_loader': target_GAN_loader,
        'H36M_test': valid_loader,
        'mpi3d_loader': mpi3d_loader,
        'action_filter': action_filter,
        'subjects_test': subjects_test,
        'keypoints': keypoints,

        'poses_train': poses_train,        #
        'poses_train_2d': poses_train_2d,  #
        'actions_train': actions_train,     #
        'cams_train': cams_train     #
    }, receptive_field



def video_mode_train_posenet(model_pos, data_loader, optimizer, criterion, device, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    epoch_loss_3d_pos = AverageMeter()
    flip_epoch_loss_3d_pos = AverageMeter()

    back_epoch_loss_3d_pos = AverageMeter()
    back_flip_epoch_loss_3d_pos = AverageMeter()

    torch.set_grad_enabled(True)
    set_grad([model_pos], True)
    model_pos.train()
    end = time.time()

    max_temp = data_loader.num_batches

    bar = Bar('Train on real pose det2d', max=max_temp)

    i = 0
    for cam, batch_3d, batch_2d in data_loader.next_epoch():
        data_time.update(time.time() - end)

        num_poses = batch_3d.shape[0]

        if num_poses == 1:
            break

        inputs_3d = torch.from_numpy(batch_3d.astype('float32'))
        inputs_2d = torch.from_numpy(batch_2d.astype('float32'))

        inputs_3d, inputs_2d = inputs_3d.to(device), inputs_2d.to(device)

        inputs_3d = inputs_3d[:, :, :, :] - inputs_3d[:, :, :1, :]

        outputs_3d = model_pos(inputs_2d)

        optimizer.zero_grad()

        loss_3d_pos = criterion(outputs_3d, inputs_3d)
        loss_3d_pos.backward()
        nn.utils.clip_grad_norm_(model_pos.parameters(), max_norm=1)
        optimizer.step()

        epoch_loss_3d_pos.update(loss_3d_pos.item(), num_poses)

        if args.GAN_video_playback_input == True:
            back_inputs_2d = torch.clone(inputs_2d)
            back_inputs_2d = back_inputs_2d.view(num_poses, -1, 16, 2)
            back_inputs_2d = torch.flip(back_inputs_2d, dims=[1])

            back_outputs_3d = model_pos(back_inputs_2d)

            optimizer.zero_grad()
            back_loss_3d_pos = criterion(back_outputs_3d, inputs_3d)
            back_loss_3d_pos.backward()
            nn.utils.clip_grad_norm_(model_pos.parameters(), max_norm=1)
            optimizer.step()

            back_epoch_loss_3d_pos.update(back_loss_3d_pos.item(), num_poses)

        if args.flip_pos_model_input:  # flip the 2D pose Left <-> Right
            joints_left = [4, 5, 6, 10, 11, 12]
            joints_right = [1, 2, 3, 13, 14, 15]
            out_left = [4, 5, 6, 10, 11, 12]
            out_right = [1, 2, 3, 13, 14, 15]

            inputs_2d_flip = inputs_2d.detach().clone()
            inputs_2d_flip[:, :, :, 0] *= -1
            inputs_2d_flip[:, :, joints_left + joints_right, :] = inputs_2d_flip[:, :, joints_right + joints_left, :]

            outputs_3d_flip = model_pos(inputs_2d_flip)

            inputs_3d_flip = inputs_3d.detach().clone()
            inputs_3d_flip[:, :, :, 0] *= -1
            inputs_3d_flip[:, :, out_left + out_right, :] = inputs_3d_flip[:, :, out_right + out_left, :]

            optimizer.zero_grad()
            flip_loss_3d_pos = criterion(outputs_3d_flip, inputs_3d_flip)
            flip_loss_3d_pos.backward()
            nn.utils.clip_grad_norm_(model_pos.parameters(), max_norm=1)
            optimizer.step()

            flip_epoch_loss_3d_pos.update(loss_3d_pos.item(), num_poses)

            if args.GAN_video_playback_input == True:
                back_inputs_2d_flip = torch.clone(inputs_2d_flip)
                back_inputs_2d_flip = back_inputs_2d_flip.view(num_poses, -1, 16, 2)
                back_inputs_2d_flip = torch.flip(back_inputs_2d_flip, dims=[1])

                back_outputs_3d_flip = model_pos(back_inputs_2d_flip)

                optimizer.zero_grad()
                back_flip_loss_3d_pos = criterion(back_outputs_3d_flip, inputs_3d_flip)
                back_flip_loss_3d_pos.backward()
                nn.utils.clip_grad_norm_(model_pos.parameters(), max_norm=1)
                optimizer.step()

                back_flip_epoch_loss_3d_pos.update(back_flip_loss_3d_pos.item(), num_poses)

            batch_time.update(time.time() - end)
            end = time.time()

            bar.suffix = 'use flip: ({batch}/{size}) Data: {data:.6f}s | Batch: {bt:.3f}s ' \
                         '| Total: {ttl:} | ETA: {eta:} ' \
                         '| Loss: {loss: .4f}  | flip_Loss: {flip_Loss: .4f} ' \
                         '| back_Loss: {back_Loss: .4f}  | back_flip_Loss: {back_flip_Loss: .4f}' \
                .format(batch=i + 1, size=max_temp, data=data_time.avg, bt=batch_time.avg,
                        ttl=bar.elapsed_td, eta=bar.eta_td, loss=epoch_loss_3d_pos.avg,
                        flip_Loss=flip_epoch_loss_3d_pos.avg, back_Loss=back_epoch_loss_3d_pos.avg,
                        back_flip_Loss=back_flip_epoch_loss_3d_pos.avg)
            bar.next()

        i += 1

    bar.finish()

    return



def GAN_dataSet_video_mode_train_posenet(model_pos, data_loader, optimizer, criterion,
                                         device, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    epoch_loss_3d_pos = AverageMeter()
    flip_epoch_loss_3d_pos = AverageMeter()
    back_epoch_loss_3d_pos = AverageMeter()
    back_flip_epoch_loss_3d_pos = AverageMeter()

    torch.set_grad_enabled(True)
    set_grad([model_pos], True)
    model_pos.train()
    end = time.time()

    max_temp = len(data_loader)

    bar = Bar('Train on real pose det2d', max=max_temp)

    for i, (cam, batch_3d, batch_2d) in enumerate(data_loader):

        data_time.update(time.time() - end)

        num_poses = batch_3d.shape[0]

        if num_poses == 1:
            break

        batch_3d = batch_3d.contiguous().view(-1, 1, 16, 3)

        inputs_3d = batch_3d.to(device)
        inputs_2d = batch_2d.to(device)

        inputs_3d = inputs_3d[:, :, :, :] - inputs_3d[:, :, :1, :]

        outputs_3d = model_pos(inputs_2d)

        optimizer.zero_grad()

        loss_3d_pos = criterion(outputs_3d, inputs_3d)
        loss_3d_pos.backward()
        nn.utils.clip_grad_norm_(model_pos.parameters(), max_norm=1)    #
        optimizer.step()

        epoch_loss_3d_pos.update(loss_3d_pos.item(), num_poses)

        if args.GAN_video_playback_input == True:
            back_inputs_2d = torch.clone(inputs_2d)
            back_inputs_2d = back_inputs_2d.view(num_poses, -1, 16, 2)
            back_inputs_2d = torch.flip(back_inputs_2d, dims=[1])  #

            back_outputs_3d = model_pos(back_inputs_2d)

            optimizer.zero_grad()
            back_loss_3d_pos = criterion(back_outputs_3d, inputs_3d)  #
            back_loss_3d_pos.backward()
            nn.utils.clip_grad_norm_(model_pos.parameters(), max_norm=1)  #
            optimizer.step()

            back_epoch_loss_3d_pos.update(back_loss_3d_pos.item(), num_poses)

        if args.flip_pos_model_input:  # flip the 2D pose Left <->
            joints_left = [4, 5, 6, 10, 11, 12]
            joints_right = [1, 2, 3, 13, 14, 15]
            out_left = [4, 5, 6, 10, 11, 12]
            out_right = [1, 2, 3, 13, 14, 15]

            inputs_2d_flip = inputs_2d.detach().clone()
            inputs_2d_flip[:, :, :, 0] *= -1
            inputs_2d_flip[:, :, joints_left + joints_right, :] = inputs_2d_flip[:, :, joints_right + joints_left, :]

            outputs_3d_flip = model_pos(inputs_2d_flip)

            inputs_3d_flip = inputs_3d.detach().clone()
            inputs_3d_flip[:, :, :, 0] *= -1
            inputs_3d_flip[:, :, out_left + out_right, :] = inputs_3d_flip[:, :, out_right + out_left, :]

            optimizer.zero_grad()
            flip_loss_3d_pos = criterion(outputs_3d_flip, inputs_3d_flip)
            flip_loss_3d_pos.backward()
            nn.utils.clip_grad_norm_(model_pos.parameters(), max_norm=1)  #
            optimizer.step()

            flip_epoch_loss_3d_pos.update(loss_3d_pos.item(), num_poses)

            if args.GAN_video_playback_input == True:
                back_inputs_2d_flip = torch.clone(inputs_2d_flip)
                back_inputs_2d_flip = back_inputs_2d_flip.view(num_poses, -1, 16, 2)
                back_inputs_2d_flip = torch.flip(back_inputs_2d_flip, dims=[1])

                back_outputs_3d_flip = model_pos(back_inputs_2d_flip)

                optimizer.zero_grad()
                back_flip_loss_3d_pos = criterion(back_outputs_3d_flip, inputs_3d_flip)
                back_flip_loss_3d_pos.backward()
                nn.utils.clip_grad_norm_(model_pos.parameters(), max_norm=1)
                optimizer.step()

                back_flip_epoch_loss_3d_pos.update(back_flip_loss_3d_pos.item(), num_poses)

            batch_time.update(time.time() - end)
            end = time.time()

            bar.suffix = 'use flip: ({batch}/{size}) Data: {data:.6f}s | Batch: {bt:.3f}s | Total: {ttl:} | ETA: {eta:} ' \
                         '| Loss: {loss: .4f}  | flip_Loss: {flip_Loss: .4f} ' \
                         '| back_Loss: {back_Loss: .4f}  | back_flip_Loss: {back_flip_Loss: .4f}' \
                .format(batch=i + 1, size=max_temp, data=data_time.avg, bt=batch_time.avg,
                        ttl=bar.elapsed_td, eta=bar.eta_td, loss=epoch_loss_3d_pos.avg,
                        flip_Loss=flip_epoch_loss_3d_pos.avg, back_Loss=back_epoch_loss_3d_pos.avg,
                        back_flip_Loss=back_flip_epoch_loss_3d_pos.avg)
            bar.next()

    bar.finish()

    return



def video_mode_evaluate(args, data_loader, model_pos_eval, device, summary=None,
                        writer=None, key='', tag='', flipaug='', get_pck_auc=False):

    filter_widths = [int(x) for x in args.architecture.split(',')]
    receptive_field = video_receptive_field(filter_widths)

    batch_time = AverageMeter()
    data_time = AverageMeter()
    epoch_p1 = AverageMeter()
    epoch_p2 = AverageMeter()
    epoch_auc = AverageMeter()
    epoch_pck = AverageMeter()

    model_pos_eval.eval()
    end = time.time()

    bar = Bar('Eval posenet on {}'.format(key), max=data_loader.num_frames())

    i = 0
    for cam, batch_3d, batch_2d in data_loader.next_epoch():
        targets_3d = torch.from_numpy(batch_3d.astype('float32'))
        inputs_2d = torch.from_numpy(batch_2d.astype('float32'))

        if args.posenet_name != 'mulit_farme_videopose':
            inputs_2d = inputs_2d.view(-1, receptive_field, 16, 2)
            targets_3d = targets_3d.view(-1, 1, 16, 3)

        targets_3d, inputs_2d = targets_3d.to(device), inputs_2d.to(device)

        data_time.update(time.time() - end)
        num_poses = targets_3d.shape[0]
        inputs_2d = inputs_2d.to(device)

        with torch.no_grad():
            #######
            if flipaug:  # flip the 2D pose Left <-> Right
                joints_left = [4, 5, 6, 10, 11, 12]
                joints_right = [1, 2, 3, 13, 14, 15]
                out_left = [4, 5, 6, 10, 11, 12]
                out_right = [1, 2, 3, 13, 14, 15]

                inputs_2d_flip = inputs_2d.detach().clone()
                inputs_2d_flip[:, :, :, 0] *= -1
                inputs_2d_flip[:, :, joints_left + joints_right, :] = \
                        inputs_2d_flip[:, :, joints_right + joints_left, :]
                outputs_3d_flip = model_pos_eval(inputs_2d_flip)
                outputs_3d_flip[:, :, :, 0] *= -1
                outputs_3d_flip[:, :, out_left + out_right, :] = outputs_3d_flip[:, :, out_right + out_left, :]

                outputs_3d = model_pos_eval(inputs_2d)
                outputs_3d = (outputs_3d + outputs_3d_flip) / 2.0

            else:    #
                outputs_3d = model_pos_eval(inputs_2d)

        targets_3d = targets_3d[:, :, :, :] - targets_3d[:, :, :1, :]  # the output is relative to the 0 joint
        outputs_3d = outputs_3d[:, :, :, :] - outputs_3d[:, :, :1, :]  # the output is relative to the 0 joint

        p1score = mpjpe(outputs_3d, targets_3d).item() * 1000.0
        epoch_p1.update(p1score, num_poses)

        outputs_3d = outputs_3d.cpu().numpy().reshape(-1, outputs_3d.shape[-2], outputs_3d.shape[-1])
        targets_3d = targets_3d.cpu().numpy().reshape(-1, targets_3d.shape[-2], targets_3d.shape[-1])

        p2score = p_mpjpe(outputs_3d, targets_3d).item() * 1000.0
        epoch_p2.update(p2score, num_poses)

        if get_pck_auc == True:
            pck = compute_PCK(targets_3d, outputs_3d)
            epoch_pck.update(pck, num_poses)
            auc = compute_AUC(targets_3d, outputs_3d)
            epoch_auc.update(auc, num_poses)

        batch_time.update(time.time() - end)
        end = time.time()

        bar.suffix = '({batch}/{size}) Data: {data:.6f}s | Batch: {bt:.3f}s | Total: {ttl:} | ETA: {eta:} ' \
                     '| MPJPE: {e1: .4f} | P-MPJPE: {e2: .4f} | PCK: {pck: .4f} | AUC: {auc: .4f}' \
            .format(batch=i + 1, size=data_loader.num_frames(), data=data_time.avg, bt=batch_time.avg,
                    ttl=bar.elapsed_td, eta=bar.eta_td, e1=epoch_p1.avg, e2=epoch_p2.avg,
                    pck=epoch_pck.avg, auc=epoch_auc.avg)

        bar.next()

        i += 1

    if writer:
        writer.add_scalar('posenet_{}'.format(key) + flipaug + '/p1score' + tag, epoch_p1.avg, summary.epoch)
        writer.add_scalar('posenet_{}'.format(key) + flipaug + '/p2score' + tag, epoch_p2.avg, summary.epoch)
        writer.add_scalar('posenet_{}'.format(key) + flipaug + '/_pck' + tag, epoch_pck.avg, summary.epoch)
        writer.add_scalar('posenet_{}'.format(key) + flipaug + '/_auc' + tag, epoch_auc.avg, summary.epoch)

    bar.finish()
    return epoch_p1.avg, epoch_p2.avg, epoch_pck.avg, epoch_auc.avg


def video_mode_evaluate_posenet(args, data_dict, model_pos, model_pos_eval,
                                device, summary, writer, tag, get_pck_auc=False):

    with torch.no_grad():
        model_pos_eval.load_state_dict(model_pos.state_dict())
        h36m_p1, h36m_p2, _, _ = video_mode_evaluate(args, data_dict['H36M_test'], model_pos_eval, device, summary, writer,
                                             key='H36M_test', tag=tag, flipaug='')  # no flip aug for h36m

        dhp_p1, dhp_p2, PCK, AUC = video_mode_evaluate(args, data_dict['mpi3d_loader'], model_pos_eval, device, summary, writer,
                                            key='mpi3d_loader', tag=tag, flipaug='_flip',
                                            get_pck_auc=get_pck_auc)    # flipaug='_flip'
    return h36m_p1, h36m_p2, dhp_p1, dhp_p2, PCK, AUC


def video_mode_random_bl_aug(x):
    bl_15segs_templates_mdifyed = np.load('./data_extra/bone_length_npy/hm36s15678_bl_templates.npy')

    root = x[:, :1, :] * 1.0
    x = x - x[:, :1, :]

    # extract length, unit bone vec
    bones_unit = get_bone_unit_vecbypose3d(x)

    tmp_idx = np.random.choice(bl_15segs_templates_mdifyed.shape[0], 1)
    bones_length = torch.from_numpy(bl_15segs_templates_mdifyed[tmp_idx].astype('float32')).unsqueeze(2)

    modifyed_bone = bones_unit * bones_length.to(x.device)

    # convert bone vec back to pose3d
    out = get_pose3dbyBoneVec(modifyed_bone)

    return out + root  # return the pose with position information.

def video_mode_dataloader_update(args, data_dict, device):

    batch_time = AverageMeter()
    data_time = AverageMeter()

    end = time.time()

    buffer_poses_train = []
    buffer_poses_train_2d = []
    buffer_actions_train = []
    buffer_cams_train = []
    bar = Bar('Update training loader', max=len(data_dict['poses_train']))

    i = 0
    for targets_3d, _, action, cam_param in zip(
                                                    data_dict['poses_train'],
                                                    data_dict['poses_train_2d'],
                                                    data_dict['actions_train'],
                                                    data_dict['cams_train']):
        data_time.update(time.time() - end)
        num_poses = targets_3d.shape[0]

        targets_3d = torch.from_numpy(targets_3d.astype('float32'))
        cam_param = torch.from_numpy(cam_param.astype('float32'))
        targets_3d, cam_param = targets_3d.to(device), cam_param.to(device)

        targets_3d = video_mode_random_bl_aug(targets_3d)

        used_cam_param = torch.zeros(targets_3d.shape[0], 9).to(device)    #
        used_cam_param[:] = cam_param[:9]
        inputs_2d = project_to_2d(targets_3d, used_cam_param)

        buffer_poses_train.append(targets_3d.detach().cpu().numpy())
        buffer_poses_train_2d.append(inputs_2d.detach().cpu().numpy())
        buffer_actions_train.append(action)
        buffer_cams_train.append(cam_param.detach().cpu().numpy())

        batch_time.update(time.time() - end)
        end = time.time()

        bar.suffix = '({batch}/{size}) Data: {data:.6f}s | Batch: {bt:.3f}s | Total: {ttl:} | ETA: {eta:} ' \
            .format(batch=i + 1, size=len(data_dict['poses_train']), data=data_time.avg, bt=batch_time.avg,
                    ttl=bar.elapsed_td, eta=bar.eta_td)
        bar.next()

        i += 1

    bar.finish()

    assert len(buffer_poses_train) == len(buffer_poses_train_2d)
    assert len(buffer_poses_train) == len(buffer_actions_train)
    assert len(buffer_poses_train) == len(buffer_cams_train)

    joints_left = [4, 5, 6, 10, 11, 12]
    joints_right = [1, 2, 3, 13, 14, 15]
    out_left = [4, 5, 6, 10, 11, 12]
    out_right = [1, 2, 3, 13, 14, 15]

    filter_widths = [int(x) for x in args.architecture.split(',')]
    receptive_field = video_receptive_field(filter_widths)

    pad = (receptive_field - 1) // 2

    data_dict['target_GAN_loader'] = \
        GAN_video_ChunkedGenerator(args.batch_size//1, copy.deepcopy(buffer_cams_train),
                                   copy.deepcopy(buffer_poses_train),
                         copy.deepcopy(buffer_poses_train_2d), chunk_length=1,
                        pad=pad, causal_shift=0, shuffle=True, augment=False,
                        kps_left=out_left, kps_right=out_right, joints_left=joints_left, joints_right=joints_right)

    return
