import os
import sys
sys.path.append(os.getcwd())

import time

import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
#import sklearn.datasets

# import tflib as lib
# import tflib.save_images
# import tflib.mnist
# import tflib.plot

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import collections
import time
import _pickle as pickle
#import cPickle as pickle

#from main_generator import MyFigure

from models_Fk_GAN import special_operate
from models_Fk_GAN.forward_kinematics_DH_model import used_16key_15bone_len_table
from models_Fk_GAN.special_operate import myResNet, Fk_get_boneVecByPose3d


def special_KCS_Input_transform(pos_16_3d, device):
    pos_16_3d = pos_16_3d.view(-1, 16, 3)

    bone_vector_from_mat = Fk_get_boneVecByPose3d(pos_16_3d)
    bone_vector_from_mat = bone_vector_from_mat.transpose(1, 0)   # 维度转换成15*N*3
    bone_length_from_mat = torch.sqrt(torch.sum(bone_vector_from_mat ** 2, dim=-1))
    bone_vector = bone_vector_from_mat
    bone_length = bone_length_from_mat

    left_small_leg_vector = bone_vector[0]
    right_small_leg_vector = bone_vector[1]
    left_big_leg_vector = bone_vector[2]
    right_big_leg_vector = bone_vector[3]
    left_hip_vector = bone_vector[4]
    right_hip_vector = bone_vector[5]
    waist_vector = bone_vector[6]
    thorax_vector = bone_vector[7]
    left_shoulder_vector = bone_vector[8]
    right_shoulder_vector = bone_vector[9]
    left_big_arm_vector = bone_vector[10]
    right_big_arm_vector = bone_vector[11]
    left_small_arm_vector = bone_vector[12]
    right_small_arm_vector = bone_vector[13]
    neck_vector = bone_vector[14]

    left_small_leg_len = bone_length[0]
    right_small_leg_len = bone_length[1]
    left_big_leg_len = bone_length[2]
    right_big_leg_len = bone_length[3]
    left_hip_len = bone_length[4]
    right_hip_len = bone_length[5]
    waist_len = bone_length[6]
    thorax_len = bone_length[7]
    left_shoulder_len = bone_length[8]
    right_shoulder_len = bone_length[9]
    left_big_arm_len = bone_length[10]
    right_big_arm_len = bone_length[11]
    left_small_arm_len = bone_length[12]
    right_small_arm_len = bone_length[13]
    neck_len = bone_length[14]

    special_KCS_input = torch.zeros((30, pos_16_3d.shape[0]), dtype=torch.float32)

    special_KCS_input = special_KCS_input.to(device)

    special_KCS_input[0] = \
        torch.sum(left_small_leg_vector * left_big_leg_vector, dim=-1) / (   # 左小腿和左大腿
                        left_small_leg_len * left_big_leg_len)

    special_KCS_input[1] = \
        torch.sum(right_small_leg_vector * right_big_leg_vector, dim=-1) / (  # 右小腿和右大腿
                        right_small_leg_len * right_big_leg_len)

    special_KCS_input[2] = \
        torch.sum(left_big_leg_vector * left_hip_vector, dim=-1) / (  # 左大腿和左髋
                        left_big_leg_len * left_hip_len)

    special_KCS_input[3] = \
        torch.sum(right_big_leg_vector * right_hip_vector, dim=-1) / (   # 右大腿和右髋
                        right_big_leg_len * right_hip_len)

    special_KCS_input[4] = \
        torch.sum(left_hip_vector * right_hip_vector, dim=-1) / (   # 左髋和右髋
                        left_hip_len * right_hip_len)

    special_KCS_input[5] = \
        torch.sum(left_hip_vector * waist_vector, dim=-1) / (    # 左髋和腰部
                        left_hip_len * waist_len)

    special_KCS_input[6] = \
        torch.sum(right_hip_vector * waist_vector, dim=-1) / (    # 右髋和腰部
                        right_hip_len * waist_len)

    special_KCS_input[7] = \
        torch.sum(waist_vector * thorax_vector, dim=-1) / (  # 腰部和胸部
                        waist_len * thorax_len)

    special_KCS_input[8] = \
        torch.sum(thorax_vector * neck_vector, dim=-1) / (    # 胸部和颈部
                        thorax_len * neck_len)


    special_KCS_input[9] = \
        torch.sum(thorax_vector * left_shoulder_vector, dim=-1) / (    # 胸部和左肩膀
                        thorax_len * left_shoulder_len)

    special_KCS_input[10] = \
        torch.sum(thorax_vector * right_shoulder_vector, dim=-1) / (    # 胸部和右肩膀
                        thorax_len * right_shoulder_len)

    special_KCS_input[11] = \
        torch.sum(left_shoulder_vector * left_big_arm_vector, dim=-1) / (    # 左肩膀和左大臂
                        left_shoulder_len * left_big_arm_len)

    special_KCS_input[12] = \
        torch.sum(right_shoulder_vector * right_big_arm_vector, dim=-1) / (    # 右肩膀和右大臂
                        right_shoulder_len * right_big_arm_len)

    special_KCS_input[13] = \
        torch.sum(left_big_arm_vector * left_small_arm_vector, dim=-1) / (    # 左大臂和左小臂
                        left_big_arm_len * left_small_arm_len)

    special_KCS_input[14] = \
        torch.sum(right_big_arm_vector * right_small_arm_vector, dim=-1) / (  # 右大臂和右小臂
                    right_big_arm_len * right_small_arm_len)

    special_KCS_input[15:] = bone_length[0:]

    special_KCS_input = special_KCS_input.transpose(0, 1)

    return special_KCS_input


class Fk_3D_Discriminator(nn.Module):
    def __init__(self, device, args):
        super(Fk_3D_Discriminator, self).__init__()

        self.device = device
        self.args = args

        self.previous = nn.Sequential(
            nn.Linear(16*3, self.args.Dis_DenseDim_3D),
            nn.ReLU(True),
        )
        self.block1 = myResNet(self.args.Dis_DenseDim_3D)
        self.block2 = myResNet(self.args.Dis_DenseDim_3D)
        self.block3 = myResNet(self.args.Dis_DenseDim_3D)

        self.special_KCS_previous = nn.Sequential(
            nn.Linear(30, self.args.Dis_DenseDim_3D),
            nn.ReLU(True),
        )

        self.special_KCS_block1 = myResNet(self.args.Dis_DenseDim_3D)
        self.special_KCS_block2 = myResNet(self.args.Dis_DenseDim_3D)
        self.special_KCS_block3 = myResNet(self.args.Dis_DenseDim_3D)

        self.merge_previous = nn.Sequential(
            nn.Linear((self.args.Dis_DenseDim_3D + self.args.Dis_DenseDim_3D), 100),
            nn.ReLU(True),
        )
        self.merge_block1 = myResNet(100)
        self.output = nn.Linear(100, 1)    #

    def forward(self, input):
        special_KCS_input = special_KCS_Input_transform(torch.clone(input), self.device)

        special_KCS_input = special_KCS_input.contiguous().view(-1, 30)
        special_KCS_out = self.special_KCS_previous(special_KCS_input)
        special_KCS_out = self.special_KCS_block1(special_KCS_out)
        special_KCS_out = self.special_KCS_block2(special_KCS_out)
        special_KCS_out = self.special_KCS_block3(special_KCS_out)

        pos_input = input.contiguous().view(-1, 16*3)
        pos_out = self.previous(pos_input)
        pos_out = self.block1(pos_out)
        pos_out = self.block2(pos_out)
        pos_out = self.block3(pos_out)

        out = torch.cat((special_KCS_out, pos_out), dim=-1)

        out = self.merge_previous(out)
        out = self.merge_block1(out)
        out = self.output(out)

        return out



def calc_gradient_penalty(netD, real_data, fake_data, BATCH_SIZE, LAMBDA, device):

    real_data = real_data.view(BATCH_SIZE, -1)
    fake_data = fake_data.view(BATCH_SIZE, -1)

    alpha = torch.rand(BATCH_SIZE, 1)


    alpha = alpha.expand(real_data.size())
    alpha = alpha.to(device)

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    interpolates = interpolates.to(device)

    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=
                              torch.ones(disc_interpolates.size(), dtype=torch.float32).to(device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA

    return gradient_penalty

# ==================Definition End======================


class Fk_2D_Discriminator(nn.Module):
    def __init__(self, args, num_joints=16):
        super(Fk_2D_Discriminator, self).__init__()

        self.args = args

        # Pose path
        self.pose_layer_1 = nn.Linear(num_joints * 2, self.args.Dis_DenseDim_2D)
        self.pose_layer_2 = nn.Linear(self.args.Dis_DenseDim_2D, self.args.Dis_DenseDim_2D)
        self.pose_layer_3 = nn.Linear(self.args.Dis_DenseDim_2D, self.args.Dis_DenseDim_2D)
        self.pose_layer_4 = nn.Linear(self.args.Dis_DenseDim_2D, self.args.Dis_DenseDim_2D)

        self.layer_last = nn.Linear(self.args.Dis_DenseDim_2D, self.args.Dis_DenseDim_2D)
        self.layer_pred = nn.Linear(self.args.Dis_DenseDim_2D, 1)

        self.relu = nn.LeakyReLU()

    def forward(self, x):
        # Pose path

        x = x.contiguous().view(-1, 16 * 2)  # (1024, 27, 16, 2) => (1024*27, 32)

        d1 = self.relu(self.pose_layer_1(x))
        d2 = self.relu(self.pose_layer_2(d1))
        d3 = self.relu(self.pose_layer_3(d2) + d1)
        d4 = self.pose_layer_4(d3)

        d_last = self.relu(self.layer_last(d4))
        d_out = self.layer_pred(d_last)

        return d_out


def video_mode_special_KCS_Input_transform(pos_16_3d, device):
    pos_16_3d = pos_16_3d.view(-1, 16, 3)

    bone_vector_from_mat = Fk_get_boneVecByPose3d(pos_16_3d)
    bone_vector_from_mat = bone_vector_from_mat.transpose(1, 0)
    bone_length_from_mat = torch.sqrt(torch.sum(bone_vector_from_mat ** 2, dim=-1))
    bone_vector = bone_vector_from_mat
    bone_length = bone_length_from_mat

    left_small_leg_vector = bone_vector[0]
    right_small_leg_vector = bone_vector[1]
    left_big_leg_vector = bone_vector[2]
    right_big_leg_vector = bone_vector[3]
    left_hip_vector = bone_vector[4]
    right_hip_vector = bone_vector[5]
    waist_vector = bone_vector[6]
    thorax_vector = bone_vector[7]
    left_shoulder_vector = bone_vector[8]
    right_shoulder_vector = bone_vector[9]
    left_big_arm_vector = bone_vector[10]
    right_big_arm_vector = bone_vector[11]
    left_small_arm_vector = bone_vector[12]
    right_small_arm_vector = bone_vector[13]
    neck_vector = bone_vector[14]

    left_small_leg_len = bone_length[0]
    right_small_leg_len = bone_length[1]
    left_big_leg_len = bone_length[2]
    right_big_leg_len = bone_length[3]
    left_hip_len = bone_length[4]
    right_hip_len = bone_length[5]
    waist_len = bone_length[6]
    thorax_len = bone_length[7]
    left_shoulder_len = bone_length[8]
    right_shoulder_len = bone_length[9]
    left_big_arm_len = bone_length[10]
    right_big_arm_len = bone_length[11]
    left_small_arm_len = bone_length[12]
    right_small_arm_len = bone_length[13]
    neck_len = bone_length[14]

    special_KCS_input = torch.zeros((15, pos_16_3d.shape[0]), dtype=torch.float32)

    special_KCS_input = special_KCS_input.to(device)

    special_KCS_input[0] = \
        torch.sum(left_small_leg_vector * left_big_leg_vector, dim=-1) / (   # 左小腿和左大腿
                        left_small_leg_len * left_big_leg_len)

    special_KCS_input[1] = \
        torch.sum(right_small_leg_vector * right_big_leg_vector, dim=-1) / (  # 右小腿和右大腿
                        right_small_leg_len * right_big_leg_len)

    special_KCS_input[2] = \
        torch.sum(left_big_leg_vector * left_hip_vector, dim=-1) / (  # 左大腿和左髋
                        left_big_leg_len * left_hip_len)

    special_KCS_input[3] = \
        torch.sum(right_big_leg_vector * right_hip_vector, dim=-1) / (   # 右大腿和右髋
                        right_big_leg_len * right_hip_len)

    special_KCS_input[4] = \
        torch.sum(left_hip_vector * right_hip_vector, dim=-1) / (   # 左髋和右髋
                        left_hip_len * right_hip_len)

    special_KCS_input[5] = \
        torch.sum(left_hip_vector * waist_vector, dim=-1) / (    # 左髋和腰部
                        left_hip_len * waist_len)

    special_KCS_input[6] = \
        torch.sum(right_hip_vector * waist_vector, dim=-1) / (    # 右髋和腰部
                        right_hip_len * waist_len)

    special_KCS_input[7] = \
        torch.sum(waist_vector * thorax_vector, dim=-1) / (  # 腰部和胸部
                        waist_len * thorax_len)

    special_KCS_input[8] = \
        torch.sum(thorax_vector * neck_vector, dim=-1) / (    # 胸部和颈部
                        thorax_len * neck_len)


    special_KCS_input[9] = \
        torch.sum(thorax_vector * left_shoulder_vector, dim=-1) / (    # 胸部和左肩膀
                        thorax_len * left_shoulder_len)

    special_KCS_input[10] = \
        torch.sum(thorax_vector * right_shoulder_vector, dim=-1) / (    # 胸部和右肩膀
                        thorax_len * right_shoulder_len)

    special_KCS_input[11] = \
        torch.sum(left_shoulder_vector * left_big_arm_vector, dim=-1) / (    # 左肩膀和左大臂
                        left_shoulder_len * left_big_arm_len)

    special_KCS_input[12] = \
        torch.sum(right_shoulder_vector * right_big_arm_vector, dim=-1) / (    # 右肩膀和右大臂
                        right_shoulder_len * right_big_arm_len)

    special_KCS_input[13] = \
        torch.sum(left_big_arm_vector * left_small_arm_vector, dim=-1) / (    # 左大臂和左小臂
                        left_big_arm_len * left_small_arm_len)

    special_KCS_input[14] = \
        torch.sum(right_big_arm_vector * right_small_arm_vector, dim=-1) / (  # 右大臂和右小臂
                    right_big_arm_len * right_small_arm_len)

    special_KCS_input = special_KCS_input.transpose(0, 1)

    return special_KCS_input



class Video_motion_Fk_3D_Discriminator(nn.Module):
    def __init__(self, device, args, video_frame_num):
        super(Video_motion_Fk_3D_Discriminator, self).__init__()

        self.video_frame_num = video_frame_num
        self.device = device
        self.args = args

        ########### 1：
        self.special_KCS_previous = nn.Sequential(
            nn.Linear(self.video_frame_num * 15, self.args.video_Dis_DenseDim_3D),
            nn.ReLU(True),
        )

        self.special_KCS_block1 = myResNet(self.args.video_Dis_DenseDim_3D)
        self.special_KCS_block2 = myResNet(self.args.video_Dis_DenseDim_3D)
        self.special_KCS_block3 = myResNet(self.args.video_Dis_DenseDim_3D)

        self.diff_special_KCS_previous = nn.Sequential(
            nn.Linear((self.video_frame_num - 1) * 15, self.args.video_Dis_DenseDim_3D),
            nn.ReLU(True),
        )
        self.diff_special_KCS_block1 = myResNet(self.args.video_Dis_DenseDim_3D)
        self.diff_special_KCS_block2 = myResNet(self.args.video_Dis_DenseDim_3D)
        self.diff_special_KCS_block3 = myResNet(self.args.video_Dis_DenseDim_3D)

        self.pos_3d_previous = nn.Sequential(
            nn.Linear(self.video_frame_num * 16 * 3, self.args.video_Dis_DenseDim_3D),
            nn.ReLU(True),
        )
        self.pos_3d_block1 = myResNet(self.args.video_Dis_DenseDim_3D)
        self.pos_3d_block2 = myResNet(self.args.video_Dis_DenseDim_3D)
        self.pos_3d_block3 = myResNet(self.args.video_Dis_DenseDim_3D)

        self.diff_pos_3d_previous = nn.Sequential(
            nn.Linear((self.video_frame_num - 1) * 16 * 3, self.args.video_Dis_DenseDim_3D),
            nn.ReLU(True),
        )
        self.diff_pos_3d_block1 = myResNet(self.args.video_Dis_DenseDim_3D)
        self.diff_pos_3d_block2 = myResNet(self.args.video_Dis_DenseDim_3D)
        self.diff_pos_3d_block3 = myResNet(self.args.video_Dis_DenseDim_3D)

        self.branch_num = 2
        if (args.motion_Dis_whether_use_3dPos_branch == True and args.motion_Dis_whether_use_3dDiff_branch==True):
            self.branch_num = 4
        elif (args.motion_Dis_whether_use_3dPos_branch == True and args.motion_Dis_whether_use_3dDiff_branch==False
        ) or (args.motion_Dis_whether_use_3dPos_branch == False and args.motion_Dis_whether_use_3dDiff_branch==True):
            self.branch_num = 3

        self.kcs_merge_previous = nn.Sequential(
            nn.Linear((self.args.video_Dis_DenseDim_3D * self.branch_num), 100),
            nn.ReLU(True),
        )
        self.kcs_merge_block1 = myResNet(100)
        self.kcs_output = nn.Linear(100, 1)

    def forward(self, input):

        input = input.view(-1, 16*3)    # (1024, 27, 16, 3) => (1024*27, 48)

        #  (1024*27, 15)
        special_KCS_input = video_mode_special_KCS_Input_transform(torch.clone(input), self.device)

        special_KCS_input = special_KCS_input.contiguous().view(-1, self.video_frame_num*15)

        special_KCS_out = self.special_KCS_previous(special_KCS_input)
        special_KCS_out = self.special_KCS_block1(special_KCS_out)
        special_KCS_out = self.special_KCS_block2(special_KCS_out)
        special_KCS_out = self.special_KCS_block3(special_KCS_out)

        diff_input_temp = torch.clone(special_KCS_input)
        diff_input_temp = diff_input_temp.view(-1, self.video_frame_num, 15)  #(1024, 27, 15)

        diff_special_KCS_input = torch.zeros(diff_input_temp.shape[0],
                                             diff_input_temp.shape[1]-1,
                                             diff_input_temp.shape[2], dtype=torch.float32)
        diff_special_KCS_input = diff_special_KCS_input.to(self.device)
        for (diff_first_id, diff_second_id) in zip(range(0, self.video_frame_num-1), range(1, self.video_frame_num)):   # 是多帧之间做差(相邻帧)
            diff_special_KCS_input[:, diff_first_id, :] =   \
                torch.clone(diff_input_temp[:, diff_second_id, :]) - torch.clone(diff_input_temp[:, diff_first_id, :])

        diff_special_KCS_input = diff_special_KCS_input.view(-1, (self.video_frame_num-1)*15)  #(1024, 26*15)

        diff_special_KCS_out = self.diff_special_KCS_previous(diff_special_KCS_input)
        diff_special_KCS_out = self.diff_special_KCS_block1(diff_special_KCS_out)
        diff_special_KCS_out = self.diff_special_KCS_block2(diff_special_KCS_out)
        diff_special_KCS_out = self.diff_special_KCS_block3(diff_special_KCS_out)

        pos_3d_out = 0
        if self.args.motion_Dis_whether_use_3dPos_branch == True:
            pos_3d_input = torch.clone(input)
            pos_3d_input = pos_3d_input.contiguous().view(-1, self.video_frame_num * 16 * 3)

            pos_3d_out = self.pos_3d_previous(pos_3d_input)
            pos_3d_out = self.pos_3d_block1(pos_3d_out)
            pos_3d_out = self.pos_3d_block2(pos_3d_out)
            pos_3d_out = self.pos_3d_block3(pos_3d_out)

        diff_3d_out = 0
        if self.args.motion_Dis_whether_use_3dDiff_branch == True:
            diff_3d_input_temp = torch.clone(input)
            diff_3d_input_temp = \
                diff_3d_input_temp.contiguous().view(-1, self.video_frame_num, 16 * 3)
            diff_3d_input = torch.zeros(diff_3d_input_temp.shape[0],
                                            diff_3d_input_temp.shape[1] - 1,
                                            diff_3d_input_temp.shape[2], dtype=torch.float32)

            diff_3d_input = diff_3d_input.to(self.device)
            for (diff_first_id, diff_second_id) in zip(range(0, self.video_frame_num-1), range(1, self.video_frame_num)):
                diff_3d_input[:, diff_first_id, :] = \
                    torch.clone(diff_3d_input_temp[:, diff_second_id, :]) - \
                    torch.clone(diff_3d_input_temp[:, diff_first_id, :])

            diff_3d_input = diff_3d_input.view(-1, (self.video_frame_num - 1) * 16 * 3)  # (1024, 26*16*3)

            diff_3d_out = self.diff_pos_3d_previous(diff_3d_input)
            diff_3d_out = self.diff_pos_3d_block1(diff_3d_out)
            diff_3d_out = self.diff_pos_3d_block2(diff_3d_out)
            diff_3d_out = self.diff_pos_3d_block3(diff_3d_out)

        out = torch.cat((special_KCS_out, diff_special_KCS_out), dim=-1)

        if self.args.motion_Dis_whether_use_3dPos_branch == True:
            out = torch.cat((out, pos_3d_out), dim=-1)

        if self.args.motion_Dis_whether_use_3dDiff_branch == True:
            out = torch.cat((out, diff_3d_out), dim=-1)
        out = self.kcs_merge_previous(out)
        out = self.kcs_merge_block1(out)
        out = self.kcs_output(out)

        return out


########### 2D
class Video_motion_Fk_2D_Discriminator(nn.Module):  #####
    def __init__(self, device, args, video_frame_num):
        super(Video_motion_Fk_2D_Discriminator, self).__init__()

        self.video_frame_num = video_frame_num
        self.device = device
        self.args = args

        self.pos_2d_previous = nn.Sequential(
            nn.Linear(self.video_frame_num*16*2, self.args.video_Dis_DenseDim_2D),
            nn.ReLU(True),
        )
        self.pos_2d_block1 = myResNet(self.args.video_Dis_DenseDim_2D)
        self.pos_2d_block2 = myResNet(self.args.video_Dis_DenseDim_2D)
        self.pos_2d_block3 = myResNet(self.args.video_Dis_DenseDim_2D)

        self.root_diff_2d_previous = nn.Sequential(
            nn.Linear((self.video_frame_num-1)*1*2, self.args.video_Dis_DenseDim_2D),
            nn.ReLU(True),
        )
        self.root_diff_2d_block1 = myResNet(self.args.video_Dis_DenseDim_2D)
        self.root_diff_2d_block2 = myResNet(self.args.video_Dis_DenseDim_2D)
        self.root_diff_2d_block3 = myResNet(self.args.video_Dis_DenseDim_2D)

        self.merge_previous = nn.Sequential(

            nn.Linear((self.args.video_Dis_DenseDim_2D + self.args.video_Dis_DenseDim_2D), 100),
            nn.ReLU(True),
        )
        self.merge_block1 = myResNet(100)
        self.merge_output = nn.Linear(100, 1)  #

    def forward(self, input):

        input = input.contiguous().view(-1, 16*2)  # (1024, 27, 16, 2) => (1024*27, 32)

        pos_2d_input = torch.clone(input)

        pos_2d_input = pos_2d_input.contiguous().view(-1, self.video_frame_num * 16 * 2)

        pos_2d_out = self.pos_2d_previous(pos_2d_input)
        pos_2d_out = self.pos_2d_block1(pos_2d_out)
        pos_2d_out = self.pos_2d_block2(pos_2d_out)
        pos_2d_out = self.pos_2d_block3(pos_2d_out)

        input = input.contiguous().view(-1, 16, 2)
        root_diff_2d_input_temp = torch.clone(input[:, 0, :])
        root_diff_2d_input_temp = \
            root_diff_2d_input_temp.contiguous().view(-1, self.video_frame_num, 2)  # (1024, 27, 2)
        root_diff_2d_input = torch.zeros(root_diff_2d_input_temp.shape[0],
                                        root_diff_2d_input_temp.shape[1] - 1,
                                        root_diff_2d_input_temp.shape[2], dtype=torch.float32)

        root_diff_2d_input = root_diff_2d_input.to(self.device)
        for (diff_first_id, diff_second_id) in zip(range(0, self.video_frame_num-1), range(1, self.video_frame_num)):
            root_diff_2d_input[:, diff_first_id, :] = \
                torch.clone(root_diff_2d_input_temp[:, diff_second_id, :]) - \
                torch.clone(root_diff_2d_input_temp[:, diff_first_id, :])

        root_diff_2d_input = root_diff_2d_input.view(-1, (self.video_frame_num - 1) * 2)  # (1024, 26*15)

        root_diff_2d_out = self.root_diff_2d_previous(root_diff_2d_input)
        root_diff_2d_out = self.root_diff_2d_block1(root_diff_2d_out)
        root_diff_2d_out = self.root_diff_2d_block2(root_diff_2d_out)
        root_diff_2d_out = self.root_diff_2d_block3(root_diff_2d_out)

        out = torch.cat((pos_2d_out, root_diff_2d_out), dim=-1)
        out = self.merge_previous(out)
        out = self.merge_block1(out)
        out = self.merge_output(out)

        return out