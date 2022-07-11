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


from models_Fk_GAN import special_operate

from common.h36m_dataset import H36M_32_To_16_Table
from models_Fk_GAN.forward_kinematics_DH_model import used_16key_15bone_len_table
from models_Fk_GAN.special_operate import myResNet, Fk_get_boneVecByPose3d

GAN_global_rotation_table = {
    'angle_x': {'range': (-180, 180), 'changeRate': (-5, 5)},
    'angle_y': {'range': (-180, 180), 'changeRate': (-5, 5)},
    'angle_z': {'range': (-180, 180), 'changeRate': (-5, 5)}
}

GAN_angle_range_table = {
    'joint1': {'range': (-90-20, 45+20), 'changeRate': (-20, 20)},
    'joint2': {'range': (-90-20, 45+20), 'changeRate': (-20, 20)},
    'joint3': {'range': (-90-20, 180), 'changeRate': (-30, 30)},
    'joint4': {'range': (-180, 0), 'changeRate': (-40, 40)},
    'joint5': {'range': (0, 0), 'changeRate': (-20, 20)},
    'joint6': {'range': (-45-20, 90+20), 'changeRate': (-20, 20)},
    'joint7': {'range': (-45-20, 90+20), 'changeRate': (-20, 20)},
    'joint8': {'range': (-90-20, 180), 'changeRate': (-30, 30)},
    'joint9': {'range': (-180, 0), 'changeRate': (-40, 40)},
    'joint10': {'range': (0, 0), 'changeRate': (-20, 20)},
    'joint11': {'range': (-180, 180), 'changeRate': (-20, 20)},
    'joint12': {'range': (-180, 180), 'changeRate': (-20, 20)},    #
    'joint13': {'range': (-180, 180), 'changeRate': (-20, 20)},    #
    'joint14': {'range': (-180, 180), 'changeRate': (-20, 20)},    #
    'joint15': {'range': (-180, 180), 'changeRate': (-20, 20)},    #
    'joint16': {'range': (-180, 180), 'changeRate': (-20, 20)},    #
    'joint17': {'range': (-180, 180), 'changeRate': (-20, 20)},    # 从
    'joint18': {'range': (-180, 180), 'changeRate': (-20, 20)},  #
    'joint19': {'range': (-180, 180), 'changeRate': (-20, 20)},    #
    'joint20': {'range': (-180, 180), 'changeRate': (-20, 20)},
    'joint21': {'range': (-180, 180), 'changeRate': (-20, 20)},    #
    'joint22': {'range': (-180, 180), 'changeRate': (-20, 20)},    #
    'joint23': {'range': (0, 0), 'changeRate': (-20, 20)},    #
    'joint24': {'range': (0, 0), 'changeRate': (-20, 20)},    #
    'joint25': {'range': (-135-20, 45+20), 'changeRate': (-20, 20)},
    'joint26': {'range': (-135-20, 45+20), 'changeRate': (-20, 20)},
    'joint27': {'range': (-80-20, 180), 'changeRate': (-30, 30)},
    'joint28': {'range': (0, 180), 'changeRate': (-40, 40)},  #
    'joint29': {'range': (0, 0), 'changeRate': (-20, 20)},
    'joint30': {'range': (-45-20, 135+20), 'changeRate': (-20, 20)},
    'joint31': {'range': (-45-20, 135+20), 'changeRate': (-20, 20)},
    'joint32': {'range': (-80-20, 180), 'changeRate': (-30, 30)},
    'joint33': {'range': (0, 180), 'changeRate': (-40, 40)},  #
    'joint34': {'range': (0, 0), 'changeRate': (-20, 20)}
}


class Fk_Generator(nn.Module):
    def __init__(self, FK_DH_Class, args, device, INPUT_VEC_DIM=128):
        super(Fk_Generator, self).__init__()

        self.OUTPUT_DIM = args.GAN_OUTPUT_DIM
        self.BATCH_SIZE = args.batch_size
        self.FK_DH_Class = FK_DH_Class
        self.train_num = 0
        self.args = args
        self.INPUT_VEC_DIM = INPUT_VEC_DIM

        self.boneLength = torch.zeros((self.BATCH_SIZE, 15), dtype=torch.float32)
        self.device = device

        self.distribute_angle = []

        self.preprocess = nn.Sequential(
            nn.Linear(INPUT_VEC_DIM, self.args.Gen_DenseDim),
            nn.ReLU(True),
        )

        self.block1 = myResNet(self.args.Gen_DenseDim)    # 1000
        self.block2 = myResNet(self.args.Gen_DenseDim)
        self.block3 = myResNet(self.args.Gen_DenseDim)
        self.deconv_out = nn.Linear(self.args.Gen_DenseDim, self.OUTPUT_DIM)
        self.sigmoid = nn.Sigmoid()
        self.Tanh = nn.Tanh()

    def GAN_generator_get_bone_length(self, input):

        bone_vector_from_mat = Fk_get_boneVecByPose3d(input)
        bone_length_from_mat = torch.sqrt(torch.sum(bone_vector_from_mat ** 2, dim=-1))
        self.boneLength = bone_length_from_mat


    def forward(self, input):
        output = self.preprocess(input)
        output = self.block1(output)
        output = self.block2(output)
        output = self.block3(output)
        output = self.deconv_out(output)

        output[:, :-3] = self.Tanh(output[:, :-3])
        output[:, -3:] = self.Tanh(output[:, -3:]) * 10.0

        output.contiguous().view(-1, self.OUTPUT_DIM)

        root_3d_pos = output[:, -3:]
        if torch.cuda.is_available():
            root_3d_pos = root_3d_pos.to(self.device)

        generator_angle = torch.zeros((output.shape[0], self.OUTPUT_DIM + 5 - 3), dtype=torch.float32)

        generator_angle = generator_angle.to(self.device)

        output_index = 0
        for index in range(self.OUTPUT_DIM + 5 - 3):
            if (index==4) or (index==9) or (index==23) or (index==28) or (index==33) or (index==22):
                generator_angle[:, index] = 0

            else:
                generator_angle[:, index] = output[:, output_index]
                output_index += 1

        if self.args.GAN_whether_use_preAngle == True:
            for index in range(self.OUTPUT_DIM + 5 - 3):
                if index < 34:
                    generator_angle[:, index] = \
                        generator_angle[:, index] * \
                        (GAN_angle_range_table['joint' + str(index + 1)]['range'][1] -
                         GAN_angle_range_table['joint' + str(index + 1)]['range'][0])/2 + \
                        (GAN_angle_range_table['joint' + str(index + 1)]['range'][1] +
                         GAN_angle_range_table['joint' + str(index + 1)]['range'][0])/2

                else:  #
                    char_temp = 'x'
                    if index == 34:
                        char_temp = 'x'
                    elif index == 35:
                        char_temp = 'y'
                    elif index == 36:
                        char_temp = 'z'
                    generator_angle[:, index] = \
                        generator_angle[:, index] * \
                        (GAN_global_rotation_table['angle_' + char_temp]['range'][1] -
                         GAN_global_rotation_table['angle_' + char_temp]['range'][0]) / 2 + \
                        (GAN_global_rotation_table['angle_' + char_temp]['range'][1] +
                         GAN_global_rotation_table['angle_' + char_temp]['range'][0]) / 2
        else:
            generator_angle = generator_angle * 180

        self.distribute_angle.append(generator_angle)

        self.train_num += 1
        if self.train_num % 500 == 1:
            special_operate.my_draw_DOF_angle_distribute(
                    torch.clone(generator_angle), '34DOF_3GlobalRot_heatmap_'+str(self.train_num), self.args)

            generator_angle = generator_angle.to(self.device)

        right_leg_joints_angle = generator_angle[:, 0: 5]
        left_leg_joints_angle = generator_angle[:, 5: 10]

        body_joints_angle = generator_angle[:, 10: 23]  # [:, 10: 24]
        right_hand_joints_angle = generator_angle[:, 23: 28]  # [:, 24: 29]
        left_hand_joints_angle = generator_angle[:, 28: 33]  # [:, 29: 34]

        generator_global_rot_3d_pos_angle = generator_angle[:, -3:]
        if self.args.whether_use_RT == False:
            print(generator_angle[:, -3:].shape)
            generator_global_rot_3d_pos_angle = torch.zeros(generator_angle[:, -3:].shape)

            if self.train_num % 500 == 1:
                print('不使用全局旋转')

        record_bone_len = self.boneLength
        bone_len_scaler = torch.zeros(8, dtype=torch.float32)
        if self.args.bone_len_scaler == 'different':
            bone_len_scaler = torch.randint(-200, 200, size=(self.BATCH_SIZE, 8))
            bone_len_scaler = bone_len_scaler/1000.0

        elif self.args.bone_len_scaler == 'same':
            bone_len_scaler = self.FK_DH_Class.random.randint(-200, 200, size=(self.BATCH_SIZE, 8))
            bone_len_scaler = np.array(bone_len_scaler).reshape(self.BATCH_SIZE, 1)
            bone_len_scaler = bone_len_scaler.repeat(8, axis=-1)
            bone_len_scaler = bone_len_scaler / 1000.0

        elif self.args.bone_len_scaler == '':
            bone_len_scaler = np.zeros((self.BATCH_SIZE, 8))

        else:
            raise ("args.bone_len_scaler")

        if torch.cuda.is_available():
            record_bone_len = record_bone_len.to(self.device)
            bone_len_scaler = bone_len_scaler.to(self.device)

        left_small_leg_len = record_bone_len[:, 0] * (1 + bone_len_scaler[:, 0])
        right_small_leg_len = record_bone_len[:, 1] * (1 + bone_len_scaler[:, 0])
        left_big_leg_len = record_bone_len[:, 2] * (1 + bone_len_scaler[:, 1])
        right_big_leg_len = record_bone_len[:, 3] * (1 + bone_len_scaler[:, 1])
        left_hip_len = record_bone_len[:, 4] * (1 + bone_len_scaler[:, 2])
        right_hip_len = record_bone_len[:, 5] * (1 + bone_len_scaler[:, 2])
        waist_len = record_bone_len[:, 6] * (1 + bone_len_scaler[:, 3])
        thorax_len = record_bone_len[:, 7]    #######
        left_shoulder_len = record_bone_len[:, 8] * (1 + bone_len_scaler[:, 4])
        right_shoulder_len = record_bone_len[:, 9] * (1 + bone_len_scaler[:, 4])
        left_big_arm_len = record_bone_len[:, 10] * (1 + bone_len_scaler[:, 5])
        right_big_arm_len = record_bone_len[:, 11] * (1 + bone_len_scaler[:, 5])
        left_small_arm_len = record_bone_len[:, 12] * (1 + bone_len_scaler[:, 6])
        right_small_arm_len = record_bone_len[:, 13] * (1 + bone_len_scaler[:, 6])
        neck_len = record_bone_len[:, 14] * (1 + bone_len_scaler[:, 7])

        single_generator_3d_world_32keyPoint = \
            self.FK_DH_Class.change_3d_joint_angle(
                left_leg_joints_angle=left_leg_joints_angle,
                right_leg_joints_angle=right_leg_joints_angle,
                body_joints_angle=body_joints_angle,
                left_hand_joints_angle=left_hand_joints_angle,
                right_hand_joints_angle=right_hand_joints_angle,
                generator_global_rot_3d_pos_angle=generator_global_rot_3d_pos_angle,
                left_small_leg_len=left_small_leg_len,
                right_small_leg_len=right_small_leg_len,
                left_big_leg_len=left_big_leg_len,
                right_big_leg_len=right_big_leg_len,
                left_hip_len=left_hip_len,
                right_hip_len=right_hip_len,
                waist_len=waist_len,
                thorax_len=thorax_len,
                left_shoulder_len=left_shoulder_len,
                right_shoulder_len=right_shoulder_len,
                left_big_arm_len=left_big_arm_len,
                right_big_arm_len=right_big_arm_len,
                left_small_arm_len=left_small_arm_len,
                right_small_arm_len=right_small_arm_len,
                neck_len=neck_len,
                #head_len=head_len,
                root_3d_pos=root_3d_pos
            )

        fake_Pos = single_generator_3d_world_32keyPoint[:, H36M_32_To_16_Table].view(-1, 16 * 3)

        return fake_Pos


class Video_Fk_Generator(nn.Module):
    def __init__(self, video_frame_num, FK_DH_Class, args, device, INPUT_VEC_DIM=128):
        super(Video_Fk_Generator, self).__init__()

        self.video_frame_num = video_frame_num
        self.OUTPUT_DIM = args.GAN_OUTPUT_DIM
        self.BATCH_SIZE = args.batch_size
        self.FK_DH_Class = FK_DH_Class
        self.train_num = 0
        self.args = args
        self.INPUT_VEC_DIM = INPUT_VEC_DIM

        self.boneLength = torch.zeros((self.BATCH_SIZE, 15), dtype=torch.float32)
        self.device = device

        self.distribute_angle = []

        self.preprocess = nn.Sequential(
            nn.Linear(self.INPUT_VEC_DIM, self.args.Gen_DenseDim),
            nn.ReLU(True),
        )

        self.block1 = myResNet(self.args.Gen_DenseDim)    # 1000
        self.block2 = myResNet(self.args.Gen_DenseDim)
        self.block3 = myResNet(self.args.Gen_DenseDim)

        self.deconv_out = nn.Linear(self.args.Gen_DenseDim, self.video_frame_num * self.OUTPUT_DIM)
        self.sigmoid = nn.Sigmoid()
        self.Tanh = nn.Tanh()

    def GAN_generator_get_bone_length(self, input):
        input = input.view(-1, 16, 3)
        bone_vector_from_mat = Fk_get_boneVecByPose3d(input)
        bone_length_from_mat = torch.sqrt(torch.sum(bone_vector_from_mat ** 2, dim=-1))
        self.boneLength = bone_length_from_mat

        self.boneLength = self.boneLength.view(-1, 15)

    def forward(self, input):
        output = self.preprocess(input)
        output = self.block1(output)
        output = self.block2(output)
        output = self.block3(output)
        output = self.deconv_out(output)
        output = output.contiguous().view(-1, self.video_frame_num, self.OUTPUT_DIM)

        output[:, :, :-3] = self.Tanh(output[:, :, :-3])
        output[:, :, -3:] = self.Tanh(output[:, :, -3:]) * 10.0

        root_3d_pos = output[:, :, -3:]
        if torch.cuda.is_available():
            root_3d_pos = root_3d_pos.to(self.device)

        output = output.contiguous().view(-1, self.OUTPUT_DIM)

        generator_angle = torch.zeros((output.shape[0], self.OUTPUT_DIM + 5 - 3), dtype=torch.float32)

        if torch.cuda.is_available():
            generator_angle = generator_angle.to(self.device)

        output_index = 0
        for index in range(self.OUTPUT_DIM + 5 - 3):

            if (index==4) or (index==9) or (index==23) or (index==28) or (index==33) or (index==22):
                generator_angle[:, index] = 0

            else:
                generator_angle[:, index] = output[:, output_index]
                output_index += 1

        if self.args.GAN_whether_use_preAngle == True:
            for index in range(self.OUTPUT_DIM + 5 - 3):
                if index < 34:
                    generator_angle[:, index] = \
                        generator_angle[:, index] * \
                        (GAN_angle_range_table['joint' + str(index + 1)]['range'][1] -
                         GAN_angle_range_table['joint' + str(index + 1)]['range'][0])/2 + \
                        (GAN_angle_range_table['joint' + str(index + 1)]['range'][1] +
                            GAN_angle_range_table['joint' + str(index + 1)]['range'][0])/2

                else:
                    char_temp = 'x'
                    if index == 34:
                        char_temp = 'x'
                    elif index == 35:
                        char_temp = 'y'
                    elif index == 36:
                        char_temp = 'z'
                    generator_angle[:, index] = \
                        generator_angle[:, index] * \
                        (GAN_global_rotation_table['angle_' + char_temp]['range'][1] -
                         GAN_global_rotation_table['angle_' + char_temp]['range'][0]) / 2 + \
                        (GAN_global_rotation_table['angle_' + char_temp]['range'][1] +
                         GAN_global_rotation_table['angle_' + char_temp]['range'][0]) / 2
        else:
            generator_angle = generator_angle * 180

        self.distribute_angle.append(generator_angle)

        self.train_num += 1
        if self.train_num % 500 == 1:
            special_operate.my_draw_DOF_angle_distribute(
                    torch.clone(generator_angle), '34DOF_3GlobalRot_heatmap_'+str(self.train_num), self.args)

            generator_angle = generator_angle.to(self.device)

        right_leg_joints_angle = generator_angle[:, 0: 5]
        left_leg_joints_angle = generator_angle[:, 5: 10]

        body_joints_angle = generator_angle[:, 10: 23]  # [:, 10: 24]
        right_hand_joints_angle = generator_angle[:, 23: 28]  # [:, 24: 29]
        left_hand_joints_angle = generator_angle[:, 28: 33]  # [:, 29: 34]

        generator_global_rot_3d_pos_angle = generator_angle[:, -3:]

        record_bone_len = self.boneLength

        bone_len_scaler = np.zeros(8)  #
        if self.args.bone_len_scaler == 'different':
            bone_len_scaler = self.FK_DH_Class.random.randint(-200, 200, size=(self.BATCH_SIZE, 8))

            bone_len_scaler = np.array(bone_len_scaler).reshape(self.BATCH_SIZE, 8)

            bone_len_scaler = bone_len_scaler / 1000.0

            bone_len_scaler = np.expand_dims(bone_len_scaler, axis=1).repeat(self.video_frame_num, axis=1)
            bone_len_scaler = bone_len_scaler.reshape(self.BATCH_SIZE * self.video_frame_num, 8)  #

        elif self.args.bone_len_scaler == 'same':  #
            bone_len_scaler = self.FK_DH_Class.random.randint(-200, 200, size=(self.BATCH_SIZE, 8))
            bone_len_scaler = np.array(bone_len_scaler).reshape(self.BATCH_SIZE, 1)
            bone_len_scaler = bone_len_scaler.repeat(8, axis=-1)
            bone_len_scaler = bone_len_scaler / 1000.0

        elif self.args.bone_len_scaler == '':
            bone_len_scaler = np.zeros((self.BATCH_SIZE, 8))

        else:
            raise ("args.bone_len_scaler")

        bone_len_scaler = torch.tensor(bone_len_scaler, dtype=torch.float32)

        if torch.cuda.is_available():
            record_bone_len = record_bone_len.to(self.device)
            bone_len_scaler = bone_len_scaler.to(self.device)

        left_small_leg_len = record_bone_len[:, 0] * (1 + bone_len_scaler[:, 0])
        right_small_leg_len = record_bone_len[:, 1] * (1 + bone_len_scaler[:, 0])
        left_big_leg_len = record_bone_len[:, 2] * (1 + bone_len_scaler[:, 1])
        right_big_leg_len = record_bone_len[:, 3] * (1 + bone_len_scaler[:, 1])
        left_hip_len = record_bone_len[:, 4] * (1 + bone_len_scaler[:, 2])
        right_hip_len = record_bone_len[:, 5] * (1 + bone_len_scaler[:, 2])
        waist_len = record_bone_len[:, 6] * (1 + bone_len_scaler[:, 3])
        thorax_len = record_bone_len[:, 7]
        left_shoulder_len = record_bone_len[:, 8] * (1 + bone_len_scaler[:, 4])
        right_shoulder_len = record_bone_len[:, 9] * (1 + bone_len_scaler[:, 4])
        left_big_arm_len = record_bone_len[:, 10] * (1 + bone_len_scaler[:, 5])
        right_big_arm_len = record_bone_len[:, 11] * (1 + bone_len_scaler[:, 5])
        left_small_arm_len = record_bone_len[:, 12] * (1 + bone_len_scaler[:, 6])
        right_small_arm_len = record_bone_len[:, 13] * (1 + bone_len_scaler[:, 6])
        neck_len = record_bone_len[:, 14] * (1 + bone_len_scaler[:, 7])

        single_generator_3d_world_32keyPoint = \
            self.FK_DH_Class.change_3d_joint_angle(
                left_leg_joints_angle=left_leg_joints_angle,
                right_leg_joints_angle=right_leg_joints_angle,
                body_joints_angle=body_joints_angle,
                left_hand_joints_angle=left_hand_joints_angle,
                right_hand_joints_angle=right_hand_joints_angle,
                generator_global_rot_3d_pos_angle=generator_global_rot_3d_pos_angle,
                left_small_leg_len=left_small_leg_len,
                right_small_leg_len=right_small_leg_len,
                left_big_leg_len=left_big_leg_len,
                right_big_leg_len=right_big_leg_len,
                left_hip_len=left_hip_len,
                right_hip_len=right_hip_len,
                waist_len=waist_len,
                thorax_len=thorax_len,
                left_shoulder_len=left_shoulder_len,
                right_shoulder_len=right_shoulder_len,
                left_big_arm_len=left_big_arm_len,
                right_big_arm_len=right_big_arm_len,
                left_small_arm_len=left_small_arm_len,
                right_small_arm_len=right_small_arm_len,
                neck_len=neck_len,
                #head_len=head_len,
                root_3d_pos=root_3d_pos
            )

        fake_Pos = single_generator_3d_world_32keyPoint[:, H36M_32_To_16_Table].view(-1, 16 * 3)

        if self.video_frame_num > 1:
            fake_Pos = fake_Pos.view(self.BATCH_SIZE, self.video_frame_num, 16*3)

        return fake_Pos