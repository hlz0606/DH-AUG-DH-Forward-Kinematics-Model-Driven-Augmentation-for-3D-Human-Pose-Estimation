#from models_Fk_GAN import myMainWindow
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec  #
import matplotlib
matplotlib.use("Qt5Agg")  #
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import Axes3D


# from PyQt5.QtCore import *
# from PyQt5.QtGui import *
# from PyQt5.QtWidgets import *
#from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout

from models_Fk_GAN import special_operate
from common import viz
from common.h36m_dataset import H36M_32_To_16_Table
from common.camera import *
from utils.utils import wrap

from pylab import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import loguru
import sys
import os
import zipfile
import h5py
import argparse
from glob import glob
from shutil import rmtree
import copy
from thop import profile

import copy
from progress.bar import Bar
from utils.utils import AverageMeter
from models_Fk_GAN.video_mode_operate import video_receptive_field

used_16key_15bone_len_table = [(5, 6), (2, 3), (4, 5), (1, 2),
                               (0, 4), (0, 1), (0, 7), (7, 8), (8, 10), (8, 13),
                               (10, 11), (13, 14), (11, 12), (14, 15),
                               (8, 9)]

H36M_POINTS_LEFT = [6, 7, 8, 17, 18, 19]  #
H36M_POINTS_RIGHT = [1, 2, 3, 25, 26, 27]  #

def dh_matrix(alpha, a, d, theta, args):

    if torch.is_tensor(theta) == False:
        alpha = alpha / 180 * np.pi
        theta = theta / 180 * np.pi
        matrix = np.identity(4)

        matrix[0, 0] = cos(theta)
        matrix[0, 1] = -sin(theta)
        matrix[0, 2] = 0
        matrix[0, 3] = a
        matrix[1, 0] = sin(theta) * cos(alpha)
        matrix[1, 1] = cos(theta) * cos(alpha)
        matrix[1, 2] = -sin(alpha)
        matrix[1, 3] = -sin(alpha) * d
        matrix[2, 0] = sin(theta) * sin(alpha)
        matrix[2, 1] = cos(theta) * sin(alpha)
        matrix[2, 2] = cos(alpha)
        matrix[2, 3] = cos(alpha) * d
        matrix[3, 0] = 0
        matrix[3, 1] = 0
        matrix[3, 2] = 0
        matrix[3, 3] = 1

        return matrix

    if torch.is_tensor(theta) == True:  #
        real_used_num = 1     #
        if args.single_or_multi_train_mode == 'multi':
            filter_widths = [int(x) for x in args.architecture.split(',')]  #
            receptive_field = video_receptive_field(filter_widths)
            real_used_num = receptive_field
        else:
            real_used_num = 1

        alpha = alpha / 180 * np.pi
        theta = theta / 180 * torch.tensor(np.pi, dtype=torch.float32)  # torch.tensor(theta / 180 * np.pi)
        matrix = torch.autograd.Variable(torch.tensor(np.zeros((args.batch_size * real_used_num, 4, 4)),
                                                      dtype=torch.float32))

        if torch.cuda.is_available():
            alpha = alpha.cuda()
            theta = theta.cuda()
            matrix = matrix.cuda()

        matrix[:, 0, 0] = torch.cos(theta)
        matrix[:, 0, 1] = -torch.sin(theta)
        matrix[:, 0, 2] = 0
        matrix[:, 0, 3] = a
        matrix[:, 1, 0] = torch.sin(theta) * torch.cos(alpha)
        matrix[:, 1, 1] = torch.cos(theta) * torch.cos(alpha)
        matrix[:, 1, 2] = -torch.sin(alpha)
        matrix[:, 1, 3] = -torch.sin(alpha) * d
        matrix[:, 2, 0] = torch.sin(theta) * torch.sin(alpha)
        matrix[:, 2, 1] = torch.cos(theta) * torch.sin(alpha)
        matrix[:, 2, 2] = torch.cos(alpha)
        matrix[:, 2, 3] = torch.cos(alpha) * d
        matrix[:, 3, 0] = 0
        matrix[:, 3, 1] = 0
        matrix[:, 3, 2] = 0
        matrix[:, 3, 3] = 1

        return matrix



def rotationMatrix(angle_x, angle_y, angle_z, args):

    if torch.is_tensor(angle_x) == False:
        angle_x = angle_x / 180 * np.pi
        angle_y = angle_y / 180 * np.pi
        angle_z = angle_z / 180 * np.pi
        R1, R2, R3 = [np.zeros((3, 3)) for _ in range(3)]
        R1[0:] = [1, 0, 0]
        R1[1:] = [0, np.cos(angle_x), -np.sin(angle_x)]
        R1[2:] = [0, np.sin(angle_x), np.cos(angle_x)]

        R2[0:] = [np.cos(angle_y), 0, np.sin(angle_y)]
        R2[1:] = [0, 1, 0]
        R2[2:] = [-np.sin(angle_y), 0, np.cos(angle_y)]

        R3[0:] = [np.cos(angle_z), -np.sin(angle_z), 0]
        R3[1:] = [np.sin(angle_z), np.cos(angle_z), 0]
        R3[2:] = [0, 0, 1]

        return (R1.dot(R2).dot(R3))

    elif torch.is_tensor(angle_x) == True:
        real_used_num = 1
        if args.single_or_multi_train_mode == 'multi':
            filter_widths = [int(x) for x in args.architecture.split(',')]
            receptive_field = video_receptive_field(filter_widths)
            real_used_num = receptive_field
        else:
            real_used_num = 1

        angle_x = angle_x / 180 * np.pi
        angle_y = angle_y / 180 * np.pi
        angle_z = angle_z / 180 * np.pi

        R1 = torch.zeros((args.batch_size * real_used_num, 3, 3), dtype=torch.float32)
        R2 = torch.zeros((args.batch_size * real_used_num, 3, 3), dtype=torch.float32)
        R3 = torch.zeros((args.batch_size * real_used_num, 3, 3), dtype=torch.float32)

        # [1 0 0; 0 cos(obj.Params(1)) -sin(obj.Params(1)); 0 sin(obj.Params(1)) cos(obj.Params(1))]
        R1[:, 0:] = torch.tensor([1, 0, 0], dtype=torch.float32)
        R1[:, 1, 0] = 0
        R1[:, 1, 1] = torch.cos(angle_x)
        R1[:, 1, 2] = -torch.sin(angle_x)
        # R1[:, 1:] = torch.tensor((0, torch.cos(angle_x[:]), -torch.sin(angle_x[:])))
        R1[:, 2, 0] = 0
        R1[:, 2, 1] = torch.sin(angle_x)
        R1[:, 2, 2] = torch.cos(angle_x)
        # R1[:, 2:] = torch.tensor(0, torch.sin(angle_x), torch.cos(angle_x))

        # [cos(obj.Params(2)) 0 sin(obj.Params(2)); 0 1 0; -sin(obj.Params(2)) 0 cos(obj.Params(2))]
        R2[:, 0, 0] = torch.cos(angle_y)
        R2[:, 0, 1] = 0
        R2[:, 0, 2] = torch.sin(angle_y)
        # R2[:, 0:] = torch.tensor([torch.cos(angle_y), 0, torch.sin(angle_y)])
        R2[:, 1:] = torch.tensor([0, 1, 0], dtype=torch.float32)
        R2[:, 2, 0] = -torch.sin(angle_y)
        R2[:, 2, 1] = 0
        R2[:, 2, 2] = torch.cos(angle_y)
        # R2[:, 2:] = torch.tensor([-torch.sin(angle_y), 0, torch.cos(angle_y)])

        # [cos(obj.Params(3)) -sin(obj.Params(3)) 0; sin(obj.Params(3)) cos(obj.Params(3)) 0; 0 0 1];%
        R3[:, 0, 0] = torch.cos(angle_z)
        R3[:, 0, 1] = -torch.sin(angle_z)
        R3[:, 0, 2] = 0
        # R3[:, 0:] = torch.tensor([torch.cos(angle_z), -torch.sin(angle_z), 0])
        R3[:, 1, 0] = torch.sin(angle_z)
        R3[:, 1, 1] = torch.cos(angle_z)
        R3[:, 1, 2] = 0
        # R3[:, 1:] = torch.tensor([torch.sin(angle_z), torch.cos(angle_z), 0])
        R3[:, 2:] = torch.tensor([0, 0, 1], dtype=torch.float32)

        return (R1.bmm(R2).bmm(R3))


class Forward_Kinematics_DH_Model():

    def __init__(self, args, train_subjects, dataset):
        super(Forward_Kinematics_DH_Model, self).__init__()

        self.args = args
        self.train_subjects = train_subjects
        self.GAN_BATCH_SIZE = self.args.batch_size
        self.dataset = dataset

        random_seed = self.args.random_seed
        self.random = np.random.RandomState(random_seed)

        self.generator_3d_pos_angle = []
        self.generator_global_rot_3d_pos_angle = []
        self.generator_bone_len = []
        self.generator_root = []

        self.show_3d_pos_num = 0

        self.record_bone_len = []    #np.zeros(15)

        self.camera_parameters = {}

        self.root_3d_pos = np.array([0, 0, 0])

        self.dataSet_world_3d_pos = {}
        self.dataSet_2d_pos = {}

        self.choice_subject = []
        self.choice_action = []
        self.choice_cam = []

        self.right_leg_joint_num = 5
        self.left_leg_joint_num = 5
        self.body_joint_num = 13  # 14
        self.right_hand_joint_num = 5 + self.body_joint_num - 4
        self.left_hand_joint_num = 5 + self.body_joint_num - 4

        # --- Robotic Arm construction ---
        self.right_leg_joints_alpha = [0.0, -90.0, -90.0, 0.0, 0.0]
        self.right_leg_joints_a = [0.25, 0.0, 0.0, 0.6, 0.5]
        self.right_leg_joints_d = [0.0, 0.0, 0.0, 0.0, 0.0]
        self.right_leg_joints_theta = [0.0, -90.0, 180.0, 0.0, 0.0]


        self.left_leg_joints_alpha = [0.0, 90.0, 90.0, 0.0, 0.0]
        self.left_leg_joints_a = [-0.25, 0.0, 0.0, 0.6, 0.5]
        self.left_leg_joints_d = [0.0, 0.0, 0.0, 0.0, 0.0]
        self.left_leg_joints_theta = [180.0, -90.0, 0.0, 0.0, 0.0]

        self.body_joints_alpha = [0.0, -90.0, -90.0, -90.0, -90.0, -90.0, -90.0,
                                  -90.0, -90.0, -90.0, -90.0, -90.0, 90.0]  # , 0.0]
        self.body_joints_a = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.15]  # , 0.10]
        self.body_joints_d = [0.0, 0.0, 0.0, 0.25, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # , 0.0]
        self.body_joints_theta = [90.0, -90.0, -90.0, -90.0, -90.0, -90.0, -90.0,
                                  -90.0, -90.0, -90.0, -90.0, 0.0, 0.0]  # , 0.0]

        self.right_hand_joints_alpha = [-90.0, -90.0, -90.0, 0.0, 0.0]
        self.right_hand_joints_a = [-0.3, 0.0, 0, 0.4, 0.35]
        self.right_hand_joints_d = [0.0, 0.0, 0.0, 0.0, 0.0]
        self.right_hand_joints_theta = [-180.0, -90.0, 180.0, 0.0, 0.0]

        ########
        self.left_hand_joints_alpha = [-90.0, 90.0, 90.0, 0.0, 0.0]
        self.left_hand_joints_a = [0.3, 0.0, 0, 0.4, 0.35]
        self.left_hand_joints_d = [0.0, 0.0, 0.0, 0.0, 0.0]
        self.left_hand_joints_theta = [0.0, -90.0, 0.0, 0.0, 0.0]
        ##################################################################

        ###############################
        self.real_used_num = 1
        if args.single_or_multi_train_mode == 'multi':
            filter_widths = [int(x) for x in args.architecture.split(',')]
            receptive_field = video_receptive_field(filter_widths)
            self.real_used_num = receptive_field
        else:
            self.real_used_num = 1

        # --- Robotic Arm construction ---
        self.GAN_right_leg_joints_alpha = \
            torch.tensor([copy.deepcopy(self.right_leg_joints_alpha) for _ in range(self.GAN_BATCH_SIZE * self.real_used_num)], dtype=torch.float32)
        self.GAN_right_leg_joints_a = \
            torch.tensor([copy.deepcopy(self.right_leg_joints_a) for _ in range(self.GAN_BATCH_SIZE * self.real_used_num)], dtype=torch.float32)
        self.GAN_right_leg_joints_d = \
            torch.tensor([copy.deepcopy(self.right_leg_joints_d) for _ in range(self.GAN_BATCH_SIZE * self.real_used_num)], dtype=torch.float32)
        self.GAN_right_leg_joints_theta = \
            torch.tensor([copy.deepcopy(self.right_leg_joints_theta) for _ in range(self.GAN_BATCH_SIZE * self.real_used_num)], dtype=torch.float32)

        ########
        self.GAN_left_leg_joints_alpha = \
            torch.tensor([copy.deepcopy(self.left_leg_joints_alpha) for _ in range(self.GAN_BATCH_SIZE * self.real_used_num)], dtype=torch.float32)
        self.GAN_left_leg_joints_a = \
            torch.tensor([copy.deepcopy(self.left_leg_joints_a) for _ in range(self.GAN_BATCH_SIZE * self.real_used_num)], dtype=torch.float32)
        self.GAN_left_leg_joints_d = \
            torch.tensor([copy.deepcopy(self.left_leg_joints_d) for _ in range(self.GAN_BATCH_SIZE * self.real_used_num)], dtype=torch.float32)
        self.GAN_left_leg_joints_theta = \
            torch.tensor([copy.deepcopy(self.left_leg_joints_theta) for _ in range(self.GAN_BATCH_SIZE * self.real_used_num)], dtype=torch.float32)

        ####################
        self.GAN_body_joints_alpha = \
            torch.tensor([copy.deepcopy(self.body_joints_alpha) for _ in range(self.GAN_BATCH_SIZE * self.real_used_num)], dtype=torch.float32)
        self.GAN_body_joints_a = \
            torch.tensor([copy.deepcopy(self.body_joints_a) for _ in range(self.GAN_BATCH_SIZE * self.real_used_num)], dtype=torch.float32)
        self.GAN_body_joints_d = \
            torch.tensor([copy.deepcopy(self.body_joints_d) for _ in range(self.GAN_BATCH_SIZE * self.real_used_num)], dtype=torch.float32)
        self.GAN_body_joints_theta = \
            torch.tensor([copy.deepcopy(self.body_joints_theta) for _ in range(self.GAN_BATCH_SIZE * self.real_used_num)], dtype=torch.float32)

        #######
        self.GAN_right_hand_joints_alpha = \
            torch.tensor([copy.deepcopy(self.right_hand_joints_alpha) for _ in range(self.GAN_BATCH_SIZE * self.real_used_num)], dtype=torch.float32)
        self.GAN_right_hand_joints_a = \
            torch.tensor([copy.deepcopy(self.right_hand_joints_a) for _ in range(self.GAN_BATCH_SIZE * self.real_used_num)], dtype=torch.float32)
        self.GAN_right_hand_joints_d = \
            torch.tensor([copy.deepcopy(self.right_hand_joints_d) for _ in range(self.GAN_BATCH_SIZE * self.real_used_num)], dtype=torch.float32)
        self.GAN_right_hand_joints_theta = \
            torch.tensor([copy.deepcopy(self.right_hand_joints_theta) for _ in range(self.GAN_BATCH_SIZE * self.real_used_num)], dtype=torch.float32)

        ########
        self.GAN_left_hand_joints_alpha = \
            torch.tensor([copy.deepcopy(self.left_hand_joints_alpha) for _ in range(self.GAN_BATCH_SIZE * self.real_used_num)], dtype=torch.float32)
        self.GAN_left_hand_joints_a = \
            torch.tensor([copy.deepcopy(self.left_hand_joints_a) for _ in range(self.GAN_BATCH_SIZE * self.real_used_num)], dtype=torch.float32)
        self.GAN_left_hand_joints_d = \
            torch.tensor([copy.deepcopy(self.left_hand_joints_d) for _ in range(self.GAN_BATCH_SIZE * self.real_used_num)], dtype=torch.float32)
        self.GAN_left_hand_joints_theta = \
            torch.tensor([copy.deepcopy(self.left_hand_joints_theta) for _ in range(self.GAN_BATCH_SIZE * self.real_used_num)], dtype=torch.float32)
        ##################################################################

        if torch.cuda.is_available():
            self.GAN_right_leg_joints_alpha = self.GAN_right_leg_joints_alpha.cuda()
            self.GAN_right_leg_joints_a = self.GAN_right_leg_joints_a.cuda()
            self.GAN_right_leg_joints_d = self.GAN_right_leg_joints_d.cuda()
            self.GAN_right_leg_joints_theta = self.GAN_right_leg_joints_theta.cuda()

            #######
            self.GAN_left_leg_joints_alpha = self.GAN_left_leg_joints_alpha.cuda()
            self.GAN_left_leg_joints_a = self.GAN_left_leg_joints_a.cuda()
            self.GAN_left_leg_joints_d = self.GAN_left_leg_joints_d.cuda()
            self.GAN_left_leg_joints_theta = self.GAN_left_leg_joints_theta.cuda()

            ###############
            self.GAN_body_joints_alpha = self.GAN_body_joints_alpha.cuda()
            self.GAN_body_joints_a = self.GAN_body_joints_a.cuda()
            self.GAN_body_joints_d = self.GAN_body_joints_d.cuda()
            self.GAN_body_joints_theta = self.GAN_body_joints_theta.cuda()

            ###
            self.GAN_right_hand_joints_alpha = self.GAN_right_hand_joints_alpha.cuda()
            self.GAN_right_hand_joints_a = self.GAN_right_hand_joints_a.cuda()
            self.GAN_right_hand_joints_d = self.GAN_right_hand_joints_d.cuda()
            self.GAN_right_hand_joints_theta = self.GAN_right_hand_joints_theta.cuda()

            ####
            self.GAN_left_hand_joints_alpha = self.GAN_left_hand_joints_alpha.cuda()
            self.GAN_left_hand_joints_a = self.GAN_left_hand_joints_a.cuda()
            self.GAN_left_hand_joints_d = self.GAN_left_hand_joints_d.cuda()
            self.GAN_left_hand_joints_theta = self.GAN_left_hand_joints_theta.cuda()

    def change_3d_joint_angle(self, left_leg_joints_angle, right_leg_joints_angle, body_joints_angle,
                              left_hand_joints_angle, right_hand_joints_angle, generator_global_rot_3d_pos_angle,

                              left_small_leg_len, right_small_leg_len, left_big_leg_len, right_big_leg_len,
                              left_hip_len, right_hip_len, waist_len, thorax_len, left_shoulder_len,
                              right_shoulder_len,
                              left_big_arm_len, right_big_arm_len, left_small_arm_len, right_small_arm_len,
                              neck_len,  # head_len,

                              root_3d_pos
                              ):

        if torch.is_tensor(left_leg_joints_angle) == False:  # numpy

            self.left_leg_joints_angle = left_leg_joints_angle
            self.right_leg_joints_angle = right_leg_joints_angle
            self.body_joints_angle = body_joints_angle
            self.right_hand_joints_angle = right_hand_joints_angle
            self.left_hand_joints_angle = left_hand_joints_angle

            self.global_rot_angle = generator_global_rot_3d_pos_angle

            self.global_rot_mat = rotationMatrix(self.global_rot_angle[0],
                                                      self.global_rot_angle[1],
                                                      self.global_rot_angle[2],
                                                        self.args)

            self.left_small_leg_len = left_small_leg_len
            self.right_small_leg_len = right_small_leg_len
            self.left_big_leg_len = left_big_leg_len
            self.right_big_leg_len = right_big_leg_len
            self.left_hip_len = left_hip_len
            self.right_hip_len = right_hip_len
            self.waist_len = waist_len
            self.thorax_len = thorax_len
            self.left_shoulder_len = left_shoulder_len
            self.right_shoulder_len = right_shoulder_len
            self.left_big_arm_len = left_big_arm_len
            self.right_big_arm_len = right_big_arm_len
            self.left_small_arm_len = left_small_arm_len
            self.right_small_arm_len = right_small_arm_len
            self.neck_len = neck_len
            # self.head_len = head_len

            self.left_leg_joints_a[0] = -self.left_hip_len
            self.left_leg_joints_a[3] = self.left_big_leg_len
            self.left_leg_joints_a[4] = self.left_small_leg_len

            self.right_leg_joints_a[0] = self.right_hip_len
            self.right_leg_joints_a[3] = self.right_big_leg_len
            self.right_leg_joints_a[4] = self.right_small_leg_len

            self.body_joints_a[12] = self.neck_len
            # self.body_joints_a[13] = self.head_len
            self.body_joints_d[3] = self.waist_len
            self.body_joints_d[6] = self.thorax_len

            self.left_hand_joints_a[0] = self.left_shoulder_len
            self.left_hand_joints_a[3] = self.left_big_arm_len
            self.left_hand_joints_a[4] = self.left_small_arm_len

            self.right_hand_joints_a[0] = -self.right_shoulder_len
            self.right_hand_joints_a[3] = self.right_big_arm_len
            self.right_hand_joints_a[4] = self.right_small_arm_len

            #    DH参数转转换矩阵T---------------------
            left_leg_joint_hm = []
            for i in range(self.left_leg_joint_num):
                left_leg_joint_hm.append(dh_matrix(self.left_leg_joints_alpha[i], self.left_leg_joints_a[i],
                                                   self.left_leg_joints_d[i],
                                                   self.left_leg_joints_theta[i] + left_leg_joints_angle[i],
                                                   self.args))
            right_leg_joint_hm = []
            for i in range(self.right_leg_joint_num):
                right_leg_joint_hm.append(dh_matrix(self.right_leg_joints_alpha[i], self.right_leg_joints_a[i],
                                                    self.right_leg_joints_d[i],
                                                    self.right_leg_joints_theta[i] + right_leg_joints_angle[i],
                                                    self.args))

            body_joint_hm = []
            for i in range(self.body_joint_num):
                body_joint_hm.append(dh_matrix(self.body_joints_alpha[i], self.body_joints_a[i],
                                               self.body_joints_d[i],
                                               self.body_joints_theta[i] + body_joints_angle[i],
                                               self.args))

            right_hand_joint_hm = copy.deepcopy(body_joint_hm[0: 9])
            for i in range(self.right_hand_joint_num - (self.body_joint_num - 4)):
                right_hand_joint_hm.append(dh_matrix(self.right_hand_joints_alpha[i], self.right_hand_joints_a[i],
                                                     self.right_hand_joints_d[i],
                                                     self.right_hand_joints_theta[i] + right_hand_joints_angle[i],
                                                     self.args))
            left_hand_joint_hm = copy.deepcopy(body_joint_hm[0: 9])
            for i in range(self.left_hand_joint_num - (self.body_joint_num - 4)):
                left_hand_joint_hm.append(dh_matrix(self.left_hand_joints_alpha[i], self.left_hand_joints_a[i],
                                                    self.left_hand_joints_d[i],
                                                    self.left_hand_joints_theta[i] + left_hand_joints_angle[i],
                                                    self.args))

            # -----------连乘计算----------------------
            for i in range(self.left_leg_joint_num - 1):
                left_leg_joint_hm[i + 1] = np.dot(left_leg_joint_hm[i], left_leg_joint_hm[i + 1])

            for i in range(self.right_leg_joint_num - 1):
                right_leg_joint_hm[i + 1] = np.dot(right_leg_joint_hm[i], right_leg_joint_hm[i + 1])

            for i in range(self.body_joint_num - 1):
                body_joint_hm[i + 1] = np.dot(body_joint_hm[i], body_joint_hm[i + 1])

            for i in range(self.right_hand_joint_num - 1):
                right_hand_joint_hm[i + 1] = np.dot(right_hand_joint_hm[i], right_hand_joint_hm[i + 1])

            for i in range(self.left_hand_joint_num - 1):
                left_hand_joint_hm[i + 1] = np.dot(left_hand_joint_hm[i], left_hand_joint_hm[i + 1])

            ########  获取坐标值
            left_leg_X = [hm[0, 3] for hm in left_leg_joint_hm]
            left_leg_Y = [hm[1, 3] for hm in left_leg_joint_hm]
            left_leg_Z = [hm[2, 3] for hm in left_leg_joint_hm]

            right_leg_X = [hm[0, 3] for hm in right_leg_joint_hm]
            right_leg_Y = [hm[1, 3] for hm in right_leg_joint_hm]
            right_leg_Z = [hm[2, 3] for hm in right_leg_joint_hm]

            body_X = [hm[0, 3] for hm in body_joint_hm]
            body_Y = [hm[1, 3] for hm in body_joint_hm]
            body_Z = [hm[2, 3] for hm in body_joint_hm]

            left_hand_X = [hm[0, 3] for hm in left_hand_joint_hm]
            left_hand_Y = [hm[1, 3] for hm in left_hand_joint_hm]
            left_hand_Z = [hm[2, 3] for hm in left_hand_joint_hm]

            right_hand_X = [hm[0, 3] for hm in right_hand_joint_hm]
            right_hand_Y = [hm[1, 3] for hm in right_hand_joint_hm]
            right_hand_Z = [hm[2, 3] for hm in right_hand_joint_hm]

            left_leg_pos = np.array([left_leg_X, left_leg_Y, left_leg_Z])
            left_leg_pos = self.global_rot_mat.dot(left_leg_pos)

            right_leg_pos = np.array([right_leg_X, right_leg_Y, right_leg_Z])
            right_leg_pos = self.global_rot_mat.dot(right_leg_pos)

            body_pos = np.array([body_X, body_Y, body_Z])
            body_pos = self.global_rot_mat.dot(body_pos)

            left_hand_pos = np.array([left_hand_X, left_hand_Y, left_hand_Z])
            left_hand_pos = self.global_rot_mat.dot(left_hand_pos)

            right_hand_pos = np.array([right_hand_X, right_hand_Y, right_hand_Z])
            right_hand_pos = self.global_rot_mat.dot(right_hand_pos)

            self.single_generator_3d_world_32keyPoint = np.zeros((32, 3))

            self.single_generator_3d_world_32keyPoint[0] = \
                np.array([body_pos[0][0], body_pos[1][0], body_pos[2][0]])  # 'Hip'

            self.single_generator_3d_world_32keyPoint[1] = \
                np.array([right_leg_pos[0][0], right_leg_pos[1][0], right_leg_pos[2][0]])  # 'RHip'

            self.single_generator_3d_world_32keyPoint[2] = \
                np.array([right_leg_pos[0][3], right_leg_pos[1][3], right_leg_pos[2][3]])  # 'RKnee'

            self.single_generator_3d_world_32keyPoint[3] = \
                np.array([right_leg_pos[0][4], right_leg_pos[1][4], right_leg_pos[2][4]])  # 'RAnkle'

            self.single_generator_3d_world_32keyPoint[6] = \
                np.array([left_leg_pos[0][0], left_leg_pos[1][0], left_leg_pos[2][0]])  # 'LHip'

            self.single_generator_3d_world_32keyPoint[7] = \
                np.array([left_leg_pos[0][3], left_leg_pos[1][3], left_leg_pos[2][3]])  # 'LKnee'

            self.single_generator_3d_world_32keyPoint[8] = \
                np.array([left_leg_pos[0][4], left_leg_pos[1][4], left_leg_pos[2][4]])  # 'LAnkle'

            self.single_generator_3d_world_32keyPoint[12] = \
                np.array([body_pos[0][3], body_pos[1][3], body_pos[2][3]])  # 'Spine'

            self.single_generator_3d_world_32keyPoint[13] = \
                np.array([body_pos[0][6], body_pos[1][6], body_pos[2][6]])  # 'Thorax'

            self.single_generator_3d_world_32keyPoint[14] = \
                np.array([body_pos[0][12], body_pos[1][12], body_pos[2][12]])  # 'Neck/Nose'

            self.single_generator_3d_world_32keyPoint[15] = \
                np.array([body_pos[0][12], body_pos[1][12], body_pos[2][12]])  # 'Head'

            self.single_generator_3d_world_32keyPoint[17] = \
                np.array([left_hand_pos[0][9], left_hand_pos[1][9], left_hand_pos[2][9]])  # 'LShoulder'

            self.single_generator_3d_world_32keyPoint[18] = \
                np.array([left_hand_pos[0][12], left_hand_pos[1][12], left_hand_pos[2][12]])  # 'LElbow'

            self.single_generator_3d_world_32keyPoint[19] = \
                np.array([left_hand_pos[0][13], left_hand_pos[1][13], left_hand_pos[2][13]])  # 'LWrist'

            self.single_generator_3d_world_32keyPoint[25] = \
                np.array([right_hand_pos[0][9], right_hand_pos[1][9], right_hand_pos[2][9]])  # 'RShoulder'

            self.single_generator_3d_world_32keyPoint[26] = \
                np.array([right_hand_pos[0][12], right_hand_pos[1][12], right_hand_pos[2][12]])  # 'RElbow'

            self.single_generator_3d_world_32keyPoint[27] = \
                np.array([right_hand_pos[0][13], right_hand_pos[1][13], right_hand_pos[2][13]])  # 'RWrist'

            self.single_generator_3d_world_32keyPoint += root_3d_pos

            return self.single_generator_3d_world_32keyPoint.astype(float32)

        else:

            self.global_rot_angle = generator_global_rot_3d_pos_angle

            self.global_rot_mat = rotationMatrix(self.global_rot_angle[:, 0],
                                                      self.global_rot_angle[:, 1],
                                                      self.global_rot_angle[:, 2],
                                                      self.args)

            self.GAN_left_leg_joints_a[:, 0] = -left_hip_len
            self.GAN_left_leg_joints_a[:, 3] = left_big_leg_len
            self.GAN_left_leg_joints_a[:, 4] = left_small_leg_len

            self.GAN_right_leg_joints_a[:, 0] = right_hip_len
            self.GAN_right_leg_joints_a[:, 3] = right_big_leg_len
            self.GAN_right_leg_joints_a[:, 4] = right_small_leg_len

            self.GAN_body_joints_a[:, 12] = neck_len
            self.GAN_body_joints_d[:, 3] = waist_len
            self.GAN_body_joints_d[:, 6] = thorax_len

            self.GAN_left_hand_joints_a[:, 0] = left_shoulder_len
            self.GAN_left_hand_joints_a[:, 3] = left_big_arm_len
            self.GAN_left_hand_joints_a[:, 4] = left_small_arm_len

            self.GAN_right_hand_joints_a[:, 0] = -right_shoulder_len  #
            self.GAN_right_hand_joints_a[:, 3] = right_big_arm_len
            self.GAN_right_hand_joints_a[:, 4] = right_small_arm_len

            #    DH参数转转换矩阵T---------------------
            left_leg_joint_hm = \
                torch.autograd.Variable(torch.zeros((self.GAN_BATCH_SIZE * self.real_used_num,
                                                     self.left_leg_joint_num, 4, 4), dtype=torch.float32))
            if torch.cuda.is_available():
                left_leg_joint_hm = left_leg_joint_hm.cuda()
            for i in range(self.left_leg_joint_num):
                left_leg_joint_hm[:, i] = \
                    dh_matrix(self.GAN_left_leg_joints_alpha[:, i], self.GAN_left_leg_joints_a[:, i],
                              self.GAN_left_leg_joints_d[:, i],
                              self.GAN_left_leg_joints_theta[:, i] + left_leg_joints_angle[:, i],
                              self.args)

            right_leg_joint_hm = \
                torch.autograd.Variable(torch.zeros((self.GAN_BATCH_SIZE * self.real_used_num,
                                                     self.right_leg_joint_num, 4, 4), dtype=torch.float32))
            if torch.cuda.is_available():
                right_leg_joint_hm = right_leg_joint_hm.cuda()
            for i in range(self.right_leg_joint_num):
                right_leg_joint_hm[:, i] = \
                    dh_matrix(self.GAN_right_leg_joints_alpha[:, i], self.GAN_right_leg_joints_a[:, i],
                              self.GAN_right_leg_joints_d[:, i],
                              self.GAN_right_leg_joints_theta[:, i] + right_leg_joints_angle[:, i],
                              self.args)

            body_joint_hm = \
                torch.autograd.Variable(
                    torch.zeros((self.GAN_BATCH_SIZE * self.real_used_num ,
                                 self.body_joint_num, 4, 4), dtype=torch.float32))  # body_joint_hm = []
            if torch.cuda.is_available():
                body_joint_hm = body_joint_hm.cuda()
            for i in range(self.body_joint_num):
                body_joint_hm[:, i] = \
                    dh_matrix(self.GAN_body_joints_alpha[:, i], self.GAN_body_joints_a[:, i],
                              self.GAN_body_joints_d[:, i],
                              self.GAN_body_joints_theta[:, i] + body_joints_angle[:, i],
                              self.args)

            right_hand_joint_hm = \
                torch.autograd.Variable(torch.zeros((
                    self.GAN_BATCH_SIZE * self.real_used_num,
                    9 + self.right_hand_joint_num - (self.body_joint_num - 4), 4, 4), dtype=torch.float32))
            right_hand_joint_hm[:, 0: 9] = torch.clone(body_joint_hm[:, 0: 9])  ####
            if torch.cuda.is_available():
                right_hand_joint_hm = right_hand_joint_hm.cuda()
            for i in range(self.right_hand_joint_num - (self.body_joint_num - 4)):

                right_hand_joint_hm[:, i + 9] = \
                    dh_matrix(self.GAN_right_hand_joints_alpha[:, i], self.GAN_right_hand_joints_a[:, i],
                              self.GAN_right_hand_joints_d[:, i],
                              self.GAN_right_hand_joints_theta[:, i] + right_hand_joints_angle[:, i],
                              self.args)

            left_hand_joint_hm = \
                torch.autograd.Variable(torch.zeros((
                    self.GAN_BATCH_SIZE * self.real_used_num,
                    9 + self.left_hand_joint_num - (self.body_joint_num - 4), 4, 4), dtype=torch.float32))
            left_hand_joint_hm[:, 0: 9] = torch.clone(body_joint_hm[:, 0: 9])
            if torch.cuda.is_available():
                left_hand_joint_hm = left_hand_joint_hm.cuda()
            for i in range(self.left_hand_joint_num - (self.body_joint_num - 4)):
                left_hand_joint_hm[:, i + 9] = \
                    dh_matrix(self.GAN_left_hand_joints_alpha[:, i], self.GAN_left_hand_joints_a[:, i],
                              self.GAN_left_hand_joints_d[:, i],
                              self.GAN_left_hand_joints_theta[:, i] + left_hand_joints_angle[:, i],
                              self.args)

            # -----------连乘计算----------------------
            for i in range(self.left_leg_joint_num - 1):
                left_leg_joint_hm[:, i + 1] = \
                    torch.bmm(torch.clone(left_leg_joint_hm[:, i]), torch.clone(left_leg_joint_hm[:, i + 1]))

            for i in range(self.right_leg_joint_num - 1):
                right_leg_joint_hm[:, i + 1] = \
                    torch.bmm(torch.clone(right_leg_joint_hm[:, i]), torch.clone(right_leg_joint_hm[:, i + 1]))

            for i in range(self.body_joint_num - 1):
                body_joint_hm[:, i + 1] = \
                    torch.bmm(torch.clone(body_joint_hm[:, i]), torch.clone(body_joint_hm[:, i + 1]))

            for i in range(self.right_hand_joint_num - 1):
                right_hand_joint_hm[:, i + 1] = \
                    torch.bmm(torch.clone(right_hand_joint_hm[:, i]), torch.clone(right_hand_joint_hm[:, i + 1]))

            for i in range(self.left_hand_joint_num - 1):
                left_hand_joint_hm[:, i + 1] = \
                    torch.bmm(torch.clone(left_hand_joint_hm[:, i]), torch.clone(left_hand_joint_hm[:, i + 1]))

            left_leg_X = torch.clone(left_leg_joint_hm[:, :, 0, 3])
            left_leg_Y = torch.clone(left_leg_joint_hm[:, :, 1, 3])
            left_leg_Z = torch.clone(left_leg_joint_hm[:, :, 2, 3])

            right_leg_X = torch.clone(right_leg_joint_hm[:, :, 0, 3])
            right_leg_Y = torch.clone(right_leg_joint_hm[:, :, 1, 3])
            right_leg_Z = torch.clone(right_leg_joint_hm[:, :, 2, 3])

            body_X = torch.clone(body_joint_hm[:, :, 0, 3])
            body_Y = torch.clone(body_joint_hm[:, :, 1, 3])
            body_Z = torch.clone(body_joint_hm[:, :, 2, 3])

            left_hand_X = torch.clone(left_hand_joint_hm[:, :, 0, 3])
            left_hand_Y = torch.clone(left_hand_joint_hm[:, :, 1, 3])
            left_hand_Z = torch.clone(left_hand_joint_hm[:, :, 2, 3])

            right_hand_X = torch.clone(right_hand_joint_hm[:, :, 0, 3])
            right_hand_Y = torch.clone(right_hand_joint_hm[:, :, 1, 3])
            right_hand_Z = torch.clone(right_hand_joint_hm[:, :, 2, 3])

            left_leg_pos = torch.autograd.Variable(
                torch.zeros((self.GAN_BATCH_SIZE * self.real_used_num , 3, self.left_leg_joint_num), dtype=torch.float32))

            left_leg_pos[:, 0, :] = left_leg_X[:, :]
            left_leg_pos[:, 1, :] = left_leg_Y[:, :]
            left_leg_pos[:, 2, :] = left_leg_Z[:, :]

            left_leg_pos = self.global_rot_mat.bmm(left_leg_pos)

            right_leg_pos = torch.autograd.Variable(torch.zeros(
                (self.GAN_BATCH_SIZE * self.real_used_num , 3, self.right_leg_joint_num), dtype=torch.float32))

            right_leg_pos[:, 0, :] = right_leg_X[:, :]
            right_leg_pos[:, 1, :] = right_leg_Y[:, :]
            right_leg_pos[:, 2, :] = right_leg_Z[:, :]

            right_leg_pos = self.global_rot_mat.bmm(right_leg_pos)

            body_pos = torch.autograd.Variable(torch.zeros(
                (self.GAN_BATCH_SIZE * self.real_used_num , 3, self.body_joint_num), dtype=torch.float32))

            body_pos[:, 0, :] = body_X[:, :]
            body_pos[:, 1, :] = body_Y[:, :]
            body_pos[:, 2, :] = body_Z[:, :]

            body_pos = self.global_rot_mat.bmm(body_pos)

            left_hand_pos = torch.autograd.Variable(torch.zeros(
                (self.GAN_BATCH_SIZE * self.real_used_num , 3, self.left_hand_joint_num), dtype=torch.float32))

            left_hand_pos[:, 0, :] = left_hand_X[:, :]
            left_hand_pos[:, 1, :] = left_hand_Y[:, :]
            left_hand_pos[:, 2, :] = left_hand_Z[:, :]

            left_hand_pos = self.global_rot_mat.bmm(left_hand_pos)

            right_hand_pos = torch.autograd.Variable(
                torch.zeros((self.GAN_BATCH_SIZE * self.real_used_num, 3, self.right_hand_joint_num),
                            dtype=torch.float32))

            right_hand_pos[:, 0, :] = right_hand_X[:, :]
            right_hand_pos[:, 1, :] = right_hand_Y[:, :]
            right_hand_pos[:, 2, :] = right_hand_Z[:, :]

            right_hand_pos = self.global_rot_mat.bmm(right_hand_pos)

            self.single_generator_3d_world_32keyPoint = \
                torch.autograd.Variable(torch.zeros((self.GAN_BATCH_SIZE * self.real_used_num, 32, 3),
                                                    dtype=torch.float32))
            if torch.cuda.is_available():
                self.single_generator_3d_world_32keyPoint = self.single_generator_3d_world_32keyPoint.cuda()

            self.single_generator_3d_world_32keyPoint[:, 0, 0] = body_pos[:, 0, 0]
            self.single_generator_3d_world_32keyPoint[:, 0, 1] = body_pos[:, 1, 0]
            self.single_generator_3d_world_32keyPoint[:, 0, 2] = body_pos[:, 2, 0]  # 'Hip'

            self.single_generator_3d_world_32keyPoint[:, 1, 0] = right_leg_pos[:, 0, 0]
            self.single_generator_3d_world_32keyPoint[:, 1, 1] = right_leg_pos[:, 1, 0]
            self.single_generator_3d_world_32keyPoint[:, 1, 2] = right_leg_pos[:, 2, 0]  # 'RHip'

            self.single_generator_3d_world_32keyPoint[:, 2, 0] = right_leg_pos[:, 0, 3]
            self.single_generator_3d_world_32keyPoint[:, 2, 1] = right_leg_pos[:, 1, 3]
            self.single_generator_3d_world_32keyPoint[:, 2, 2] = right_leg_pos[:, 2, 3]  # 'RKnee'

            self.single_generator_3d_world_32keyPoint[:, 3, 0] = right_leg_pos[:, 0, 4]
            self.single_generator_3d_world_32keyPoint[:, 3, 1] = right_leg_pos[:, 1, 4]
            self.single_generator_3d_world_32keyPoint[:, 3, 2] = right_leg_pos[:, 2, 4]  # 'RAnkle'

            self.single_generator_3d_world_32keyPoint[:, 6, 0] = left_leg_pos[:, 0, 0]
            self.single_generator_3d_world_32keyPoint[:, 6, 1] = left_leg_pos[:, 1, 0]
            self.single_generator_3d_world_32keyPoint[:, 6, 2] = left_leg_pos[:, 2, 0]  # 'LHip'

            self.single_generator_3d_world_32keyPoint[:, 7, 0] = left_leg_pos[:, 0, 3]
            self.single_generator_3d_world_32keyPoint[:, 7, 1] = left_leg_pos[:, 1, 3]
            self.single_generator_3d_world_32keyPoint[:, 7, 2] = left_leg_pos[:, 2, 3]  # 'LKnee'

            self.single_generator_3d_world_32keyPoint[:, 8, 0] = left_leg_pos[:, 0, 4]
            self.single_generator_3d_world_32keyPoint[:, 8, 1] = left_leg_pos[:, 1, 4]
            self.single_generator_3d_world_32keyPoint[:, 8, 2] = left_leg_pos[:, 2, 4]  # 'LAnkle'

            self.single_generator_3d_world_32keyPoint[:, 12, 0] = body_pos[:, 0, 3]
            self.single_generator_3d_world_32keyPoint[:, 12, 1] = body_pos[:, 1, 3]
            self.single_generator_3d_world_32keyPoint[:, 12, 2] = body_pos[:, 2, 3]  # 'Spine'

            self.single_generator_3d_world_32keyPoint[:, 13, 0] = body_pos[:, 0, 6]
            self.single_generator_3d_world_32keyPoint[:, 13, 1] = body_pos[:, 1, 6]
            self.single_generator_3d_world_32keyPoint[:, 13, 2] = body_pos[:, 2, 6]  # 'Thorax'

            self.single_generator_3d_world_32keyPoint[:, 14, 0] = body_pos[:, 0, 12]
            self.single_generator_3d_world_32keyPoint[:, 14, 1] = body_pos[:, 1, 12]
            self.single_generator_3d_world_32keyPoint[:, 14, 2] = body_pos[:, 2, 12]  # 'Neck/Nose'

            self.single_generator_3d_world_32keyPoint[:, 15, 0] = body_pos[:, 0, 12]  # body_pos[:, 0, 13]
            self.single_generator_3d_world_32keyPoint[:, 15, 1] = body_pos[:, 1, 12]  # body_pos[:, 1, 13]
            self.single_generator_3d_world_32keyPoint[:, 15, 2] = body_pos[:, 2, 12]  # body_pos[:, 2, 13]  # 'Head'

            self.single_generator_3d_world_32keyPoint[:, 17, 0] = left_hand_pos[:, 0, 9]
            self.single_generator_3d_world_32keyPoint[:, 17, 1] = left_hand_pos[:, 1, 9]
            self.single_generator_3d_world_32keyPoint[:, 17, 2] = left_hand_pos[:, 2, 9]  # 'LShoulder'

            self.single_generator_3d_world_32keyPoint[:, 18, 0] = left_hand_pos[:, 0, 12]
            self.single_generator_3d_world_32keyPoint[:, 18, 1] = left_hand_pos[:, 1, 12]
            self.single_generator_3d_world_32keyPoint[:, 18, 2] = left_hand_pos[:, 2, 12]  # 'LElbow'

            self.single_generator_3d_world_32keyPoint[:, 19, 0] = left_hand_pos[:, 0, 13]
            self.single_generator_3d_world_32keyPoint[:, 19, 1] = left_hand_pos[:, 1, 13]
            self.single_generator_3d_world_32keyPoint[:, 19, 2] = left_hand_pos[:, 2, 13]  # 'LWrist'

            self.single_generator_3d_world_32keyPoint[:, 25, 0] = right_hand_pos[:, 0, 9]
            self.single_generator_3d_world_32keyPoint[:, 25, 1] = right_hand_pos[:, 1, 9]
            self.single_generator_3d_world_32keyPoint[:, 25, 2] = right_hand_pos[:, 2, 9]  # 'RShoulder'

            self.single_generator_3d_world_32keyPoint[:, 26, 0] = right_hand_pos[:, 0, 12]
            self.single_generator_3d_world_32keyPoint[:, 26, 1] = right_hand_pos[:, 1, 12]
            self.single_generator_3d_world_32keyPoint[:, 26, 2] = right_hand_pos[:, 2, 12]  # 'RElbow'

            self.single_generator_3d_world_32keyPoint[:, 27, 0] = right_hand_pos[:, 0, 13]
            self.single_generator_3d_world_32keyPoint[:, 27, 1] = right_hand_pos[:, 1, 13]
            self.single_generator_3d_world_32keyPoint[:, 27, 2] = right_hand_pos[:, 2, 13]  # 'RWrist'

            root_3d_pos = root_3d_pos.view(-1, 1, 3)
            self.single_generator_3d_world_32keyPoint = self.single_generator_3d_world_32keyPoint + root_3d_pos

            return self.single_generator_3d_world_32keyPoint

    def init_Fk_DH_angle(self):
        left_leg_joints_angle = [0, 0, 0, 0, 0]
        right_leg_joints_angle = [0, 0, 0, 0, 0]
        body_joints_angle = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # 13
        left_hand_joints_angle = [0, 0, 0, 0, 0]
        right_hand_joints_angle = [0, 0, 0, 0, 0]
        generator_global_rot_3d_pos_angle = (0.0, 0.0, 0.0)

        single_generator_3d_world_32keyPoint = \
            self.change_3d_joint_angle(
                left_leg_joints_angle=left_leg_joints_angle,
                right_leg_joints_angle=right_leg_joints_angle,
                body_joints_angle=body_joints_angle,
                left_hand_joints_angle=left_hand_joints_angle,
                right_hand_joints_angle=right_hand_joints_angle,
                generator_global_rot_3d_pos_angle=generator_global_rot_3d_pos_angle,
                left_small_leg_len=0.5,
                right_small_leg_len=0.5,
                left_big_leg_len=0.6,
                right_big_leg_len=0.6,
                left_hip_len=0.25,
                right_hip_len=0.25,
                waist_len=0.25,
                thorax_len=0.2,
                left_shoulder_len=0.4,
                right_shoulder_len=0.4,
                left_big_arm_len=0.4,
                right_big_arm_len=0.4,
                left_small_arm_len=0.35,
                right_small_arm_len=0.35,
                neck_len=0.15,
                # head_len=0.1,
                root_3d_pos=(0.0, 0.0, 0.0)
            )
        return single_generator_3d_world_32keyPoint


    def random_state(self):
        return self.random

    def set_random_state(self, random):
        self.random = random

    def get_dataSet_3d_and_2d_pose(self, dataset, pix_2d):

        world_3d = {}
        for subject in dataset.subjects():
            world_3d[subject] = {}

            for action in dataset[subject].keys():
                anim = dataset[subject][action]
                world_3d[subject][action] = {}

                for cam_idx, kps in enumerate(pix_2d[subject][action]):
                    world_3d[subject][action][cam_idx] = anim['positions']

        self.dataSet_world_3d_pos = world_3d  # (N,16,3)
        self.dataSet_2d_pos = pix_2d

    def my_random_get_sigle_frame_data(self):
        choice_subject_id = self.random.randint(0, len(self.train_subjects))
        choice_subject = self.train_subjects[choice_subject_id]

        action_name = list(self.dataSet_world_3d_pos[choice_subject].keys())
        choice_action_id = self.random.randint(0, len(action_name))
        choice_action = action_name[choice_action_id]

        cam_name = list(self.dataSet_world_3d_pos[choice_subject][choice_action].keys())
        choice_cam_id = self.random.randint(0, len(cam_name))
        choice_cam = cam_name[choice_cam_id]

        used_frame = self.random.randint(0,
                            self.dataSet_world_3d_pos[choice_subject][choice_action][choice_cam].shape[0])

        return choice_subject, choice_action, choice_cam, used_frame

    def get_bone_len_from_dataSet(self):
        choice_subject, choice_action, choice_cam, used_frame = self.my_random_get_sigle_frame_data()

        generator_input_single_world_3d_16_keypoint = \
            copy.deepcopy(
                self.dataSet_world_3d_pos[choice_subject][choice_action][choice_cam][used_frame])
        generator_input_single_2d = \
            copy.deepcopy(self.dataSet_2d_pos[choice_subject][choice_action][choice_cam][used_frame])

        dataSet_bone_len = []
        for bone_len_temp in used_16key_15bone_len_table:  #
            dataSet_bone_len.append(
                special_operate.get_single_bone_length(
                    generator_input_single_world_3d_16_keypoint[bone_len_temp[0]],
                    generator_input_single_world_3d_16_keypoint[bone_len_temp[1]]))

        self.record_bone_len = copy.deepcopy(dataSet_bone_len)

    def get_root_3d_pos_from_dataSet(self):

        choice_subject, choice_action, choice_cam, used_frame = self.my_random_get_sigle_frame_data()

        generator_input_single_world_3d = \
            copy.deepcopy(
                self.dataSet_world_3d_pos[choice_subject][choice_action][choice_cam][used_frame])
        generator_input_single_2d = \
            copy.deepcopy(self.dataSet_2d_pos[choice_subject][choice_action][choice_cam][used_frame])

        root_3d_pos = generator_input_single_world_3d[0]
        self.root_3d_pos = copy.deepcopy(root_3d_pos)

    def handler_but_generater(self):

        end = time.time()

        angle_range_table = {
            'joint1': {'range': (-90, 45), 'changeRate': (-20, 20)},
            'joint2': {'range': (-90, 45), 'changeRate': (-20, 20)},
            'joint3': {'range': (-45, 180 - 60), 'changeRate': (-30, 30)},
            'joint4': {'range': (-135, 0), 'changeRate': (-40, 40)},  #
            'joint5': {'range': (0, 0), 'changeRate': (-20, 20)},
            'joint6': {'range': (-45, 90), 'changeRate': (-20, 20)},
            'joint7': {'range': (-45, 90), 'changeRate': (-20, 20)},
            'joint8': {'range': (-45, 180 - 60), 'changeRate': (-30, 30)},
            'joint9': {'range': (-135, 0), 'changeRate': (-40, 40)},  #
            'joint10': {'range': (0, 0), 'changeRate': (-20, 20)},
            'joint11': {'range': (-25, 25), 'changeRate': (-20, 20)},
            'joint12': {'range': (-10, 90), 'changeRate': (-20, 20)},
            'joint13': {'range': (-20, 20), 'changeRate': (-20, 20)},
            'joint14': {'range': (-20, 20), 'changeRate': (-20, 20)},
            'joint15': {'range': (-10, 45), 'changeRate': (-20, 20)},
            'joint16': {'range': (-25, 25), 'changeRate': (-20, 20)},
            'joint17': {'range': (-20, 20), 'changeRate': (-20, 20)},
            'joint18': {'range': (0, 0), 'changeRate': (-20, 20)},  # 18
            'joint19': {'range': (-20, 20), 'changeRate': (-20, 20)},
            'joint20': {'range': (-90, 90), 'changeRate': (-20, 20)},
            'joint21': {'range': (-20, 90), 'changeRate': (-20, 20)},
            'joint22': {'range': (-45, 45), 'changeRate': (-20, 20)},
            'joint23': {'range': (0, 0), 'changeRate': (-20, 20)},
            'joint24': {},  # {'range': (0, 0), 'changeRate': (0, 0)},
            'joint25': {'range': (-135, 45), 'changeRate': (-20, 20)},
            'joint26': {'range': (-135, 45), 'changeRate': (-20, 20)},
            'joint27': {'range': (-80 + 35, 180), 'changeRate': (-30, 30)},
            'joint28': {'range': (0, 135), 'changeRate': (-40, 40)},
            'joint29': {'range': (0, 0), 'changeRate': (-20, 20)},
            'joint30': {'range': (-45, 135), 'changeRate': (-20, 20)},
            'joint31': {'range': (-45, 135), 'changeRate': (-20, 20)},
            'joint32': {'range': (-80 + 35, 180), 'changeRate': (-30, 30)},
            'joint33': {'range': (0, 135), 'changeRate': (-40, 40)},
            'joint34': {'range': (0, 0), 'changeRate': (-20, 20)}

        }
        global_rotation_table = {
            'angle_x': {'range': (-20, 20), 'changeRate': (-5, 5)},
            'angle_y': {'range': (-20, 20), 'changeRate': (-5, 5)},
            'angle_z': {'range': (-180, 180), 'changeRate': (-5, 5)}
        }

        self.generator_3d_pos_angle = []  #
        self.generator_global_rot_3d_pos_angle = []  #
        self.generator_bone_len = []  #
        self.generator_root = []  #

        generator_whole_number = self.args.generator_whole_number

        bar = Bar('Train pose gan', max=generator_whole_number)
        batch_time = AverageMeter()
        data_time = AverageMeter()

        for frame_num in range(generator_whole_number):  ####

            data_time.update(time.time() - end)

            if self.args.generator_choose_BoneLen == True:
                self.get_bone_len_from_dataSet()

            self.generator_bone_len.append(self.record_bone_len)

            if self.args.generator_choose_root_pos == True:
                self.get_root_3d_pos_from_dataSet()
            self.generator_root.append(self.root_3d_pos)

            change_angle_num = self.random.randint(0, len(angle_range_table))

            change_angle = self.random.choice(np.arange(len(angle_range_table)), size=change_angle_num,
                                              replace=False)

            generator_angle = []
            for angle_temp in range(len(angle_range_table)):

                if (angle_temp + 1) == 24:
                    continue

                if (angle_temp in change_angle) and (frame_num > 0):

                    angle_min_temp = angle_range_table['joint' + str(angle_temp + 1)]['range'][0]
                    angle_max_temp = angle_range_table['joint' + str(angle_temp + 1)]['range'][1]
                    mu = (angle_min_temp + angle_max_temp) / 2
                    sigma = 60  # mean and standard deviation
                    angle = self.random.normal(mu, sigma)
                    if angle > angle_max_temp:
                        angle = angle_max_temp
                    elif angle < angle_min_temp:
                        angle = angle_min_temp
                    generator_angle.append(angle)
                else:
                    generator_angle.append(0)

            angle_index = 0
            generator_global_angle = []
            for angle_key in global_rotation_table.keys():

                if frame_num > 0:

                    if self.args.generator_global_rot == True:

                        global_angle_min_temp = global_rotation_table[angle_key]['range'][0]
                        global_angle_max_temp = global_rotation_table[angle_key]['range'][1]
                        mu = (global_angle_min_temp + global_angle_max_temp) / 2  #
                        sigma = 60  # mean and standard deviation
                        global_angle = self.random.normal(mu, sigma)

                        if global_angle > global_angle_max_temp:
                            global_angle = global_angle_max_temp
                        elif global_angle < global_angle_min_temp:
                            global_angle = global_angle_min_temp

                        generator_global_angle.append(global_angle)

                    else:
                        generator_global_angle.append(0)
                else:
                    generator_global_angle.append(0)

                angle_index += 1

            self.generator_global_rot_3d_pos_angle.append(generator_global_angle)

            self.generator_3d_pos_angle.append(generator_angle)

            batch_time.update(time.time() - end)
            end = time.time()
            bar.suffix = '({batch}/{size}) Data: {data:.6f}s | Batch: {bt:.3f}s | Total: {ttl:} | ETA: {eta:} ' \
                .format(batch=frame_num + 1, size=generator_whole_number, data=data_time.avg, bt=batch_time.avg,
                        ttl=bar.elapsed_td, eta=bar.eta_td)
            bar.next()
        bar.finish()

        self.generator_3d_pos_angle = np.array(self.generator_3d_pos_angle).reshape(-1, len(generator_angle))

        new_generator_3d_pos = []
        for frame_num in range(generator_whole_number):
            generator_angle = self.generator_3d_pos_angle[frame_num]

            right_leg_joints_angle = generator_angle[0: 5]
            left_leg_joints_angle = generator_angle[5: 10]
            body_joints_angle = generator_angle[10: 23]    # [10: 24]
            right_hand_joints_angle = generator_angle[23: 28]  # [24: 29]
            left_hand_joints_angle = generator_angle[28: 33]  # [29: 34]

            generator_global_rot_3d_pos_angle = self.generator_global_rot_3d_pos_angle[frame_num]

            self.record_bone_len = self.generator_bone_len[frame_num]

            bone_len_scaler = np.zeros(8)
            if self.args.bone_len_scaler == 'different':
                bone_len_scaler = self.random.randint(-200, 200, size=(8))
                bone_len_scaler = np.array(bone_len_scaler).reshape(8)
                bone_len_scaler = bone_len_scaler / 1000.0
            elif self.args.bone_len_scaler == 'same':
                bone_len_scaler = self.random.randint(-200, 200, size=(1))
                bone_len_scaler = np.array(bone_len_scaler).reshape(1)
                bone_len_scaler = bone_len_scaler.repeat(8)
                bone_len_scaler = bone_len_scaler / 1000.0
            elif self.args.bone_len_scaler == '':
                pass
            else:
                raise ("args.bone_len_scaler")

            left_small_leg_len = self.record_bone_len[0] * (1 + bone_len_scaler[0])
            right_small_leg_len = self.record_bone_len[1] * (1 + bone_len_scaler[0])
            left_big_leg_len = self.record_bone_len[2] * (1 + bone_len_scaler[1])
            right_big_leg_len = self.record_bone_len[3] * (1 + bone_len_scaler[1])
            left_hip_len = self.record_bone_len[4] * (1 + bone_len_scaler[2])
            right_hip_len = self.record_bone_len[5] * (1 + bone_len_scaler[2])
            waist_len = self.record_bone_len[6] * (1 + bone_len_scaler[3])
            thorax_len = self.record_bone_len[7]
            left_shoulder_len = self.record_bone_len[8] * (1 + bone_len_scaler[4])
            right_shoulder_len = self.record_bone_len[9] * (1 + bone_len_scaler[4])
            left_big_arm_len = self.record_bone_len[10] * (1 + bone_len_scaler[5])
            right_big_arm_len = self.record_bone_len[11] * (1 + bone_len_scaler[5])
            left_small_arm_len = self.record_bone_len[12] * (1 + bone_len_scaler[6])
            right_small_arm_len = self.record_bone_len[13] * (1 + bone_len_scaler[6])
            neck_len = self.record_bone_len[14] * (1 + bone_len_scaler[7])

            self.root_3d_pos = self.generator_root[frame_num]
            root_3d_pos = self.root_3d_pos

            single_generator_3d_world_32keyPoint = \
                        self.change_3d_joint_angle(
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
            new_generator_3d_pos.append(single_generator_3d_world_32keyPoint)

        new_generator_3d_pos = np.array(new_generator_3d_pos)

        return new_generator_3d_pos, \
               self.generator_3d_pos_angle, \
               self.generator_global_rot_3d_pos_angle, \
               self.generator_bone_len, \
               self.generator_root

