from models_Fk_GAN import myMainWindow
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec  #
import matplotlib
matplotlib.use("Qt5Agg")  #
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D



from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
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


class MyFigure(FigureCanvas):

    def __init__(self, title, dimension, args, train_subjects, parent=None, width=5, height=4, dpi=100):

        self.myFigure = Figure(figsize=(width, height), dpi=dpi)

        self.dimension = dimension

        if dimension == 2:
            self.axes = self.myFigure.add_subplot(111)
        else:
            self.axes = Axes3D(self.myFigure)

        super(MyFigure, self).__init__(self.myFigure)

        self.setParent(parent)
        self.title = title

    def plot_groundTruth_point(self, keyPoint):

        if self.dimension == 2:
            self.axes.set_title(self.title)
            viz.show2Dpose(keyPoint, self.axes)

        if self.dimension == 3:
            self.axes.set_title(self.title)
            viz.show3Dpose_world(keyPoint, self.axes, gt=False)

        self.draw()

    def plot_forward_kinematics_point(self, dataset, choice_subject,
                                      choice_action, choice_cam, pos_3d_world):

        if self.dimension == 2:
            cam_para_temp = dataset[choice_subject][choice_action]['cameras'][choice_cam]['intrinsic']
            res_w = dataset[choice_subject][choice_action]['cameras'][choice_cam]['res_w']
            res_h = dataset[choice_subject][choice_action]['cameras'][choice_cam]['res_h']
            cam_R = dataset[choice_subject][choice_action]['cameras'][choice_cam]['orientation']
            cam_t = dataset[choice_subject][choice_action]['cameras'][choice_cam]['translation']

            show_16key_3D_temp = np.array(pos_3d_world).reshape((1, -1, 3))
            pos_3d_cam = world_to_camera(show_16key_3D_temp, R=cam_R, t=cam_t)

            show_16key_2D_picture = wrap(project_to_2d, True, pos_3d_cam, cam_para_temp)

            show_16key_2D_pix = image_coordinates(show_16key_2D_picture, w=res_w, h=res_h)

            show_16key_2D_pix_norm = normalize_screen_coordinates(show_16key_2D_pix, w=res_w, h=res_h)

            viz.show2Dpose(show_16key_2D_pix_norm, self.axes)

            self.draw()

            return show_16key_2D_pix_norm

        elif self.dimension == 3:
            show_16key_3D_temp = pos_3d_world[H36M_32_To_16_Table, :]

            viz.show3Dpose_world(show_16key_3D_temp, self.axes, gt=False)

            self.draw()

            return show_16key_3D_temp


class MyWindow(QMainWindow, myMainWindow.Ui_MainWindow):
    def __init__(self, args, dataset, train_subjects,
                 func_change_3d_joint_angle, fk_handler_but_generater,
                 parent=None):

        super(MyWindow, self).__init__(parent)
        self.setupUi(self)

        self.dataset = dataset
        self.args = args
        self.train_subjects = train_subjects

        self.func_change_3d_joint_angle = func_change_3d_joint_angle
        self.fk_handler_but_generater = fk_handler_but_generater

        self.figure1 = MyFigure(title='gt_3d', dimension=3, args=self.args, train_subjects=self.train_subjects)
        self.figure2 = MyFigure(title='gt_2d', dimension=2, args=self.args, train_subjects=self.train_subjects)
        self.figure3 = MyFigure(title='generator_3d', dimension=3, args=self.args, train_subjects=self.train_subjects)
        self.figure4 = MyFigure(title='generator_2d', dimension=2, args=self.args, train_subjects=self.train_subjects)

        self.but_showMode_flag = 0
        self.timer = QTimer(self)
        random_seed = self.args.random_seed
        self.random = np.random.RandomState(random_seed)

        self.generator_3d_pos_angle = []
        self.generator_global_rot_3d_pos_angle = []
        self.generator_bone_len = []
        self.generator_root = []

        self.show_3d_pos_num = 0

        self.record_bone_len = np.zeros(15)

        self.camera_parameters = {}

        self.root_3d_pos = np.array([0, 0, 0])

        self.dataSet_world_3d_pos = {}
        self.dataSet_2d_pos = {}

        self.choice_subject = []
        self.choice_action = []
        self.choice_cam = []

    def random_state(self):
        return self.random

    def set_random_state(self, random):
        self.random = random

    def get_single_bone_len(self, single_bone_len):
        self.record_bone_len = single_bone_len
        print(self.record_bone_len)


    def handle_init_subFigure_and_show_Fk_pose(self, point_3d, point_2d, FK_point_3d,
                                               choice_subject, choice_action, choice_cam, used_frame):
        self.choice_subject = choice_subject
        self.choice_action = choice_action
        self.choice_cam = choice_cam

        self.figure1.plot_groundTruth_point(point_3d[choice_subject][choice_action][choice_cam][used_frame])
        self.figure2.plot_groundTruth_point(point_2d[choice_subject][choice_action][choice_cam][used_frame])

        fk_pose_3d_world = self.figure3.plot_forward_kinematics_point(self.dataset,
                                                                      choice_subject, choice_action, choice_cam,
                                                                      pos_3d_world=FK_point_3d
                                                                      )
        fk_pos_2d_pix_norm = self.figure4.plot_forward_kinematics_point(self.dataset,
                                                                        choice_subject, choice_action, choice_cam,
                                                                        pos_3d_world=fk_pose_3d_world
                                                                        )

        self.layout_gt = QGridLayout(self.groupBox_groundTruth)
        self.layout_fk = QGridLayout(self.groupBox_generator)

        self.layout_gt.addWidget(self.figure1)
        self.layout_gt.addWidget(self.figure2)
        self.layout_fk.addWidget(self.figure3)
        self.layout_fk.addWidget(self.figure4)

        self.show()

    def update_joint_angle(self):
        if self.but_showMode_flag == 0:
            self.figure3.axes.cla()
            self.figure4.axes.cla()

            right_leg_joints_angle = [float(self.bar_joint1.value()), float(self.bar_joint2.value()),
                                      float(self.bar_joint3.value()), float(self.bar_joint4.value()),
                                      float(self.bar_joint5.value())]

            left_leg_joints_angle = [float(self.bar_joint6.value()), float(self.bar_joint7.value()),
                                     float(self.bar_joint8.value()), float(self.bar_joint9.value()),
                                     float(self.bar_joint10.value())]

            body_joints_angle = [float(self.bar_joint11.value()), float(self.bar_joint12.value()),
                                 float(self.bar_joint13.value()), float(self.bar_joint14.value()),
                                 float(self.bar_joint15.value()), float(self.bar_joint16.value()),
                                 float(self.bar_joint17.value()), float(self.bar_joint18.value()),
                                 float(self.bar_joint19.value()), float(self.bar_joint20.value()),
                                 float(self.bar_joint21.value()), float(self.bar_joint22.value()),
                                 float(self.bar_joint23.value())]

            right_hand_joints_angle = [float(self.bar_joint25.value()), float(self.bar_joint26.value()),
                                       float(self.bar_joint27.value()), float(self.bar_joint28.value()),
                                       float(self.bar_joint29.value())]
            left_hand_joints_angle = [float(self.bar_joint30.value()), float(self.bar_joint31.value()),
                                      float(self.bar_joint32.value()), float(self.bar_joint33.value()),
                                      float(self.bar_joint34.value())]

            generator_global_rot_3d_pos_angle = np.array(
                [float(self.bar_global_x.value()), float(self.bar_global_y.value()),
                 float(self.bar_global_z.value())])

            left_small_leg_len = self.record_bone_len[0]
            right_small_leg_len = self.record_bone_len[1]
            left_big_leg_len = self.record_bone_len[2]
            right_big_leg_len = self.record_bone_len[3]
            left_hip_len = self.record_bone_len[4]
            right_hip_len = self.record_bone_len[5]
            waist_len = self.record_bone_len[6]
            thorax_len = self.record_bone_len[7]
            left_shoulder_len = self.record_bone_len[8]
            right_shoulder_len = self.record_bone_len[9]
            left_big_arm_len = self.record_bone_len[10]
            right_big_arm_len = self.record_bone_len[11]
            left_small_arm_len = self.record_bone_len[12]
            right_small_arm_len = self.record_bone_len[13]
            neck_len = self.record_bone_len[14]
            # head_len = self.record_bone_len[15]

            root_3d_pos = self.root_3d_pos

            single_generator_3d_world_32keyPoint = \
                self.func_change_3d_joint_angle(
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
                    # head_len=head_len,
                    root_3d_pos=root_3d_pos
                )

            show_16key_3D_temp = single_generator_3d_world_32keyPoint[H36M_32_To_16_Table, :]

            viz.show3Dpose_world(show_16key_3D_temp, self.figure3.axes, gt=False)

            self.figure4.plot_forward_kinematics_point(self.dataset,
                                                       self.choice_subject, self.choice_action, self.choice_cam,
                                                       show_16key_3D_temp
                                                       )

            self.figure3.draw()
            self.figure4.draw()


    def handler_but_showMode(self):

        if self.but_showMode_flag % 2 == 0:
            self.but_showMode_flag = 1
            self.timer.stop()

            self.but_showMode.setText('前后键 更新模式')
        else:
            self.but_showMode_flag = 0
            self.timer.start(200)

            self.but_showMode.setText('滑动条 更新模式')

    def QT_handler_but_generater(self):
        new_generator_3d_pos, self.generator_3d_pos_angle, \
        self.generator_global_rot_3d_pos_angle, self.generator_bone_len, \
        self.generator_root = self.fk_handler_but_generater()

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Up:
            if self.show_3d_pos_num > 0:
                self.show_3d_pos_num -= 1

        elif e.key() == Qt.Key_Down:
            if self.show_3d_pos_num < len(self.generator_3d_pos_angle) - 1:
                self.show_3d_pos_num += 1

        elif e.key() == Qt.Key_Left:
            print("左")
        elif e.key() == Qt.Key_Right:
            print("右边")

        if self.but_showMode_flag == 1 and self.show_3d_pos_num > 0:
            self.figure3.axes.cla()  #
            self.figure4.axes.cla()

            generator_angle = self.generator_3d_pos_angle[self.show_3d_pos_num]

            right_leg_joints_angle = generator_angle[0: 5]
            left_leg_joints_angle = generator_angle[5: 10]

            body_joints_angle = generator_angle[10: 23]    # [10: 24]
            right_hand_joints_angle = generator_angle[23: 28]    # [24: 29]
            left_hand_joints_angle = generator_angle[28: 33]    # [29: 34]

            generator_global_rot_3d_pos_angle = self.generator_global_rot_3d_pos_angle[self.show_3d_pos_num]

            self.record_bone_len = self.generator_bone_len[self.show_3d_pos_num]

            left_small_leg_len = self.record_bone_len[0]
            right_small_leg_len = self.record_bone_len[1]
            left_big_leg_len = self.record_bone_len[2]
            right_big_leg_len = self.record_bone_len[3]
            left_hip_len = self.record_bone_len[4]
            right_hip_len = self.record_bone_len[5]
            waist_len = self.record_bone_len[6]
            thorax_len = self.record_bone_len[7]
            left_shoulder_len = self.record_bone_len[8]
            right_shoulder_len = self.record_bone_len[9]
            left_big_arm_len = self.record_bone_len[10]
            right_big_arm_len = self.record_bone_len[11]
            left_small_arm_len = self.record_bone_len[12]
            right_small_arm_len = self.record_bone_len[13]
            neck_len = self.record_bone_len[14]

            self.root_3d_pos = self.generator_root[self.show_3d_pos_num]
            root_3d_pos = self.root_3d_pos

            single_generator_3d_world_32keyPoint = \
                self.func_change_3d_joint_angle(
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
                    # head_len=head_len,
                    root_3d_pos=root_3d_pos
                )

            show_16key_3D_temp = single_generator_3d_world_32keyPoint[H36M_32_To_16_Table, :]

            viz.show3Dpose_world(show_16key_3D_temp, self.figure3.axes, gt=False)

            self.figure4.plot_forward_kinematics_point(self.dataset,
                                                       self.choice_subject, self.choice_action, self.choice_cam,
                                                       show_16key_3D_temp
                                                       )

            self.figure3.draw()
            self.figure4.draw()