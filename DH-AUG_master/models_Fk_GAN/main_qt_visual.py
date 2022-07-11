from pylab import *
import torch.backends.cudnn as cudnn
import numpy as np
import loguru

import myMainWindow
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout

####################################
import zipfile
import h5py
from glob import glob
from shutil import rmtree
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import sys
import errno
import cdflib

import copy

import matplotlib
matplotlib.use("Qt5Agg")  # 声明使用QT5

from function_aug import config
from forward_kinematics_DH_model import *
from visual_Fk_DH_byQt import *
import datetime
import os.path as path
import random

if __name__ == '__main__':

    args = config.get_parse_args()

    whole_subjects = ['S1', 'S5', 'S6', 'S7', 'S8', 'S9', 'S11']
    train_subjects = []
    if args.s1only:
        train_subjects = ['S1']  #
    elif args.s1s5only:
        train_subjects = ['S1', 'S5']  #
    else:
        train_subjects = ['S1', 'S5', 'S6', 'S7', 'S8']

    if args.s1only == True and args.s1s5only == True:
        raise KeyError(' args.s1only and args.s1s5only both set true')

    if train_subjects == []:
        raise 'train_subjects   error'

    test_subjects = ['S9', 'S11']

    random_seed = args.random_seed
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    # copy from #https://pytorch.org/docs/stable/notes/randomness.html
    torch.backends.cudnn.deterministic = True

    data_dict = special_operate.fk_data_preparation(args, r'..')

    dataset = data_dict['dataset']
    pose_2d_pix = data_dict['keypoints']

    app = QApplication(sys.argv)

    FK_DH_Class = Forward_Kinematics_DH_Model(args, train_subjects, dataset)

    myWin = MyWindow(args, dataset=dataset, train_subjects=train_subjects,
                     func_change_3d_joint_angle=FK_DH_Class.change_3d_joint_angle,
                     fk_handler_but_generater=FK_DH_Class.handler_but_generater)

    FK_DH_Class.get_dataSet_3d_and_2d_pose(dataset, pose_2d_pix)

    FK_DH_Class.get_bone_len_from_dataSet()

    ##############
    FK_DH_Class.record_bone_len[0] = FK_DH_Class.record_bone_len[10]
    FK_DH_Class.record_bone_len[1] = FK_DH_Class.record_bone_len[11]
    FK_DH_Class.record_bone_len[2] = FK_DH_Class.record_bone_len[12]
    FK_DH_Class.record_bone_len[3] = FK_DH_Class.record_bone_len[13]
    ############

    myWin.get_single_bone_len(FK_DH_Class.record_bone_len)

    FK_DH_Class.get_root_3d_pos_from_dataSet()

    choice_subject, choice_action, choice_cam, used_frame = FK_DH_Class.my_random_get_sigle_frame_data()

    show_FK_point_3d = FK_DH_Class.init_Fk_DH_angle()
    myWin.handle_init_subFigure_and_show_Fk_pose(FK_DH_Class.dataSet_world_3d_pos, FK_DH_Class.dataSet_2d_pos,
                                                    show_FK_point_3d,
                                                    choice_subject, choice_action, choice_cam, used_frame)

    myWin.but_showMode.setText('toolbar updata mode')
    myWin.but_showMode.clicked.connect(myWin.handler_but_showMode)

    myWin.but_generater.setText('generate 3D pose')
    myWin.but_generater.clicked.connect(myWin.QT_handler_but_generater)

    myWin.timer.timeout.connect(myWin.update_joint_angle)
    myWin.timer.start(200)
    myWin.show()

    sys.exit(app.exec_())
