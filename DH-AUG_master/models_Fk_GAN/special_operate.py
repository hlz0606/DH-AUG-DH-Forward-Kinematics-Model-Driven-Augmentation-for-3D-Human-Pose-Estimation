import numpy as np
import copy
import matplotlib
#matplotlib.use('Agg')
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.pyplot import MultipleLocator
import torch
#from __future__ import print_function, absolute_import, division

import os.path as path
from torch.utils.data import DataLoader
from common.data_loader import PoseDataSet, PoseBuffer, PoseTarget, CmuDatasetPoseTarget
from utils.data_utils import fetch, read_3d_data, create_2d_data

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import os

from common.camera import normalize_screen_coordinates
import cv2

PW3D_TO_16POINTS_TABLE = {1:4, 2:1, 6:7, 4:5, 5:2, 7:6, 8:3, 15:9,
                          16:10, 17:13, 18:11, 19:14, 20:12, 21:15}


PW3D_TO_16POINTS_TABLE_FROM_VIBE = {0:3, 1:2, 2:1, 3:4, 4:5, 5:6, 6:15, 7:14, 8:13,
                                    9:10, 10:11, 11:12, 12:9}

PW3D_TO_16POINTS_TABLE_FROM_COCO = {0:9, 1:8, 2:13, 3:14, 4:15, 5:10, 6:11, 7:12,
                                    8:1, 9:2, 10:3, 11:4, 12:5, 13:6}


def fk_data_preparation(args, file_path_prefix):

    dataset_path = path.join(file_path_prefix, 'data', 'data_3d_' + args.dataset + '.npz')

    if args.dataset == 'h36m':
        from common.h36m_dataset import Human36mDataset, TEST_SUBJECTS
        dataset = Human36mDataset(dataset_path)
        if args.s1only:
            subjects_train = ['S1']
        elif args.s1s5only:
            subjects_train = ['S1', 'S5']
        else:
            subjects_train = ['S1', 'S5', 'S6', 'S7', 'S8']
        subjects_test = TEST_SUBJECTS

        if args.s1only == True and args.s1s5only == True:
            raise KeyError(' args.s1only and args.s1s5only both set true')

    else:
        raise KeyError('Invalid dataset')


    print('==> Loading 3D data...')
    dataset = read_3d_data(dataset)

    keypoints = create_2d_data(path.join(file_path_prefix, 'data', 'data_2d_' + args.dataset + '_' + args.keypoints + '.npz'), dataset)

    action_filter = None if args.actions == '*' else args.actions.split(',')
    if action_filter is not None:
        action_filter = map(lambda x: dataset.define_actions(x)[0], action_filter)
        print('==> Selected actions: {}'.format(action_filter))

    stride = args.downsample

    ############################################
    # general 2D-3D pair dataset
    ############################################
    poses_train, poses_train_2d, actions_train, cams_train = fetch(subjects_train, dataset, keypoints,
                                                                   args, 'train',  action_filter,
                                                                   stride, whether_need_cam_external=True)

    poses_valid, poses_valid_2d, actions_valid, cams_valid = fetch(subjects_test, dataset, keypoints,
                                                                   args, 'test',  action_filter,
                                                                   stride)

    train_det2d3d_loader = DataLoader(PoseDataSet(poses_train, poses_train_2d, actions_train, cams_train),
                                      batch_size=args.batch_size,
                                      shuffle=True, num_workers=args.num_workers, pin_memory=True)

    train_gt2d3d_loader = DataLoader(PoseDataSet(poses_train, poses_train_2d, actions_train, cams_train),
                                 batch_size=args.batch_size,
                                 shuffle=True, num_workers=args.num_workers, pin_memory=True)

    target_2d_loader = DataLoader(PoseTarget(poses_train_2d),
                                  batch_size=args.batch_size,
                                  shuffle=True, num_workers=args.num_workers, pin_memory=True)

    target_3d_loader = DataLoader(PoseTarget(poses_train),
                                  batch_size=args.batch_size,
                                  shuffle=True, num_workers=args.num_workers, pin_memory=True)

    valid_loader = DataLoader(PoseDataSet(poses_valid, poses_valid_2d, actions_valid, cams_valid),
                              batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers, pin_memory=True)


    # 3DHP -  2929 version
    mpi3d_npz = np.load(file_path_prefix + '/data_extra/test_set/test_3dhp.npz')    # this is the 2929 version
    tmp = mpi3d_npz
    mpi3d_loader = DataLoader(PoseBuffer([tmp['pose3d']], [tmp['pose2d']]),
                              batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers, pin_memory=True)


    return {
        'dataset': dataset,
        'train_det2d3d_loader': train_det2d3d_loader,
        'train_gt2d3d_loader': train_gt2d3d_loader,
        'target_2d_loader': target_2d_loader,
        'target_3d_loader': target_3d_loader,
        'H36M_test': valid_loader,
        'mpi3d_loader': mpi3d_loader,
        'action_filter': action_filter,
        'subjects_test': subjects_test,
        'keypoints': keypoints,    # 2D

    }


def get_batch_bone_vector(first_point, second_point):
    # 3d 或 2d
    bone_vector = second_point - first_point

    return bone_vector



def get_single_bone_length(first_point, second_point):
    bone_vector = second_point - first_point

    if torch.is_tensor(first_point) == True and torch.is_tensor(second_point) == True:
        bone_len = torch.pow(torch.pow(bone_vector, 2).sum(dim=-1), 0.5)

    else:
        bone_len = np.sqrt(np.sum(bone_vector ** 2))

    return bone_len


def normalize(vector):
    return vector/np.linalg.norm(vector)


def get_upper_part_basis(skeleton):

    left_shoulder = skeleton[17]
    right_shoulder = skeleton[25]
    axis_x = normalize(right_shoulder - left_shoulder)
    thorax = skeleton[13]
    spine = skeleton[12]
    axis_y = normalize(spine - thorax)

    axis_z = normalize(np.cross(axis_x, axis_y))

    return axis_x, axis_y, axis_z


def Rodrigues_rotation_formula_get_RotationMatrix(angle, u):
    angle = angle * np.pi / 180

    norm = np.linalg.norm(u)
    rotatinMatrix = np.zeros((3, 3))

    u[0] = u[0] / norm
    u[1] = u[1] / norm
    u[2] = u[2] / norm

    rotatinMatrix[0:] = [
        np.cos(angle) + u[0] * u[0] * (1 - np.cos(angle)),
        u[0] * u[1] * (1 - np.cos(angle) - u[2] * np.sin(angle)),
        u[1] * np.sin(angle) + u[0] * u[2] * (1 - np.cos(angle))
    ]
    rotatinMatrix[1:] = [
        u[2] * np.sin(angle) + u[0] * u[1] * (1 - np.cos(angle)),
        np.cos(angle) + u[1] * u[1] * (1 - np.cos(angle)),
        -u[0] * np.sin(angle) + u[1] * u[2] * (1 - np.cos(angle))
    ]
    rotatinMatrix[2:] = [
        -u[1] * np.sin(angle) + u[0] * u[2] * (1 - np.cos(angle)),
        u[0] * np.sin(angle) + u[1] * u[2] * (1 - np.cos(angle)),
        np.cos(angle) + u[2] * u[2] * (1 - np.cos(angle))
    ]

    return rotatinMatrix


def gram_schmidt_columns(X):
    B = np.zeros(X.shape)
    B[:, 0] = (1/np.linalg.norm(X[:, 0]))*X[:, 0]
    for i in range(1, 3):
        v = X[:, i]
        U = B[:, 0:i]
        pc = U.T @ v
        p = U@pc
        v = v - p
        if np.linalg.norm(v) < 2e-16:
            raise ValueError
        else:
            v = normalize(v)
            B[:, i] = v
    return B


def my_visual_3D_pos(args, used_3D_pos, iter, name, showNum=15):

    if args.record_all_picture == True:
        import matplotlib as mpl
        mpl.use('Agg')
        import matplotlib.pyplot as plt
        plt.switch_backend('agg')

        from common.h36m_dataset import H36M_32_To_16_Table

        I = np.array([0, 1, 2, 0, 4, 5, 0, 7, 8, 8, 10, 11, 8, 13, 14])  # start points
        J = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])  # end points
        LR = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0], dtype=bool)

        lcolor = "red"
        rcolor = "blue"

        figure = plt.figure(figsize=(19.2, 10.8))
        if showNum==15:
            gs1 = gridspec.GridSpec(3, 5)  #
        else:
            gs1 = gridspec.GridSpec(1, 1)  #
        plt.subplots_adjust(left=None, bottom=None, right=None,  #
                            top=None, wspace=1, hspace=0.5)

        used_3D_16pos = np.zeros((used_3D_pos.shape[0], 16, 3))

        if used_3D_pos.shape[1] == 16 * 3:
            used_3D_16pos = used_3D_pos.reshape(-1, 16, 3)
        elif used_3D_pos.shape[1] == 32:
            used_3D_16pos = used_3D_pos[H36M_32_To_16_Table].reshape(-1, 16, 3)
        else:     #
            used_3D_16pos = used_3D_pos

        show_column = 0
        for show_num in range(0, showNum, 1):
            show_vals = copy.deepcopy(used_3D_16pos[show_num])  #

            ax1 = plt.subplot(gs1[show_column], projection='3d')
            ax1.set_title('Pos_3d')  #

            radius = 1.7
            ax1.set_xlim3d([-radius / 2 + show_vals[0, 0], radius / 2 + show_vals[0, 0]])
            ax1.set_ylim3d([-radius / 2 + show_vals[0, 1], radius / 2 + show_vals[0, 1]])
            ax1.set_zlim3d([-radius / 2 + show_vals[0, 2], radius / 2 + show_vals[0, 2]])

            for i in np.arange(len(I)):
                x, y, z = [np.array([show_vals[I[i], j], show_vals[J[i], j]]) for j in range(3)]
                ax1.plot(x, y, z, lw=2, c=lcolor if LR[i] else rcolor)  #

            show_column += 1

        fig_path = args.checkpoint + r'/tmp/' + name + str(iter) + '.jpg'
        plt.savefig(fig_path)

        plt.close()


def my_visual_2D_pos(args, used_2D_pos, iter, name, showNum=15):
    if args.record_all_picture == True:
        import matplotlib as mpl
        mpl.use('Agg')
        import matplotlib.pyplot as plt
        plt.switch_backend('agg')

        from common.h36m_dataset import H36M_32_To_16_Table

        I = np.array([0, 1, 2, 0, 4, 5, 0, 7, 8, 8, 10, 11, 8, 13, 14])  # start points
        J = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])  # end points
        LR = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0], dtype=bool)

        lcolor = "red"
        rcolor = "blue"

        figure = plt.figure(figsize=(19.2, 10.8))
        gs1 = gridspec.GridSpec(3, 5)  #
        plt.subplots_adjust(left=None, bottom=None, right=None,
                            top=None, wspace=1, hspace=0.5)

        used_2D_16pos = np.zeros((used_2D_pos.shape[0], 16, 2))

        if used_2D_pos.shape[1] == 16 * 2:
            used_2D_16pos = used_2D_pos.reshape(-1, 16, 2)
        elif used_2D_pos.shape[1] == 32:
            used_2D_16pos = used_2D_pos[H36M_32_To_16_Table].reshape(-1, 16, 2)
        else:
            used_2D_16pos = used_2D_pos

        show_column = 0
        for show_num in range(0, showNum, 1):
            show_vals = copy.deepcopy(used_2D_16pos[show_num])  #

            ax1 = plt.subplot(gs1[show_column])  #
            ax1.set_title('Pos_2d')  #
            ax1.set_aspect('equal')
            ax1.set_xlim([-1, 1])
            ax1.set_ylim([-1, 1])

            for i in np.arange(len(I)):
                x, y = [np.array([show_vals[I[i], j], show_vals[J[i], j]]) for j in range(2)]
                ax1.plot(x, -y, lw=2, c=lcolor if LR[i] else rcolor)

            show_column += 1

        fig_path = args.checkpoint + r'/tmp/' + name + str(iter) + '.jpg'
        plt.savefig(fig_path)

        plt.close()


def my_draw_loss_picture(loss_list, color, name, iter, args, scale_num):

    if args.record_all_picture == True:
        import matplotlib as mpl
        mpl.use('Agg')
        import matplotlib.pyplot as plt
        plt.switch_backend('agg')

        if iter == 0:
            return

        plt.figure()
        epoch_x = np.arange(0, len(loss_list))

        plt.plot(epoch_x, loss_list[0:], color=color)
        plt.ylabel('Loss')
        plt.xlabel('iter * ' + str(scale_num))
        plt.xlim((0, iter))

        fig_path = args.checkpoint + r'/tmp/' + name + '.jpg'
        plt.savefig(fig_path)

        plt.close()


def my_draw_DOF_angle_distribute(angle_data, name, args):
    if args.record_all_picture == True:
        import matplotlib as mpl
        mpl.use('Agg')
        import matplotlib.pyplot as plt
        plt.switch_backend('agg')

        data_num = angle_data.shape[0]
        DOF_num = angle_data.shape[1]

        angle_data = angle_data.view(data_num, DOF_num)

        each_DOF_distribution = torch.zeros((DOF_num, 361))  #
        angle_data = angle_data.transpose(1, 0)

        for DOF_index in range(DOF_num):  #
            for data_index in range(data_num):
                each_DOF_distribution[DOF_index][int(angle_data[DOF_index][data_index]) + 180] += 1

        each_DOF_maxAngle_num, _ = torch.max(each_DOF_distribution, dim=-1)
        each_DOF_maxAngle_num = each_DOF_maxAngle_num.view(-1, 1)

        each_DOF_distribution = each_DOF_distribution / each_DOF_maxAngle_num

        draw_each_DOF_distribution_mat = torch.zeros((DOF_num * 10, 361))
        for i in range(DOF_num):
            draw_each_DOF_distribution_mat[i * 10: (i + 1) * 10] = each_DOF_distribution[i]

        fig = plt.figure()
        ax = fig.add_subplot(111)

        ax.matshow(draw_each_DOF_distribution_mat, cmap=plt.cm.Blues)
        plt.title('DOF angle heatmap', fontsize=18)
        ax.set_xlabel('angle')
        ax.set_ylabel('DOF ID')

        x_major_locator = MultipleLocator(60)
        y_major_locator = MultipleLocator(10)
        ax.xaxis.set_major_locator(x_major_locator)
        ax.yaxis.set_major_locator(y_major_locator)
        ax.set_xticks([0, 60, 120, 180, 240, 300, 360])
        ax.set_xticklabels([-180, -120, -60, 0, 60, 120, 180])
        show_temp = []
        show_temp_x10 = []
        for i in range(DOF_num):
            show_temp.append(i)
            show_temp_x10.append(i*10)
        ax.set_yticks(show_temp_x10)
        ax.set_yticklabels(show_temp, fontsize=8)
        fig_path = args.checkpoint + r'/tmp/' + name + '.jpg'
        plt.savefig(fig_path)
        plt.close()


def angle_to_radian(angle):
    return (angle / 180) * np.pi


def norm_image(image):
    image = image.copy()
    image -= np.min(image)
    image /= np.max(image)
    image *= 255.
    return np.uint8(image)


def gen_heatmap(mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    fam = heatmap
    return norm_image(fam), (heatmap * 255).astype(np.uint8)


def my_draw_distribute_for_paper(all_angle_data, name, args):
    if args.record_all_picture == True:
        import matplotlib as mpl
        mpl.use('Agg')
        import matplotlib.pyplot as plt
        plt.switch_backend('agg')

        data_num = all_angle_data.shape[0]
        DOF_num = 2

        angle_data = torch.zeros((data_num, 2))
        angle_data[:, 0] = all_angle_data[:, 8]
        angle_data[:, 1] = all_angle_data[:, 3]

        angle_data = angle_data.view(data_num, DOF_num)

        DOF_distribution_img = torch.zeros((361, 361))

        for data_index in range(data_num):
            left_joint_index = 0
            right_joint_index = 1
            DOF_distribution_img[int(angle_data[data_index][left_joint_index]) + 180][
                int(angle_data[data_index][right_joint_index]) + 180] += 1

        each_DOF_maxAngle_num, _ = torch.max(DOF_distribution_img.view(-1), dim=-1)

        _, heatmap = gen_heatmap(DOF_distribution_img)

        heatmap = cv2.flip(heatmap, 0)

        fig_path = args.checkpoint + r'/tmp/' + name + '.jpg'
        cv2.imwrite(fig_path, heatmap)


def my_draw_original_dataset_distribute_for_paper(all_angle_data, name, args):
    if args.record_all_picture == True:
        import matplotlib as mpl
        mpl.use('Agg')
        import matplotlib.pyplot as plt
        plt.switch_backend('agg')

        data_num = all_angle_data.shape[0]
        DOF_num = 2

        angle_data = torch.zeros((data_num, 2))
        angle_data[:, 0] = all_angle_data[:, 0]    #
        angle_data[:, 1] = all_angle_data[:, 1]    #

        angle_data = angle_data.view(data_num, DOF_num)

        DOF_distribution_img = torch.zeros((361, 361))

        for data_index in range(data_num):
            left_joint_index = 0
            right_joint_index = 1
            DOF_distribution_img[int(angle_data[data_index][left_joint_index]) + 180][
                int(angle_data[data_index][right_joint_index]) + 180] += 1

        each_DOF_maxAngle_num, _ = torch.max(DOF_distribution_img.view(-1), dim=-1)

        _, heatmap = gen_heatmap(DOF_distribution_img)

        heatmap = cv2.flip(heatmap, 0)

        fig_path = args.checkpoint + r'/tmp/' + name + '.jpg'
        cv2.imwrite(fig_path, heatmap)



# ==================Definition Start======================
class myResNet(nn.Module):
    def __init__(self, DIM):
        super(myResNet, self).__init__()

        self.fc1 = nn.Linear(DIM, DIM)
        self.fc2 = nn.Linear(DIM, DIM)

        self.relu = nn.ReLU(True)


    def forward(self, input):
        output = self.fc1(input)
        output = self.relu(output)

        output = self.fc2(output)

        output += input    #

        output = self.relu(output)

        return output


def Fk_get_boneVecByPose3d(x, num_joints=16):

    Ct = torch.Tensor([
        [0, 0, 0, 0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 5 6      左小腿
        [0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 2 3      右小腿
        [0, 0, 0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 4 5      左大腿
        [0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 1 2      右大腿
        [-1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 0 4      左髋
        [-1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 0 1      右髋
        [-1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # 0 7      腰部
        [0, 0, 0, 0, 0, 0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0],  # 7 8      胸部
        [0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 1, 0, 0, 0, 0, 0],  # 8 10     左肩
        [0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1, 0, 0],  # 8 13     右肩
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 1, 0, 0, 0, 0],  # 10 11    左大臂
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 1, 0],  # 13 14    右大臂
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 1, 0, 0, 0],  # 11 12    左小臂
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 1],  # 14 15    右小臂
        [0, 0, 0, 0, 0, 0, 0, 0, -1, 1, 0, 0, 0, 0, 0, 0],  # 8 9      头部
    ]).transpose(1, 0)

    Ct = Ct.to(x.device)
    C = Ct.repeat([x.size(0), 1, 1]).view(-1, num_joints, num_joints - 1)
    pose3 = x.permute(0, 2, 1).contiguous()
    B = torch.matmul(pose3, C)
    B = B.permute(0, 2, 1)

    return B



def my_visual_GAN_video(args, used_3D_pos, used_2D_pos, name, showNum, now_epoch):

    if args.record_all_picture == True:
        import matplotlib as mpl
        mpl.use('Agg')
        import matplotlib.pyplot as plt
        plt.switch_backend('agg')

        from common.h36m_dataset import H36M_32_To_16_Table

        I = np.array([0, 1, 2, 0, 4, 5, 0, 7, 8, 8, 10, 11, 8, 13, 14])  # start points
        J = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])  # end points
        LR = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0], dtype=bool)

        lcolor = "red"
        rcolor = "blue"

        figure = plt.figure(figsize=(19.2, 10.8))
        gs1 = gridspec.GridSpec(1, 2)  #
        plt.subplots_adjust(left=None, bottom=None, right=None,  #
                            top=None, wspace=1, hspace=0.5)

        used_3D_16pos = copy.deepcopy(used_3D_pos[0])
        used_2D_16pos = copy.deepcopy(used_2D_pos[0])

        for show_num in range(0, showNum, 1):
            plt.clf()  #

            show_vals = copy.deepcopy(used_3D_16pos[show_num])

            ax1 = plt.subplot(gs1[0], projection='3d')  #
            ax1.set_title('Pos_3d')

            radius = 1.7
            ax1.set_xlim3d([-radius / 2 + show_vals[0, 0], radius / 2 + show_vals[0, 0]])
            ax1.set_ylim3d([-radius / 2 + show_vals[0, 1], radius / 2 + show_vals[0, 1]])
            ax1.set_zlim3d([-radius / 2 + show_vals[0, 2], radius / 2 + show_vals[0, 2]])

            for i in np.arange(len(I)):
                x, y, z = [np.array([show_vals[I[i], j], show_vals[J[i], j]]) for j in range(3)]
                ax1.plot(x, y, z, lw=2, c=lcolor if LR[i] else rcolor)  #

            ################ 2D
            show_vals = copy.deepcopy(used_2D_16pos[show_num])  #

            ax1 = plt.subplot(gs1[1])  #
            ax1.set_title('Pos_2d')  #
            ax1.set_aspect('equal')
            ax1.set_xlim([-1, 1])
            ax1.set_ylim([-1, 1])

            for i in np.arange(len(I)):
                x, y = [np.array([show_vals[I[i], j], show_vals[J[i], j]]) for j in range(2)]
                ax1.plot(x, -y, lw=2, c=lcolor if LR[i] else rcolor)

            fig_path = args.checkpoint + r'/tmp/video_' + str(now_epoch) + '/' + name + str(show_num) + '.jpg'
            plt.savefig(fig_path)

        plt.close()


def my_visual_3D_pos_for_parer(args, used_3D_pos, iter, name, showNum=15):

    if args.record_all_picture == True:
        import matplotlib as mpl
        mpl.use('Agg')
        import matplotlib.pyplot as plt
        plt.switch_backend('agg')

        from common.h36m_dataset import H36M_32_To_16_Table

        I = np.array([0, 1, 2, 0, 4, 5, 0, 7, 8, 8, 10, 11, 8, 13, 14])  # start points
        J = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])  # end points
        LR = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0], dtype=bool)
        white = (1.0, 1.0, 1.0, 0.0)

        lcolor = "red"
        rcolor = "blue"

        figure = plt.figure(figsize=(19.2, 10.8))
        if showNum==15:
            gs1 = gridspec.GridSpec(1,3)  #
        else:
            gs1 = gridspec.GridSpec(4, 4)  #
        plt.subplots_adjust(left=None, bottom=None, right=None,  #
                            top=None, wspace=1, hspace=0.5)

        used_3D_16pos = np.zeros((used_3D_pos.shape[0], 16, 3))

        if used_3D_pos.shape[1] == 16 * 3:
            used_3D_16pos = used_3D_pos.reshape(-1, 16, 3)
        elif used_3D_pos.shape[1] == 32:
            used_3D_16pos = used_3D_pos[H36M_32_To_16_Table].reshape(-1, 16, 3)
        else:
            used_3D_16pos = used_3D_pos

        show_column = 0
        for show_num in range(50, 53, 1):

            show_vals = copy.deepcopy(used_3D_16pos[show_num])

            ax1 = plt.subplot(gs1[show_column], projection='3d')

            radius = 1.7
            ax1.set_xlim3d([-radius / 2 + show_vals[0, 0], radius / 2 + show_vals[0, 0]])
            ax1.set_ylim3d([-radius / 2 + show_vals[0, 1], radius / 2 + show_vals[0, 1]])
            ax1.set_zlim3d([-radius / 2 + show_vals[0, 2], radius / 2 + show_vals[0, 2]])

            ax1.w_xaxis.set_pane_color(white)
            ax1.w_yaxis.set_pane_color(white)

            ax1.w_xaxis.line.set_color(white)
            ax1.w_yaxis.line.set_color(white)
            ax1.w_zaxis.line.set_color(white)

            ax1.set_xticks([])
            ax1.set_yticks([])
            ax1.set_zticks([])
            ax1.get_xaxis().set_ticklabels([])
            ax1.get_yaxis().set_ticklabels([])
            ax1.set_zticklabels([])

            for i in np.arange(len(I)):
                x, y, z = [np.array([show_vals[I[i], j], show_vals[J[i], j]]) for j in range(3)]
                ax1.plot(x, y, z, lw=2, c=lcolor if LR[i] else rcolor)

            show_column += 1

        fig_path = args.checkpoint + r'/tmp/' + name + str(iter) + '.jpg'
        plt.savefig(fig_path)
        plt.close()




def my_visual_2D_pos_for_paper(args, used_2D_pos, iter, name, showNum=15):
    if args.record_all_picture == True:
        import matplotlib as mpl
        mpl.use('Agg')
        import matplotlib.pyplot as plt
        plt.switch_backend('agg')

        from common.h36m_dataset import H36M_32_To_16_Table

        I = np.array([0, 1, 2, 0, 4, 5, 0, 7, 8, 8, 10, 11, 8, 13, 14])  # start points
        J = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])  # end points
        LR = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0], dtype=bool)
        white = (1.0, 1.0, 1.0, 0.0)

        lcolor = "red"
        rcolor = "blue"

        figure = plt.figure(figsize=(19.2, 10.8))
        gs1 = gridspec.GridSpec(1, 1)  #
        plt.subplots_adjust(left=None, bottom=None, right=None,  #
                            top=None, wspace=1, hspace=0.5)

        used_2D_16pos = np.zeros((used_2D_pos.shape[0], 16, 2))

        if used_2D_pos.shape[1] == 16 * 2:
            used_2D_16pos = used_2D_pos.reshape(-1, 16, 2)
        elif used_2D_pos.shape[1] == 32:
            used_2D_16pos = used_2D_pos[H36M_32_To_16_Table].reshape(-1, 16, 2)
        else:     #
            used_2D_16pos = used_2D_pos

        show_column = 0
        for show_num in range(60, 61, 1):
            show_vals = copy.deepcopy(used_2D_16pos[show_num][13])

            ax1 = plt.subplot(gs1[show_column])  #
            ax1.set_title('Pos_2d')  #
            ax1.set_aspect('equal')
            ax1.set_xlim([-1, 1])
            ax1.set_ylim([-1, 1])

            ax1.set_xticks([])
            ax1.set_yticks([])

            ax1.get_xaxis().set_ticklabels([])
            ax1.get_yaxis().set_ticklabels([])

            for i in np.arange(len(I)):
                x, y = [np.array([show_vals[I[i], j], show_vals[J[i], j]]) for j in range(2)]
                ax1.plot(x, -y, lw=2, c=lcolor if LR[i] else rcolor)

            show_column += 1

        fig_path = args.checkpoint + r'/tmp/' + name + str(iter) + '.jpg'
        plt.savefig(fig_path)

        plt.close()    #
