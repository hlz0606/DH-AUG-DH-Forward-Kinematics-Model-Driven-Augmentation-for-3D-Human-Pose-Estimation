from __future__ import absolute_import, division

import copy

import numpy as np

from common.camera import world_to_camera, normalize_screen_coordinates


########
def create_2d_data(data_path, dataset):
    keypoints = np.load(data_path, allow_pickle=True)
    keypoints = keypoints['positions_2d'].item()

    for subject in keypoints.keys():
        for action in keypoints[subject]:
            for cam_idx, kps in enumerate(keypoints[subject][action]):

                cam = dataset.cameras()[subject][cam_idx]
                kps[..., :2] = normalize_screen_coordinates(kps[..., :2], w=cam['res_w'], h=cam['res_h'])
                keypoints[subject][action][cam_idx] = kps

    return keypoints


def read_3d_data(dataset):
    for subject in dataset.subjects():
        for action in dataset[subject].keys():
            anim = dataset[subject][action]

            positions_3d = []
            for cam in anim['cameras']:

                pos_3d = world_to_camera(anim['positions'], R=cam['orientation'], t=cam['translation'])

                positions_3d.append(pos_3d)
            anim['positions_3d'] = positions_3d

    return dataset


def fetch(subjects, dataset, keypoints, args, train_or_test, action_filter=None, stride=1, parse_3d_poses=True,
          whether_need_cam_external=False):
    out_poses_3d = []
    out_poses_2d = []
    out_actions = []
    out_cam = []

    for subject in subjects:
        for action in keypoints[subject].keys():

            if action_filter is not None:
                found = False
                for a in action_filter:
                    # if action.startswith(a):
                    if action.split(' ')[0] == a:
                        found = True
                        break
                if not found:
                    continue

            poses_2d = keypoints[subject][action]

            for i in range(len(poses_2d)):  # Iterate across cameras
                out_poses_2d.append(poses_2d[i])
                out_actions.append([action.split(' ')[0]] * poses_2d[i].shape[0])

            if parse_3d_poses and 'positions_3d' in dataset[subject][action]:
                poses_3d = dataset[subject][action]['positions_3d']
                assert len(poses_3d) == len(poses_2d), 'Camera count mismatch'

                for i in range(len(poses_3d)):  # Iterate across cameras
                    out_poses_3d.append(poses_3d[i])
                    cam = dataset[subject][action]['cameras'][i]['intrinsic']

                    if whether_need_cam_external == True:
                        cam = list(cam)
                        cam.extend(dataset[subject][action]['cameras'][i]['orientation'])
                        cam.extend(dataset[subject][action]['cameras'][i]['translation'])
                        cam = np.array(cam)

                    if args.single_or_multi_train_mode == 'single':
                        out_cam.append([cam] * poses_3d[i].shape[0])
                    elif args.single_or_multi_train_mode == 'multi':
                        out_cam.append(cam)

    if len(out_poses_3d) == 0:
        out_poses_3d = None

    if stride > 1 and train_or_test == 'train':

        for i in range(len(out_poses_2d)):
            out_poses_2d[i] = out_poses_2d[i][::stride]
            out_actions[i] = out_actions[i][::stride]
            if out_poses_3d is not None:
                out_poses_3d[i] = out_poses_3d[i][::stride]

    if args.video_over_200mm == True and args.single_or_multi_train_mode == 'multi'\
            and train_or_test == 'train':

        adjacent_frame_displacement_less_than_200_num = 0
        out_poses_3d_temp = copy.deepcopy(out_poses_3d)
        for video_itr, video_kp3d in enumerate(out_poses_3d_temp):
            all_gt2ds = []
            all_gt3ds = []

            prev_kp3d = copy.deepcopy(out_poses_3d[video_itr][0])

            for itr, kp3d in enumerate(video_kp3d):

                if itr > 0:
                    if not np.any(np.linalg.norm(prev_kp3d - kp3d, axis=1)*1000 >= 200):
                        adjacent_frame_displacement_less_than_200_num += 1
                        continue

                    all_gt2ds.append(out_poses_2d[video_itr][itr])
                    all_gt3ds.append(out_poses_3d[video_itr][itr])
                else:  #
                    all_gt2ds.append(out_poses_2d[video_itr][itr])
                    all_gt3ds.append(out_poses_3d[video_itr][itr])

                prev_kp3d = kp3d
            out_poses_2d[video_itr] = np.array(all_gt2ds)
            out_poses_3d[video_itr] = np.array(all_gt3ds)

    return out_poses_3d, out_poses_2d, out_actions, out_cam
