from __future__ import print_function, absolute_import, division

import time

from torch.utils.data import DataLoader

from common.camera import project_to_2d
from common.data_loader import PoseDataSet, PoseTarget

from progress.bar import Bar
from utils.utils import AverageMeter

import numpy as np
from utils.gan_utils import get_bone_lengthbypose3d, get_bone_unit_vecbypose3d, \
    get_pose3dbyBoneVec, blaugment9to15
import torch

def random_bl_aug(x):
    '''
    :param x: nx16x3
    :return: nx16x3
    '''

    bl_15segs_templates_mdifyed = np.load('./data_extra/bone_length_npy/hm36s15678_bl_templates.npy')

    root = x[:, :1, :] * 1.0
    x = x - x[:, :1, :]

    # extract length, unit bone vec
    bones_unit = get_bone_unit_vecbypose3d(x)

    # prepare a bone length list for augmentation.
    tmp_idx = np.random.choice(bl_15segs_templates_mdifyed.shape[0], x.shape[0])
    bones_length = torch.from_numpy(bl_15segs_templates_mdifyed[tmp_idx].astype('float32')).unsqueeze(2)

    modifyed_bone = bones_unit * bones_length.to(x.device)

    # convert bone vec back to pose3d
    out = get_pose3dbyBoneVec(modifyed_bone)

    return out + root  # return the pose with position information.

def dataloader_update(args, data_dict, device):
    """
    this function load the train loader and do swap bone length augment for train loader, target 3D loader,
     and target2D from hm3.6, for more stable GAN training.
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()

    end = time.time()

    buffer_poses_train = []
    buffer_poses_train_2d = []
    buffer_actions_train = []
    buffer_cams_train = []
    bar = Bar('Update training loader', max=len(data_dict['train_gt2d3d_loader']))

    for i, (targets_3d, _, action, cam_param) in enumerate(data_dict['train_gt2d3d_loader']):

        data_time.update(time.time() - end)
        num_poses = targets_3d.size(0)

        targets_3d, cam_param = targets_3d.to(device), cam_param.to(device)

        # do bone length random swap argumentation.
        targets_3d = random_bl_aug(targets_3d)

        inputs_2d = project_to_2d(targets_3d, cam_param)

        ################
        buffer_poses_train.append(targets_3d.detach().cpu().numpy())
        buffer_poses_train_2d.append(inputs_2d.detach().cpu().numpy())
        buffer_actions_train.append(action)
        buffer_cams_train.append(cam_param.detach().cpu().numpy())

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        bar.suffix = '({batch}/{size}) Data: {data:.6f}s | Batch: {bt:.3f}s | Total: {ttl:} | ETA: {eta:} ' \
            .format(batch=i + 1, size=len(data_dict['train_gt2d3d_loader']), data=data_time.avg, bt=batch_time.avg,
                    ttl=bar.elapsed_td, eta=bar.eta_td)
        bar.next()
    bar.finish()

    assert len(buffer_poses_train) == len(buffer_poses_train_2d)
    assert len(buffer_poses_train) == len(buffer_actions_train)
    assert len(buffer_poses_train) == len(buffer_cams_train)

    print('==> Random Bone Length (S15678) swap completed')

    data_dict['train_gt2d3d_loader'] = DataLoader(PoseDataSet(buffer_poses_train, buffer_poses_train_2d,
                                                              buffer_actions_train, buffer_cams_train),
                                                  batch_size=args.batch_size,
                                                  shuffle=True, num_workers=args.num_workers, pin_memory=True)

    # 3D GAN
    data_dict['target_3d_loader'] = DataLoader(PoseTarget(buffer_poses_train),
                                               batch_size=args.batch_size,
                                               shuffle=True, num_workers=args.num_workers, pin_memory=True)

    # 2D GAN
    data_dict['target_2d_loader'] = DataLoader(PoseTarget(buffer_poses_train_2d),
                                               batch_size=args.batch_size,
                                               shuffle=True, num_workers=args.num_workers, pin_memory=True)
    return
