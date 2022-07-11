from __future__ import print_function, absolute_import, division

import glob

import torch

from models_baseline.gcn.graph_utils import adj_mx_from_skeleton
from models_baseline.gcn.sem_gcn import SemGCN
from models_baseline.mlp.linear_model import LinearModel, init_weights
from models_baseline.videopose.model_VideoPose3D import TemporalModelOptimized1f

from models_Fk_GAN.mulit_farme_videopose import multiFrame_TemporalModelOptimized1f, multiFrame_TemporalModel
from models_baseline.poseformer.model_poseformer import PoseTransformer

from models_Fk_GAN.video_mode_operate import video_receptive_field


def model_pos_preparation(args, dataset, device, flag='train'):
    """
    return a posenet Model: with Bx16x2 --> posenet --> Bx16x3
    #####
    """

    num_joints = dataset.skeleton().num_joints()   # num_joints = 16 fix
    print('create model: {}'.format(args.posenet_name))

    if args.posenet_name == 'gcn':
        adj = adj_mx_from_skeleton(dataset.skeleton())
        model_pos = SemGCN(adj, 128, num_layers=args.stages, p_dropout=args.dropout, nodes_group=None).to(device)

    elif args.posenet_name == 'mlp':
        model_pos = LinearModel(num_joints * 2, (num_joints - 1) * 3, num_stage=args.stages, p_dropout=args.dropout)

    elif args.posenet_name == 'videopose':
        filter_widths = [1]
        for stage_id in range(args.stages):
            filter_widths.append(1)

        model_pos = TemporalModelOptimized1f(16, 2, 15, filter_widths=filter_widths, causal=False,
                                             dropout=0.25, channels=1024)

    elif args.posenet_name == 'mulit_farme_videopose':

        filter_widths = [int(x) for x in args.architecture.split(',')]
        if flag == 'train':
            model_pos = multiFrame_TemporalModelOptimized1f(16, 2, 16, filter_widths=filter_widths, causal=False,
                                             dropout=0.25, channels=1024)
        elif flag == 'test':
            model_pos = multiFrame_TemporalModel(16, 2, 16, filter_widths=filter_widths, causal=False,
                                                            dropout=0.25, channels=1024)

    elif args.posenet_name == 'mulit_farme_poseformer':    # ICCV2021
        filter_widths = [int(x) for x in args.architecture.split(',')]  #
        receptive_field = video_receptive_field(filter_widths)
        if flag == 'train':
            model_pos = PoseTransformer(num_frame=receptive_field, num_joints=num_joints, in_chans=2,
                                              embed_dim_ratio=32, depth=4,
                                              num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                                              drop_path_rate=0.1)

        elif flag == 'test':
            model_pos = PoseTransformer(num_frame=receptive_field, num_joints=num_joints, in_chans=2,
                                        embed_dim_ratio=32, depth=4,
                                        num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None, drop_path_rate=0)
    else:
        assert False, 'posenet_name invalid'

    model_pos = model_pos.to(device)
    print("==> Total parameters for model {}: {:.2f}M"
          .format(args.posenet_name, sum(p.numel() for p in model_pos.parameters()) / 1000000.0))

    if args.pretrain:
        tmp_path = r''

        posenet_pretrain_path = glob.glob(tmp_path)
        print(posenet_pretrain_path)
        assert len(posenet_pretrain_path) == 1, 'suppose only 1 pretrain path for each model setting, ' \
                                                'please delete the redundant file'
        tmp_ckpt = torch.load(posenet_pretrain_path[0])
        print(tmp_ckpt.keys())
        #model_pos.load_state_dict(tmp_ckpt['state_dict'])
        model_pos.load_state_dict(tmp_ckpt['model_pos'])
        print('==> Pretrained posenet loaded')
    else:
        model_pos.apply(init_weights)

    return model_pos
