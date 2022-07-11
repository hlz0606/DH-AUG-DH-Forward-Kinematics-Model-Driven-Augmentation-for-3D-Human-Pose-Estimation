import argparse



def get_parse_args():
    parser = argparse.ArgumentParser(description='PyTorch training script')

    # General arguments
    parser.add_argument('--dataset', default='h36m', type=str, metavar='NAME', help='target dataset')
    parser.add_argument('--keypoints', default='gt', type=str, metavar='NAME', help='2D detections to use, \
    gt/hr/cpn_ft_h36m_dbb/detectron_ft_h36m')
    parser.add_argument('--actions', default='*', type=str, metavar='LIST',
                        help='actions to train/test on, separated by comma, or * for all')
    parser.add_argument('--checkpoint', default='checkpoint/debug', type=str, metavar='PATH',
                        help='checkpoint directory')
    parser.add_argument('--snapshot', default=2, type=int, help='save gan rlt for every #snapshot epochs')

    #
    parser.add_argument('--note', default='debug', type=str, help='additional name on checkpoint directory')

    # Evaluate choice
    parser.add_argument('--evaluate', default='', type=str, metavar='FILENAME',
                        help='checkpoint to evaluate (file name)')  # not in used here.
    parser.add_argument('--resume', default='', type=str, metavar='FILENAME',
                        help='checkpoint to resume (file name)')

    # Model arguments    'gcn'    'mlp'    'videopose'(single)  'poseformer'
    # 'mulit_farme_videopose'(video)    'mulit_farme_poseformer’(video)
    parser.add_argument('--posenet_name', default='videopose', type=str, help='posenet: gcn/stgcn/videopose/mlp')
    parser.add_argument('--stages', default=4, type=int, metavar='N', help='stages of baseline model')
    parser.add_argument('--dropout', default=0.25, type=float, help='dropout rate')

    # Training detail
    parser.add_argument('--batch_size', default=1024, type=int, metavar='N',
                        help='batch size in terms of predicted frames')
    parser.add_argument('--epochs', default=50, type=int, metavar='N', help='number of training epochs')
    parser.add_argument('--decay_epoch', default=0, type=int, metavar='N', help='number of decay epochs')

    # Learning rate
    parser.add_argument('--lr_g', default=1.0e-4, type=float, metavar='LR', help='initial learning rate for augmentor/generator')
    parser.add_argument('--lr_d', default=1.0e-4, type=float, metavar='LR', help='initial learning rate for discriminator')
    parser.add_argument('--lr_p', default=1.0e-4, type=float, metavar='LR', help='initial learning rate for posenet')
    parser.add_argument('--no_max', dest='max_norm', action='store_false', help='if use max_norm clip on grad')
    parser.set_defaults(max_norm=True)

    # Experimental setting
    parser.add_argument('--random_seed', type=int, default=0)  # change this if GAN collapse


    parser.add_argument('--downsample', default=1, type=int, metavar='FACTOR', help='downsample frame rate by factor')
    parser.add_argument('--pretrain', default=False, type=lambda x: (str(x).lower() == 'true'), help='pretrain model')
    parser.add_argument('--s1only', default=False, type=lambda x: (str(x).lower() == 'true'), help='train S1 only')
    parser.add_argument('--s1s5only', default=False, type=lambda x: (str(x).lower() == 'true'), help='train S1 S5 only')
    parser.add_argument('--num_workers', default=0, type=int, metavar='N', help='num of workers for data loading')

    # Training PoseAug detail
    parser.add_argument('--warmup', default=2, type=int, help='train gan only at the beginning')
    parser.add_argument('--df', default=2, type=int, help='update discriminator frequency')


    #######  normal    GAN    NO_enhance
    parser.add_argument('--data_enhancement_method', default='GAN', type=str, metavar='NAME', help='')

    ########
    parser.add_argument('--generator_whole_number', default=10000, type=int, metavar='NAME', help='')

    #######
    parser.add_argument('--generator_choose_BoneLen', default=True,
                        type=lambda x: (str(x).lower() == 'true'),
                        metavar='NAME', help='')

    ######## 0%-20%
    # 'different'：
    # 'same'：
    parser.add_argument('--bone_len_scaler', default='different', type=str, metavar='NAME', help='')

    parser.add_argument('--generator_choose_root_pos', default=True,
                        type=lambda x: (str(x).lower() == 'true'),
                        metavar='NAME', help='')

    parser.add_argument('--generator_global_rot', default=True,
                        type=lambda x: (str(x).lower() == 'true'),
                        metavar='NAME', help='')

    parser.add_argument('--GAN_OUTPUT_DIM', default=32+3, type=int, metavar='NAME', help='')

    parser.add_argument('--GAN_LAMBDA', default=10, type=int, metavar='NAME', help='')

    parser.add_argument('--GAN_whether_use_preAngle', default=True,
                        type=lambda x: (str(x).lower() == 'true'),
                        metavar='NAME', help='')

    parser.add_argument('--motion_Dis_whether_use_3dPos_branch', default=True,
                        type=lambda x: (str(x).lower() == 'true'),
                        metavar='NAME', help='')

    parser.add_argument('--motion_Dis_whether_use_3dDiff_branch', default=True,
                        type=lambda x: (str(x).lower() == 'true'),
                        metavar='NAME', help='')

    parser.add_argument('--Dis_DenseDim_3D', default=1000, type=int, metavar='NAME', help='')

    parser.add_argument('--Dis_DenseDim_2D', default=1000, type=int, metavar='NAME', help='')

    parser.add_argument('--Gen_DenseDim', default=1000, type=int, metavar='NAME', help='')

    parser.add_argument('--video_Dis_DenseDim_3D', default=1000, type=int, metavar='NAME', help='')

    parser.add_argument('--video_Dis_DenseDim_2D', default=1000, type=int, metavar='NAME', help='')

    parser.add_argument('--GAN_3d_loss_weight', default=1, type=float, metavar='NAME', help='')

    parser.add_argument('--GAN_2d_loss_weight', default=0.2, type=float, metavar='NAME', help='')

    parser.add_argument('--GAN_3d_motion_loss_weight', default=1, type=float, metavar='NAME', help='')

    parser.add_argument('--GAN_2d_motion_loss_weight', default=1, type=float, metavar='NAME', help='')

    parser.add_argument('--GAN_whether_rand_root', default=True,
                        type=lambda x: (str(x).lower() == 'true'),
                        metavar='NAME', help='')

    parser.add_argument('--set_demo_mode', default=False,
                        type=lambda x: (str(x).lower() == 'true'),
                        metavar='NAME', help='')

    parser.add_argument('--GAN_checkpoint',
                        default='/media/hlz/3c948a72-4c5c-40ee-a460-ad3d9f94922f/checkpoint',
                        type=str, metavar='NAME', help='')

    parser.add_argument('--GAN_resume', default='', type=str, metavar='FILENAME',
                        help='GAN_checkpoint to resume (file name)')

    parser.add_argument('--record_all_picture', default=True,
                        type=lambda x: (str(x).lower() == 'true'),
                        metavar='FILENAME',
                        help='GAN_checkpoint to resume (file name)')

    parser.add_argument('--additional_train_epoch', default=60, type=int, metavar='NAME', help='')
    parser.add_argument('--additional_LR_decay', default=0.95, type=float, metavar='NAME', help='')

    parser.add_argument('--single_dis_warmup_epoch', default=4, type=int, metavar='NAME', help='')

    parser.add_argument('--video_over_200mm', default=False,
                        type=lambda x: (str(x).lower() == 'true'),
                        metavar='FILENAME',
                        help='GAN_checkpoint to resume (file name)')

    parser.add_argument('--whether_use_RT', default=True,
                        type=lambda x: (str(x).lower() == 'true'),
                        metavar='FILENAME',
                        help='GAN_checkpoint to resume (file name)')

    parser.add_argument('--flip_pos_model_input', default=True,
                        type=lambda x: (str(x).lower() == 'true'),
                        metavar='FILENAME',
                        help='GAN_checkpoint to resume (file name)')

    parser.add_argument('--flip_GAN_model_input', default=True,
                        type=lambda x: (str(x).lower() == 'true'),
                        metavar='FILENAME',
                        help='GAN_checkpoint to resume (file name)')

    parser.add_argument('--Pos_video_playback_input', default=True,
                        type=lambda x: (str(x).lower() == 'true'),
                        metavar='FILENAME',
                        help='GAN_checkpoint to resume (file name)')

    parser.add_argument('--GAN_video_playback_input', default=True,
                        type=lambda x: (str(x).lower() == 'true'),
                        metavar='FILENAME',
                        help='GAN_checkpoint to resume (file name)')

    parser.add_argument('--gpu_id', default='0', type=str, metavar='FILENAME',
                        help='GAN_checkpoint to resume (file name)')

    parser.add_argument('--Path_3DPW', default='/media/hlz/3c948a72-4c5c-40ee-a460-ad3d9f94922f/'
                                               '3DPW_dataSet',
                        type=str, metavar='FILENAME',
                        help='GAN_checkpoint to resume (file name)')

    parser.add_argument('--single_or_multi_train_mode', default='single', type=str, metavar='FILENAME',
                       help='GAN_checkpoint to resume (file name)')

    parser.add_argument('--architecture', default='3,3,3', type=str,
                        metavar='LAYERS', help='filter widths separated by comma')

    args = parser.parse_args()

    # Check invalid configuration
    if args.resume and args.evaluate:
        print('Invalid flags: --resume and --evaluate cannot be set at the same time')
        exit()

    return args

