from __future__ import print_function, absolute_import, division

import time

import torch
import torch.nn as nn

from progress.bar import Bar
from utils.utils import AverageMeter, set_grad

from models_Fk_GAN.special_operate import my_visual_2D_pos, my_visual_3D_pos

def train_posenet(model_pos, data_loader, optimizer, criterion, device, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    epoch_loss_3d_pos = AverageMeter()
    flip_epoch_loss_3d_pos = AverageMeter()

    torch.set_grad_enabled(True)
    set_grad([model_pos], True)
    model_pos.train()
    end = time.time()

    bar = Bar('Train on real pose det2d', max=len(data_loader))

    for i, (targets_3d, inputs_2d, _, _) in enumerate(data_loader):

        data_time.update(time.time() - end)
        num_poses = targets_3d.size(0)

        if num_poses == 1:
            break

        targets_3d, inputs_2d = targets_3d.to(device), inputs_2d.to(device)
        targets_3d = targets_3d[:, :, :] - targets_3d[:, :1, :]  # the output is relative to the 0 joint

        outputs_3d = model_pos(inputs_2d)

        optimizer.zero_grad()
        loss_3d_pos = criterion(outputs_3d, targets_3d)
        loss_3d_pos.backward()
        nn.utils.clip_grad_norm_(model_pos.parameters(), max_norm=1)
        optimizer.step()

        epoch_loss_3d_pos.update(loss_3d_pos.item(), num_poses)

        if args.flip_pos_model_input:  # flip the 2D pose Left <-> Right    #
            joints_left = [4, 5, 6, 10, 11, 12]
            joints_right = [1, 2, 3, 13, 14, 15]
            out_left = [4, 5, 6, 10, 11, 12]
            out_right = [1, 2, 3, 13, 14, 15]

            inputs_2d_flip = inputs_2d.detach().clone()
            inputs_2d_flip[:, :, 0] *= -1
            inputs_2d_flip[:, joints_left + joints_right, :] = inputs_2d_flip[:, joints_right + joints_left, :]

            outputs_3d_flip = model_pos(inputs_2d_flip.view(num_poses, -1))

            targets_3d_flip = targets_3d.detach().clone()
            targets_3d_flip[:, :, 0] *= -1
            targets_3d_flip[:, out_left + out_right, :] = targets_3d_flip[:, out_right + out_left, :]

            optimizer.zero_grad()
            flip_loss_3d_pos = criterion(outputs_3d_flip, targets_3d_flip)
            flip_loss_3d_pos.backward()
            nn.utils.clip_grad_norm_(model_pos.parameters(), max_norm=1)  #
            optimizer.step()

            flip_epoch_loss_3d_pos.update(loss_3d_pos.item(), num_poses)

            batch_time.update(time.time() - end)
            end = time.time()

            bar.suffix = 'use flip: ({batch}/{size}) Data: {data:.6f}s | Batch: {bt:.3f}s | Total: {ttl:} | ETA: {eta:} ' \
                         '| Loss: {loss: .4f}  | flip_Loss: {flip_Loss: .4f}' \
                .format(batch=i + 1, size=len(data_loader), data=data_time.avg, bt=batch_time.avg,
                        ttl=bar.elapsed_td, eta=bar.eta_td, loss=epoch_loss_3d_pos.avg,
                        flip_Loss=flip_epoch_loss_3d_pos.avg)
            bar.next()

    bar.finish()

    return