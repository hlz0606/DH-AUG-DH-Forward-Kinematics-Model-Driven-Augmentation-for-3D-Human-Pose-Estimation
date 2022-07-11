from __future__ import print_function, absolute_import, division

import time

import numpy as np
import torch
import torch.nn as nn
# add model for generator and discriminator
from torch.autograd import Variable
from torch.utils.data import DataLoader

from common.camera import project_to_2d
from common.data_loader import PoseDataSet
from progress.bar import Bar
from utils.gan_utils import get_discriminator_accuracy
from utils.loss import diff_range_loss, rectifiedL2loss
from utils.utils import AverageMeter, set_grad

def get_adv_loss(model_dis, data_real, data_fake, criterion, summary, writer, writer_name):
    device = torch.device("cuda")

    real_3d = model_dis(data_real)
    fake_3d = model_dis(data_fake)

    real_label_3d = Variable(torch.ones(real_3d.size())).to(device)
    fake_label_3d = Variable(torch.zeros(fake_3d.size())).to(device)

    adv_3d_real_loss = criterion(real_3d, fake_label_3d)
    adv_3d_fake_loss = criterion(fake_3d, real_label_3d)

    adv_3d_loss = (adv_3d_real_loss + adv_3d_fake_loss) * 0.5

    ###################################################
    real_acc = get_discriminator_accuracy(real_3d.reshape(-1), real_label_3d.reshape(-1))
    fake_acc = get_discriminator_accuracy(fake_3d.reshape(-1), fake_label_3d.reshape(-1))
    writer.add_scalar('train_G_iter_PoseAug/{}_real_acc'.format(writer_name), real_acc, summary.train_iter_num)
    writer.add_scalar('train_G_iter_PoseAug/{}_fake_acc'.format(writer_name), fake_acc, summary.train_iter_num)
    writer.add_scalar('train_G_iter_PoseAug/{}_adv_loss'.format(writer_name), adv_3d_loss.item(),
                      summary.train_iter_num)
    return adv_3d_loss


def train_dis(model_dis, data_real, data_fake, criterion, summary, writer, writer_name, fake_data_pool, optimizer):
    device = torch.device("cuda")
    optimizer.zero_grad()

    data_real = data_real.clone().detach().to(device)
    data_fake = data_fake.clone().detach().to(device)

    data_fake = Variable(torch.Tensor(fake_data_pool(data_fake.cpu().detach().data.numpy()))).to(device)

    # predicte the label
    real_pre = model_dis(data_real)
    fake_pre = model_dis(data_fake)

    real_label = Variable(torch.ones(real_pre.size())).to(device)
    fake_label = Variable(torch.zeros(fake_pre.size())).to(device)

    dis_real_loss = criterion(real_pre, real_label)
    dis_fake_loss = criterion(fake_pre, fake_label)

    # Total discriminators losse
    dis_loss = (dis_real_loss + dis_fake_loss) * 0.5

    real_acc = get_discriminator_accuracy(real_pre.reshape(-1), real_label.reshape(-1))
    fake_acc = get_discriminator_accuracy(fake_pre.reshape(-1), fake_label.reshape(-1))

    writer.add_scalar('train_G_iter_PoseAug/{}_real_acc'.format(writer_name), real_acc, summary.train_iter_num)
    writer.add_scalar('train_G_iter_PoseAug/{}_fake_acc'.format(writer_name), fake_acc, summary.train_iter_num)
    writer.add_scalar('train_G_iter_PoseAug/{}_dis_loss'.format(writer_name), dis_loss.item(), summary.train_iter_num)

    dis_loss.backward()
    nn.utils.clip_grad_norm_(model_dis.parameters(), max_norm=1)    # 梯度裁剪
    optimizer.step()

    return real_acc, fake_acc

