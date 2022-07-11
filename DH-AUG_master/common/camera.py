from __future__ import absolute_import, division

import numpy as np
import torch

from utils.utils import wrap
from common.quaternion import qrot, qinverse


def normalize_screen_coordinates(point, w, h):

    point[..., 0] = point[..., 0] / w * 2 - 1
    point[..., 1] = point[..., 1] / w * 2 - h/w

    return point


def image_coordinates(X, w, h):
    assert X.shape[-1] == 2

    # Reverse camera frame normalization
    X[..., 0] = (X[..., 0] + 1) * w / 2
    X[..., 1] = (X[..., 1] + h / w) * w / 2

    return X


def world_to_camera(X, R, t):
    Rt = wrap(qinverse, False, R)  # Invert rotation
    # print('world_to_camera :{0}   {1}   {2}'.format(Rt.shape, X.shape[:-1] + (1,),X.shape))
    # (4,)    (1552, 16, 1)    (1552, 16, 3)

    return wrap(qrot, False, np.tile(Rt, X.shape[:-1] + (1,)), X - t)  # Rotate and translate


def GAN_torch_world_to_camera(X, R, t):
    Rt = qinverse(R)  # Invert rotation
    return qrot(Rt.repeat(X.shape[:-1] + (1,)), X - t)  # Rotate and translate



def camera_to_world(X, R, t):
    return wrap(qrot, False, np.tile(R, X.shape[:-1] + (1,)), X) + t


def video_GAN_torch_camera_to_world(X, R, t):
    R = R.view(-1, 1, 4)
    R = R.repeat(1, 16, 1)
    t = t.view(-1, 1, 3)
    t = t.repeat(1, 16, 1)
    return qrot(R, X) + t

def GAN_torch_camera_to_world_batch(X, R, t):
    R = R.view(-1, 1, 4)
    R = R.repeat(1, 16, 1)
    t = t.view(-1, 1, 3)
    t = t.repeat(1, 16, 1)

    return qrot(R, X) + t


def project_to_2d(X, camera_params):
    """
    Project 3D points to 2D using the Human3.6M camera projection function.
    This is a differentiable and batched reimplementation of the original MATLAB script.

    Arguments:
    X -- 3D points in *camera space* to transform (N, *, 3)
    camera_params -- intrinsic parameteres (N, 2+2+3+2=9)
    """
    assert X.shape[-1] == 3
    assert len(camera_params.shape) == 2
    assert camera_params.shape[-1] == 9 or camera_params.shape[-1] == 16
    assert X.shape[0] == camera_params.shape[0]

    while len(camera_params.shape) < len(X.shape):
        camera_params = camera_params.unsqueeze(1)

    f = camera_params[..., :2]
    c = camera_params[..., 2:4]
    k = camera_params[..., 4:7]
    #p = camera_params[..., 7:]
    p = camera_params[..., 7:9]

    XX = torch.clamp(X[..., :2] / X[..., 2:], min=-1, max=1)
    r2 = torch.sum(XX[..., :2] ** 2, dim=len(XX.shape) - 1, keepdim=True)

    radial = 1 + torch.sum(k * torch.cat((r2, r2 ** 2, r2 ** 3), dim=len(r2.shape) - 1), dim=len(r2.shape) - 1,
                           keepdim=True)
    tan = torch.sum(p * XX, dim=len(XX.shape) - 1, keepdim=True)

    XXX = XX * (radial + tan) + p * r2

    return f * XXX + c


def project_to_2d_linear(X, camera_params):
    """
    Project 3D points to 2D using only linear parameters (focal length and principal point).

    Arguments:
    X -- 3D points in *camera space* to transform (N, *, 3)
    camera_params -- intrinsic parameteres (N, 2+2+3+2=9)
    """
    assert X.shape[-1] == 3
    assert len(camera_params.shape) == 2
    assert camera_params.shape[-1] == 9
    assert X.shape[0] == camera_params.shape[0]

    while len(camera_params.shape) < len(X.shape):
        camera_params = camera_params.unsqueeze(1)

    f = camera_params[..., :2]
    c = camera_params[..., 2:4]

    XX = torch.clamp(X[..., :2] / X[..., 2:], min=-1, max=1)

    return f * XX + c
