# Some parts of this code are based on implementation from:
# https://github.com/graphdeco-inria/gaussian-splatting/tree/main (Accessed June 16th 2025)
#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

# Some parts of this code are based on shape-of-motion:
# https://github.com/vye16/shape-of-motion (Accessed Jun 18th 2025)
#
# The original license applies.

import os
from PIL import Image
import torch
import numpy as np

import torch.nn.functional as F
from torch.autograd import Variable
from math import exp

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def readGT(rgb_dir, intrinsic_dir, pose_dir, device):
    scene_dict = {}
    # Read intrinsics
    intrinsics = []
    with open(intrinsic_dir, 'r') as f:
        for line in f:
            intrinsics.append([float(x) for x in line.split()])
    intrinsics = np.array(intrinsics)
    scene_dict["intrinsics"] = torch.from_numpy(intrinsics).float().to(device)
    # print("Intrinsics are read...")

    # Read poses
    scannet_to_py3d_T = torch.zeros((4, 4))
    scannet_to_py3d_T[0, 1] = 1
    scannet_to_py3d_T[1, 2] = 1
    scannet_to_py3d_T[2, 0] = 1
    scannet_to_py3d_T[3, 3] = 1

    pose_files = os.listdir(pose_dir)
    poses = []
    for pf in pose_files:
        pose_path = os.path.join(pose_dir, pf)
        pose = []
        with open(pose_path, 'r') as f:
            for line in f:
                pose.append([float(x) for x in line.split()])
            pose = torch.from_numpy(np.array(pose)).float()
            pose = torch.matmul(scannet_to_py3d_T, pose)
        poses.append(pose)
    poses = torch.cat(poses, dim=0)
    scene_dict["pose"] = poses.to(device)
    # print("Poses are read...")

    # Read images
    images = []
    for fname in os.listdir(rgb_dir):
        image = np.array(Image.open(os.path.join(rgb_dir, fname)))
        images.append(torch.from_numpy(image))
    images = torch.cat(images, dim=0)
    scene_dict["images"] = images.to(device)

    return scene_dict