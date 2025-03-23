import torch 
import math
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import BaseModule, auto_fp16

def normalize_bbox(bboxes):
    cx = bboxes[..., 0:1]
    cy = bboxes[..., 1:2]
    cz = bboxes[..., 2:3]
    w = bboxes[..., 3:4].log()
    l = bboxes[..., 4:5].log()
    h = bboxes[..., 5:6].log()
    rot = bboxes[..., 6:7]

    if bboxes.size(-1) > 7:
        vx = bboxes[..., 7:8]
        vy = bboxes[..., 8:9]
        out = torch.cat([cx, cy, w, l, cz, h, rot.sin(), rot.cos(), vx, vy], dim=-1)
    else:
        out = torch.cat([cx, cy, w, l, cz, h, rot.sin(), rot.cos()], dim=-1)

    return out


def denormalize_bbox(normalized_bboxes):
    rot_sin = normalized_bboxes[..., 6:7]
    rot_cos = normalized_bboxes[..., 7:8]
    rot = torch.atan2(rot_sin, rot_cos)

    cx = normalized_bboxes[..., 0:1]
    cy = normalized_bboxes[..., 1:2]
    cz = normalized_bboxes[..., 4:5]

    w = normalized_bboxes[..., 2:3].exp()
    l = normalized_bboxes[..., 3:4].exp()
    h = normalized_bboxes[..., 5:6].exp()

    if normalized_bboxes.size(-1) > 8:
        vx = normalized_bboxes[..., 8:9]
        vy = normalized_bboxes[..., 9:10]
        out = torch.cat([cx, cy, cz, w, l, h, rot, vx, vy], dim=-1)
    else:
        out = torch.cat([cx, cy, cz, w, l, h, rot], dim=-1)

    return out


def encode_bbox(bboxes, pc_range=None):
    xyz = bboxes[..., 0:3].clone()
    wlh = bboxes[..., 3:6].log()
    rot = bboxes[..., 6:7]

    if pc_range is not None:
        xyz[..., 0] = (xyz[..., 0] - pc_range[0]) / (pc_range[3] - pc_range[0])
        xyz[..., 1] = (xyz[..., 1] - pc_range[1]) / (pc_range[4] - pc_range[1])
        xyz[..., 2] = (xyz[..., 2] - pc_range[2]) / (pc_range[5] - pc_range[2])

    if bboxes.shape[-1] > 7:
        vel = bboxes[..., 7:9].clone()
        return torch.cat([xyz, wlh, rot.sin(), rot.cos(), vel], dim=-1)
    else:
        return torch.cat([xyz, wlh, rot.sin(), rot.cos()], dim=-1)


def decode_bbox(bboxes, pc_range=None):
    xyz = bboxes[..., 0:3].clone()
    wlh = bboxes[..., 3:6].exp()
    rot = torch.atan2(bboxes[..., 6:7], bboxes[..., 7:8])

    if pc_range is not None:
        xyz[..., 0] = xyz[..., 0] * (pc_range[3] - pc_range[0]) + pc_range[0]
        xyz[..., 1] = xyz[..., 1] * (pc_range[4] - pc_range[1]) + pc_range[1]
        xyz[..., 2] = xyz[..., 2] * (pc_range[5] - pc_range[2]) + pc_range[2]

    if bboxes.shape[-1] > 8:
        vel = bboxes[..., 8:10].clone()
        return torch.cat([xyz, wlh, rot, vel], dim=-1)
    else:
        return torch.cat([xyz, wlh, rot], dim=-1)

def theta_d2xy_coods(theta_d_coords, map_size=102.4, r=65.0):
    B, Q = theta_d_coords.shape[:2]
    center = map_size / 2
    xy_coords = theta_d_coords[..., :2].clone()
    xy_coords[..., 0:1] = (center + theta_d_coords[..., 1:2]*r * torch.cos(theta_d_coords[..., 0:1]*(2 * torch.pi))) / map_size
    xy_coords[..., 1:2] = (center + theta_d_coords[..., 1:2]*r * torch.sin(theta_d_coords[..., 0:1]*(2 * torch.pi))) / map_size
    xy_coords = torch.clamp(xy_coords, min=0, max=1)

    return torch.cat([xy_coords, theta_d_coords[..., 2:]], dim=-1)


def xy2theta_d_coods(xy_coords_norm, map_size=102.4, r=65.0, norm=True):
    xy_coords = xy_coords_norm.clone()
    center = map_size / 2
    if norm:
        distances = torch.sqrt((xy_coords[..., 0:1]*map_size - center) ** 2 + (xy_coords[..., 1:2]*map_size - center) ** 2) / r
        theta = torch.atan2(xy_coords[..., 1:2]*map_size - center, xy_coords[..., 0:1]*map_size - center)
        theta = ((theta + 2 * torch.pi) % (2 * torch.pi)) / (2 * torch.pi)
        theta_d_coods = torch.cat((theta, distances), dim=-1)
    else:
        distances = torch.sqrt(xy_coords[..., 0:1] ** 2 + xy_coords[..., 1:2] ** 2)
        theta = torch.atan2(xy_coords[..., 1:2], xy_coords[..., 0:1])
        theta = ((theta + 2 * torch.pi) % (2 * torch.pi))
        theta_d_coods = torch.cat((theta, distances), dim=-1)        
    return torch.cat([theta_d_coods, xy_coords[..., 2:]], dim=-1)

