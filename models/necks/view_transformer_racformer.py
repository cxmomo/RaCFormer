# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_conv_layer, build_norm_layer
from mmcv.runner import BaseModule, force_fp32

from models.csrc.bev_pool_v2.bev_pool import bev_pool_v2
from mmdet.models.backbones.resnet import BasicBlock
from mmdet.models.builder import NECKS

from .focalloss import FocalLoss
import numpy as np


@NECKS.register_module()
class LSSViewTransformer_racformer(BaseModule):
    r"""Lift-Splat-Shoot view transformer with BEVPoolv2 implementation.

    Please refer to the `paper <https://arxiv.org/abs/2008.05711>`_ and
        `paper <https://arxiv.org/abs/2211.17111>`

    Args:
        grid_config (dict): Config of grid alone each axis in format of
            (lower_bound, upper_bound, interval). axis in {x,y,z,depth}.
        input_size (tuple(int)): Size of input images in format of (height,
            width).
        downsample (int): Down sample factor from the input size to the feature
            size.
        in_channels (int): Channels of input feature.
        out_channels (int): Channels of transformed feature.
        accelerate (bool): Whether the view transformation is conducted with
            acceleration. Note: the intrinsic and extrinsic of cameras should
            be constant when 'accelerate' is set true.
    """

    def __init__(
        self,
        grid_config,
        input_size,
        downsample=16,
        in_channels=512,
        out_channels=64,
        accelerate=False,
        norm_cfg=dict(type='BN'),
        depth_only=False
    ):
        super(LSSViewTransformer_racformer, self).__init__()
        self.grid_config = grid_config
        self.downsample = downsample
        # Model     
        self.bin_size = 2 * (grid_config['depth'][1] - grid_config['depth'][0]) / (grid_config['depth'][2] * (1 + grid_config['depth'][2]))
        bin_indice = torch.linspace(0, grid_config['depth'][2] - 1, int(grid_config['depth'][2]), requires_grad=False)
        self.bin_value = (bin_indice + 0.5).pow(2) * self.bin_size / 2 - self.bin_size / 8 + grid_config['depth'][0]
        
        self.create_grid_infos(**grid_config)
        self.create_frustum(input_size, downsample)
        self.out_channels = out_channels
        self.in_channels = in_channels
        if depth_only:
            self.depth_net = nn.Conv2d(
                in_channels, self.D, kernel_size=1, padding=0)
        else:
            self.depth_net = nn.Conv2d(
                in_channels, self.D + self.out_channels, kernel_size=1, padding=0)
        self.accelerate = accelerate
        self.initial_flag = True

    def create_grid_infos(self, x, y, z, **kwargs):
        """Generate the grid information including the lower bound, interval,
        and size.

        Args:
            x (tuple(float)): Config of grid alone x axis in format of
                (lower_bound, upper_bound, interval).
            y (tuple(float)): Config of grid alone y axis in format of
                (lower_bound, upper_bound, interval).
            z (tuple(float)): Config of grid alone z axis in format of
                (lower_bound, upper_bound, interval).
            **kwargs: Container for other potential parameters
        """
        self.grid_lower_bound = torch.Tensor([cfg[0] for cfg in [x, y, z]])
        self.grid_interval = torch.Tensor([cfg[2] for cfg in [x, y, z]])
        self.grid_size = torch.Tensor([(cfg[1] - cfg[0]) / cfg[2]
                                       for cfg in [x, y, z]])

    def create_frustum(self, input_size, downsample):
        """Generate the frustum template for each image.

        Args:
            depth_cfg (tuple(float)): Config of grid alone depth axis in format
                of (lower_bound, upper_bound, interval).
            input_size (tuple(int)): Size of input images in format of (height,
                width).
            downsample (int): Down sample scale factor from the input size to
                the feature size.
        """
        H_in, W_in = input_size
        H_feat, W_feat = H_in // downsample, W_in // downsample

        d = self.bin_value.view(-1, 1, 1).expand(-1, H_feat, W_feat)
        self.D = d.shape[0]
        x = torch.linspace(0, W_in - 1, W_feat,  dtype=torch.float)\
            .view(1, 1, W_feat).expand(self.D, H_feat, W_feat)
        y = torch.linspace(0, H_in - 1, H_feat,  dtype=torch.float)\
            .view(1, H_feat, 1).expand(self.D, H_feat, W_feat)

        # D x H x W x 3
        self.frustum = nn.Parameter(torch.stack((x, y, d), -1), requires_grad=False)


    def get_lidar_coor(self, img, img_metas):
        """Calculate the locations of the frustum points in the lidar
        coordinate system.

        Args:
            rots (torch.Tensor): Rotation from camera coordinate system to
                lidar coordinate system in shape (B, N_cams, 3, 3).
            trans (torch.Tensor): Translation from camera coordinate system to
                lidar coordinate system in shape (B, N_cams, 3).
            cam2imgs (torch.Tensor): Camera intrinsic matrixes in shape
                (B, N_cams, 3, 3).
            post_rots (torch.Tensor): Rotation in camera coordinate system in
                shape (B, N_cams, 3, 3). It is derived from the image view
                augmentation.
            post_trans (torch.Tensor): Translation in camera coordinate system
                derived from image view augmentation in shape (B, N_cams, 3).

        Returns:
            torch.tensor: Point coordinates in shape
                (B, N_cams, D, ownsample, 3)
        """
        eps = 1e-5
        B, N, C, H, W = img.shape
        uvd = self.frustum.to(img)
        coords = torch.cat((uvd, torch.ones_like(uvd[..., :1])), -1)
        coords[..., :2] = coords[..., :2] * torch.maximum(coords[..., 2:3], torch.ones_like(coords[..., 2:3])*eps)

        img2lidars = []
        for img_meta in img_metas:
            img2lidar = []
            for i in range(len(img_meta['lidar2img'])):
                img2lidar.append(np.linalg.inv(img_meta['lidar2img'][i]))
                # torch.linalg.inv
            img2lidars.append(np.asarray(img2lidar))
        img2lidars = np.asarray(img2lidars).astype(np.float32)
        img2lidars = coords.new_tensor(img2lidars).to(img) # (B, N, 4, 4)

        # coords = coords.view(1, 1, W, H, self.D, 4, 1).repeat(B, N, 1, 1, 1, 1, 1)
        coords = coords.view(1,1,self.D,H,W,4,1).repeat(B,N,1,1,1,1,1)
        img2lidars = img2lidars.view(B, N, 1, 1, 1, 4, 4).repeat(1, 1, self.D, H, W, 1, 1)
        coords3d = torch.matmul(img2lidars, coords).squeeze(-1)[..., :3]
        return coords3d
    
    def init_acceleration_v2(self, coor):
        """Pre-compute the necessary information in acceleration including the
        index of points in the final feature.

        Args:
            coor (torch.tensor): Coordinate of points in lidar space in shape
                (B, N_cams, D, H, W, 3).
            x (torch.tensor): Feature of points in shape
                (B, N_cams, D, H, W, C).
        """

        ranks_bev, ranks_depth, ranks_feat, \
            interval_starts, interval_lengths = \
            self.voxel_pooling_prepare_v2(coor)

        self.ranks_bev = ranks_bev.int().contiguous()
        self.ranks_feat = ranks_feat.int().contiguous()
        self.ranks_depth = ranks_depth.int().contiguous()
        self.interval_starts = interval_starts.int().contiguous()
        self.interval_lengths = interval_lengths.int().contiguous()

    def voxel_pooling_v2(self, coor, depth, feat):
        ranks_bev, ranks_depth, ranks_feat, \
            interval_starts, interval_lengths = \
            self.voxel_pooling_prepare_v2(coor)
        if ranks_feat is None:
            print('warning ---> no points within the predefined '
                  'bev receptive field')
            dummy = torch.zeros(size=[
                feat.shape[0], feat.shape[2],
                int(self.grid_size[2]),
                int(self.grid_size[0]),
                int(self.grid_size[1])
            ]).to(feat)
            dummy = torch.cat(dummy.unbind(dim=2), 1)
            return dummy
        feat = feat.permute(0, 1, 3, 4, 2)
        bev_feat_shape = (depth.shape[0], int(self.grid_size[2]),
                          int(self.grid_size[1]), int(self.grid_size[0]),
                          feat.shape[-1])  # (B, Z, Y, X, C)
        bev_feat = bev_pool_v2(depth, feat, ranks_depth, ranks_feat, ranks_bev,
                               bev_feat_shape, interval_starts,
                               interval_lengths)
        # collapse Z
        bev_feat = torch.cat(bev_feat.unbind(dim=2), 1)
        return bev_feat

    def voxel_pooling_prepare_v2(self, coor):
        """Data preparation for voxel pooling.

        Args:
            coor (torch.tensor): Coordinate of points in the lidar space in
                shape (B, N, D, H, W, 3).

        Returns:
            tuple[torch.tensor]: Rank of the voxel that a point is belong to
                in shape (N_Points); Reserved index of points in the depth
                space in shape (N_Points). Reserved index of points in the
                feature space in shape (N_Points).
        """
        B, N, D, H, W, _ = coor.shape
        num_points = B * N * D * H * W
        # record the index of selected points for acceleration purpose
        ranks_depth = torch.arange(
            0, num_points, dtype=torch.int, device=coor.device)
        ranks_feat = torch.arange(
            0, num_points // D, dtype=torch.int, device=coor.device)
        ranks_feat = ranks_feat.reshape(B, N, 1, H, W)
        ranks_feat = ranks_feat.expand(B, N, D, H, W).flatten()
        # convert coordinate into the voxel space
        coor = ((coor - self.grid_lower_bound.to(coor)) /
                self.grid_interval.to(coor))
        coor = coor.long().view(num_points, 3)
        batch_idx = torch.arange(0, B).reshape(B, 1). \
            expand(B, num_points // B).reshape(num_points, 1).to(coor)
        coor = torch.cat((coor, batch_idx), 1)

        # filter out points that are outside box
        kept = (coor[:, 0] >= 0) & (coor[:, 0] < self.grid_size[0]) & \
               (coor[:, 1] >= 0) & (coor[:, 1] < self.grid_size[1]) & \
               (coor[:, 2] >= 0) & (coor[:, 2] < self.grid_size[2])
        if len(kept) == 0:
            return None, None, None, None, None
        coor, ranks_depth, ranks_feat = \
            coor[kept], ranks_depth[kept], ranks_feat[kept]
        # get tensors from the same voxel next to each other
        ranks_bev = coor[:, 3] * (
            self.grid_size[2] * self.grid_size[1] * self.grid_size[0])
        ranks_bev += coor[:, 2] * (self.grid_size[1] * self.grid_size[0])
        ranks_bev += coor[:, 1] * self.grid_size[0] + coor[:, 0]
        order = ranks_bev.argsort()
        ranks_bev, ranks_depth, ranks_feat = \
            ranks_bev[order], ranks_depth[order], ranks_feat[order]

        kept = torch.ones(
            ranks_bev.shape[0], device=ranks_bev.device, dtype=torch.bool)
        kept[1:] = ranks_bev[1:] != ranks_bev[:-1]
        interval_starts = torch.where(kept)[0].int()
        if len(interval_starts) == 0:
            return None, None, None, None, None
        interval_lengths = torch.zeros_like(interval_starts)
        interval_lengths[:-1] = interval_starts[1:] - interval_starts[:-1]
        interval_lengths[-1] = ranks_bev.shape[0] - interval_starts[-1]
        return ranks_bev.int().contiguous(), ranks_depth.int().contiguous(
        ), ranks_feat.int().contiguous(), interval_starts.int().contiguous(
        ), interval_lengths.int().contiguous()

    def pre_compute(self, img, img_metas):
        if self.initial_flag:
            coor = self.get_lidar_coor(img, img_metas)
            self.init_acceleration_v2(coor)
            self.initial_flag = False

    def view_transform_core(self, x, depth_digit, tran_feat, img_metas):
        B = len(img_metas)
        # Lift-Splat
        depth = depth_digit.softmax(dim=1)
        if self.accelerate:
            B, N, C, H, W = x.shape
            feat = tran_feat.view(B, N, self.out_channels, H, W)
            feat = feat.permute(0, 1, 3, 4, 2) # (B, N, H, W, self.out_channels)
            
            depth = depth.view(B, N, self.D, H, W)
            bev_feat_shape = (depth.shape[0], int(self.grid_size[2]),
                              int(self.grid_size[1]), int(self.grid_size[0]),
                              feat.shape[-1])  # (B, Z, Y, X, C)
            bev_feat = bev_pool_v2(depth, feat, self.ranks_depth,
                                   self.ranks_feat, self.ranks_bev,
                                   bev_feat_shape, self.interval_starts,
                                   self.interval_lengths)

            bev_feat = bev_feat.squeeze(2)
        else:
            BN, C, H, W = x.shape
            N = BN // B
            x = x.view(B,N,C,H,W)
            coor = self.get_lidar_coor(x, img_metas)
            bev_feat = self.voxel_pooling_v2(
                coor, depth.view(B, N, self.D, H, W),
                tran_feat.view(B, N, self.out_channels, H, W))
        return bev_feat, depth_digit

    def view_transform(self, x, depth_digit, tran_feat, img_metas):
        if self.accelerate:
            B = len(img_metas)
            BN, C, H, W = x.shape
            N = BN // B
            x = x.view(B,N,C,H,W)
            self.pre_compute(x, img_metas)
        return self.view_transform_core(x, depth_digit, tran_feat, img_metas)

    def forward(self, x, img_metas):
        """Transform image-view feature into bird-eye-view feature.

        Args:
            input (list(torch.tensor)): of (image-view feature, rots, trans,
                intrins, post_rots, post_trans)

        Returns:
            torch.tensor: Bird-eye-view feature in shape (B, C, H_BEV, W_BEV)
        """
        B, N, C, H, W = x.shape
        x = x.view(B * N, C, H, W)
        x = self.depth_net(x)

        depth_digit = x[:, :self.D, ...]
        tran_feat = x[:, self.D:self.D + self.out_channels, ...]

        return self.view_transform(x, depth_digit, tran_feat, img_metas)

    def get_mlp_input(self, rot, tran, intrin, post_rot, post_tran, bda):
        return None


class _ASPPModule(nn.Module):

    def __init__(self, inplanes, planes, kernel_size, padding, dilation,
                 norm_cfg=dict(type='BN')):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(
            inplanes,
            planes,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            dilation=dilation,
            bias=False)
        self.bn = build_norm_layer(
                norm_cfg, planes, postfix=0)[1]
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class ASPP(nn.Module):

    def __init__(self, inplanes, mid_channels=256, norm_cfg=dict(type='BN')):
        super(ASPP, self).__init__()

        dilations = [1, 6, 12, 18]

        self.aspp1 = _ASPPModule(
            inplanes,
            mid_channels,
            1,
            padding=0,
            dilation=dilations[0],
            norm_cfg=norm_cfg)
        self.aspp2 = _ASPPModule(
            inplanes,
            mid_channels,
            3,
            padding=dilations[1],
            dilation=dilations[1],
            norm_cfg=norm_cfg)
        self.aspp3 = _ASPPModule(
            inplanes,
            mid_channels,
            3,
            padding=dilations[2],
            dilation=dilations[2],
            norm_cfg=norm_cfg)
        self.aspp4 = _ASPPModule(
            inplanes,
            mid_channels,
            3,
            padding=dilations[3],
            dilation=dilations[3],
            norm_cfg=norm_cfg)

        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(inplanes, mid_channels, 1, stride=1, bias=False),
            build_norm_layer(
                norm_cfg, mid_channels, postfix=0)[1],
            nn.ReLU(),
        )
        self.conv1 = nn.Conv2d(
            int(mid_channels * 5), mid_channels, 1, bias=False)
        self.bn1 = build_norm_layer(
                norm_cfg, mid_channels, postfix=0)[1]
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self._init_weight()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(
            x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return self.dropout(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class Mlp(nn.Module):

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.ReLU,
                 drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class SELayer(nn.Module):

    def __init__(self, channels, act_layer=nn.ReLU, gate_layer=nn.Sigmoid):
        super().__init__()
        self.conv_reduce = nn.Conv2d(channels, channels, 1, bias=True)
        self.act1 = act_layer()
        self.conv_expand = nn.Conv2d(channels, channels, 1, bias=True)
        self.gate = gate_layer()

    def forward(self, x, x_se):
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        return x * self.gate(x_se)


class DepthNet(nn.Module):

    def __init__(self,
                 in_channels,
                 mid_channels,
                 context_channels,
                 depth_channels,
                 use_dcn=True,
                 use_aspp=True,
                 depth_only=False,
                 norm_cfg=dict(type='BN')):
        super(DepthNet, self).__init__()
        self.depth_only = depth_only
        self.reduce_conv = nn.Sequential(
            nn.Conv2d(
                in_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            build_norm_layer(
                norm_cfg, mid_channels, postfix=0)[1],
            nn.ReLU(inplace=True),
        )
        self.context_conv = nn.Conv2d(
            mid_channels, context_channels, kernel_size=1, stride=1, padding=0)
        if norm_cfg['type']=='GN' or norm_cfg['type']=='SyncBN':
            self.bn = nn.LayerNorm(9)
        else:
            self.bn = nn.BatchNorm1d(9)
        self.depth_mlp = Mlp(9, mid_channels, mid_channels)
        self.depth_se = SELayer(mid_channels)  # NOTE: add camera-aware

        self.dep_proj = nn.Conv2d(mid_channels+depth_channels+1+32, mid_channels, kernel_size=1, stride=1, padding=0)

        self.context_mlp = Mlp(9, mid_channels, mid_channels)
        self.context_se = SELayer(mid_channels)  # NOTE: add camera-aware
        depth_conv_list = [
            BasicBlock(mid_channels, mid_channels, norm_cfg=norm_cfg),
            BasicBlock(mid_channels, mid_channels, norm_cfg=norm_cfg),
            BasicBlock(mid_channels, mid_channels, norm_cfg=norm_cfg),
        ]
        if use_aspp:
            depth_conv_list.append(ASPP(mid_channels, mid_channels, norm_cfg=norm_cfg))
        if use_dcn:
            depth_conv_list.append(
                build_conv_layer(
                    cfg=dict(
                        type='DCN',
                        in_channels=mid_channels,
                        out_channels=mid_channels,
                        kernel_size=3,
                        padding=1,
                        groups=4,
                        im2col_step=128,
                    )))
        depth_conv_list.append(
            nn.Conv2d(
                mid_channels,
                depth_channels,
                kernel_size=1,
                stride=1,
                padding=0))
        self.depth_conv = nn.Sequential(*depth_conv_list)
        if depth_only:
            del self.context_mlp
            del self.context_se
            del self.context_conv

    def forward(self, x, radar_feats, rcs_embedding, mlp_input):
        BN, C, H, W = x.shape
        B = BN //6
        mlp_input = self.bn(mlp_input.reshape(-1, mlp_input.shape[-1]))
        x = self.reduce_conv(x)
        if not self.depth_only:
            context_se = self.context_mlp(mlp_input)[..., None, None]
            context = self.context_se(x, context_se)
            context = self.context_conv(context)
            depth_se = self.depth_mlp(mlp_input)[..., None, None]
            depth = self.depth_se(x, depth_se)
            
            depth = torch.cat((depth, radar_feats, rcs_embedding), dim=1)           
            depth = self.dep_proj(depth)

            depth = self.depth_conv(depth)
            return torch.cat([depth, context], dim=1)
        else:
            depth_se = self.depth_mlp(mlp_input)[..., None, None]
            depth = self.depth_se(x, depth_se)
            depth = self.depth_conv(depth)
            return depth



@NECKS.register_module()
class LSSViewTransformerBEVDepth_racformer(LSSViewTransformer_racformer):

    def __init__(self, loss_depth_weight=3.0, loss_reg_depth_weight=1.0, depthnet_cfg=dict(), **kwargs):
        super(LSSViewTransformerBEVDepth_racformer, self).__init__(**kwargs)
        self.loss_depth_weight = loss_depth_weight
        self.depth_net = DepthNet(self.in_channels, self.in_channels,
                                  self.out_channels, self.D, **depthnet_cfg)
        
        # self.depth_embedding = nn.Conv2d(self.D, 32, kernel_size=1, padding=0)
        self.rcs_embedding = nn.Conv2d(64, 32, kernel_size=1, padding=0)
        self.loss_func = FocalLoss(alpha=0.25, gamma=2.0, reduction="none")

    def get_mlp_input(self, img_metas):
        B = len(img_metas)
        N = 6
        T = len(img_metas[0]['lidar2img']) // N    
        lidar2imgs = [img_meta['lidar2img'] for img_meta in img_metas]
        lidar2imgs = np.linalg.inv(np.stack(lidar2imgs))
        mlp_input = torch.from_numpy(lidar2imgs[:,:,:3,:3]).to(torch.float32).contiguous().view(B, N*T, 9).cuda()
        return mlp_input
    
    def get_downsampled_depth(self, depths, downsample):
        """
        Input:
            depths: [B, N, H, W]
        Output:
            depths: [B*N*h*w, d]
        """
        downsample = self.downsample if downsample <= 0 else downsample
        B, N, H, W = depths.shape
        depths = depths.view(B * N, H // downsample,
                                   downsample, W // downsample,
                                   downsample, 1)
        depths = depths.permute(0, 1, 3, 5, 2, 4).contiguous()
        depths = depths.view(-1, downsample * downsample)
        depths_tmp = torch.where(depths == 0.0,
                                    1e5 * torch.ones_like(depths),
                                    depths)
        depths = torch.min(depths_tmp, dim=-1).values
        depths = depths.view(B * N, H // downsample,
                                   W // downsample)
        
        bin_size = 2 * (self.grid_config['depth'][1] - self.grid_config['depth'][0]) / (self.grid_config['depth'][2] * (1 + self.grid_config['depth'][2]))
        indices = -0.5 + 0.5 * torch.sqrt(1 + 8 * (depths - self.grid_config['depth'][0]) / bin_size)
        mask = (indices < 0) | (indices > self.grid_config['depth'][2]) | (~torch.isfinite(indices))
        
        indices[mask] = self.grid_config['depth'][2]

        # Convert to integer
        indices = indices.type(torch.int64)

        # bin_size2 = (self.grid_config['depth'][1] - self.grid_config['depth'][0]) / self.grid_config['depth'][2]
        # one_hot_depths = (depths - (self.grid_config['depth'][0] - bin_size2)) / bin_size2
        # one_hot_depths = torch.where((one_hot_depths < self.D + 1) & (one_hot_depths >= 0.0),
        #                         one_hot_depths, torch.zeros_like(one_hot_depths))
        # one_hot_depths = F.one_hot(
        #     one_hot_depths.long(), num_classes=self.D + 1).view(B*N, H // downsample, W // downsample, self.D + 1)[..., 1:]
        
        return indices, depths
        # return indices, depths, one_hot_depths.float()

    def get_downsampled_rcs(self, rcs, downsample):
        """
        Input:
            gt_depths: [B, N, H, W]
        Output:
            gt_depths: [B*N*h*w, d]
        """
        downsample = self.downsample if downsample <= 0 else downsample
        B, N, H, W = rcs.shape
        rcs = rcs.view(B * N, H // downsample,
                                   downsample, W // downsample,
                                   downsample, 1)
        rcs = rcs.permute(0, 1, 3, 5, 2, 4).contiguous()
        rcs = rcs.view(-1, downsample * downsample)

        rcs_tmp = torch.where(rcs < -64,
                                    -1e5 * torch.ones_like(rcs),
                                    rcs)
        rcs = torch.max(rcs_tmp, dim=-1).values

        rcs = rcs.view(B * N, H // downsample,
                                   W // downsample)

        bin_size = (self.grid_config['rcs'][1] - self.grid_config['rcs'][0]) / self.grid_config['rcs'][2]
        rcs = (rcs - (self.grid_config['rcs'][0] - bin_size)) / bin_size
        rcs = torch.where((rcs < 64 + 1) & (rcs >= -1),
                                rcs, torch.zeros_like(rcs)-1)
        rcs = F.one_hot(
            rcs.long(), num_classes=64 + 1).view(B*N,  H // downsample, W // downsample, 64 + 1)[..., 1:]
        return rcs.float()

    
    @force_fp32()
    def get_depth_loss(self, depth_labels, depth_preds, downsample=0):
        BN, _, H, W = depth_preds.shape
        depth_labels, depth_values = self.get_downsampled_depth(depth_labels, downsample)
        depth_preds = depth_preds.permute(0, 2, 3,
                                          1).contiguous()
        fg_mask = depth_labels < self.grid_config['depth'][2]
        depth_labels = depth_labels[fg_mask]
        depth_preds = depth_preds[fg_mask]
        depth_values = depth_values[fg_mask]

        dep_logits_loss = self.loss_depth_weight * self.loss_func(depth_preds, depth_labels).sum() / max(1.0, fg_mask.sum())

        return dep_logits_loss
    
    def forward(self, x, radar_depth, radar_rcs, img_metas, mlp_input):

        B, N, C, H, W = x.shape
        x = x.reshape(B * N, C, H, W)
        rad_inds, radar_depth = self.get_downsampled_depth(radar_depth, downsample=self.downsample)
        one_hot_rcs= self.get_downsampled_rcs(radar_rcs, downsample=self.downsample)

        # dep_embedding = self.depth_embedding(one_hot_depths.permute(0,3,1,2))
        rcs_embedding = self.rcs_embedding(one_hot_rcs.permute(0,3,1,2))
        
        ones = torch.ones_like(radar_depth).unsqueeze(-1)
        rad_dep_grids = torch.zeros_like(radar_depth).unsqueeze(-1).repeat(1,1,1,int(self.grid_config['depth'][2]+1))
        rad_dep_grids = rad_dep_grids.scatter_(dim=-1, index=rad_inds.unsqueeze(-1), src=ones)
        rad_dep_grids = rad_dep_grids.permute(0,3,1,2)
        
        x = self.depth_net(x, rad_dep_grids, rcs_embedding, mlp_input)
        depth_digit = x[:, :self.D, ...]
        tran_feat = x[:, self.D:self.D + self.out_channels, ...]
        
        return self.view_transform(x, depth_digit, tran_feat, img_metas)


