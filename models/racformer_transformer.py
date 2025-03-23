import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from mmcv.runner import BaseModule
from mmcv.cnn import bias_init_with_prob, xavier_init
from mmcv.cnn.bricks.transformer import MultiheadAttention, FFN, build_positional_encoding
from mmdet.models.utils.builder import TRANSFORMER
from .bbox.utils import decode_bbox, theta_d2xy_coods, xy2theta_d_coods
from .utils import inverse_sigmoid, DUMP
from .sparsebev_sampling import sampling_4d, make_sample_points
from .checkpoint import checkpoint as cp
from .csrc.wrapper import MSMV_CUDA

from .bev_self_attention import BEVSelfAttention

@TRANSFORMER.register_module()
class RaCFormerTransformer(BaseModule):
    def __init__(self, 
                 embed_dims, 
                 num_frames=8, 
                 num_points=4, 
                 num_points_bev=4, 
                 num_layers=6, 
                 num_levels=4, 
                 num_classes=10, 
                 code_size=10, 
                 img_depth_num=3, 
                 bev_depth_num=5, 
                 pc_range=[], 
                 num_ray=150, 
                 d_region_list = [0.15, 0.1, 0.1, 0.08, 0.08, 0.05], 
                 spatial_shapes=(128, 128), 
                 init_cfg=None):
        assert init_cfg is None, 'To prevent abnormal initialization ' \
                            'behavior, init_cfg is not allowed to be set'
        super(RaCFormerTransformer, self).__init__(init_cfg=init_cfg)

        self.embed_dims = embed_dims
        self.pc_range = pc_range

        self.decoder = RaCFormerTransformerDecoder(embed_dims, num_frames, num_points, num_points_bev, num_layers, num_levels, num_classes, code_size, \
                                                   img_depth_num=img_depth_num, bev_depth_num=bev_depth_num, pc_range=pc_range, num_ray=num_ray, \
                                                    d_region_list=d_region_list, spatial_shapes=spatial_shapes)

    @torch.no_grad()
    def init_weights(self):
        self.decoder.init_weights()

    def forward(self, query_bbox, query_feat, mlvl_feats, lss_bev_feats, radar_bev_feats, attn_mask, img_metas):
        cls_scores, bbox_preds = self.decoder(query_bbox, query_feat, mlvl_feats, lss_bev_feats, radar_bev_feats, attn_mask, img_metas)

        cls_scores = torch.nan_to_num(cls_scores)
        bbox_preds = torch.nan_to_num(bbox_preds)

        return cls_scores, bbox_preds


class RaCFormerTransformerDecoder(BaseModule):
    def __init__(self, 
                 embed_dims, 
                 num_frames=8, 
                 num_points=4, 
                 num_points_bev=4, 
                 num_layers=6, 
                 num_levels=4, 
                 num_classes=10, 
                 code_size=10, 
                 img_depth_num=3, 
                 bev_depth_num=5, 
                 pc_range=[], 
                 num_ray=150, 
                 d_region_list=[0.15, 0.1, 0.1, 0.08, 0.08, 0.05], 
                 spatial_shapes=(128, 128), 
                 init_cfg=None):
        super(RaCFormerTransformerDecoder, self).__init__(init_cfg)
        self.num_layers = num_layers
        self.pc_range = pc_range

        # params are shared across all decoder layers
        self.decoder_layer = RaCFormerTransformerDecoderLayer(
            embed_dims, num_frames, num_points, num_points_bev, num_levels, num_classes, code_size, \
                img_depth_num=img_depth_num, bev_depth_num=bev_depth_num, num_ray=num_ray, pc_range=pc_range, \
                    d_region_list=d_region_list, spatial_shapes=spatial_shapes,
        )

    @torch.no_grad()
    def init_weights(self):
        self.decoder_layer.init_weights()

    def forward(self, query_bbox, query_feat, mlvl_feats, lss_bev_feats, radar_bev_feats, attn_mask, img_metas):
        cls_scores, bbox_preds = [], []

        # calculate time difference according to timestamps
        timestamps = np.array([m['img_timestamp'] for m in img_metas], dtype=np.float64)
        timestamps = np.reshape(timestamps, [query_bbox.shape[0], -1, 6])
        time_diff = timestamps[:, :1, :] - timestamps
        time_diff = np.mean(time_diff, axis=-1).astype(np.float32)  # [B, F]
        time_diff = torch.from_numpy(time_diff).to(query_bbox.device)  # [B, F]
        img_metas[0]['time_diff'] = time_diff

        # organize projections matrix and copy to CUDA
        lidar2img = np.asarray([m['lidar2img'] for m in img_metas]).astype(np.float32)
        lidar2img = torch.from_numpy(lidar2img).to(query_bbox.device)  # [B, N, 4, 4]
        img_metas[0]['lidar2img'] = lidar2img

        # group image features in advance for sampling, see `sampling_4d` for more details
        for lvl, feat in enumerate(mlvl_feats):
            B, TN, GC, H, W = feat.shape  # [B, TN, GC, H, W]
            N, T, G, C = 6, TN // 6, 4, GC // 4
            feat = feat.reshape(B, T, N, G, C, H, W)

            if MSMV_CUDA:  # Our CUDA operator requires channel_last
                feat = feat.permute(0, 1, 3, 2, 5, 6, 4)  # [B, T, G, N, H, W, C]
                feat = feat.reshape(B*T*G, N, H, W, C)
            else:  # Torch's grid_sample requires channel_first
                feat = feat.permute(0, 1, 3, 4, 2, 5, 6)  # [B, T, G, C, N, H, W]
                feat = feat.reshape(B*T*G, C, N, H, W)

            mlvl_feats[lvl] = feat.contiguous()

        for i in range(self.num_layers):
            DUMP.stage_count = i

            query_feat, cls_score, bbox_pred = self.decoder_layer(
                query_bbox, query_feat, mlvl_feats, lss_bev_feats, radar_bev_feats, attn_mask, img_metas, layer=i
            )
            query_bbox = bbox_pred.clone().detach()

            bbox_pred = theta_d2xy_coods(bbox_pred)

            cls_scores.append(cls_score)
            bbox_preds.append(bbox_pred)

        cls_scores = torch.stack(cls_scores)
        bbox_preds = torch.stack(bbox_preds)

        return cls_scores, bbox_preds


class RaCFormerTransformerDecoderLayer(BaseModule):
    def __init__(self, 
                 embed_dims, 
                 num_frames=8, 
                 num_points=4, 
                 num_points_bev=4, 
                 num_levels=4, 
                 num_classes=10, 
                 code_size=10, 
                 num_cls_fcs=2, 
                 num_reg_fcs=2,
                 img_depth_num=3, 
                 bev_depth_num=5, 
                 num_ray=150, 
                 pc_range=[], 
                 d_region_list = [0.15, 0.1, 0.1, 0.08, 0.08, 0.05], 
                 spatial_shapes=(128, 128), 
                 init_cfg=None):
        super(RaCFormerTransformerDecoderLayer, self).__init__(init_cfg)

        self.embed_dims = embed_dims
        self.num_classes = num_classes
        self.code_size = code_size
        self.pc_range = pc_range

        self.position_encoder = nn.Sequential(
            nn.Linear(3, self.embed_dims), 
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
        )

        self.self_attn = ScaleAdaptiveSelfAttention(embed_dims, num_heads=8, dropout=0.1, pc_range=pc_range)
        self.sampling = RaCFormerSampling(embed_dims, num_frames=num_frames, num_groups=4, num_points=num_points, num_levels=num_levels, depth_num=img_depth_num, pc_range=pc_range)

        # self.sampling_radar_bev = BEVSampling(embed_dims, num_frames=num_frames, num_heads=4, num_points=num_points_bev, num_levels=1, pc_range=pc_range, depth_num=bev_depth_num, spatial_shapes=spatial_shapes, temp_radar=False)
        self.sampling_radar_bev = BEVSampling(embed_dims, num_frames=num_frames, num_heads=4, num_points=num_points_bev, num_levels=1, pc_range=pc_range, depth_num=bev_depth_num, spatial_shapes=spatial_shapes, temp_radar=True)
        
        self.sampling_lss_bev = BEVSampling(embed_dims, num_frames=num_frames, num_heads=4, num_points=num_points_bev, num_levels=1, pc_range=pc_range, depth_num=bev_depth_num, spatial_shapes=spatial_shapes)
        self.mixing = AdaptiveMixing(in_dim=embed_dims, in_points=num_points * num_frames * img_depth_num, n_groups=4, out_points=128)
        self.ffn = FFN(embed_dims, feedforward_channels=512, ffn_drop=0.1)

        self.norm1 = nn.LayerNorm(embed_dims)
        self.norm2 = nn.LayerNorm(embed_dims)
        self.norm3 = nn.LayerNorm(embed_dims)

        self.fusion = nn.Linear(embed_dims*3, embed_dims)

        self.norm_radar_bev = nn.LayerNorm(embed_dims)
        self.norm_lss_bev = nn.LayerNorm(embed_dims)
        self.norm_fusion = nn.LayerNorm(embed_dims)

        cls_branch = []
        for _ in range(num_cls_fcs):
            cls_branch.append(nn.Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU(inplace=True))
        cls_branch.append(nn.Linear(self.embed_dims, self.num_classes))
        self.cls_branch = nn.Sequential(*cls_branch)

        reg_branch = []
        for _ in range(num_reg_fcs):
            reg_branch.append(nn.Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU(inplace=True))
        reg_branch.append(nn.Linear(self.embed_dims, self.code_size))
        self.reg_branch = nn.Sequential(*reg_branch)
        
        self.d_region_list = d_region_list
        self.num_ray = num_ray

    @torch.no_grad()
    def init_weights(self):
        self.self_attn.init_weights()
        self.sampling.init_weights()
        self.mixing.init_weights()

        self.sampling_radar_bev.init_weights()
        self.sampling_lss_bev.init_weights()
        bias_init = bias_init_with_prob(0.01)
        nn.init.constant_(self.cls_branch[-1].bias, bias_init)
        
        xavier_init(self.fusion, distribution='uniform', bias=0.)

    def refine_bbox(self, bbox_proposal, bbox_delta):
        dz = inverse_sigmoid(bbox_proposal[..., 1:3])
        dz_delta = bbox_delta[..., 1:3]
        dz_new = torch.sigmoid(dz_delta + dz)
        theta = bbox_proposal[..., 0:1] + (torch.sigmoid(bbox_delta[..., 0:1])*2-1) / self.num_ray

        return torch.cat([theta, dz_new, bbox_delta[..., 3:]], dim=-1)


    def forward(self, query_bbox, query_feat, mlvl_feats, lss_bev_feats, radar_bev_feats, attn_mask, img_metas, layer=0):
        """
        query_bbox: [B, Q, 10] [cx, cy, cz, w, h, d, rot.sin, rot.cos, vx, vy]
        """
        query_pos = self.position_encoder(query_bbox[..., :3])
        query_feat = query_feat + query_pos

        query_feat = self.norm1(self.self_attn(query_bbox, query_feat, attn_mask))

        query_radar_feat = self.sampling_radar_bev(query_bbox, query_feat, radar_bev_feats, img_metas, d_region=self.d_region_list[layer]) # here
        query_radar_feat = self.norm_radar_bev(query_radar_feat)
        query_lss_feat = self.sampling_lss_bev(query_bbox, query_feat, lss_bev_feats, img_metas, d_region=self.d_region_list[layer]) # here
        query_lss_feat = self.norm_lss_bev(query_lss_feat)


        sampled_feat = self.sampling(query_bbox, query_feat, mlvl_feats, img_metas, d_region=self.d_region_list[layer])

        query_feat = self.norm2(self.mixing(sampled_feat, query_feat))
        query_feat = self.norm_fusion(self.fusion(torch.cat((query_feat, query_radar_feat, query_lss_feat), dim=-1)))
        query_feat = self.norm3(self.ffn(query_feat))

        cls_score = self.cls_branch(query_feat)  # [B, Q, num_classes]
        bbox_pred = self.reg_branch(query_feat)  # [B, Q, code_size]
        bbox_pred = self.refine_bbox(query_bbox, bbox_pred)

        # calculate absolute velocity according to time difference
        time_diff = img_metas[0]['time_diff']  # [B, F]
        if time_diff.shape[1] > 1:
            time_diff = time_diff.clone()
            time_diff[time_diff < 1e-5] = 1.0
            bbox_pred[..., 8:] = bbox_pred[..., 8:] / time_diff[:, 1:2, None]

        if DUMP.enabled:
            query_bbox_dec = decode_bbox(query_bbox, self.pc_range)
            bbox_pred_dec = decode_bbox(bbox_pred, self.pc_range)
            cls_score_sig = torch.sigmoid(cls_score)
            torch.save(query_bbox_dec, '{}/query_bbox_stage{}.pth'.format(DUMP.out_dir, DUMP.stage_count))
            torch.save(bbox_pred_dec, '{}/bbox_pred_stage{}.pth'.format(DUMP.out_dir, DUMP.stage_count))
            torch.save(cls_score_sig, '{}/cls_score_stage{}.pth'.format(DUMP.out_dir, DUMP.stage_count))

        return query_feat, cls_score, bbox_pred


class ScaleAdaptiveSelfAttention(BaseModule):
    """Scale-adaptive Self Attention"""
    def __init__(self, embed_dims=256, num_heads=8, dropout=0.1, pc_range=[], init_cfg=None):
        super().__init__(init_cfg)
        self.pc_range = pc_range

        self.attention = MultiheadAttention(embed_dims, num_heads, dropout, batch_first=True)
        self.gen_tau = nn.Linear(embed_dims, num_heads)

    @torch.no_grad()
    def init_weights(self):
        nn.init.zeros_(self.gen_tau.weight)
        nn.init.uniform_(self.gen_tau.bias, 0.0, 2.0)

    def inner_forward(self, query_bbox, query_feat, pre_attn_mask):
        """
        query_bbox: [B, Q, 10]
        query_feat: [B, Q, C]
        """
        query_bbox = theta_d2xy_coods(query_bbox).clone()
        dist = self.calc_bbox_dists(query_bbox)
        tau = self.gen_tau(query_feat)  # [B, Q, 8]

        if DUMP.enabled:
            torch.save(tau, '{}/sasa_tau_stage{}.pth'.format(DUMP.out_dir, DUMP.stage_count))

        tau = tau.permute(0, 2, 1)  # [B, 8, Q]
        attn_mask = dist[:, None, :, :] * tau[..., None]  # [B, 8, Q, Q]

        if pre_attn_mask is not None:  # for query denoising
            attn_mask[:, :, pre_attn_mask] = float('-inf')

        attn_mask = attn_mask.flatten(0, 1)  # [Bx8, Q, Q]
        return self.attention(query_feat, attn_mask=attn_mask)

    def forward(self, query_bbox, query_feat, pre_attn_mask):
        if self.training and query_feat.requires_grad:
            return cp(self.inner_forward, query_bbox, query_feat, pre_attn_mask, use_reentrant=False)
        else:
            return self.inner_forward(query_bbox, query_feat, pre_attn_mask)

    @torch.no_grad()
    def calc_bbox_dists(self, bboxes):
        centers = decode_bbox(bboxes, self.pc_range)[..., :2]  # [B, Q, 2]

        dist = []
        for b in range(centers.shape[0]):
            dist_b = torch.norm(centers[b].reshape(-1, 1, 2) - centers[b].reshape(1, -1, 2), dim=-1)
            dist.append(dist_b[None, ...])

        dist = torch.cat(dist, dim=0)  # [B, Q, Q]
        dist = -dist

        return dist


class RaCFormerSampling(BaseModule):
    """Adaptive Spatio-temporal Sampling"""
    def __init__(self, embed_dims=256, num_frames=4, num_groups=4, num_points=8, num_levels=4, depth_num=15, pc_range=[], init_cfg=None):
        super().__init__(init_cfg)

        self.num_frames = num_frames
        self.num_points = num_points
        self.num_groups = num_groups
        self.num_levels = num_levels
        self.pc_range = pc_range
        self.depth_num = depth_num

        self.ray_points_offset = nn.Linear(embed_dims, self.depth_num)
        self.sampling_offset = nn.Linear(embed_dims, depth_num * num_groups * num_points * 3)
        self.scale_weights = nn.Linear(embed_dims, num_groups * num_frames * depth_num * num_points * num_levels)

        
    def init_weights(self):
        bias = self.sampling_offset.bias.data.view(self.depth_num * self.num_groups * self.num_points, 3)
        nn.init.zeros_(self.sampling_offset.weight)
        nn.init.uniform_(bias[:, 0:3], -0.5, 0.5)
        
    
    def inner_forward(self, query_ray, query_feat, mlvl_feats, img_metas, d_region=0.1):
        '''
        query_bbox: [B, Q, 10]
        query_feat: [B, Q, C]
        '''
        B, Q, M = query_ray.shape
        image_h, image_w, _ = img_metas[0]['img_shape'][0]

        query_bbox = theta_d2xy_coods(query_ray).clone()

        # sampling offset of all frames
        sampling_offset = self.sampling_offset(query_feat)
        sampling_offset = sampling_offset.view(B, Q, self.num_groups * self.num_points*self.depth_num, 3)
        sampling_points = make_sample_points(query_bbox, sampling_offset, self.pc_range)  # [B, Q, GP, 3]
        sampling_points = sampling_points.reshape(B, Q, 1, self.num_groups, self.num_points*self.depth_num, 3)
        sampling_points = sampling_points.expand(B, Q, self.num_frames, self.num_groups, self.num_points*self.depth_num, 3)

        # # warp sample points based on velocity
        time_diff = img_metas[0]['time_diff']  # [B, F]
        time_diff = time_diff[:, None, :, None]  # [B, 1, F, 1]
        vel = query_ray[..., 8:].detach()  # [B, Q, 2]
        vel = vel[:, :, None, :]  # [B, Q, 1, 2]
        dist = vel * time_diff  # [B, Q, F, 2]
        dist = dist[:, :, :, None, None, :]  # [B, Q, F, 1, 1, 2]
        sampling_points = torch.cat([
            sampling_points[..., 0:2] - dist,
            sampling_points[..., 2:3]
        ], dim=-1)

        sampling_points[..., 0:1] = (sampling_points[..., 0:1] - self.pc_range[0]) / (self.pc_range[3] - self.pc_range[0])
        sampling_points[..., 1:2] = (sampling_points[..., 1:2] - self.pc_range[1]) / (self.pc_range[4] - self.pc_range[1])
        
        sampling_points = xy2theta_d_coods(sampling_points)
        sampling_points = sampling_points.reshape(B, Q, self.num_frames, self.num_groups, self.num_points, self.depth_num, 3)
        sampling_points_d = torch.linspace(-d_region, d_region, self.depth_num).view(1,1,self.depth_num).repeat(B, Q, 1).to(query_bbox.device)
        sampling_points_d = sampling_points_d + (self.ray_points_offset(query_feat).sigmoid()*2-1)*d_region/self.depth_num/2
        sampling_points_d = sampling_points_d.view(B, Q, 1, 1, 1, self.depth_num, 1).repeat(1, 1, self.num_frames, self.num_groups, self.num_points, 1, 1)

        sampling_points = torch.cat((sampling_points[..., 0:1], sampling_points[..., 1:2]+sampling_points_d, sampling_points[..., 2:]), dim=-1)
        sampling_points = sampling_points.reshape(B, Q, self.num_frames, self.num_groups, self.num_points*self.depth_num, 3) 

        sampling_points = theta_d2xy_coods(sampling_points)
        sampling_points[..., 0:1] = sampling_points[..., 0:1] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0]
        sampling_points[..., 1:2] = sampling_points[..., 1:2] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1]       

        # scale weights
        scale_weights = self.scale_weights(query_feat).view(B, Q, self.num_groups, self.num_frames, self.depth_num*self.num_points, self.num_levels).contiguous()
        scale_weights = torch.softmax(scale_weights, dim=-1)

        # sampling
        sampled_feats = sampling_4d(
            sampling_points,
            mlvl_feats,
            scale_weights,
            img_metas[0]['lidar2img'],
            image_h, image_w
        )  # [B, Q, G, FP, C]

        return sampled_feats
    
    
    
    def forward(self, query_ray, query_feat, mlvl_feats, img_metas, d_region=0.1):
        if self.training and query_feat.requires_grad:
            return cp(self.inner_forward, query_ray, query_feat, mlvl_feats, img_metas, d_region=d_region, use_reentrant=False)
        else:
            return self.inner_forward(query_ray, query_feat, mlvl_feats, img_metas, d_region=d_region)

class BEVSampling(BaseModule):
    """Adaptive Spatio-temporal Sampling"""
    def __init__(self, embed_dims=256, 
                     num_frames=4, 
                     num_points=8, 
                     num_heads=4, 
                     num_levels=4, 
                     pc_range=[],                 
                     spatial_shapes=(128, 128),
                     depth_num=30,
                     temp_radar=False,
                     init_cfg=None):
        super().__init__(init_cfg)

        self.num_frames = num_frames
        self.num_points = num_points
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.embed_dims = embed_dims
        self.pc_range = pc_range
        self.depth_num = depth_num

        self.ray_points_offset = nn.Linear(embed_dims, self.depth_num)
        self.sampling_offset = nn.Linear(embed_dims, depth_num * num_heads * num_points * 2)
        self.scale_weights = nn.Linear(embed_dims, num_heads * num_levels * depth_num * num_points)
        
        positional_encoding=dict(
        type='LearnedPositionalEncoding',
        num_feats=128,
        row_num_embed=spatial_shapes[1],
        col_num_embed=spatial_shapes[0])
        
        self.positional_encoding = build_positional_encoding(
            positional_encoding)      
        self.attention = BEVSelfAttention(embed_dims=embed_dims, num_heads=4, num_levels=1, num_points=num_points*self.depth_num, num_bev_queue=num_frames, queue_weight=True)

        self.temp_radar = temp_radar

        if temp_radar:
            self.temporal_encoder = RadarBEVTemporalEncoder(embed_dims, 64, num_frames)
        
    def init_weights(self):
        bias = self.sampling_offset.bias.data.view(self.depth_num * self.num_heads * self.num_points, 2)
        nn.init.zeros_(self.sampling_offset.weight)
        nn.init.uniform_(bias[:, 0:2], -0.5, 0.5)
        self.attention.init_weights()
        if self.temp_radar:
            self.temporal_encoder.init_weights()


    def inner_forward(self, query_ray, query_feat, bev_feats, img_metas, d_region=0.1):
        '''
        query_bbox: [B, Q, 10]
        query_feat: [B, Q, C]
        '''
        if self.temp_radar:
            bev_feats = self.temporal_encoder(bev_feats)
        
        B, Q, M = query_ray.shape
        bev_h, bev_w = bev_feats.shape[-2:]

        query_bbox = theta_d2xy_coods(query_ray).clone()

        # sampling offset of all frames
        sampling_offset = self.sampling_offset(query_feat)
        sampling_offset = sampling_offset.view(B, Q, self.num_heads*self.num_points*self.depth_num, 2)
        sampling_offset = torch.cat((sampling_offset, torch.zeros_like(sampling_offset[..., 0:1])), dim=-1)
        sampling_points = make_sample_points(query_bbox, sampling_offset, self.pc_range)  # [B, Q, GP, 3]
        sampling_points = sampling_points.reshape(B, Q, 1, self.num_heads, self.num_points*self.depth_num, 3)
        sampling_points = sampling_points.expand(B, Q, self.num_frames, self.num_heads, self.num_points*self.depth_num, 3)

        # warp sample points based on velocity
        time_diff = img_metas[0]['time_diff']  # [B, F]
        time_diff = time_diff[:, None, :, None]  # [B, 1, F, 1]
        vel = query_ray[..., 8:].detach()  # [B, Q, 2]
        vel = vel[:, :, None, :]  # [B, Q, 1, 2]
        dist = vel * time_diff  # [B, Q, F, 2]
        dist = dist[:, :, :, None, None, :]  # [B, Q, F, 1, 1, 2]
        sampling_points = sampling_points[..., 0:2] - dist
  
        sampling_points[..., 0:1] = (sampling_points[..., 0:1] - self.pc_range[0]) / (self.pc_range[3] - self.pc_range[0])
        sampling_points[..., 1:2] = (sampling_points[..., 1:2] - self.pc_range[1]) / (self.pc_range[4] - self.pc_range[1])
        
        sampling_points = xy2theta_d_coods(sampling_points)
        
        sampling_points = sampling_points.reshape(B, Q, self.num_frames, self.num_heads, self.num_points, self.depth_num, 2)
        sampling_points_d = torch.linspace(-d_region, d_region, self.depth_num).view(1,1,self.depth_num).repeat(B, Q, 1).to(query_bbox.device)
        sampling_points_d = sampling_points_d + (self.ray_points_offset(query_feat).sigmoid()*2-1)*d_region/self.depth_num/2
        sampling_points_d = sampling_points_d.view(B, Q, 1, 1, 1, self.depth_num, 1).repeat(1, 1, self.num_frames, self.num_heads, self.num_points, 1, 1)
        
        sampling_points = torch.cat((sampling_points[..., 0:1], sampling_points[..., 1:2]+sampling_points_d), dim=-1)
        sampling_points = sampling_points.reshape(B, Q, self.num_frames, self.num_heads, self.num_points*self.depth_num, 2)

        sampling_points = theta_d2xy_coods(sampling_points)
                
        # scale weights
        sampling_points = sampling_points.permute(0,1,3,2,4,5).contiguous()
        scale_weights = self.scale_weights(query_feat).view(B, Q, self.num_heads, 1, self.num_levels, self.depth_num*self.num_points).contiguous()
        scale_weights = torch.softmax(scale_weights, dim=-1)
        
        scale_weights = scale_weights.expand(B, Q, self.num_heads, self.num_frames, self.num_levels, self.depth_num*self.num_points).contiguous()

        # sampling
        bev_mask = torch.zeros((B, bev_h, bev_w),
                               device=bev_feats.device).to(bev_feats.dtype)
        bev_pos = self.positional_encoding(bev_mask).to(bev_feats.dtype)
        bev_pos = bev_pos.view(B, 1, self.embed_dims, bev_h, bev_w).repeat(1,self.num_frames,1,1,1)
        
        sampled_feats = self.attention(query_feat, bev_feats+bev_pos, sampling_points, scale_weights, spatial_shapes=(bev_h, bev_w))

        return sampled_feats
        
    
    def forward(self, query_ray, query_feat, bev_feats, img_metas, d_region=0.1):
        if self.training and query_feat.requires_grad:
            return cp(self.inner_forward, query_ray, query_feat, bev_feats, img_metas, d_region=d_region, use_reentrant=False)
        else:
            return self.inner_forward(query_ray, query_feat, bev_feats, img_metas, d_region=d_region)


class AdaptiveMixing(nn.Module):
    """Adaptive Mixing"""
    def __init__(self, in_dim, in_points, n_groups=1, query_dim=None, out_dim=None, out_points=None):
        super(AdaptiveMixing, self).__init__()

        out_dim = out_dim if out_dim is not None else in_dim
        out_points = out_points if out_points is not None else in_points
        query_dim = query_dim if query_dim is not None else in_dim

        self.query_dim = query_dim
        self.in_dim = in_dim
        self.in_points = in_points
        self.n_groups = n_groups
        self.out_dim = out_dim
        self.out_points = out_points

        self.eff_in_dim = in_dim // n_groups
        self.eff_out_dim = out_dim // n_groups

        self.m_parameters = self.eff_in_dim * self.eff_out_dim
        self.s_parameters = self.in_points * self.out_points
        self.total_parameters = self.m_parameters + self.s_parameters

        self.parameter_generator = nn.Linear(self.query_dim, self.n_groups * self.total_parameters)
        self.out_proj = nn.Linear(self.eff_out_dim * self.out_points * self.n_groups, self.query_dim)
        self.act = nn.ReLU(inplace=True)

    @torch.no_grad()
    def init_weights(self):
        nn.init.zeros_(self.parameter_generator.weight)

    def inner_forward(self, x, query):
        B, Q, G, P, C = x.shape
        assert G == self.n_groups
        assert P == self.in_points
        assert C == self.eff_in_dim

        '''generate mixing parameters'''
        params = self.parameter_generator(query)
        params = params.reshape(B*Q, G, -1)
        out = x.reshape(B*Q, G, P, C)

        M, S = params.split([self.m_parameters, self.s_parameters], 2)
        M = M.reshape(B*Q, G, self.eff_in_dim, self.eff_out_dim)
        S = S.reshape(B*Q, G, self.out_points, self.in_points)

        '''adaptive channel mixing'''
        out = torch.matmul(out, M)
        out = F.layer_norm(out, [out.size(-2), out.size(-1)])
        out = self.act(out)

        '''adaptive point mixing'''
        out = torch.matmul(S, out)  # implicitly transpose and matmul
        out = F.layer_norm(out, [out.size(-2), out.size(-1)])
        out = self.act(out)

        '''linear transfomation to query dim'''
        out = out.reshape(B, Q, -1)
        out = self.out_proj(out)
        out = query + out

        return out

    def forward(self, x, query):
        if self.training and x.requires_grad:
            return cp(self.inner_forward, x, query, use_reentrant=False)
        else:
            return self.inner_forward(x, query)

class RadarBEVTemporalEncoder(BaseModule):
    """Adaptive Spatio-temporal Sampling"""
    def __init__(self, embed_dims=256,
                     hidden_dims=64,
                     num_frames=8, 
                     kernel_size=3,
                     downsample_ratio=2,
                     init_cfg=None):
        super().__init__(init_cfg)

        self.num_frames = num_frames
        self.embed_dims = embed_dims
        self.hidden_dims = hidden_dims
        self.convGRU = ConvGRU(input_channels=hidden_dims, hidden_channels=hidden_dims, kernel_size=kernel_size)
        self.temporal_fusion = nn.Conv2d(embed_dims+hidden_dims, embed_dims, kernel_size, padding=kernel_size//2)

        self.downsample_ratio = downsample_ratio
        self.downsample = nn.Conv2d(embed_dims, hidden_dims, kernel_size=3, stride=downsample_ratio, padding=1)

        self.upsample = nn.Sequential(
                            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                            nn.Conv2d(hidden_dims, hidden_dims, kernel_size=3, padding=1))

        
    def init_weights(self):
        self.convGRU.init_weights()

    def inner_forward(self, bev_feats):
        B, T, C, H, W = bev_feats.shape 

        bev_feats_down = bev_feats
        bev_feats_down = self.downsample(bev_feats.flatten(0,1)).reshape(B, T, self.hidden_dims, H//self.downsample_ratio, W//self.downsample_ratio)
        bev_h_feats = self.convGRU(bev_feats_down)
        bev_h_feats = self.upsample(bev_h_feats.flatten(0,1)).reshape(B, T, self.hidden_dims, H, W)

        bev_feats = torch.cat((bev_feats, bev_h_feats), dim=2)
        bev_feats = bev_feats.flatten(0, 1)
        bev_feats = self.temporal_fusion(bev_feats).reshape(B, T, C, H, W)
        return bev_feats
        
    
    def forward(self, bev_feats):
        if self.training and bev_feats.requires_grad:
            return cp(self.inner_forward, bev_feats, use_reentrant=False)
        else:
            return self.inner_forward(bev_feats)
        
class ConvGRU(BaseModule):
    def __init__(self, input_channels, hidden_channels, kernel_size):
        super().__init__()
        self.convGRUCell = ConvGRUCell(input_channels, hidden_channels, kernel_size)
        self.hidden_channels = hidden_channels

    def forward(self, x):

        B, T, C, H, W = x.shape
        h = torch.zeros(B, self.hidden_channels, H, W, device=x.device)
        h0 = h.clone().detach()
        
        out = []
        x_unfold = x.permute(1, 0, 2, 3, 4)  # (T, B, C_in, H, W)

        num_t = 4 if T>4 else T
        for t in range(T):
            if t >= num_t:
                out.append(h0)
                continue
            x_t = x_unfold[t]
            if t > 1:
                with torch.no_grad():
                    h = self.convGRUCell(x_t, h)
            else:
                h = self.convGRUCell(x_t, h)
            out.append(h)
        
        return torch.stack(out, dim=1)

class ConvGRUCell(BaseModule):
    def __init__(self, input_channels, hidden_channels, kernel_size):
        super().__init__()
        padding = kernel_size // 2
        self.hidden_channels = hidden_channels
        # 合并所有门控的卷积计算为单个大卷积
        self.gates_conv = nn.Conv2d(
            input_channels + hidden_channels, 
            3 * hidden_channels,  # 同时计算z, r, h_candidate
            kernel_size=kernel_size,
            padding=padding
        )
        self.matching_layer = nn.Conv2d(hidden_channels, input_channels, 1)

    def forward(self, x, h_prev):
        h_matched = self.matching_layer(h_prev)
        combined = torch.cat([x, h_matched], dim=1)
        gates = self.gates_conv(combined)
        z_gate, r_gate, h_candidate = torch.split(gates, self.hidden_channels, dim=1)
        
        z = torch.sigmoid(z_gate)
        r = torch.sigmoid(r_gate)
        h_candidate = torch.tanh(h_candidate + r * h_prev)
        
        h_next = (1 - z) * h_prev + z * h_candidate
        return h_next
       
# class ConvGRU(BaseModule):
#     def __init__(self, input_channels, hidden_channels, kernel_size):
#         super(ConvGRU, self).__init__()
#         self.convGRUCell = ConvGRUCell(input_channels, hidden_channels, kernel_size)
#         self.input_channels = input_channels
#         self.hidden_channels = hidden_channels

#     def forward(self, x):
#         # 初始化隐藏状态
#         h = torch.zeros(x.size(0), self.hidden_channels, x.size(3), x.size(4), device=x.device)

#         out = []
#         # num_t = 4 if x.size(1)>4 else x.size(1)
#         for t in range(x.size(1))[: :-1]:
#             # if t >= num_t:
#             #     out.append(h)
#             #     continue
#             if t<1:
#                 h = self.convGRUCell(x[:, t, :, :, :], h)
#             else:
#                 with torch.no_grad():
#                     h = self.convGRUCell(x[:, t, :, :, :], h)
#             out.append(h)
#         reversed_out = out[: :-1]
#         return torch.stack(reversed_out, dim=1)


       
# class ConvGRUCell(BaseModule):
#     def __init__(self, input_channels, hidden_channels, kernel_size):
#         super(ConvGRUCell, self).__init__()
#         self.input_channels = input_channels
#         self.hidden_channels = hidden_channels
#         self.kernel_size = kernel_size
#         self.padding = kernel_size // 2

#         self.update_gate = nn.Conv2d(input_channels, hidden_channels, kernel_size, padding=self.padding)
#         self.reset_gate = nn.Conv2d(input_channels, hidden_channels, kernel_size, padding=self.padding)
#         self.candidate_hidden = nn.Conv2d(input_channels, hidden_channels, kernel_size, padding=self.padding)     
#         self.matching_layer = nn.Conv2d(hidden_channels, input_channels, 1)
  
#     def forward(self, x, h_prev):
#         h_prev_matching = self.matching_layer(h_prev)      
#         update = torch.sigmoid(self.update_gate(x) + self.reset_gate(h_prev_matching))
#         reset = torch.sigmoid(self.reset_gate(x))
#         new_h_cand = torch.tanh(self.candidate_hidden(x))
#         h_curr = update * h_prev + reset * new_h_cand
#         return h_curr