import torch
from mmdet.core.bbox.match_costs.builder import MATCH_COST
from models.bbox.utils import xy2theta_d_coods

@MATCH_COST.register_module()
class BBox3DL1Cost(object):
    """BBox3DL1Cost.
     Args:
         weight (int | float, optional): loss_weight
    """

    def __init__(self, weight=1.0):
        self.weight = weight

    def __call__(self, bbox_pred, gt_bboxes):
        """
        Args:
            bbox_pred (Tensor): Predicted boxes with normalized coordinates
                (cx, cy, w, h), which are all in range [0, 1]. Shape
                [num_query, 4].
            gt_bboxes (Tensor): Ground truth boxes with normalized
                coordinates (x1, y1, x2, y2). Shape [num_gt, 4].
        Returns:
            torch.Tensor: bbox_cost value with weight
        """
        bbox_cost = torch.cdist(bbox_pred, gt_bboxes, p=1)
        return bbox_cost * self.weight

@MATCH_COST.register_module()
class ThetaL1Cost(object):
    """BBox3DL1Cost.
     Args:
         weight (int | float, optional): loss_weight
    """

    def __init__(self, weight=1.0):
        self.weight = weight

    def __call__(self, bbox_pred, gt_bboxes):
        """
        Args:
            bbox_pred (Tensor): Predicted boxes with normalized coordinates
                (cx, cy, w, h), which are all in range [0, 1]. Shape
                [num_query, 4].
            gt_bboxes (Tensor): Ground truth boxes with normalized
                coordinates (x1, y1, x2, y2). Shape [num_gt, 4].
        Returns:
            torch.Tensor: bbox_cost value with weight
        """
        pc_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]

        bbox_pred[..., 0] = (bbox_pred[..., 0] - pc_range[0]) / (pc_range[3] - pc_range[0])
        bbox_pred[..., 1] = (bbox_pred[..., 1] - pc_range[1]) / (pc_range[4] - pc_range[1])
        
        gt_bboxes[..., 0] = (gt_bboxes[..., 0] - pc_range[0]) / (pc_range[3] - pc_range[0])
        gt_bboxes[..., 1] = (gt_bboxes[..., 1] - pc_range[1]) / (pc_range[4] - pc_range[1])

        theta_pred = xy2theta_d_coods(bbox_pred)[..., 0:1]
        theta_gt = xy2theta_d_coods(gt_bboxes)[..., 0:1]
        theta_cost = torch.cdist(theta_pred, theta_gt, p=1)
        
        theta_cost = torch.abs(torch.remainder(theta_cost + 0.5, 1) - 0.5)

        return theta_cost * self.weight
    
@MATCH_COST.register_module()
class BBoxBEVL1Cost(object):
    def __init__(self, weight, pc_range):
        self.weight = weight
        self.pc_range = pc_range

    def __call__(self, bboxes, gt_bboxes):
        pc_start = bboxes.new(self.pc_range[0:2])
        pc_range = bboxes.new(self.pc_range[3:5]) - bboxes.new(self.pc_range[0:2])
        # normalize the box center to [0, 1]
        normalized_bboxes_xy = (bboxes[:, :2] - pc_start) / pc_range
        normalized_gt_bboxes_xy = (gt_bboxes[:, :2] - pc_start) / pc_range
        reg_cost = torch.cdist(normalized_bboxes_xy, normalized_gt_bboxes_xy, p=1)
        return reg_cost * self.weight


@MATCH_COST.register_module()
class IoU3DCost(object):
    def __init__(self, weight):
        self.weight = weight

    def __call__(self, iou):
        iou_cost = - iou
        return iou_cost * self.weight
