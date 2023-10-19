import numpy as np
import torch
import torch.nn as nn
import math

from ...utils import box_coder_utils, common_utils, loss_utils
from .target_assigner.anchor_generator import AnchorGenerator
from .target_assigner.atss_target_assigner import ATSSTargetAssigner
from .target_assigner.axis_aligned_target_assigner import AxisAlignedTargetAssigner
from .target_assigner.axis_aligned_target_assigner_pr_importance import AxisAlignedTargetAssignerPRI
from pathlib import Path
import datetime

class AnchorHeadPRITemplate(nn.Module):
    def __init__(self, model_cfg, num_class, class_names, grid_size, point_cloud_range, predict_boxes_when_training):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_class = num_class
        self.class_names = class_names
        self.predict_boxes_when_training = predict_boxes_when_training
        self.use_multihead = self.model_cfg.get('USE_MULTIHEAD', False)

        anchor_target_cfg = self.model_cfg.TARGET_ASSIGNER_CONFIG
        # print(anchor_target_cfg)
        self.box_coder = getattr(box_coder_utils, anchor_target_cfg.BOX_CODER)(
            num_dir_bins=anchor_target_cfg.get('NUM_DIR_BINS', 6),
            **anchor_target_cfg.get('BOX_CODER_CONFIG', {})
        )

        anchor_generator_cfg = self.model_cfg.ANCHOR_GENERATOR_CONFIG
        anchors, self.num_anchors_per_location = self.generate_anchors(
            anchor_generator_cfg, grid_size=grid_size, point_cloud_range=point_cloud_range,
            anchor_ndim=self.box_coder.code_size
        )
        self.anchors = [x.cuda() for x in anchors]
        self.target_assigner = self.get_target_assigner(anchor_target_cfg)

        self.forward_ret_dict = {}
        self.build_losses(self.model_cfg.LOSS_CONFIG)
        
        # easy_original:
        self.conv_information = nn.Conv2d(6, 24, kernel_size=1)
        self.conv_cls_information = nn.Conv2d(18, 24, kernel_size=1)
        self.conv_box_information = nn.Conv2d(6 * self.box_coder.code_size, 24, kernel_size=1)
        self.conv_dir_information = nn.Conv2d(6, 24, kernel_size=1)

        # try:
        # self.conv_information = nn.Sequential(
        #     nn.Conv2d(6, 24, kernel_size=1),
        #     nn.BatchNorm2d(24),
        #     nn.ReLU()
        # )
        # self.conv_cls_information = nn.Sequential(
        #     nn.Conv2d(18, 24, kernel_size=1),
        #     nn.BatchNorm2d(24),
        #     nn.ReLU()
        # )
        # self.conv_box_information = nn.Sequential(
        #     nn.Conv2d(6 * self.box_coder.code_size, 24, kernel_size=1),
        #     nn.BatchNorm2d(24),
        #     nn.ReLU()
        # )
        # self.conv_dir_information = nn.Sequential(
        #     nn.Conv2d(6, 24, kernel_size=1),
        #     nn.BatchNorm2d(24),
        #     nn.ReLU()
        # )

        # voxelrcnn
        self.conv_importance = nn.Sequential(
            nn.Conv2d(352, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv2d(64, 6, kernel_size=1),
            nn.Dropout(0.2),
            nn.Sigmoid(), 
        )

        # only loss
        # self.conv_importance_only_loss = nn.Sequential(
        #     nn.Conv2d(328, 64, kernel_size=1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),
        #     nn.Dropout(0.2),
        #     nn.Conv2d(64, 6, kernel_size=1),
        #     nn.Dropout(0.2),
        #     nn.Sigmoid(), 
        # )

        # second
        # self.conv_importance = nn.Sequential(
        #     nn.Conv2d(480, 64, kernel_size=1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),
        #     nn.Dropout(0.2),
        #     nn.Conv2d(64, 6, kernel_size=1),
        #     nn.Dropout(0.2),
        #     nn.Sigmoid(), 
        # )
    @staticmethod
    def generate_anchors(anchor_generator_cfg, grid_size, point_cloud_range, anchor_ndim=7):
        anchor_generator = AnchorGenerator(
            anchor_range=point_cloud_range,
            anchor_generator_config=anchor_generator_cfg
        )
        feature_map_size = [grid_size[:2] // config['feature_map_stride'] for config in anchor_generator_cfg]
        anchors_list, num_anchors_per_location_list = anchor_generator.generate_anchors(feature_map_size)

        if anchor_ndim != 7:
            for idx, anchors in enumerate(anchors_list):
                pad_zeros = anchors.new_zeros([*anchors.shape[0:-1], anchor_ndim - 7])
                new_anchors = torch.cat((anchors, pad_zeros), dim=-1)
                anchors_list[idx] = new_anchors

        return anchors_list, num_anchors_per_location_list

    def get_target_assigner(self, anchor_target_cfg):
        if anchor_target_cfg.NAME == 'ATSS':
            target_assigner = ATSSTargetAssigner(
                model_cfg=self.model_cfg,
                topk=anchor_target_cfg.TOPK,
                box_coder=self.box_coder,
                match_height=anchor_target_cfg.MATCH_HEIGHT
            )
        elif anchor_target_cfg.NAME == 'AxisAlignedTargetAssigner':
            target_assigner = AxisAlignedTargetAssigner(
                model_cfg=self.model_cfg,
                class_names=self.class_names,
                box_coder=self.box_coder,
                match_height=anchor_target_cfg.MATCH_HEIGHT
            )
        elif anchor_target_cfg.NAME == 'AxisAlignedTargetAssignerPRI':
            target_assigner = AxisAlignedTargetAssignerPRI(
                model_cfg=self.model_cfg,
                class_names=self.class_names,
                box_coder=self.box_coder,
                match_height=anchor_target_cfg.MATCH_HEIGHT
            )
        else:
            raise NotImplementedError
        return target_assigner

    def build_losses(self, losses_cfg):
        self.add_module(
            'cls_loss_func',
            loss_utils.SigmoidFocalClassificationLoss(alpha=0.25, gamma=2.0)
        )
        reg_loss_name = 'WeightedSmoothL1Loss' if losses_cfg.get('REG_LOSS_TYPE', None) is None \
            else losses_cfg.REG_LOSS_TYPE
        self.add_module(
            'reg_loss_func',
            getattr(loss_utils, reg_loss_name)(code_weights=losses_cfg.LOSS_WEIGHTS['code_weights'])
        )
        self.add_module(
            'dir_loss_func',
            loss_utils.WeightedCrossEntropyLoss()
        )

    def assign_targets(self, gt_boxes, pts_ratio):
        """
        Args:
            gt_boxes: (B, M, 8)
        Returns:

        """
        # print(gt_boxes.shape)
        # print(pts_ratio.shape)
        # exit()
        targets_dict = self.target_assigner.assign_targets(
            self.anchors, gt_boxes, pts_ratio
        )
        return targets_dict

    def get_cls_layer_loss(self):
        # print(self.forward_ret_dict.keys())
        cls_preds = self.forward_ret_dict['cls_preds'] # [8, 200, 176, 18]
        box_cls_labels = self.forward_ret_dict['box_cls_labels'] # [8, 211200]
        # pts_ratio_weight = self.forward_ret_dict['pts_ratio_weight'] + 0.1 # [8, 211200]
        
        # print(pts_ratio_weight.max())
        # print(pts_ratio_weight.min())
        batch_size = int(cls_preds.shape[0])
        cared = box_cls_labels >= 0  # [N, num_anchors]
        positives = box_cls_labels > 0

        negatives = box_cls_labels == 0
        negative_cls_weights = negatives * 1.0
        cls_weights = (negative_cls_weights + 1.0 * positives).float()
        if self.num_class == 1:
            # class agnostic
            box_cls_labels[positives] = 1

        pos_normalizer = positives.sum(1, keepdim=True).float()
        cls_weights /= torch.clamp(pos_normalizer, min=1.0)

        cls_targets = box_cls_labels * cared.type_as(box_cls_labels)
        cls_targets = cls_targets.unsqueeze(dim=-1) # [8, 211200, 1]
        cls_targets = cls_targets.squeeze(dim=-1) # [8, 211200]
        one_hot_targets = torch.zeros(
            *list(cls_targets.shape), self.num_class + 1, dtype=cls_preds.dtype, device=cls_targets.device
        )   # [8, 211200, 4]
        one_hot_targets.scatter_(-1, cls_targets.unsqueeze(dim=-1).long(), 1.0)
        # [8, 211200, 4]
        # print(cls_preds.shape)
        cls_preds = cls_preds.view(batch_size, -1, self.num_class) #[8, 211200, 3]
        # print(cls_preds.shape)
        # exit()
        one_hot_targets = one_hot_targets[..., 1:] # [8, 211200, 3]

        cls_loss_src = self.cls_loss_func(cls_preds, one_hot_targets, weights=cls_weights)  # [N, M]
        # reweight by pts_ratio:
        # print(pts_ratio_weight.min())
        # print(pts_ratio_weight.max())
        # pts_ratio_weight = pts_ratio_weight.unsqueeze(dim=-1)
        # cls_loss_src = cls_loss_src * pts_ratio_weight

        cls_loss = cls_loss_src.sum() / batch_size
        cls_loss = cls_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['cls_weight']
        tb_dict = {
            'rpn_loss_cls': cls_loss.item()
        }
        return cls_loss, tb_dict

    def get_cls_layer_dr_loss(self):
        # print(self.forward_ret_dict.keys())
        cls_preds = self.forward_ret_dict['cls_preds'] # [8, 200, 176, 18]
        box_cls_labels = self.forward_ret_dict['box_cls_labels'] # [8, 211200]
        box_cls_labels_vis = box_cls_labels.cpu().numpy().reshape(200,176,3)
        importance = self.forward_ret_dict['importance']
        batch_size = int(cls_preds.shape[0])
        cared = box_cls_labels >= 0  # [N, num_anchors]
        positives = box_cls_labels > 0

        negatives = box_cls_labels == 0
        negative_cls_weights = negatives * 1.0
        cls_weights = (negative_cls_weights + 1.0 * positives).float()
        if self.num_class == 1:
            # class agnostic
            box_cls_labels[positives] = 1

        pos_normalizer = positives.sum(1, keepdim=True).float()
        cls_weights /= torch.clamp(pos_normalizer, min=1.0)

        cls_targets = box_cls_labels * cared.type_as(box_cls_labels)
        cls_targets = cls_targets.unsqueeze(dim=-1) # [8, 211200, 1]
        cls_targets = cls_targets.squeeze(dim=-1) # [8, 211200]
        one_hot_targets = torch.zeros(
            *list(cls_targets.shape), self.num_class + 1, dtype=cls_preds.dtype, device=cls_targets.device
        )   # [8, 211200, 4]
        one_hot_targets.scatter_(-1, cls_targets.unsqueeze(dim=-1).long(), 1.0)
        cls_preds = cls_preds.view(batch_size, -1, self.num_class) #[8, 211200, 3]
        one_hot_targets = one_hot_targets[..., 1:] # [8, 211200, 3]
        cls_loss_src = self.cls_loss_func(cls_preds, one_hot_targets, weights=cls_weights)  # [N, M]

        # importance and regularization
        importance = importance.view(batch_size, -1, 1)
        # cls_loss_src = cls_loss_src * importance
        cls_loss_src = cls_loss_src * importance

        cls_loss = cls_loss_src.sum() / batch_size
        cls_loss = cls_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['cls_weight']
        tb_dict = {
            'rpn_loss_cls': cls_loss.item()
        }
        return cls_loss, tb_dict

    def get_cls_loss(self):
        cls_preds = self.forward_ret_dict['cls_preds'] # [8, 200, 176, 18]
        box_cls_labels = self.forward_ret_dict['box_cls_labels'] # [8, 211200]
        batch_size = int(cls_preds.shape[0])
        cared = box_cls_labels >= 0  # [N, num_anchors]
        positives = box_cls_labels > 0
        negatives = box_cls_labels == 0
        negative_cls_weights = negatives * 1.0
        cls_weights = (negative_cls_weights + 1.0 * positives).float()
        if self.num_class == 1:
            # class agnostic
            box_cls_labels[positives] = 1

        pos_normalizer = positives.sum(1, keepdim=True).float()
        cls_weights /= torch.clamp(pos_normalizer, min=1.0)

        cls_targets = box_cls_labels * cared.type_as(box_cls_labels)
        cls_targets = cls_targets.unsqueeze(dim=-1) # [8, 211200, 1]
        cls_targets = cls_targets.squeeze(dim=-1) # [8, 211200]
        one_hot_targets = torch.zeros(
            *list(cls_targets.shape), self.num_class + 1, dtype=cls_preds.dtype, device=cls_targets.device
        )   # [8, 211200, 4]
        one_hot_targets.scatter_(-1, cls_targets.unsqueeze(dim=-1).long(), 1.0)
        cls_preds = cls_preds.view(batch_size, -1, self.num_class) #[8, 211200, 3]
        one_hot_targets = one_hot_targets[..., 1:] # [8, 211200, 3]
        cls_loss_src = self.cls_loss_func(cls_preds, one_hot_targets, weights=cls_weights)    # [N, M]
        # cls_loss_src[positives] = cls_loss_src[positives] / cls_loss_src[positives].mean()
        return cls_loss_src

    @staticmethod
    def add_sin_difference(boxes1, boxes2, dim=6):
        assert dim != -1
        rad_pred_encoding = torch.sin(boxes1[..., dim:dim + 1]) * torch.cos(boxes2[..., dim:dim + 1])
        rad_tg_encoding = torch.cos(boxes1[..., dim:dim + 1]) * torch.sin(boxes2[..., dim:dim + 1])
        boxes1 = torch.cat([boxes1[..., :dim], rad_pred_encoding, boxes1[..., dim + 1:]], dim=-1)
        boxes2 = torch.cat([boxes2[..., :dim], rad_tg_encoding, boxes2[..., dim + 1:]], dim=-1)
        return boxes1, boxes2

    @staticmethod
    def get_direction_target(anchors, reg_targets, one_hot=True, dir_offset=0, num_bins=2):
        batch_size = reg_targets.shape[0]
        anchors = anchors.view(batch_size, -1, anchors.shape[-1])
        rot_gt = reg_targets[..., 6] + anchors[..., 6]
        offset_rot = common_utils.limit_period(rot_gt - dir_offset, 0, 2 * np.pi)
        dir_cls_targets = torch.floor(offset_rot / (2 * np.pi / num_bins)).long()
        dir_cls_targets = torch.clamp(dir_cls_targets, min=0, max=num_bins - 1)

        if one_hot:
            dir_targets = torch.zeros(*list(dir_cls_targets.shape), num_bins, dtype=anchors.dtype,
                                      device=dir_cls_targets.device)
            dir_targets.scatter_(-1, dir_cls_targets.unsqueeze(dim=-1).long(), 1.0)
            dir_cls_targets = dir_targets
        return dir_cls_targets

    def get_box_reg_layer_loss(self):
        box_preds = self.forward_ret_dict['box_preds']
        box_dir_cls_preds = self.forward_ret_dict.get('dir_cls_preds', None)
        box_reg_targets = self.forward_ret_dict['box_reg_targets']
        box_cls_labels = self.forward_ret_dict['box_cls_labels']

        pts_ratio_weight = self.forward_ret_dict['pts_ratio_weight'] + 0.1 # [8, 211200]
        batch_size = int(box_preds.shape[0])

        positives = box_cls_labels > 0
        # reg_weights = positives.float()
        reg_weights = self.forward_ret_dict['reg_weights']
        # print(reg_weights[positives])
        pos_normalizer = positives.sum(1, keepdim=True).float()
        # print(pos_normalizer)
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)
        if isinstance(self.anchors, list):
            if self.use_multihead:
                anchors = torch.cat(
                    [anchor.permute(3, 4, 0, 1, 2, 5).contiguous().view(-1, anchor.shape[-1]) for anchor in
                     self.anchors], dim=0)
            else:
                anchors = torch.cat(self.anchors, dim=-3)
        else:
            anchors = self.anchors
        anchors = anchors.view(1, -1, anchors.shape[-1]).repeat(batch_size, 1, 1)
        box_preds = box_preds.view(batch_size, -1,
                                   box_preds.shape[-1] // self.num_anchors_per_location if not self.use_multihead else
                                   box_preds.shape[-1])
        # sin(a - b) = sinacosb-cosasinb
        box_preds_sin, reg_targets_sin = self.add_sin_difference(box_preds, box_reg_targets)
        loc_loss_src = self.reg_loss_func(box_preds_sin, reg_targets_sin, weights=reg_weights)  # [N, M]

        # reweight by pts_ratio:
        # pts_ratio_weight = pts_ratio_weight.unsqueeze(dim=-1)
        # loc_loss_src = loc_loss_src * pts_ratio_weight
        loc_loss = loc_loss_src.sum() / batch_size

        loc_loss = loc_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['loc_weight']
        box_loss = loc_loss
        tb_dict = {
            'rpn_loss_loc': loc_loss.item()
        }

        if box_dir_cls_preds is not None:
            dir_targets = self.get_direction_target(
                anchors, box_reg_targets,
                dir_offset=self.model_cfg.DIR_OFFSET,
                num_bins=self.model_cfg.NUM_DIR_BINS
            )

            dir_logits = box_dir_cls_preds.view(batch_size, -1, self.model_cfg.NUM_DIR_BINS)
            weights = positives.type_as(dir_logits)
            weights /= torch.clamp(weights.sum(-1, keepdim=True), min=1.0)
            dir_loss = self.dir_loss_func(dir_logits, dir_targets, weights=weights)
            dir_loss = dir_loss.sum() / batch_size
            dir_loss = dir_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['dir_weight']
            box_loss += dir_loss
            tb_dict['rpn_loss_dir'] = dir_loss.item()

        return box_loss, tb_dict

    def get_box_loss(self):
        box_preds = self.forward_ret_dict['box_preds']
        box_dir_cls_preds = self.forward_ret_dict.get('dir_cls_preds', None)
        box_reg_targets = self.forward_ret_dict['box_reg_targets']
        box_cls_labels = self.forward_ret_dict['box_cls_labels']
        batch_size = int(box_preds.shape[0])
        positives = box_cls_labels > 0
        reg_weights = self.forward_ret_dict['reg_weights']
        pos_normalizer = positives.sum(1, keepdim=True).float()
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)
        if isinstance(self.anchors, list):
            if self.use_multihead:
                anchors = torch.cat(
                    [anchor.permute(3, 4, 0, 1, 2, 5).contiguous().view(-1, anchor.shape[-1]) for anchor in
                     self.anchors], dim=0)
            else:
                anchors = torch.cat(self.anchors, dim=-3)
        else:
            anchors = self.anchors
        anchors = anchors.view(1, -1, anchors.shape[-1]).repeat(batch_size, 1, 1)
        box_preds = box_preds.view(batch_size, -1,
                                   box_preds.shape[-1] // self.num_anchors_per_location if not self.use_multihead else
                                   box_preds.shape[-1])
        # sin(a - b) = sinacosb-cosasinb
        box_preds_sin, reg_targets_sin = self.add_sin_difference(box_preds, box_reg_targets)
        loc_loss_src = self.reg_loss_func(box_preds_sin, reg_targets_sin, weights=reg_weights)  # [N, M]

        if box_dir_cls_preds is not None:
            dir_targets = self.get_direction_target(
                anchors, box_reg_targets,
                dir_offset=self.model_cfg.DIR_OFFSET,
                num_bins=self.model_cfg.NUM_DIR_BINS
            )

            dir_logits = box_dir_cls_preds.view(batch_size, -1, self.model_cfg.NUM_DIR_BINS)
            weights = positives.type_as(dir_logits)
            weights /= torch.clamp(weights.sum(-1, keepdim=True), min=1.0)
            dir_loss = self.dir_loss_func(dir_logits, dir_targets, weights=weights)

        return loc_loss_src, dir_loss

    def get_importance(self, cls_loss_src, loc_loss_src, dir_loss):
        cls_preds = self.forward_ret_dict['cls_preds'] # [8, 200, 176, 18]
        spatial_features_2d = self.forward_ret_dict['spatial_features_2d']
        spatial_features = spatial_features_2d.detach()
        batch_size = int(cls_preds.shape[0])

        pr_information = self.forward_ret_dict['pts_ratio_weight'].view(batch_size, 200, 176, -1) # [8, 200, 176, 6]
        cls_loss_src = cls_loss_src.view(batch_size, 200, 176, -1)

        loc_loss_src = loc_loss_src.view(batch_size, 200, 176, -1)
        dir_loss = dir_loss.view(batch_size, 200, 176, -1)

        pr_information = pr_information.permute(0, 3, 1, 2).contiguous().detach() # [8, 6, 200, 176]
        cls_loss_src = cls_loss_src.permute(0, 3, 1, 2).contiguous().detach()
        loc_loss_src = loc_loss_src.permute(0, 3, 1, 2).contiguous().detach()
        dir_loss = dir_loss.permute(0, 3, 1, 2).contiguous().detach()
        pr_information = self.conv_information(pr_information)
        cls_information = self.conv_cls_information(cls_loss_src)
        box_information = self.conv_box_information(loc_loss_src)
        dir_information = self.conv_dir_information(dir_loss)
        feature = torch.cat((spatial_features, pr_information, cls_information, box_information, dir_information), dim=1)
        importance = self.conv_importance(feature)  # [8, 6, 200, 176]
        importance = importance.permute(0, 2, 3, 1).contiguous()
        # print(importance[0,0,0,:])
        return importance

    def get_importance_only_loss(self, cls_loss_src, loc_loss_src, dir_loss):
        cls_preds = self.forward_ret_dict['cls_preds'] # [8, 200, 176, 18]
        spatial_features_2d = self.forward_ret_dict['spatial_features_2d']
        spatial_features = spatial_features_2d.detach()
        batch_size = int(cls_preds.shape[0])

        cls_loss_src = cls_loss_src.view(batch_size, 200, 176, -1)
        loc_loss_src = loc_loss_src.view(batch_size, 200, 176, -1)
        dir_loss = dir_loss.view(batch_size, 200, 176, -1)

        cls_loss_src = cls_loss_src.permute(0, 3, 1, 2).contiguous().detach()
        loc_loss_src = loc_loss_src.permute(0, 3, 1, 2).contiguous().detach()
        dir_loss = dir_loss.permute(0, 3, 1, 2).contiguous().detach()

        cls_information = self.conv_cls_information(cls_loss_src)
        box_information = self.conv_box_information(loc_loss_src)
        dir_information = self.conv_dir_information(dir_loss)
        feature = torch.cat((spatial_features, cls_information, box_information, dir_information), dim=1)
        importance = self.conv_importance_only_loss(feature)
        importance = importance.permute(0, 2, 3, 1).contiguous()
        # print(importance[0,0,0,:])
        return importance

    def get_box_reg_layer_adaptive_loss(self):
        # box_loss
        box_preds = self.forward_ret_dict['box_preds']
        # print(box_preds.shape)
        box_dir_cls_preds = self.forward_ret_dict.get('dir_cls_preds', None)
        box_reg_targets = self.forward_ret_dict['box_reg_targets']
        box_cls_labels = self.forward_ret_dict['box_cls_labels']
        # pts_ratio_weight = self.forward_ret_dict['pts_ratio_weight'] + 0.1
        importance = self.forward_ret_dict['importance']
        # print(importance.shape)
        batch_size = int(box_preds.shape[0])

        positives = box_cls_labels > 0

        reg_weights = positives.float()
        pos_normalizer = positives.sum(1, keepdim=True).float()
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)

        if isinstance(self.anchors, list):
            if self.use_multihead:
                anchors = torch.cat(
                    [anchor.permute(3, 4, 0, 1, 2, 5).contiguous().view(-1, anchor.shape[-1]) for anchor in
                     self.anchors], dim=0)
            else:
                anchors = torch.cat(self.anchors, dim=-3)
        else:
            anchors = self.anchors
        anchors = anchors.view(1, -1, anchors.shape[-1]).repeat(batch_size, 1, 1)
        # print(box_preds.shape)
        box_preds = box_preds.view(batch_size, -1,
                                   box_preds.shape[-1] // self.num_anchors_per_location if not self.use_multihead else
                                   box_preds.shape[-1])
        # reg_weights = reg_weights * importance
        # sin(a - b) = sinacosb-cosasinb
        box_preds_sin, reg_targets_sin = self.add_sin_difference(box_preds, box_reg_targets)
        loc_loss_src = self.reg_loss_func(box_preds_sin, reg_targets_sin, weights=reg_weights)  # [N, M]

        # pts_ratio_weight = pts_ratio_weight.unsqueeze(dim=-1)
        # loc_loss_src = loc_loss_src * pts_ratio_weight

        # importance and regularization
        importance = importance.view(batch_size, -1, 1)
        # loc_loss_src = loc_loss_src * importance + (1-importance) * (1-importance)
        loc_loss_src = loc_loss_src * importance - 1 * (importance - 0.5) * (importance - 0.5)

        # loc_loss_src = loc_loss_src * 2 * torch.exp(-iou_loss_src*iou_loss_src/0.25) * difficulty_factor
        loc_loss = loc_loss_src.sum() / batch_size
        loc_loss = loc_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['loc_weight']
        box_loss = loc_loss
        tb_dict = {
            'rpn_loss_loc': loc_loss.item()
        }

        if box_dir_cls_preds is not None:
            dir_targets = self.get_direction_target(
                anchors, box_reg_targets,
                dir_offset=self.model_cfg.DIR_OFFSET,
                num_bins=self.model_cfg.NUM_DIR_BINS
            )

            dir_logits = box_dir_cls_preds.view(batch_size, -1, self.model_cfg.NUM_DIR_BINS)
            weights = positives.type_as(dir_logits)
            weights /= torch.clamp(weights.sum(-1, keepdim=True), min=1.0)
            dir_loss = self.dir_loss_func(dir_logits, dir_targets, weights=weights)
            dir_loss = dir_loss.sum() / batch_size
            dir_loss = dir_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['dir_weight']
            box_loss += dir_loss
            tb_dict['rpn_loss_dir'] = dir_loss.item()

        return box_loss, tb_dict

    def get_box_reg_layer_pri_loss(self, importance):
        box_preds = self.forward_ret_dict['box_preds']
        box_dir_cls_preds = self.forward_ret_dict.get('dir_cls_preds', None)
        box_reg_targets = self.forward_ret_dict['box_reg_targets']
        box_cls_labels = self.forward_ret_dict['box_cls_labels']
        # print(importance.shape)
        batch_size = int(box_preds.shape[0])

        positives = box_cls_labels > 0

        reg_weights = positives.float()
        pos_normalizer = positives.sum(1, keepdim=True).float()
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)

        if isinstance(self.anchors, list):
            if self.use_multihead:
                anchors = torch.cat(
                    [anchor.permute(3, 4, 0, 1, 2, 5).contiguous().view(-1, anchor.shape[-1]) for anchor in
                     self.anchors], dim=0)
            else:
                anchors = torch.cat(self.anchors, dim=-3)
        else:
            anchors = self.anchors
        anchors = anchors.view(1, -1, anchors.shape[-1]).repeat(batch_size, 1, 1)
        # print(box_preds.shape)
        box_preds = box_preds.view(batch_size, -1,
                                   box_preds.shape[-1] // self.num_anchors_per_location if not self.use_multihead else
                                   box_preds.shape[-1])

        # sin(a - b) = sinacosb-cosasinb
        box_preds_sin, reg_targets_sin = self.add_sin_difference(box_preds, box_reg_targets)
        loc_loss_src = self.reg_loss_func(box_preds_sin, reg_targets_sin, weights=reg_weights)  # [N, M]
        frame_id = self.forward_ret_dict['frame_id']
        if '000049' in frame_id:
            i = np.where(frame_id == '000049')
            a = {'importance':importance[i], 'box_cls_labels':box_cls_labels[i], "box_reg_targets":box_reg_targets[i]}
            file_path = Path("/mnt/HDD2/chun/OpenPCDet/output/imp_vis/000049")
            file_name = '%s.pth' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
            torch.save(a, file_path/file_name)
            print(f'Dictionary saved to {file_path}')
        if '000112' in frame_id:
            i = np.where(frame_id == '000112')
            a = {'importance':importance[i], 'box_cls_labels':box_cls_labels[i], "box_reg_targets":box_reg_targets[i]}
            file_path = Path('/mnt/HDD2/chun/OpenPCDet/output/imp_vis/000112')
            file_name = '%s.pth' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
            torch.save(a, file_path/file_name)
            print(f'Dictionary saved to {file_path}')
        if '000120' in frame_id:
            i = np.where(frame_id == '000120')
            a = {'importance':importance[i], 'box_cls_labels':box_cls_labels[i], "box_reg_targets":box_reg_targets[i]}
            file_path = Path('/mnt/HDD2/chun/OpenPCDet/output/imp_vis/000120')
            file_name = '%s.pth' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
            torch.save(a, file_path/file_name)
            print(f'Dictionary saved to {file_path}')
        
        # for i in range(len(frame_id)):
        #     a = {'importance':importance[i], 'box_cls_labels':box_cls_labels[i]}
        #     file_path = '/mnt/HDD2/chun/OpenPCDet/output/ei_models/voxel_rcnn_ei_loss/default/imp/'
        #     torch.save(a, file_path+f'{frame_id[i]}.pth')
        # importance and regularization
        importance = importance.view(batch_size, -1, 1)
        loc_loss_src = loc_loss_src * importance

        # regularization_term = ((1 - importance[positives]) * (1 - importance[positives])).sum()
        regularization_term = ((importance[positives] - 0.5) * (importance[positives] - 0.5)).sum()
        loc_loss = (loc_loss_src.sum() - self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['rt'] * regularization_term)/ batch_size
        # loc_loss = (loc_loss_src.sum() - regularization_term)/ batch_size
        loc_loss = loc_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['loc_weight']
        box_loss = loc_loss
        tb_dict = {
            'rpn_loss_loc': loc_loss.item()
        }

        if box_dir_cls_preds is not None:
            dir_targets = self.get_direction_target(
                anchors, box_reg_targets,
                dir_offset=self.model_cfg.DIR_OFFSET,
                num_bins=self.model_cfg.NUM_DIR_BINS
            )

            dir_logits = box_dir_cls_preds.view(batch_size, -1, self.model_cfg.NUM_DIR_BINS)
            weights = positives.type_as(dir_logits)
            weights /= torch.clamp(weights.sum(-1, keepdim=True), min=1.0)
            dir_loss = self.dir_loss_func(dir_logits, dir_targets, weights=weights)
            dir_loss = dir_loss.sum() / batch_size
            dir_loss = dir_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['dir_weight']
            box_loss += dir_loss
            tb_dict['rpn_loss_dir'] = dir_loss.item()

        return box_loss, tb_dict

    def get_box_reg_layer_dr_loss(self):
        # box_loss
        box_preds = self.forward_ret_dict['box_preds']
        # print(box_preds.shape)
        box_dir_cls_preds = self.forward_ret_dict.get('dir_cls_preds', None)
        box_reg_targets = self.forward_ret_dict['box_reg_targets']
        box_cls_labels = self.forward_ret_dict['box_cls_labels']
        importance = self.forward_ret_dict['importance']
        # print(importance.shape)
        batch_size = int(box_preds.shape[0])

        positives = box_cls_labels > 0

        reg_weights = positives.float()
        pos_normalizer = positives.sum(1, keepdim=True).float()
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)

        if isinstance(self.anchors, list):
            if self.use_multihead:
                anchors = torch.cat(
                    [anchor.permute(3, 4, 0, 1, 2, 5).contiguous().view(-1, anchor.shape[-1]) for anchor in
                     self.anchors], dim=0)
            else:
                anchors = torch.cat(self.anchors, dim=-3)
        else:
            anchors = self.anchors
        anchors = anchors.view(1, -1, anchors.shape[-1]).repeat(batch_size, 1, 1)
        # print(box_preds.shape)
        box_preds = box_preds.view(batch_size, -1,
                                   box_preds.shape[-1] // self.num_anchors_per_location if not self.use_multihead else
                                   box_preds.shape[-1])

        # sin(a - b) = sinacosb-cosasinb
        box_preds_sin, reg_targets_sin = self.add_sin_difference(box_preds, box_reg_targets)
        loc_loss_src = self.reg_loss_func(box_preds_sin, reg_targets_sin, weights=reg_weights)  # [N, M]

        # importance and regularization
        importance = importance.view(batch_size, -1, 1)
        # loc_loss_src = loc_loss_src * importance + (1-importance) * (1-importance)
        loc_loss_src = loc_loss_src * importance - self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['dr_weight'] * (importance - 0.5) * (importance - 0.5)

        # loc_loss_src = loc_loss_src * 2 * torch.exp(-iou_loss_src*iou_loss_src/0.25) * difficulty_factor
        loc_loss = loc_loss_src.sum() / batch_size
        loc_loss = loc_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['loc_weight']
        box_loss = loc_loss
        tb_dict = {
            'rpn_loss_loc': loc_loss.item()
        }

        if box_dir_cls_preds is not None:
            dir_targets = self.get_direction_target(
                anchors, box_reg_targets,
                dir_offset=self.model_cfg.DIR_OFFSET,
                num_bins=self.model_cfg.NUM_DIR_BINS
            )

            dir_logits = box_dir_cls_preds.view(batch_size, -1, self.model_cfg.NUM_DIR_BINS)
            weights = positives.type_as(dir_logits)
            weights /= torch.clamp(weights.sum(-1, keepdim=True), min=1.0)
            dir_loss = self.dir_loss_func(dir_logits, dir_targets, weights=weights)
            dir_loss = dir_loss.sum() / batch_size
            dir_loss = dir_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['dir_weight']
            box_loss += dir_loss
            tb_dict['rpn_loss_dir'] = dir_loss.item()

        return box_loss, tb_dict

    def get_loss(self):
        cls_loss, tb_dict = self.get_cls_layer_loss()
        box_loss, tb_dict_box = self.get_box_reg_layer_adaptive_loss()
        tb_dict.update(tb_dict_box)
        rpn_loss = cls_loss + box_loss

        tb_dict['rpn_loss'] = rpn_loss.item()
        return rpn_loss, tb_dict

    def get_dr_loss(self):
        cls_loss, tb_dict = self.get_cls_layer_loss()
        box_loss, tb_dict_box = self.get_box_reg_layer_dr_loss()
        tb_dict.update(tb_dict_box)
        rpn_loss = cls_loss + box_loss
        tb_dict['rpn_loss'] = rpn_loss.item()
        return rpn_loss, tb_dict

        
    def get_adaptive_loss(self):
        # print("here!")
        cls_loss_src = self.get_cls_loss()
        loc_loss_src, dir_loss = self.get_box_loss()
        importance = self.get_importance(cls_loss_src, loc_loss_src, dir_loss)
        cls_loss, tb_dict = self.get_cls_layer_loss()
        box_loss, tb_dict_box = self.get_box_reg_layer_pri_loss(importance)
        tb_dict.update(tb_dict_box)
        rpn_loss = cls_loss + box_loss

        tb_dict['rpn_loss'] = rpn_loss.item()
        return rpn_loss, tb_dict

    def generate_predicted_boxes(self, batch_size, cls_preds, box_preds, dir_cls_preds=None):
        """
        Args:
            batch_size:
            cls_preds: (N, H, W, C1)
            box_preds: (N, H, W, C2)
            dir_cls_preds: (N, H, W, C3)

        Returns:
            batch_cls_preds: (B, num_boxes, num_classes)
            batch_box_preds: (B, num_boxes, 7+C)

        """
        if isinstance(self.anchors, list):
            if self.use_multihead:
                anchors = torch.cat([anchor.permute(3, 4, 0, 1, 2, 5).contiguous().view(-1, anchor.shape[-1])
                                     for anchor in self.anchors], dim=0)
            else:
                anchors = torch.cat(self.anchors, dim=-3)
        else:
            anchors = self.anchors
        num_anchors = anchors.view(-1, anchors.shape[-1]).shape[0]
        batch_anchors = anchors.view(1, -1, anchors.shape[-1]).repeat(batch_size, 1, 1)
        # print(cls_preds.shape)
        batch_cls_preds = cls_preds.view(batch_size, num_anchors, -1).float() \
            if not isinstance(cls_preds, list) else cls_preds
        # print(batch_cls_preds.shape)
        # exit()
        batch_box_preds = box_preds.view(batch_size, num_anchors, -1) if not isinstance(box_preds, list) \
            else torch.cat(box_preds, dim=1).view(batch_size, num_anchors, -1)
        batch_box_preds = self.box_coder.decode_torch(batch_box_preds, batch_anchors)

        if dir_cls_preds is not None:
            dir_offset = self.model_cfg.DIR_OFFSET
            dir_limit_offset = self.model_cfg.DIR_LIMIT_OFFSET
            dir_cls_preds = dir_cls_preds.view(batch_size, num_anchors, -1) if not isinstance(dir_cls_preds, list) \
                else torch.cat(dir_cls_preds, dim=1).view(batch_size, num_anchors, -1)
            dir_labels = torch.max(dir_cls_preds, dim=-1)[1]

            period = (2 * np.pi / self.model_cfg.NUM_DIR_BINS)
            dir_rot = common_utils.limit_period(
                batch_box_preds[..., 6] - dir_offset, dir_limit_offset, period
            )
            batch_box_preds[..., 6] = dir_rot + dir_offset + period * dir_labels.to(batch_box_preds.dtype)

        if isinstance(self.box_coder, box_coder_utils.PreviousResidualDecoder):
            batch_box_preds[..., 6] = common_utils.limit_period(
                -(batch_box_preds[..., 6] + np.pi / 2), offset=0.5, period=np.pi * 2
            )

        return batch_cls_preds, batch_box_preds

    def forward(self, **kwargs):
        raise NotImplementedError
