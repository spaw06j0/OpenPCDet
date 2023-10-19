import numpy as np
import torch
import torch.nn as nn
from torch.nn import Parameter
from .anchor_head_pr_importance_template import AnchorHeadPRITemplate

import datetime

class AnchorHeadPRI(AnchorHeadPRITemplate):
    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range,
                 predict_boxes_when_training=True, **kwargs):
        super().__init__(
            model_cfg=model_cfg, num_class=num_class, class_names=class_names, grid_size=grid_size, point_cloud_range=point_cloud_range,
            predict_boxes_when_training=predict_boxes_when_training
        )

        self.num_anchors_per_location = sum(self.num_anchors_per_location)

        # original:
        self.conv_cls = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.num_class,
            kernel_size=1
        )

        self.conv_box = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.box_coder.code_size,
            kernel_size=1
        )

        if self.model_cfg.get('USE_DIRECTION_CLASSIFIER', None) is not None:
            self.conv_dir_cls = nn.Conv2d(
                input_channels,
                self.num_anchors_per_location * self.model_cfg.NUM_DIR_BINS,
                kernel_size=1
            )
        else:
            self.conv_dir_cls = None
            
        # self.conv_information = nn.Conv2d(self.num_anchors_per_location, 24, kernel_size=1)

        # self.conv_importance = nn.Sequential(
        #     nn.Conv2d(input_channels+24, 64, kernel_size=1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),
        #     nn.Dropout(0.2),
        #     nn.Conv2d(64, self.num_anchors_per_location, kernel_size=1),
        #     nn.Dropout(0.2),
        #     nn.Sigmoid(), 
        # )
        self.init_weights()

    def init_weights(self):
        pi = 0.01
        # nn.init.constant_(self.conv_cls.bias, -np.log((1 - pi) / pi))
        # nn.init.xavier_uniform(self.conv_cls)
        nn.init.normal_(self.conv_box.weight, mean=0, std=0.001)

    def forward(self, data_dict):
        # print(data_dict.keys())
        # frame_id
        spatial_features_2d = data_dict['spatial_features_2d'] # [8, 256, 200, 176]
        batch_size = data_dict['batch_size']
        # print(self.num_anchors_per_location)
        cls_preds = self.conv_cls(spatial_features_2d)
        box_preds = self.conv_box(spatial_features_2d)

        if self.training:
            targets_dict = self.assign_targets(
                gt_boxes=data_dict['gt_boxes'],
                pts_ratio=data_dict['pts_ratio']
            )
            self.forward_ret_dict.update(targets_dict)
        
            # pts_ratio_weight.shape [8, 211200]
            # spatial_features = spatial_features_2d.detach()
            # pr_information = targets_dict['pts_ratio_weight'].view(batch_size, 200, 176, -1) # [8, 200, 176, 6]
            # pr_information = pr_information.permute(0, 3, 1, 2).contiguous().detach()
            # pr_information = self.conv_information(pr_information)
            # feature = torch.cat((spatial_features, pr_information), dim=1)
            # importance = self.conv_importance(feature)
            # importance = importance.permute(0, 2, 3, 1).contiguous()
            # self.forward_ret_dict['importance'] = importance

        # [8, 200, 176, 18] -> [8, 211200, 3]
        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]

        self.forward_ret_dict['cls_preds'] = cls_preds
        self.forward_ret_dict['box_preds'] = box_preds
        self.forward_ret_dict['spatial_features_2d'] = spatial_features_2d    
        # self.forward_ret_dict['norms'] = norms
        frame_id = data_dict['frame_id']
        self.forward_ret_dict['frame_id'] = frame_id
        if self.conv_dir_cls is not None:
            dir_cls_preds = self.conv_dir_cls(spatial_features_2d)
            dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
            self.forward_ret_dict['dir_cls_preds'] = dir_cls_preds
        else:
            dir_cls_preds = None
        # print(data_dict.keys())
        # exit()
        
        if not self.training or self.predict_boxes_when_training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=data_dict['batch_size'],
                cls_preds=cls_preds, box_preds=box_preds, dir_cls_preds=dir_cls_preds
            )
            data_dict['batch_cls_preds'] = batch_cls_preds
            data_dict['batch_box_preds'] = batch_box_preds
            data_dict['cls_preds_normalized'] = False
        # print(data_dict.keys())
        # print(data_dict['batch_cls_preds'].shape)
        return data_dict
