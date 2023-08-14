import numpy as np
import torch
import torch.nn as nn
from torch.nn import Parameter
from .anchor_adaptive_head_template import AnchorAdaptiveHeadTemplate


class AnchorAdaptiveHead(AnchorAdaptiveHeadTemplate):
    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range,
                 predict_boxes_when_training=True, **kwargs):
        super().__init__(
            model_cfg=model_cfg, num_class=num_class, class_names=class_names, grid_size=grid_size, point_cloud_range=point_cloud_range,
            predict_boxes_when_training=predict_boxes_when_training
        )

        self.num_anchors_per_location = sum(self.num_anchors_per_location)

        self.conv_cls = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.num_class,
            kernel_size=1, bias=False
        )

        self.conv_box = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.box_coder.code_size,
            kernel_size=1
        )

        # self.conv_iou = nn.Conv2d(
        #     input_channels, self.num_anchors_per_location,
        #     kernel_size=1
        # )

        self.conv_importance = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv2d(64, self.num_anchors_per_location, kernel_size=1),
            nn.Dropout(0.2),
            nn.Sigmoid(), 
        )        
        # self.conv_importance = nn.Conv2d(
        #     input_channels, 1,
        #     kernel_size=1
        # )

        if self.model_cfg.get('USE_DIRECTION_CLASSIFIER', None) is not None:
            self.conv_dir_cls = nn.Conv2d(
                input_channels,
                self.num_anchors_per_location * self.model_cfg.NUM_DIR_BINS,
                kernel_size=1
            )
        else:
            self.conv_dir_cls = None
        self.init_weights()

    def init_weights(self):
        pi = 0.01
        # nn.init.constant_(self.conv_cls.bias, -np.log((1 - pi) / pi))
        # nn.init.xavier_uniform(self.conv_cls)
        nn.init.normal_(self.conv_box.weight, mean=0, std=0.001)

    def forward(self, data_dict):
        spatial_features_2d = data_dict['spatial_features_2d']
        # print(spatial_features_2d.shape)
        # cls_preds = self.conv_cls(spatial_features_2d)

        # weight_normalization
        # with torch.no_grad():
        #     df_features_2d = spatial_features_2d / torch.norm(spatial_features_2d, dim=1, keepdim=True)
        #     self.conv_cls.weight.div_(torch.norm(self.conv_cls.weight, dim = 1, keepdim=True))
        #     difficulty_factor = (1 - self.conv_cls(nn.functional.normalize(df_features_2d)))/2

        cls_preds = self.conv_cls(spatial_features_2d)
        box_preds = self.conv_box(spatial_features_2d)
        # iou_preds = self.conv_iou(spatial_features_2d)
        feature = spatial_features_2d.detach()
        importance = self.conv_importance(feature)

        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
        # iou_preds = iou_preds.permute(0, 2, 3, 1).contiguous()
        importance = importance.permute(0, 2, 3, 1).contiguous()

        self.forward_ret_dict['cls_preds'] = cls_preds
        self.forward_ret_dict['box_preds'] = box_preds
        self.forward_ret_dict['importance'] = importance
        # self.forward_ret_dict['iou_preds'] = iou_preds
        # self.forward_ret_dict['difficulty_factor'] = difficulty_factor
        # self.forward_ret_dict['fn'] = fn

        if self.conv_dir_cls is not None:
            dir_cls_preds = self.conv_dir_cls(spatial_features_2d)
            dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
            self.forward_ret_dict['dir_cls_preds'] = dir_cls_preds
        else:
            dir_cls_preds = None

        if self.training:
            targets_dict = self.assign_targets(
                gt_boxes=data_dict['gt_boxes']
            )
            self.forward_ret_dict.update(targets_dict)

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
