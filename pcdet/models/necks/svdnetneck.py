from xml.sax.handler import property_interning_dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .dyrelu import DyReLU, DyReLUA, DyReLUB
from ...ops.iou3d_nms import iou3d_nms_utils
# from pcdet.models.necks.dyrelu import DyReLU, DyReLUA, DyReLUB
# from pcdet.ops.iou3d_nms import iou3d_nms_utils

dyrelu_dict = {"DYNAMIC_RELU": DyReLU,
               "DYNAMIC_RELUA": DyReLUA, "DYNAMIC_RELUB": DyReLUB}


class SVDNeck(nn.Module):
    def __init__(self, model_cfg, num_upsamples, upsample_filters, grid_size, voxel_size, point_cloud_range, num_classes):
        super().__init__()
        self.model_cfg = model_cfg
        self.upsample_blocks = self.model_cfg.get("UPSAMPLE_BLOCKS", [0])
        self.grid_size = torch.tensor(grid_size)
        self.voxel_size = torch.tensor(voxel_size)

        self.point_cloud_range = torch.tensor(point_cloud_range)
        # print(self.grid_size, self.voxel_size, self.point_cloud_range)
        self.num_classes = num_classes
        # DARN
        self.polynomial_num = self.model_cfg.get("POLYNOMIAL_NUM", 0)
        self.use_DARN = self.polynomial_num != 0
        # dynamic ReLU
        dynamic_relu_type = self.model_cfg.get("DYNAMIC_RELU", None)
        self.use_dyrelu = dynamic_relu_type != None

        input_channels = sum(
            [upsample_filters[idx] if idx <= num_upsamples else 0 for idx in self.upsample_blocks])

        if self.use_DARN != None:
            self.DARN = nn.Conv2d(
                input_channels, self.polynomial_num, self.model_cfg.DARN_KERNEL)
        if self.use_dyrelu != None:
            self.dyrelu = dyrelu_dict[dynamic_relu_type](
                input_channels, conv_type=self.model_cfg.DYNAMIC_RELU_CONV_TYPE)
        self.channel_range = [
            0] + [sum(upsample_filters[:idx+1])for idx in range(num_upsamples)]

        self.num_bev_features = input_channels // 2 * 3
        self.voxel_based = grid_size[-1] != 1
        if self.voxel_based:
            self.grid_size = self.grid_size // 8
            self.voxel_size = self.voxel_size * 8

    def get_svd_mask(self, data_dict):
        voxel_coords, batch_sizes, device = data_dict['voxel_coords'], data_dict[
            'batch_size'], data_dict['gt_boxes'].device
        # size : 176, 200, 5
        nx, ny, nz = self.grid_size
        # size : 432, 496, 1
        xs, ys, zs = torch.meshgrid(torch.arange(
            nx), torch.arange(ny), torch.arange(nz))
        coord = torch.cat(
            (xs.unsqueeze(3), ys.unsqueeze(3), zs.unsqueeze(3)), 3)
        coord = (coord + 0.5) * \
            self.voxel_size[None, None, None, :] + \
            self.point_cloud_range[None, None, None, :3]

        bbox_info = torch.zeros((4))
        bbox_info[:3] = self.voxel_size
        bbox_info = bbox_info.repeat(
            nx, ny, nz, 1)
        voxel_bbox = torch.cat((coord, bbox_info), 3)
        voxel_bbox = voxel_bbox.view(-1, 7)
        svd_masks = torch.zeros((batch_sizes, self.num_classes, nx, ny, nz),
                                dtype=torch.bool, device=device)
        for batch_idx in range(batch_sizes):
            spatial_map = torch.zeros(
                nx * ny * nz, dtype=torch.bool, device=device)
            this_coords = voxel_coords[voxel_coords[:, 0] == batch_idx, :]
            this_coords = this_coords // 8 if self.voxel_based else this_coords
            # nz, ny, nx
            indices = (this_coords[:, 1] * nx * ny +
                       this_coords[:, 2] * nx + this_coords[:, 3]).type(torch.long)
            spatial_map[indices] = 1
            spatial_map = spatial_map.reshape(nz, ny, nx).permute(2, 1, 0)
            gt_boxes = data_dict['gt_boxes'][batch_idx]
            for num_class in range(self.num_classes):
                gt_class_boxes = gt_boxes[torch.logical_and(
                    gt_boxes[:, 7] == num_class + 1, gt_boxes[:, 3] != 0)]
                if len(gt_class_boxes) > 0:
                    # print(voxel_bbox.size(), gt_class_boxes.size())
                    voxel_by_gt_overlap = iou3d_nms_utils.boxes_iou3d_gpu(
                        voxel_bbox.to(device), gt_class_boxes[:, :7])

                    voxel_to_gt_max, _ = torch.max(voxel_by_gt_overlap, dim=1)
                    svd_mask = (voxel_to_gt_max != 0).view(nx, ny, nz)
                    svd_masks[batch_idx, num_class, :, :,
                              :] = torch.logical_and(svd_mask, spatial_map)
        return svd_masks

    def forward(self, data_dict):
        # """
        # Args:
        #     data_dict:
        #         spatial_features_2d
        # Returns:
        # """
        spatial_features_2d = data_dict['spatial_features_2d']
        x = torch.cat([spatial_features_2d[:, self.channel_range[idx]:self.channel_range[idx+1], :, :]
                       for idx in self.upsample_blocks], dim=1)

        if self.use_DARN:
            # use x axis
            depth_index = torch.arange(
                spatial_features_2d.size(-1), dtype=torch.float).to(x.device)
            depth_poly = torch.stack(
                [torch.pow(depth_index, idx)for idx in range(1, self.polynomial_num + 1)], dim=0)[None, :, None, :]
            polynomial_coefficient = F.softmax(self.DARN(x), dim=1)
            # need to refine
            combine_ratio_map = F.sigmoid(
                torch.sum(depth_poly * polynomial_coefficient, dim=1)).unsqueeze(1)
            up2, up3 = x[:, :x.size(1)//2, :, :], x[:, x.size(1)//2:, :, :]
            distance_aware_feature = up2 * combine_ratio_map + \
                up3 * (1 - combine_ratio_map)

        if self.use_dyrelu:
            x = self.dyrelu(x)

        if self.use_DARN:
            x = torch.cat((x, distance_aware_feature), dim=1)
        data_dict['spatial_features_2d'] = x
        if self.training:
            data_dict['svd_masks'] = self.get_svd_mask(data_dict)
        return data_dict


if __name__ == "__main__":
    data_dict = {'voxel_coords': torch.tensor([[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 0, 1, 1]]), 'batch_size': 1,
                 'gt_boxes': torch.tensor([[[0.2400,  0.0800, -1.0000,  0.1600,  0.1600,  4.0000,  30.0000, 2], [0.2400,  0.2400, -1.0000,  0.1600,  0.1600,  4.0000,  0.0000, 2]]]).cuda()}
    voxel_coords, batch_sizes, device = data_dict['voxel_coords'], data_dict[
        'batch_size'], data_dict['gt_boxes'].device
    nx, ny, nz = 3, 3, 1
    voxel_size = torch.tensor([0.16, 0.16, 4])
    point_cloud_range = torch.tensor([0, 0, -3, 0.32, 0.32, 1])
    # size : 432, 496, 1
    xs, ys, zs = torch.meshgrid(torch.arange(
        nx), torch.arange(ny), torch.arange(nz))
    coord = torch.cat(
        (xs.unsqueeze(3), ys.unsqueeze(3), zs.unsqueeze(3)), 3)
    coord = (coord + 0.5) * \
        voxel_size[None, None, None, :] + \
        point_cloud_range[None, None, None, :3]

    bbox_info = torch.zeros((4))
    bbox_info[:3] = voxel_size
    bbox_info = bbox_info.repeat(
        nx, ny, nz, 1)
    voxel_bbox = torch.cat((coord, bbox_info), 3)
    voxel_bbox = voxel_bbox.view(-1, 7)
    svd_masks = torch.zeros((batch_sizes, 3, nx, ny, nz),
                            dtype=torch.bool, device=device)

    for batch_idx in range(batch_sizes):
        spatial_map = torch.zeros(
            nx * ny * nz, dtype=torch.bool, device=device)
        this_coords = voxel_coords[voxel_coords[:, 0] == batch_idx, :]
        # nz, ny, nx
        indices = (this_coords[:, 1] * nx * ny +
                   this_coords[:, 2] * nx + this_coords[:, 3]).type(torch.long)
        spatial_map[indices] = 1
        spatial_map = spatial_map.reshape(nz, ny, nx).permute(2, 1, 0)
        gt_boxes = data_dict['gt_boxes'][batch_idx]
        for num_class in range(3):
            gt_class_boxes = gt_boxes[torch.logical_and(
                gt_boxes[:, 7] == num_class, gt_boxes[:, 3] != 0)]
            if len(gt_class_boxes) > 0:
                voxel_by_gt_overlap = iou3d_nms_utils.boxes_iou3d_gpu(
                    voxel_bbox.to(device), gt_class_boxes[:, 0:7])
                voxel_to_gt_max, _ = torch.max(voxel_by_gt_overlap, dim=1)
                svd_mask = (voxel_to_gt_max != 0).view(nx, ny, nz)
                svd_masks[batch_idx, num_class, :, :,
                          :] = torch.logical_and(svd_mask, spatial_map)
    import pdb
    pdb.set_trace()
