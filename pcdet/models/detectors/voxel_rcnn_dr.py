from .detector3d_template import Detector3DTemplate
import torch


class VoxelRCNN_DR(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()

    def forward(self, batch_dict):
        # print(batch_dict.keys())
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)
        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss(batch_dict)

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def get_training_loss(self, batch_dict):
        disp_dict = {}
        loss = 0

        loss_rpn, tb_dict = self.dense_head.get_dr_loss()
        loss_rcnn, tb_dict = self.roi_head.get_loss(tb_dict)

        loss = loss + loss_rpn + loss_rcnn

        if hasattr(self.backbone_3d, 'get_loss'):
            loss_backbone3d, tb_dict = self.backbone_3d.get_loss(tb_dict)
            loss += loss_backbone3d

        return loss, tb_dict, disp_dict

