from .detector3d_template import Detector3DTemplate


class VoxelRCNN_IOU_loss(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()
        # self.adaptive_all = self.model_cfg.get("USE_SVD", False)
    def forward(self, batch_dict):
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        if self.training:
            if self.model_cfg.DENSE_HEAD.NAME == 'AnchorAdaptiveHead':
                loss, tb_dict, disp_dict = self.get_training_adaptive_loss(batch_dict)
            else:
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

        loss_rpn, tb_dict = self.dense_head.get_loss()
        # loss_rcnn, tb_dict = self.roi_head.get_adaptive_loss()
        loss_rcnn, tb_dict = self.roi_head.get_loss()
        loss = loss + loss_rpn + loss_rcnn

        if hasattr(self.backbone_3d, 'get_loss'):
            loss_backbone3d, tb_dict = self.backbone_3d.get_loss(tb_dict)
            loss += loss_backbone3d

        return loss, tb_dict, disp_dict

    def get_training_adaptive_loss(self, batch_dict):
        disp_dict = {}
        loss = 0

        loss_rpn, tb_dict = self.dense_head.get_adaptive_loss()
        if self.model_cfg.DENSE_HEAD.NAME == 'VoxelRCNNHeadADA':
            loss_rcnn, tb_dict = self.roi_head.get_adaptive_loss()
        else:
            loss_rcnn, tb_dict = self.roi_head.get_loss()
        loss = loss + loss_rpn + loss_rcnn

        if hasattr(self.backbone_3d, 'get_loss'):
            loss_backbone3d, tb_dict = self.backbone_3d.get_loss(tb_dict)
            loss += loss_backbone3d

        return loss, tb_dict, disp_dict