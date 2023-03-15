from .detector3d_template import Detector3DTemplate
import torch


class SECONDNet(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()
        self.use_svd = self.model_cfg.get("USE_SVD", False)

    def forward(self, batch_dict):
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

        loss_rpn, tb_dict = self.dense_head.get_loss()
        tb_dict = {
            'loss_rpn': loss_rpn.item(),
            **tb_dict
        }
        if self.use_svd:
            rank_loss, tb_dict_rank = self.get_rank_loss(batch_dict)
            tb_dict.update(tb_dict_rank)
            loss = loss_rpn + rank_loss
        else:
            loss = loss_rpn

        return loss, tb_dict, disp_dict

    def get_rank_loss(self, batch_dict):
        rank_losses = [[] for i in range(batch_dict['svd_masks'].size(1))]
        for spatial_feature_2d, svd_mask in zip(batch_dict['spatial_features'], batch_dict['svd_masks']):
            spatial_feature_2d = spatial_feature_2d.permute(2, 1, 0)
            svd_mask = torch.any(svd_mask, -1)
            for class_id, class_svd_mask in enumerate(svd_mask):
                class_feature = spatial_feature_2d[class_svd_mask, :]
                if class_feature.size(0) > 1:
                    _, eigenvalue, _ = torch.svd(class_feature)
                    if eigenvalue[0] != 0:
                        rank_losses[class_id].append(
                            eigenvalue[1] / eigenvalue[0])
        rank_loss = torch.tensor([torch.tensor(class_rank_loss).mean()
                                  for class_rank_loss in rank_losses]).sum() * 0.25
        return rank_loss, {'rank_loss': rank_loss}
