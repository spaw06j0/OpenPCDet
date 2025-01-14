from .detector3d_template import Detector3DTemplate
from .PartA2_net import PartA2Net
from .point_rcnn import PointRCNN
from .pointpillar import PointPillar
from .pv_rcnn import PVRCNN
from .second_net import SECONDNet
from .second_ei import SECONDNetEI
from .second_net_iou import SECONDNetIoU
from .caddn import CaDDN
from .voxel_rcnn import VoxelRCNN
from .voxel_rcnn_adaloss import VoxelRCNN_ADALOSS
from .voxel_rcnn_combine import VoxelRCNN_COMBINE
from .voxel_rcnn_iou_loss import VoxelRCNN_IOU_loss
from .voxel_rcnn_pr import VoxelRCNN_PR
from .voxel_rcnn_ei import VoxelRCNN_EI
from .voxel_rcnn_selective import VoxelRCNN_Selective
from .voxel_rcnn_dr import VoxelRCNN_DR
from .voxel_rcnn_df_dr import VoxelRCNN_DFDR

from .centerpoint import CenterPoint
from .pv_rcnn_plusplus import PVRCNNPlusPlus
from .mppnet import MPPNet
from .mppnet_e2e import MPPNetE2E
from .pillarnet import PillarNet

__all__ = {
    'Detector3DTemplate': Detector3DTemplate,
    'SECONDNet': SECONDNet,
    'SECONDNetEI': SECONDNetEI,
    'PartA2Net': PartA2Net,
    'PVRCNN': PVRCNN,
    'PointPillar': PointPillar,
    'PointRCNN': PointRCNN,
    'SECONDNetIoU': SECONDNetIoU,
    'CaDDN': CaDDN,
    'VoxelRCNN': VoxelRCNN,
    'CenterPoint': CenterPoint,

    'PVRCNNPlusPlus': PVRCNNPlusPlus,
    'VoxelRCNNADALOSS': VoxelRCNN_ADALOSS,
    'VoxelRCNN_COMBINE': VoxelRCNN_COMBINE,
    'VoxelRCNN_IOU_LOSS': VoxelRCNN_IOU_loss,
    'VoxelRCNN_PR': VoxelRCNN_PR,
    'VoxelRCNN_DR': VoxelRCNN_DR,
    'VoxelRCNN_EI': VoxelRCNN_EI,
    'VoxelRCNN_DFDR': VoxelRCNN_DFDR,
    
    'VoxelRCNN_Selective':VoxelRCNN_Selective,

    'PillarNet': PillarNet,
    'PVRCNNPlusPlus': PVRCNNPlusPlus,
    'MPPNet': MPPNet,
    'MPPNetE2E': MPPNetE2E,
    'PillarNet': PillarNet

}


def build_detector(model_cfg, num_class, dataset):
    model = __all__[model_cfg.NAME](
        model_cfg=model_cfg, num_class=num_class, dataset=dataset
    )

    return model
