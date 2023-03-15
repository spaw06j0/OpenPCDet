from .detector3d_template import Detector3DTemplate
from .PartA2_net import PartA2Net
from .point_rcnn import PointRCNN
from .pointpillar import PointPillar
from .pv_rcnn import PVRCNN
from .second_net import SECONDNet
from .second_net_iou import SECONDNetIoU
from .caddn import CaDDN
from .voxel_rcnn import VoxelRCNN
from .voxel_rcnn_adaloss import VoxelRCNN_ADALOSS
from .voxel_rcnn_combine import VoxelRCNN_COMBINE
from .voxel_rcnn_iou_loss import VoxelRCNN_IOU_loss
from .centerpoint import CenterPoint
from .pv_rcnn_plusplus import PVRCNNPlusPlus

__all__ = {
    'Detector3DTemplate': Detector3DTemplate,
    'SECONDNet': SECONDNet,
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
}


def build_detector(model_cfg, num_class, dataset):
    model = __all__[model_cfg.NAME](
        model_cfg=model_cfg, num_class=num_class, dataset=dataset
    )

    return model
