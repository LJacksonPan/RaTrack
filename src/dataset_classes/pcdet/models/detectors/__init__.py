from .detector3d_template import Detector3DTemplate
from .PartA2_net import PartA2Net
from .point_rcnn import PointRCNN
from .pointpillar import PointPillar
from .pv_rcnn import PVRCNN
from .second_net import SECONDNet
from .centerpoint import CenterPoint 
from .centerpoint_rcnn import CenterPointRCNN
from .IASSD import IASSD
from .detectorX_template import DetectorX_template
from .IASSD_X import IASSD_X
from .IASSD_GAN import IASSD_GAN
from .point_3DSSD import Point3DSSD
from .RaDet import RaDet

__all__ = {
    'Detector3DTemplate': Detector3DTemplate,
    'DetectorXTemplate': DetectorX_template,
    'SECONDNet': SECONDNet,
    'PartA2Net': PartA2Net,
    'PVRCNN': PVRCNN,
    'PointPillar': PointPillar,
    'PointRCNN': PointRCNN,
    'CenterPoint': CenterPoint,
    'CenterPointRCNN': CenterPointRCNN,
    'IASSD': IASSD,
    'IASSDX': IASSD_X,
    'IASSDGAN': IASSD_GAN,
    '3DSSD': Point3DSSD,
    'RaDet': RaDet
}


def build_detector(model_cfg, num_class, dataset, tb_log=None):
    try: 
    
        model = __all__[model_cfg.NAME](
            model_cfg=model_cfg, num_class=num_class, dataset=dataset, tb_log=tb_log
        )
    except:
            model = __all__[model_cfg.NAME](
            model_cfg=model_cfg, num_class=num_class, dataset=dataset
        )
    return model
