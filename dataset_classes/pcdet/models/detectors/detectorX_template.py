import os

import torch
import torch.nn as nn

from ...ops.iou3d_nms import iou3d_nms_utils
from .. import backbones_2d, backbones_3d, dense_heads, roi_heads
from ..backbones_2d import map_to_bev
from ..backbones_3d import pfe, vfe
from ..model_utils import model_nms_utils
from .detector3d_template import Detector3DTemplate

class DetectorX_template(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg, num_class, dataset)

        # dual backbone, fusion to fuse two backbone features
        self.module_topology = [
            'vfe', 'multibackbone', 'map_to_bev_module', 'pfe',
            'backbone_2d', 'dense_head',  'point_head', 'roi_head'
        ]
    
    def build_networks(self):
        model_info_dict = {
            'module_list': [],
            'num_rawpoint_features': self.dataset.point_feature_encoder.num_point_features,
            'num_point_features': self.dataset.point_feature_encoder.num_point_features,
            'grid_size': self.dataset.grid_size,
            'point_cloud_range': self.dataset.point_cloud_range,
            'voxel_size': self.dataset.voxel_size,
            'is_attach': False
        }
        for module_name in self.module_topology:
            module, model_info_dict = getattr(self, 'build_%s' % module_name)(
                model_info_dict=model_info_dict
            )
            self.add_module(module_name, module)
        return model_info_dict['module_list']

    def build_multibackbone(self, model_info_dict):
        if self.model_cfg.get('MULTIBACKBONE', False):
            backbone_name = 'multi_' + self.model_cfg.BACKBONE_3D.NAME
            backbone_3d_module = backbones_3d.__all__[backbone_name](
                model_cfg=self.model_cfg.BACKBONE_3D,
                input_channels=model_info_dict['num_point_features'],
                grid_size=model_info_dict['grid_size'],
                voxel_size=model_info_dict['voxel_size'],
                point_cloud_range=model_info_dict['point_cloud_range']
            )
            model_info_dict['module_list'].append(backbone_3d_module)
            model_info_dict['num_point_features'] = backbone_3d_module.num_point_features
            return backbone_3d_module, model_info_dict
        else:
            return None, model_info_dict

    def load_params_from_file_singlebranch(self, filename, logger, to_cpu=False, id='backbone'):
        if not os.path.isfile(filename):
            raise FileNotFoundError

        logger.info('==> Loading parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))
        loc_type = torch.device('cpu') if to_cpu else None
        checkpoint = torch.load(filename, map_location=loc_type)
        model_state_disk = checkpoint['model_state']

        if 'version' in checkpoint:
            logger.info('==> Checkpoint trained from version: %s' % checkpoint['version'])

        update_model_state = {}
        for key, val in model_state_disk.items():
            if key in self.state_dict() and self.state_dict()[key].shape == model_state_disk[key].shape:
                if id in key:
                    # only load part of the parameters
                    update_model_state[key] = val
                    logger.info('Update weight %s: %s' % (key, str(val.shape)))

        state_dict = self.state_dict()
        state_dict.update(update_model_state)
        self.load_state_dict(state_dict)

        for key in state_dict:
            if key not in update_model_state:
                logger.info('Not updated weight %s: %s' % (key, str(state_dict[key].shape)))

        logger.info('==> Done (loaded %d/%d)' % (len(update_model_state), len(self.state_dict())))

# class multibackbone(nn.Module):
#     def __init__(self, model_cfg, model_info_dict):
#         super().__init__()

#         self.model_cfg = model_cfg
#         self.model_info_dict = model_info_dict
#         backbone_num = self.model_cfg.BACKBONE_NUM
#         self.module_list = []
#         for i in range(backbone_num):
#             backbone_3d_module = backbones_3d.__all__[self.model_cfg.BACKBONE_3D.NAME](
#                 model_cfg=self.model_cfg.BACKBONE_3D,
#                 input_channels=model_info_dict['num_point_features'],
#                 grid_size=model_info_dict['grid_size'],
#                 voxel_size=model_info_dict['voxel_size'],
#                 point_cloud_range=model_info_dict['point_cloud_range']
#             )
#             self.module_list.append(backbone_3d_module)

#     def forward(self, batch_dict):
#         reslut_dict_list = []
#         for m in self.module_list:
#             reslut_dict_list.append(m(batch_dict))

#         # combine all values
        
