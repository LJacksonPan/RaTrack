
from .detector3d_template import Detector3DTemplate
import torch
import torch.nn as nn
from ...ops.pointnet2.pointnet2_batch import domain_fusion as df
import os
from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils import box_coder_utils, box_utils, loss_utils, common_utils
import ipdb
from ...vis_tools.vis_tools import *
import numpy as np



class IASSD_GAN(Detector3DTemplate):
    #TODO self.transfer seems useless? 
    def __init__(self, model_cfg, num_class, dataset, tb_log=None):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        
        print('Start building IA-SSD-GAN') 
        # Tensorboard + Debug 
        self.tb_log = tb_log
        self.debug = self.model_cfg.get('DEBUG', False)

        # Network Modules: 
        # Used modules: bb_3d, feature_aug, point_head
        self.module_topology = [
            'vfe', 'backbone_3d', 'map_to_bev_module', 'pfe',
            'backbone_2d', 'feature_aug', 'dense_head',  'point_head', 'roi_head'
        ]

        self.module_list = self.build_networks()
        
        # Attach Network
        self.attach_module_topology = ['backbone_3d']
        self.attach_model_cfg = model_cfg.get('ATTACH_NETWORK')
        self.attach_model_cfg.BACKBONE_3D['num_class'] = num_class        
        self.attach_model = None if model_cfg.get('DISABLE_ATTACH') else self.build_attach_network()[0] # idx becos it reutrns bb in list 
        
        # Shared Head
        self.shared_module_topology = ['point_head']
        shared_head = self.build_shared_head()
        self.shared_head = None if len(shared_head) == 0 else shared_head[0] 
        
        # ? Not sure if used
        self.cross_over_cfg = self.model_cfg.CROSS_OVER 
        
        # Feature augmentation for feature detection
        self.use_feature_aug = model_cfg.get('USE_FEAT_AUG', False)
        
        print('Done building IA-SSD-GAN')   
        
        # Visualization settings
        self.vis_cnt = 0 
        self.vis_interval = 100 # in unit batch
        self.class_names = model_cfg.get('CLASS_NAMES', None)
        


# ===========================================================================
    def load_ckpt_to_attach(self, filename, logger, to_cpu=False):
        """
        not used..?
        """
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
            attach_key = 'attach_' + key
            if attach_key in self.state_dict() and self.state_dict()[attach_key].shape == model_state_disk[key].shape:
                update_model_state[attach_key] = val
                logger.info('Update weight %s: %s' % (key, str(val.shape)))

        state_dict = self.state_dict()
        state_dict.update(update_model_state)
        self.load_state_dict(state_dict)

        # for key in state_dict:
        #     if key not in update_model_state:
        #         logger.info('Not updated weight %s: %s' % (key, str(state_dict[key].shape)))

        logger.info('==> Done (loaded %d/%d)' % (len(update_model_state), len(self.state_dict())))

    def freeze_attach(self, logger):
        for name, param in self.named_parameters():
            if 'attach' in name:
                param.requires_grad = False
                logger.info('Freeze param in ' + name)

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
            # print("XX"*150)
            # print(module)
            self.add_module(module_name, module)
        return model_info_dict['module_list']

    def build_feature_aug(self, model_info_dict, custom_cfg=None):
        feature_aug_cfg = self.model_cfg.get('FEAT_AUG', None)
        if feature_aug_cfg is None:
            return None, model_info_dict

        if self.model_cfg.POINT_HEAD.get('USE_POINT_FEATURES_BEFORE_FUSION', False):
            num_point_features = model_info_dict['num_point_features_before_fusion']
        else:
            num_point_features = model_info_dict['num_point_features'] # ====> this was changed by backbone3d

        feature_aug_cfg = self.model_cfg.get('FEAT_AUG', None) if custom_cfg is None else custom_cfg

        # model_info_dict['num_point_features'] =====> change this for detection head

        feature_aug_module = FeatureAug(feature_aug_cfg, num_point_features)
        model_info_dict['num_point_features'] = feature_aug_module.channel_out
        model_info_dict['module_list'].append(feature_aug_module)
        return feature_aug_module, model_info_dict

    def build_attach_network(self):
        num_feats = self.attach_model_cfg.get('NUM_POINT_FEATURES',4) 
        # print(f"ATTACH NUM FEATS {num_feats}")
        model_info_dict = {
            'module_list': [],
            'num_rawpoint_features': num_feats,
            'num_point_features': num_feats,
            'grid_size': self.dataset.grid_size,
            'point_cloud_range': self.dataset.point_cloud_range,
            'voxel_size': self.dataset.voxel_size,
            'is_attach': True
        }
        for module_name in self.attach_module_topology:
            module, model_info_dict = getattr(self, 'build_%s' % module_name)(
                model_info_dict=model_info_dict
            )
            full_module_name = 'attach_' + module_name
            self.add_module(full_module_name, module)
        return model_info_dict['module_list']
    
    def build_shared_head(self):
        num_feats = self.attach_model_cfg.get('NUM_POINT_FEATURES',4) 
        # print(f"SHARED HEAD NUM FEATS {num_feats}")
        model_info_dict = {
            'module_list': [],
            'num_rawpoint_features': num_feats,
            'num_point_features': self.model_cfg.SHARED_HEAD.NUM_POINT_FEATURES,
            'grid_size': self.dataset.grid_size,
            'point_cloud_range': self.dataset.point_cloud_range,
            'voxel_size': self.dataset.voxel_size,
            'is_attach': False
        }

        for module_name in self.shared_module_topology:
            self.model_cfg.SHARED_HEAD['DEBUG'] = self.debug
            module, model_info_dict = getattr(self, 'build_%s' % module_name)(
                model_info_dict=model_info_dict,
                custom_cfg=self.model_cfg.SHARED_HEAD
            )
            full_module_name = 'shared_' + module_name
            self.add_module(full_module_name, module)
        return model_info_dict['module_list']

# ===========================================================================

    def print_shapes(self, batch_dict):
        keys = batch_dict.keys()
        print('='*80)
        for k in batch_dict.keys():
            if isinstance(batch_dict[k],int):
                print(f'{k} (int): {batch_dict[k]}')
            elif isinstance(batch_dict[k],dict):
                dict2 = batch_dict[k]
                print('-'*30+f'inner dict: {k}'+'-'*30)
                for K in dict2.keys():
                    if isinstance(dict2[K],int):
                        print(f'{K}: {dict2[K]}')
                    elif isinstance(dict2[K],list):
                        print(f'{K} (len): {len(dict2[K])} , {[len(tensor) for tensor in dict2[K]]}')
                    elif dict2[K] is None:
                        print(f'{K}: IS NONE')
                    else:
                        print(f'{K}: {dict2[K].shape}')
                print('-'*60)
            elif isinstance(batch_dict[k],list):
                print(f'{k} (len): {len(batch_dict[k])}, {[len(tensor) for tensor in batch_dict[k]]}')
            elif batch_dict[k] is None:
                print(f'{k}: is NONE')    
            else:
                print(f'{k}: {batch_dict[k].shape}')
            
    def get_transfer_feature(self, batch_dict):
        attach_dict = {
            'points': torch.clone(batch_dict['attach']),
            'batch_size': batch_dict['batch_size'],
            'frame_id': batch_dict['frame_id']
        }

        attach_dict = self.attach_model(attach_dict)

        return attach_dict

    def forward(self, batch_dict):
        """
        batch_dict: dict = ['points', 'frame_id', 'attach', 'gt_boxes', 'use_lead_xyz', 'image_shape', 'batch_size']
        """
        if self.use_feature_aug & self.training:
            if self.attach_model is not None:
                transfer_dict = self.get_transfer_feature(batch_dict)
                # self.print_shapes(batch_dict)
                # print('TRANSFER DICT')
                # self.print_shapes(transfer_dict)
                batch_dict['att'] = transfer_dict
        for cur_module in self.module_list:
            
            batch_dict = cur_module(batch_dict)
            # self.print_shapes(batch_dict)
            # print('='*150)
        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()

            # get feat transfer loss
            transfer_loss, shared_tb_dict, transfer_disp_dict = self.get_transfer_loss(batch_dict)
            disp_dict['det_loss'] = loss.item()
            disp_dict['matching_loss'] = tb_dict['matching_loss']
            loss = (transfer_loss + loss) / 2
            tb_keys = ['center_loss_cls', 'center_loss_box', 'corner_loss_reg']

            ret_dict = {
                'loss': loss,
                'gan_loss': transfer_loss
            }
            disp_dict['gan_loss'] = transfer_loss.item()
            disp_dict['tatal_loss'] = loss.item()



            shared_det_list = []
            det_list = []
            for k in tb_keys:
                shared_det_list += [shared_tb_dict[k]]
                det_list += [tb_dict[k]]
            disp_dict['shared_box_loss'] = sum(shared_det_list)
            disp_dict['box_loss'] = sum(det_list)
            tb_dict['shared_box_loss'] = sum(shared_det_list)
            tb_dict['box_loss'] = sum(det_list)

            return ret_dict, tb_dict, disp_dict
        else:
            
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            if self.debug:
                loss, tb_dict, disp_dict = self.get_training_loss()
                transfer_loss = self.get_transfer_loss(batch_dict)
                batch_dict['sa_ins_labels'] = tb_dict['sa_ins_labels']
                # selected lidar points
                # selected lidar points labels
                # radar points labels
                # radar points classification
                pass
            recall_dicts['batch_dict'] = batch_dict
            return pred_dicts, recall_dicts

    def get_transfer_loss(self, batch_dict):

        attach_dict = self.get_transfer_feature(batch_dict)
        transfer_dict = {
            'att': attach_dict,
            'batch': batch_dict
        }
        radar_shared_feat = batch_dict['radar_shared']
        share_head_dict = {}
        # print(f'RAD SHARE FEAT{radar_shared_feat.shape}')
        for key in attach_dict.keys():
            if key in batch_dict:
                share_head_dict[key] = batch_dict[key]
        share_head_dict.pop('centers_features')
        share_head_dict['gt_boxes'] = batch_dict['gt_boxes']
        _, c, _ = radar_shared_feat.shape
        # print(f'RAD SHARE FEAT{radar_shared_feat.shape}')
        share_head_dict['centers_features'] = radar_shared_feat.permute(0,2,1).contiguous().view(-1, c)
        share_head_dict = self.shared_head(share_head_dict)
        share_head_loss, shared_tb_dict = self.shared_head.get_loss(share_head_dict)
        disp_dict = {
            'share_det_loss': share_head_loss.item()
        }
        return share_head_loss, shared_tb_dict, disp_dict
        
    
    def get_training_loss(self):
        disp_dict = {}
        loss_point, tb_dict = self.point_head.get_loss()
        loss_match, new_tb_dict = self.feature_aug.get_loss()

        # get loss from shared_det_head
        tb_dict.update(new_tb_dict)
        loss = (2/3)*loss_point + (1/3)*loss_match

        return loss, tb_dict, disp_dict

class CrossOverBlock(nn.Module):
    def __init__(self, mlps, channel_in, relu=True, bn=True):
        '''
        mlps: list of output channel
        channel_in: input channel
        relu: whether to use relu
        bn: whether to use bn
        '''
        super().__init__()
        self.last_channel = mlps[-1]
        self.relu = relu
        self.bn = bn
        in_chs = [channel_in]
        self.mlp = []
        for idx, ch_out in enumerate(mlps):
            self.mlp.append(
                mlp_bn_relu(
                    in_chs[idx],
                    ch_out,
                    relu=self.relu,
                    bn=self.bn
                )
            )
            in_chs += [ch_out]
        self.mlp = nn.Sequential(*self.mlp)

    def forward(self, x):
        return self.mlp(x)
        

class mlp_bn_relu(nn.Module):
    def __init__(self, ch_in, ch_out, relu=True, bn=True):
        super().__init__()
        
        temp_list = [nn.Conv1d(ch_in, ch_out, 1, 1)]
        
        if bn:
            temp_list += [nn.BatchNorm1d(ch_out)]

        if relu:
            temp_list += [nn.ReLU()]

        self.net = nn.Sequential(*temp_list)

    def forward(self, x):
        return self.net(x)

class FeatureAug(nn.Module):
    '''module for cross modal feature generation'''

    def __init__(self, model_cfg, input_channels, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        
        self.mlps = self.model_cfg.get('MLPS', [448, 384, 320, 256])
        self.bn = self.model_cfg.get('BN', True)
        self.relu = self.model_cfg.get('RELU', True)
        
        self.channel_in = input_channels
        self.channel_out = self.channel_in + self.mlps[-1]
        
        self.lidar_shared_mlp = CrossOverBlock(self.mlps, self.channel_in, relu=self.relu, bn=self.bn)
        self.radar_shared_mlp = CrossOverBlock(self.mlps, self.channel_in, relu=self.relu, bn=self.bn)
        
        self.forward_dict = {}
    
    def forward(self, x):
        batch_dict = x
        bat_feats = batch_dict['encoder_features']
        radar_feat = bat_feats[-1] # BxCxN

        shared_radar = self.radar_shared_mlp(radar_feat)
        
        if self.training:
            attach_dict = x['att']

            att_feats = attach_dict['encoder_features']
            lidar_feat = att_feats[-1]

            att_xyzs = attach_dict['encoder_xyz']
            bat_xyzs = batch_dict['encoder_xyz']            
            
            lidar_xyz = att_xyzs[-1]
            radar_xyz = bat_xyzs[-1]

            if torch.isnan(radar_xyz).sum() > 0:
                raise RuntimeError('Nan occurs in domain cross over!')
        
            shared_lidar = self.lidar_shared_mlp(lidar_feat) # [B, C, N]

            self.forward_dict['batch_size'] = batch_dict['batch_size']
            self.forward_dict['lidar_original'] = lidar_feat
            self.forward_dict['radar_original'] = radar_feat
            self.forward_dict['radar_shared'] = shared_radar
            self.forward_dict['lidar_shared'] = shared_lidar
            self.forward_dict['lidar_xyz'] = lidar_xyz
            self.forward_dict['radar_xyz'] = radar_xyz
            self.forward_dict['lidar_centers'] = attach_dict['centers']
            self.forward_dict['radar_centers'] = batch_dict['centers']

            batch_dict['radar_shared'] = shared_radar
            # print("shared_radar")
            # print(shared_radar.shape)
        
                    
        # cat augmented feature to the original feature 'centers_features'
        centers_features = batch_dict['centers_features'] # (B*N) x C
        final_radar = shared_radar.permute(0, 2, 1).contiguous().view(-1, shared_radar.shape[1])
        new_feat = torch.cat((centers_features, final_radar), dim=1)
        batch_dict['centers_features'] = new_feat
        return batch_dict
        
    def break_up_pc(self, pc):
        batch_idx = pc[:, 0]
        xyz = pc[:, 1:4].contiguous()
        features = (pc[:, 4:].contiguous() if pc.size(-1) > 4 else None)
        return batch_idx, xyz, features

    def get_loss(self):
        batch_size = self.forward_dict['batch_size']
        lidar_center = self.forward_dict['lidar_centers']
        _, lidar_center, _ = self.break_up_pc(lidar_center)
        radar_center = self.forward_dict['radar_centers']
        _, radar_center, _ = self.break_up_pc(radar_center)
        # xyz = xyz.view(batch_size, -1, 3)
        lidar_center = lidar_center.view(batch_size, -1, 3)
        radar_center = radar_center.view(batch_size, -1, 3)

        lidar_shared_feat = self.forward_dict['lidar_shared'].permute(0,2,1) # [B, C, N] -> [B, N, C]
        radar_shared_feat = self.forward_dict['radar_shared'].permute(0,2,1)
        
        
        lidar_xyz = lidar_center
        radar_xyz = radar_center    
        
        # matching loss
        self_idx, _ = df.ball_point(1, radar_xyz, radar_xyz, 1)
        cross_idx, mask = df.ball_point(1, lidar_xyz, radar_xyz, 1) # this should get the one and only result
        mask = mask.unsqueeze(-1).unsqueeze(-1)
        self_feat = df.index_points_group(radar_shared_feat, self_idx)
        cross_feat = df.index_points_group(lidar_shared_feat, cross_idx)
        self_coord = df.index_points_group(radar_xyz, self_idx)
        cross_coord = df.index_points_group(lidar_xyz, cross_idx)
        self_pts = torch.cat((self_coord, self_feat), dim=-1) * mask
        cross_pts = torch.cat((cross_coord, cross_feat), dim=-1) * mask
        
        if torch.isnan(self_pts).sum() > 0:
            print('idx error in self_pts')
            raise RuntimeError
        elif torch.isnan(cross_pts).sum() > 0:
            print('idx error in cross_pts')
            raise RuntimeError
        matching_loss = nn.functional.mse_loss(self_pts, cross_pts, reduction='sum')
        total_num = mask.sum() + 1e-7
        matching_loss = matching_loss / total_num

        tb_dict = {
            'matching_loss': matching_loss.item()
        }
        return matching_loss, tb_dict


# ===================================VISUALIZATION====================================

def draw_match(lidar, radar, mask, draw_match=True, \
    bbox=None, c_names=None):
    # draw the first batch
    lidar_pts = lidar[0, :, :].detach().cpu().numpy().reshape([-1, 3])
    radar_pts = radar[0, :, :].detach().cpu().numpy().reshape([-1, 3])
    mask_match = mask[0, :, :].detach().cpu().numpy().reshape([-1])
    match_idx = np.where(mask_match == 1)[0]
    l_pts = lidar_pts[match_idx, :2]
    r_pts = radar_pts[match_idx, :2]
    fig, ax1, ax2 = draw_two_pointcloud(lidar_pts, radar_pts, 'lidar', 'radar')
    if draw_match:
        for i in range(mask_match.sum()):
            xyA = l_pts[i, :]
            xyB = r_pts[i, :]
            draw_cross_line(xyA, xyB, fig, ax1, ax2)

    if bbox is not None:
        raw_bbox = bbox[0, :, :].cpu().numpy()
        rec_list = boxes2rec(raw_bbox, c_names)
        for rec in rec_list:
            ax1.add_patch(rec)
        rec_list = boxes2rec(raw_bbox, c_names)
        for rec in rec_list:
            ax2.add_patch(rec)
    return fig

def draw_match_in_one(lidar, radar, mask, tb_log, draw_match=True, \
        bbox=None, c_names=None, title='match'):
    # draw the first batch
    lidar_pts = lidar[0, :, :].detach().cpu().numpy().reshape([-1, 3])
    radar_pts = radar[0, :, :].detach().cpu().numpy().reshape([-1, 3])
    mask_match = mask[0, :, :].detach().cpu().numpy().reshape([-1])
    match_idx = np.where(mask_match == 1)[0]
    l_pts = lidar_pts[match_idx, :2]
    r_pts = radar_pts[match_idx, :2]
    fig = plt.figure(dpi=150)
    ax = fig.add_subplot()

    ax.scatter(l_pts[:,0], l_pts[:,1], c='cyan', s=10)
    ax.scatter(r_pts[:,0], r_pts[:,1], c='gold', s=10)
    drawBEV(ax, lidar_pts, radar_pts, None, None, title)
    if draw_match:
        for i in range(mask_match.sum()):
            xyA = l_pts[i, :]
            xyB = r_pts[i, :]
            # draw_cross_line(xyA, xyB, fig, ax, ax)
            x_values = (xyA[0], xyB[0])
            y_values = (xyA[1], xyB[1])
            ax.plot(x_values, y_values, '-', color='orange')
    if bbox is not None:
        try:
            raw_bbox = bbox[0, :, :].cpu().numpy()
        except:
            raw_bbox = bbox[0, :, :]
        rec_list = boxes2rec(raw_bbox, c_names)
        for rec in rec_list:
            ax.add_patch(rec)
    ax.set_xlim(-0, 75)
    ax.set_ylim(-30, 30)
    # if tb_log is not None:
    tb_log.add_figure('one_fig_match_' + title, fig)
    plt.close()
    
def draw_overlay(pcd1, pcd2, tb_log, title='two_pointcloud', bbox=None, c_names=None):
    pts1 = pcd1[0,:,:].detach().cpu().numpy()
    pts2 = pcd2[0,:,:].detach().cpu().numpy()
    fig = plt.figure(dpi=150)
    ax = fig.add_subplot()
    drawBEV(ax, pts1, pts2, None, None, title, set_legend=False)
    if bbox is not None:
        try:
            raw_bbox = bbox[0, :, :].cpu().numpy()
        except:
            raw_bbox = bbox[0, :, :]
        rec_list = boxes2rec(raw_bbox, c_names)
        for rec in rec_list:
            ax.add_patch(rec)
    ax.set_xlim(-0, 75)
    ax.set_ylim(-30, 30)
    tb_log.add_figure(title, fig)
    plt.close()
