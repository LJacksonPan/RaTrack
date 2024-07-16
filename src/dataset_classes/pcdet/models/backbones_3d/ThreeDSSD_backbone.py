import torch
import torch.nn as nn

from ...ops.pointnet2.pointnet2_batch import pointnet2_modules
from ...ops.pointnet2.pointnet2_stack import pointnet2_modules as pointnet2_modules_stack
from ...ops.pointnet2.pointnet2_stack import pointnet2_utils as pointnet2_utils_stack


class SSDBackbone(nn.Module):
    def __init__(self, model_cfg, input_channels, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg

        self.SA_modules = nn.ModuleList()
        channel_in = input_channels - 3
        channel_out_list = [channel_in]

        self.num_points_each_layer = []
        skip_channel_list = [input_channels - 3]

        sa_config = self.model_cfg.SA_CONFIG
        self.layer_types = self.model_cfg.SA_CONFIG.LAYER_TYPE
        self.ctr_indexes = self.model_cfg.SA_CONFIG.CTR_INDEX
        self.layer_names = self.model_cfg.SA_CONFIG.LAYER_NAME
        self.layer_inputs = self.model_cfg.SA_CONFIG.LAYER_INPUT
        self.max_translate_range = self.model_cfg.SA_CONFIG.MAX_TRANSLATE_RANGE


        for k in range(self.model_cfg.SA_CONFIG.NPOINTS.__len__()):
            channel_in = channel_out_list[self.layer_inputs[k]]
            
            if self.layer_types[k] == 'SA_Layer':
                mlps = self.model_cfg.SA_CONFIG.MLPS[k].copy()
                for idx in range(mlps.__len__()):
                    mlps[idx] = [channel_in] + mlps[idx]

                self.SA_modules.append(
                    pointnet2_modules.PointnetSAModuleMSG_SSD(
                        npoint=self.model_cfg.SA_CONFIG.NPOINTS[k],
                        radii=self.model_cfg.SA_CONFIG.RADIUS[k],
                        nsamples=self.model_cfg.SA_CONFIG.NSAMPLE[k],
                        mlps=mlps,
                        use_xyz=True,
                        out_channle=self.model_cfg.SA_CONFIG.AGGREATION_CHANNEL[k],
                        fps_type=self.model_cfg.SA_CONFIG.FPS_TYPE[k],
                        fps_range=self.model_cfg.SA_CONFIG.FPS_RANGE[k],
                        dilated_group=False,
                    )
                )

            elif self.layer_types[k] == 'Vote_Layer':
                self.SA_modules.append(pointnet2_modules.Vote_layer3DSSD(mlp_list=self.model_cfg.SA_CONFIG.MLPS[k],
                                                                    pre_channel=channel_out_list[self.layer_inputs[k]],
                                                                    max_translate_range=self.max_translate_range))

            channel_out_list.append(self.model_cfg.SA_CONFIG.AGGREATION_CHANNEL[k])

        skip_channel_list = [1, 64, 128, 256, 512]
        channel_out = 512
        
        self.num_point_features = self.model_cfg.FP_MLPS[0][-1]


    def break_up_pc(self, pc):
        batch_idx = pc[:, 0]
        xyz = pc[:, 1:4].contiguous()
        features = (pc[:, 4:].contiguous() if pc.size(-1) > 4 else None)
        return batch_idx, xyz, features

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                points: (num_points, 4 + C), [batch_idx, x, y, z, ...]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
                point_features: (N, C)
        """
        batch_size = batch_dict['batch_size']
        points = batch_dict['points']
        batch_idx, xyz, features = self.break_up_pc(points)

        xyz_batch_cnt = xyz.new_zeros(batch_size).int()
        for bs_idx in range(batch_size):
            xyz_batch_cnt[bs_idx] = (batch_idx == bs_idx).sum()

        assert xyz_batch_cnt.min() == xyz_batch_cnt.max()
        xyz = xyz.view(batch_size, -1, 3)
        features = features.view(batch_size, -1, features.shape[-1]).permute(0, 2, 1) if features is not None else None

        encoder_xyz, encoder_features = [xyz], [features]
        for i in range(len(self.SA_modules)):
            xyz_input = encoder_xyz[self.layer_inputs[i]]
            feature_input = encoder_features[self.layer_inputs[i]]
            # print(f'CURRENT LAYER name: {self.layer_names[ ci]}, ft input len{len(encoder_features)}') 
            
            if self.layer_types[i] == 'SA_Layer':
            
                ctr_xyz = None
                if self.ctr_indexes[i] != -1:
            
                    ctr_xyz = encoder_xyz[self.ctr_indexes[i]]
                li_xyz, li_features = self.SA_modules[i](xyz_input, feature_input, ctr_xyz=ctr_xyz)
            elif self.layer_types[i] == 'Vote_Layer':
            
                li_xyz, li_features, ctr_offsets = self.SA_modules[i](xyz_input, feature_input)
                centers = li_xyz
                centers_origin = xyz_input
            encoder_xyz.append(li_xyz)
            encoder_features.append(li_features)
        
        # print('all modules done:')
        # for i in range(len(encoder_xyz)):
            # print(f'idx {i:} encoder_xyz shape {encoder_xyz[i].shape}')
            # print(f'idx {i:} encoder featurs shape {encoder_features[i].shape}')
        # print(f'ctr_offsets shape {ctr_offsets.shape}')
        # print(f'bactch idx shape {batch_idx.shape}')
        # print(f'after all layers, encoder_xyz shape: {encoder_xyz.shape}')
        # print(f'after all layers, encoder_feature shape: {encoder_features.shape}')
        ctr_batch_idx = batch_idx.view(batch_size, -1)[:, :ctr_offsets.shape[1]]
        # print(f'ctr_batch_idx {ctr_batch_idx}')
        # print(f'batchsize {batch_size}')
        ctr_batch_idx = ctr_batch_idx.contiguous().view(-1)
        # print(f'ctr_batch_idx {ctr_batch_idx.shape}')
        batch_dict['ctr_offsets'] = torch.cat((ctr_batch_idx[:, None].float(), ctr_offsets.contiguous().view(-1, 3)), dim=1)
        batch_dict['centers'] = torch.cat((ctr_batch_idx[:, None].float(), centers.contiguous().view(-1, 3)), dim=1)
        batch_dict['centers_origin'] = torch.cat((ctr_batch_idx[:, None].float(), centers_origin.contiguous().view(-1, 3)), dim=1)
        center_features = encoder_features[-1].permute(0, 2, 1).contiguous().view(-1, encoder_features[-1].shape[1])
        batch_dict['centers_features'] = center_features
        batch_dict['ctr_batch_idx'] = ctr_batch_idx

        # for thing in batch_dict:
        #     # print(thing)
        #     if thing == 'batch_size':
        #         print(f'{thing}')
        #     else:
        #         print(f'{thing}, {batch_dict[thing].shape}')

        return batch_dict