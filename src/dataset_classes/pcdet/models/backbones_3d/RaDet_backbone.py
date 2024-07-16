import torch
import torch.nn as nn
import os # for debug
import torch
import torch.nn as nn

from ...ops.pointnet2.pointnet2_batch import pointnet2_modules
from ...ops.pointnet2.pointnet2_stack import pointnet2_modules as pointnet2_modules_stack
from ...ops.pointnet2.pointnet2_stack import pointnet2_utils as pointnet2_utils_stack


class RaDetBackbone(nn.Module):
    def __init__(self, model_cfg, input_channels, **kwargs):
        super().__init__()
        self.channel_in = input_channels

        self.model_cfg = model_cfg
        self.extractor_cfg = model_cfg.EXTRACTOR_CONFIG
        self.seg_cfg = model_cfg.SEG_CONFIG
        self.vote_cfg = model_cfg.VOTE_CONFIG
        self.agg_cfg = model_cfg.AGGREGATION_CONFIG
        self.num_class = model_cfg.num_class

        self.extractor = nn.ModuleList()
        self.seg_mlp = nn.ModuleList()
        self.offset_layer = nn.ModuleList()
        self.original_sampler = None
        self.center_sampler = None
        self.agg_module = nn.ModuleList()
        
        self.build_extractor()
        self.build_seg_branch()
        self.build_offset_branch()
        self.build_aggregation_module()

        self.num_point_features = self.agg_module.ch_out
        
    def break_up_pc(self, pc):
        batch_idx = pc[:, 0]
        xyz = pc[:, 1:4].contiguous()
        features = (pc[:, 4:].contiguous() if pc.size(-1) > 4 else None)
        return batch_idx, xyz, features
        
    def forward(self, batch_dict):
        batch_size = batch_dict['batch_size']
        points = batch_dict['points']
        batch_idx, xyz, features = self.break_up_pc(points)

        xyz_batch_cnt = xyz.new_zeros(batch_size).int()
        for bs_idx in range(batch_size):
            xyz_batch_cnt[bs_idx] = (batch_idx == bs_idx).sum()

        assert xyz_batch_cnt.min() == xyz_batch_cnt.max()
        xyz = xyz.view(batch_size, -1, 3)
        features = features.view(batch_size, -1, features.shape[-1]).permute(0, 2, 1) if features is not None else None

        extractor_xyz, extractor_features = [xyz], [features]

        for i in range(len(self.extractor)):
            xyz_input = extractor_xyz[self.extractor_cfg.LAYER_INPUT[i]]
            feature_input = extractor_features[self.extractor_cfg.LAYER_INPUT[i]]
            xyz_new, features_new = self.extractor[i](xyz_input, feature_input.contiguous())
            extractor_xyz += [xyz_new]
            extractor_features += [features_new]

        for i in range(len(self.seg_mlp)):
            if i == 0:
                seg_result = self.seg_mlp[i](extractor_features[-1])
            else:
                seg_result = self.seg_mlp[i](seg_result)
        
        # foreground extraction
        seg_temp = seg_result.transpose(1,2)
        seg_temp, _ = seg_temp.max(dim=-1)
        seg_socre = torch.sigmoid(seg_temp)
        _, fg_idx = torch.topk(seg_socre, self.fg_npoint, dim=-1)
        fg_idx = fg_idx.int()
        # fg_xyz = extractor_xyz[-1][fg_idx]
        # fg_feat = extractor_features[-1][fg_idx]
        xyz_flipped = extractor_xyz[-1].transpose(1, 2).contiguous()
        fg_xyz = pointnet2_modules.pointnet2_utils.gather_operation(xyz_flipped, fg_idx).transpose(1, 2).contiguous()
        fg_feat = pointnet2_modules.pointnet2_utils.gather_operation(extractor_features[-1], fg_idx)

        # concentration
        fg_xyz_ctr, fg_feat_ctr, fg_ctr_offset = self.offset_layer[0](fg_xyz, fg_feat)

        # sample before aggregation
        # (B, N, 3) -> (B, N/2, 3)
        if self.original_sampler is not None:
            sampled_ori_xyz, ori_idx = self.original_sampler.sample(fg_xyz, fg_feat) 
            sampled_ori_feat = pointnet2_modules.pointnet2_utils.gather_operation(
            fg_feat.contiguous(),
            ori_idx
        ).contiguous() 
        else:
            sampled_ori_xyz = fg_xyz
            sampled_ori_feat = fg_feat
        
        if self.center_sampler is not None:
            sampled_ctr_xyz, ctr_idx = self.center_sampler.sample(fg_xyz_ctr, fg_feat_ctr) 
            sampled_ctr_offset = pointnet2_modules.pointnet2_utils.gather_operation(
                fg_ctr_offset.transpose(1,2).contiguous(),
                ctr_idx
            ).transpose(1,2).contiguous()
            sampled_ctr_feat = pointnet2_modules.pointnet2_utils.gather_operation(
            fg_feat_ctr.contiguous(),
            ctr_idx
        ).contiguous()
            sampled_ctr_origin = pointnet2_modules.pointnet2_utils.gather_operation(
                fg_xyz.transpose(1,2).contiguous(),
                ctr_idx
            ).transpose(1,2).contiguous()
        else:
            sampled_ctr_xyz = fg_xyz_ctr
            sampled_ctr_feat = fg_feat_ctr
            sampled_ctr_offset = fg_ctr_offset
            sampled_ctr_origin = fg_xyz

        # aggregation
        if self.model_cfg.AGGREGATION_CONFIG.USE_ORIGINAL_POINT:
            agg_xyz = torch.cat((sampled_ctr_xyz, sampled_ori_xyz), dim=1) # (B, N/2, 3) -> (B, N, 3)
            agg_feat = torch.cat((sampled_ctr_feat, sampled_ori_feat), dim=-1) # (B, C, N/2) -> (B, C, N)
        else:
            agg_xyz = sampled_ctr_xyz
            agg_feat = sampled_ctr_feat
        agg_xyz_list = [agg_xyz]
        agg_feat_list = [agg_feat]
        for idx, layer_type in enumerate(self.agg_cfg.LAYER_TYPE):
            xyz_input = agg_xyz_list[self.agg_cfg.LAYER_INPUT[idx]]
            feat_input = agg_feat_list[self.agg_cfg.LAYER_INPUT[idx]]
            if layer_type == 'SA_Layer':
                xyz_new, feat_new = self.agg_module[i](xyz_input, feat_input, ctr_xyz=sampled_ctr_xyz)
            elif layer_type == 'PCT_Layer':
                xyz_new, feat_new = self.agg_module[i](xyz_input, feat_input, ctr_xyz=sampled_ctr_xyz)
            else:
                raise NotImplementedError
            pass
            agg_feat_list += [feat_new]
            agg_xyz_list += [xyz_new]

        def save_ctr(batch_size, fg_ctr_offset, fg_xyz_ctr, fg_xyz, batch_dict):
            ctr_batch_idx = batch_idx.view(batch_size, -1)[:, :fg_ctr_offset.shape[1]]
            ctr_batch_idx = ctr_batch_idx.contiguous().view(-1)
            fg_batch_idx = batch_idx.view(batch_size, -1)[:, :fg_xyz.shape[1]]
            fg_batch_idx = ctr_batch_idx.contiguous().view(-1)
            batch_dict['ctr_offsets'] = torch.cat((ctr_batch_idx[:, None].float(), fg_ctr_offset.contiguous().view(-1, 3)), dim=1)
            batch_dict['centers'] = torch.cat((ctr_batch_idx[:, None].float(), fg_xyz_ctr.contiguous().view(-1, 3)), dim=1)
            batch_dict['centers_origin'] = torch.cat((fg_batch_idx[:, None].float(), fg_xyz.contiguous().view(-1, 3)), dim=1)
            batch_dict['ctr_batch_idx'] = ctr_batch_idx

        if self.model_cfg.SAVE_SAMPLED:
            save_ctr(batch_size, sampled_ctr_offset, sampled_ctr_xyz, sampled_ctr_origin, batch_dict)
        else:
            save_ctr(batch_size, fg_ctr_offset, fg_xyz_ctr, sampled_ctr_origin, batch_dict)
        center_features = agg_feat_list[-1].permute(0, 2, 1).contiguous().view(-1, agg_feat_list[-1].shape[1])
        batch_dict['centers_features'] = center_features
        
        seg_transpose = seg_result.transpose(1,2).contiguous()
        pred_batch_idx = batch_idx.view(batch_size, -1)[:, :seg_transpose.shape[1]]
        save_preds = torch.cat([pred_batch_idx[...,None].float(), seg_transpose.view(batch_size, -1, seg_transpose.shape[-1])], dim=-1)
        batch_dict['seg_preds'] = save_preds
        seg_points_batch_idx = batch_idx.view(batch_size, -1)[:, :extractor_xyz[-1].shape[1]]
        seg_points_batch_idx = seg_points_batch_idx.contiguous().view(-1)
        encode_seg_points = torch.cat((seg_points_batch_idx[:, None].float(), extractor_xyz[-1].view(-1, 3)), dim=-1)
        batch_dict['seg_points'] = encode_seg_points

        return batch_dict


    def build_extractor(self):
        extractor_cfg = self.extractor_cfg
        layer_types = extractor_cfg.LAYER_TYPE
        # self.extractor_out = extractor_cfg.MLPS[-1][-1][-1]
        ch_in = self.channel_in - 3
        for idx, layer in enumerate(layer_types):
            mlps = extractor_cfg.MLPS[idx].copy()
            for mlp_idx in range(mlps.__len__()):
                mlps[mlp_idx] = [ch_in] + mlps[mlp_idx]
            ch_in = mlps[-1][-1]
            if layer == 'SA_Layer':
                self.extractor.append(
                    pointnet2_modules.PointnetSAModuleMSG_SSD(
                        npoint=extractor_cfg.NPOINTS[idx],
                        radii=extractor_cfg.RADIUS[idx],
                        nsamples=extractor_cfg.NSAMPLE[idx],
                        mlps=mlps,
                        use_xyz=True,
                        out_channle=extractor_cfg.AGGREATION_CHANNEL[idx],
                        fps_type=extractor_cfg.FPS_TYPE[idx],
                        fps_range=extractor_cfg.FPS_RANGE[idx],
                        dilated_group=False,
                    )
                )
            else:
                raise NotImplementedError

        setattr(self.extractor, 'ch_out', extractor_cfg.AGGREATION_CHANNEL[-1])
            

    def build_seg_branch(self):
        seg_cfg = self.seg_cfg
        mlp_spec = [self.extractor.ch_out] + seg_cfg.MLPS
        shared_mlps = []
        for k in range(len(mlp_spec) - 1):
            shared_mlps.extend([
                nn.Conv1d(mlp_spec[k], mlp_spec[k + 1], kernel_size=1, bias=False),
                nn.BatchNorm1d(mlp_spec[k + 1]),
                nn.ReLU()
            ])
        shared_mlps.extend([
            nn.Conv1d(mlp_spec[-1], self.num_class, kernel_size=1, bias=True)
            # nn.Sigmoid()
            ])
        self.seg_mlp.append(nn.Sequential(*shared_mlps))
        self.fg_npoint = seg_cfg.FG_NPOINTS

    def build_offset_branch(self):
        vote_cfg = self.vote_cfg
        self.offset_layer.append(
            pointnet2_modules.Vote_layer3DSSD(
                mlp_list=vote_cfg.MLPS,
                pre_channel=self.extractor.ch_out,
                max_translate_range=vote_cfg.MAX_TRANSLATE_RANGE
            )
        )
        pass

    def build_aggregation_module(self):
        agg_cfg = self.agg_cfg
        layer_type = agg_cfg.LAYER_TYPE
        ori_sampler_cfg = agg_cfg.get('ORIGINAL_POINT', None)
        ctr_sampler_cfg = agg_cfg.get('CENTER_POINT', None)
        # ori_sampler_cfg = None
        # ctr_sampler_cfg = None
        if ori_sampler_cfg is not None:
            self.original_sampler = pointnet2_modules.FPSampler(
                ori_sampler_cfg.SAMPLE_POINT,
                ori_sampler_cfg.SAMPLER
            )
        if ctr_sampler_cfg is not None:
            self.center_sampler = pointnet2_modules.FPSampler(
                ctr_sampler_cfg.SAMPLE_POINT,
                ctr_sampler_cfg.SAMPLER
            )
        ch_in = self.extractor.ch_out
        for idx, layer in enumerate(layer_type):
            mlps = agg_cfg.MLPS[idx].copy()
            for mlp_idx in range(mlps.__len__()):
                mlps[mlp_idx] = [ch_in] + mlps[mlp_idx]
            ch_in = self.extractor.ch_out
            if layer == 'SA_Layer':
                self.agg_module.append(
                    pointnet2_modules.PointnetSAModuleMSG_SSD(
                        npoint=agg_cfg.NPOINTS[idx],
                        radii=agg_cfg.RADIUS[idx],
                        nsamples=agg_cfg.NSAMPLE[idx],
                        mlps=mlps,
                        use_xyz=True,
                        out_channle=agg_cfg.AGGREATION_CHANNEL[idx],
                        fps_type=agg_cfg.FPS_TYPE[idx],
                        fps_range=agg_cfg.FPS_RANGE[idx],
                        dilated_group=False,
                    )
                )
            elif layer == 'PCT_Layer':
                self.agg_module.append(
                    pointnet2_modules.AttentiveSAModule(
                        npoint_list=agg_cfg.NPOINTS[idx],
                        radii=agg_cfg.RADIUS[idx],
                        nsamples=agg_cfg.NSAMPLE[idx],
                        mlps=mlps,
                        use_xyz=True,
                        out_channel=agg_cfg.AGGREATION_CHANNEL[idx]
                    )
                )
            else:
                raise NotImplementedError

        setattr(self.agg_module, 'ch_out', agg_cfg.AGGREATION_CHANNEL[-1])
        



class RaDetBackbonev2(nn.Module):
    def __init__(self, model_cfg, input_channels, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_class = model_cfg.num_class
        self.SA_modules = nn.ModuleList()
        channel_in = input_channels - 3
        channel_out_list = [channel_in]
        # self.fg_point = model_cfg.FG_POINT
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
            mlps = self.model_cfg.SA_CONFIG.MLPS[k].copy()
            try:
                for idx in range(mlps.__len__()):
                    mlps[idx] = [channel_in] + mlps[idx]
            except:
                pass
            if self.layer_types[k] == 'SA_Layer':
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
            elif self.layer_types[k] == 'PCT_Layer':
                self.SA_modules.append(
                    pointnet2_modules.AttentiveSAModule(
                        npoint_list=self.model_cfg.SA_CONFIG.NPOINTS[k],
                        radii=self.model_cfg.SA_CONFIG.RADIUS[k],
                        nsamples=self.model_cfg.SA_CONFIG.NSAMPLE[k],
                        mlps=mlps,
                        use_xyz=True,
                        out_channel=self.model_cfg.SA_CONFIG.AGGREATION_CHANNEL[k]
                    )
                )
            elif self.layer_types[k] == 'Seg_Layer':
                mlps = self.model_cfg.SA_CONFIG.MLPS[k].copy()
                mlps = [channel_in] + mlps
                shared_mlps = []
                for k in range(len(mlps) - 1):
                    shared_mlps.extend([
                        nn.Conv1d(mlps[k], mlps[k + 1], kernel_size=1, bias=False),
                        nn.BatchNorm1d(mlps[k + 1]),
                        nn.ReLU()
                    ])
                shared_mlps.extend([
                    nn.Conv1d(mlps[-1], self.num_class, kernel_size=1, bias=True)
                    ])
                self.SA_modules.append(nn.Sequential(*shared_mlps))
                
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
            if self.layer_types[i] == 'SA_Layer':
                ctr_xyz = None
                if self.ctr_indexes[i] != -1:
                    ctr_xyz = encoder_xyz[self.ctr_indexes[i]]
                li_xyz, li_features = self.SA_modules[i](xyz_input, feature_input, ctr_xyz=ctr_xyz)
            elif self.layer_types[i] == 'Vote_Layer':
                li_xyz, li_features, ctr_offsets = self.SA_modules[i](xyz_input, feature_input)
                centers = li_xyz
                centers_origin = xyz_input
            elif self.layer_types[i] == 'PCT_Layer':
                ctr_xyz = None
                if self.ctr_indexes[i] != -1:
                    ctr_xyz = encoder_xyz[self.ctr_indexes[i]]
                li_xyz, li_features = self.SA_modules[i](xyz_input, feature_input, ctr_xyz=ctr_xyz)
            elif self.layer_types[i] == 'Seg_Layer':
                seg_result = self.SA_modules[i](feature_input)
                seg_temp = seg_result.transpose(1,2)
                seg_temp, _ = seg_temp.max(dim=-1)
                seg_socre = torch.sigmoid(seg_temp)
                fg_npoint = self.model_cfg.SA_CONFIG.NPOINTS[i][0]
                _, fg_idx = torch.topk(seg_socre, fg_npoint, dim=-1)
                fg_idx = fg_idx.int()
                xyz_flipped_input = xyz_input.transpose(1, 2).contiguous()
                li_xyz = pointnet2_modules.pointnet2_utils.gather_operation(xyz_flipped_input, fg_idx).transpose(1, 2).contiguous()
                li_features = pointnet2_modules.pointnet2_utils.gather_operation(feature_input, fg_idx)
                seg_point_idx = i
            encoder_xyz.append(li_xyz)
            encoder_features.append(li_features)
            
        ctr_batch_idx = batch_idx.view(batch_size, -1)[:, :ctr_offsets.shape[1]]
        ctr_batch_idx = ctr_batch_idx.contiguous().view(-1)
        batch_dict['ctr_offsets'] = torch.cat((ctr_batch_idx[:, None].float(), ctr_offsets.contiguous().view(-1, 3)), dim=1)
        batch_dict['centers'] = torch.cat((ctr_batch_idx[:, None].float(), centers.contiguous().view(-1, 3)), dim=1)
        batch_dict['centers_origin'] = torch.cat((ctr_batch_idx[:, None].float(), centers_origin.contiguous().view(-1, 3)), dim=1)
        center_features = encoder_features[-1].permute(0, 2, 1).contiguous().view(-1, encoder_features[-1].shape[1])
        batch_dict['centers_features'] = center_features
        batch_dict['ctr_batch_idx'] = ctr_batch_idx
        
        seg_transpose = seg_result.transpose(1,2).contiguous()
        pred_batch_idx = batch_idx.view(batch_size, -1)[:, :seg_transpose.shape[1]]
        save_preds = torch.cat([pred_batch_idx[...,None].float(), seg_transpose.view(batch_size, -1, seg_transpose.shape[-1])], dim=-1)
        batch_dict['seg_preds'] = save_preds

        seg_points_batch_idx = batch_idx.view(batch_size, -1)[:, :encoder_xyz[seg_point_idx].shape[1]]
        seg_points_batch_idx = seg_points_batch_idx.contiguous().view(-1)
        encode_seg_points = torch.cat((seg_points_batch_idx[:, None].float(), encoder_xyz[seg_point_idx].view(-1, 3)), dim=-1)
        batch_dict['seg_points'] = encode_seg_points

        return batch_dict
        