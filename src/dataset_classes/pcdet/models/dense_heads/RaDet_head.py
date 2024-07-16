from random import sample
from turtle import forward
from sklearn.preprocessing import OneHotEncoder
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

from ...utils import box_coder_utils, box_utils, loss_utils, common_utils
from .point_head_box_3DSSD import PointHeadBox3DSSD
from ...ops.roiaware_pool3d import roiaware_pool3d_utils

class RaDet_Head(PointHeadBox3DSSD):
    '''
    Point-based detection head, used for RaDet
    '''
    def __init__(self, model_cfg, num_class, input_channels, **kwargs):
        super().__init__(model_cfg=model_cfg, num_class=num_class, input_channels=
        input_channels)


        target_cfg = self.model_cfg.TARGET_CONFIG
        self.box_coder = getattr(box_coder_utils, target_cfg.BOX_CODER)(
            **target_cfg.BOX_CODER_CONFIG
        )
        detector_dim = self.model_cfg.get('INPUT_DIM', input_channels) # for spec input_channel
        self.cls_center_layers = self.make_fc_layers(
            fc_cfg=self.model_cfg.CLS_FC,
            input_channels=detector_dim,
            output_channels=num_class
        )
        self.box_center_layers = self.make_fc_layers(
            fc_cfg=self.model_cfg.REG_FC,
            input_channels=detector_dim,
            output_channels=self.box_coder.code_size
        )

    def forward(self, batch_dict):

        center_features = batch_dict['centers_features']
        center_cls_preds = self.cls_center_layers(center_features)  # (total_centers, num_class)
        center_box_preds = self.box_center_layers(center_features)  # (total_centers, box_code_size)
        center_cls_preds_max, _ = center_cls_preds.max(dim=-1)
        batch_dict['center_cls_scores'] = torch.sigmoid(center_cls_preds_max)
        ret_dict = {'center_cls_preds': center_cls_preds,
                    'center_box_preds': center_box_preds,
                    'ctr_offsets': batch_dict['ctr_offsets'],
                    'centers': batch_dict['centers'],
                    'centers_origin': batch_dict['centers_origin'],
                    'seg_preds':batch_dict['seg_preds'],
                    'seg_points':batch_dict['seg_points']}

        if self.training:
            targets_dict = self.assign_targets(batch_dict)
            ret_dict['center_cls_labels'] = targets_dict['center_cls_labels']
            ret_dict['center_box_labels'] = targets_dict['center_box_labels']
            ret_dict['center_gt_box_of_fg_points'] = targets_dict['center_gt_box_of_fg_points']
            ret_dict['seg_labels'] = targets_dict['seg_labels']

        if not self.training or self.predict_boxes_when_training or \
                self.model_cfg.LOSS_CONFIG.CORNER_LOSS_REGULARIZATION or \
                self.model_cfg.LOSS_CONFIG.CENTERNESS_REGULARIZATION:

            point_cls_preds, point_box_preds = self.generate_predicted_boxes(
                    points=batch_dict['centers'][:, 1:4],
                    point_cls_preds=center_cls_preds, point_box_preds=center_box_preds
                )

            batch_dict['batch_cls_preds'] = point_cls_preds
            batch_dict['batch_box_preds'] = point_box_preds
            batch_dict['batch_index'] = batch_dict['ctr_batch_idx']
            batch_dict['cls_preds_normalized'] = False

            if self.model_cfg.LOSS_CONFIG.CORNER_LOSS_REGULARIZATION:
                ret_dict['point_box_preds'] = point_box_preds

        self.forward_ret_dict = ret_dict

        return batch_dict

    def build_losses(self, losses_cfg):
        super().build_losses(losses_cfg)
        self.add_module(
            'seg_loss_func',
            loss_utils.WeightedBinaryCrossEntropyLoss()
        )

    # def build_losses(self, losses_cfg):
    #     # box classification loss
    #     if losses_cfg.LOSS_CLS.startswith('WeightedBinaryCrossEntropy'):
    #         self.add_module(
    #             'box_cls_loss_func',
    #             loss_utils.WeightedBinaryCrossEntropyLoss()
    #         )
    #     elif losses_cfg.LOSS_CLS.startswith('WeightedCrossEntropy'):
    #         self.add_module(
    #             'box_cls_loss_func',
    #             loss_utils.WeightedClassificationLoss()
    #         )
    #     elif losses_cfg.LOSS_CLS.startswith('FocalLoss'):
    #         self.add_module(
    #             'box_cls_loss_func',
    #             loss_utils.SigmoidFocalClassificationLoss(
    #                 **losses_cfg.get('LOSS_CLS_CONFIG', {})
    #             )
    #         )
    #     else:
    #         raise NotImplementedError

    #     # regression loss
    #     if losses_cfg.LOSS_REG == 'WeightedSmoothL1Loss':
    #         self.add_module(
    #             'box_reg_loss_func',
    #             loss_utils.WeightedSmoothL1Loss(
    #                 code_weights=losses_cfg.LOSS_WEIGHTS.get('code_weights', None),
    #                 **losses_cfg.get('LOSS_REG_CONFIG', {})
    #             )
    #         )
    #     elif losses_cfg.LOSS_REG == 'WeightedL1Loss':
    #         self.add_module(
    #             'box_reg_loss_func',
    #             loss_utils.WeightedL1Loss(
    #                 code_weights=losses_cfg.LOSS_WEIGHTS.get('code_weights', None)
    #             )
    #         )
    #     else:
    #         raise NotImplementedError

    #     # poi segmentation loss
    #     if losses_cfg.get('LOSS_INS', None) is not None:
    #         if losses_cfg.LOSS_INS.startswith('WeightedBinaryCrossEntropy'):
    #             self.add_module(
    #                 'seg_loss_func',
    #                 loss_utils.WeightedBinaryCrossEntropyLoss()
    #             )
    #         elif losses_cfg.LOSS_INS.startswith('WeightedCrossEntropy'):
    #             self.add_module(
    #                 'seg_loss_func',
    #                 loss_utils.WeightedClassificationLoss()
    #             )
    #         elif losses_cfg.LOSS_INS.startswith('FocalLoss'):
    #             self.add_module(
    #                 'seg_loss_func',
    #                 loss_utils.SigmoidFocalClassificationLoss(
    #                     **losses_cfg.get('LOSS_CLS_CONFIG', {})
    #                 )
    #             )
    #         else:
    #             raise NotImplementedError

    # def get_loss(self, tb_dict=None):
    #     pass

    def assign_targets(self, input_dict):
        '''
        what to assign:
            1. hard segmentation label
            2. vote offset regression target
            3. POI confidence label
            3. POI class label
            4. POI box regression label

        Args:
            input_dict:
                seg_points: (B, N, 3)
                center_origin: (B, N, 4)
                gt_boxes: (B, M, 8)

        '''
        gt_boxes = input_dict['gt_boxes']
        batch_size = gt_boxes.shape[0]
        extend_gt_boxes = box_utils.enlarge_box3d(
            gt_boxes.view(-1, gt_boxes.shape[-1]), extra_width=self.model_cfg.TARGET_CONFIG.GT_EXTRA_WIDTH
        ).view(batch_size, -1, gt_boxes.shape[-1])
        centers = input_dict['centers'].detach()
        assert gt_boxes.shape.__len__() == 3, 'gt_boxes.shape=%s' % str(gt_boxes.shape)
        assert centers.shape.__len__() in [2], 'points.shape=%s' % str(centers.shape)

        # get segmentation targets

        seg_target_dict = self.assign_stack_targets(
            points=input_dict['seg_points'],
            gt_boxes=gt_boxes,
            extend_gt_boxes=extend_gt_boxes,
            ret_box_labels=False
        )

        # get center point targets
        boxes_target_dict = self.assign_stack_targets(
            points=input_dict['centers_origin'],
            gt_boxes=gt_boxes,
            extend_gt_boxes=extend_gt_boxes,
            ret_box_labels=True
        )

        # combine all the target into one dict
        targets_dict = {}
        targets_dict['seg_labels'] = seg_target_dict['point_cls_labels']
        targets_dict['center_cls_labels'] = boxes_target_dict['point_cls_labels']
        targets_dict['center_gt_box_of_fg_points'] = boxes_target_dict['gt_box_of_fg_points']
        targets_dict['center_box_labels'] = boxes_target_dict['point_box_labels']

        return targets_dict


    
    def assign_stack_targets(self, points, gt_boxes, extend_gt_boxes=None,
                            ret_box_labels=True,
                            set_ignore_flag=True, use_ball_constraint=False, central_radius=2.0):
        """
        Args:
            points: (N1 + N2 + N3 + ..., 4) [bs_idx, x, y, z]
            gt_boxes: (B, M, 8)
            extend_gt_boxes: [B, M, 8]
            ret_box_labels: get box regression label
            ret_part_labels:
            set_ignore_flag:
            use_ball_constraint:
            central_radius:
        Returns:
            point_cls_labels: (N1 + N2 + N3 + ...), long type, 0:background, -1:ignored
            point_box_labels: (N1 + N2 + N3 + ..., code_size)
        """
        assert len(points.shape) == 2 and points.shape[1] == 4, 'points.shape=%s' % str(points.shape)
        assert len(gt_boxes.shape) == 3 and gt_boxes.shape[2] == 8, 'gt_boxes.shape=%s' % str(gt_boxes.shape)
        assert extend_gt_boxes is None or len(extend_gt_boxes.shape) == 3 and extend_gt_boxes.shape[2] == 8, \
            'extend_gt_boxes.shape=%s' % str(extend_gt_boxes.shape)
        assert set_ignore_flag != use_ball_constraint, 'Choose one only!'
        batch_size = gt_boxes.shape[0]
        bs_idx = points[:, 0]
        point_cls_labels = points.new_zeros(points.shape[0]).long()
        point_box_labels = gt_boxes.new_zeros((points.shape[0], 8)) if ret_box_labels else None
        # point_part_labels = gt_boxes.new_zeros((points.shape[0], 3)) if ret_part_labels else None
        gt_boxes_of_fg_points = []
        for k in range(batch_size):
            bs_mask = (bs_idx == k)
            points_single = points[bs_mask][:, 1:4]
            point_cls_labels_single = point_cls_labels.new_zeros(bs_mask.sum())
            box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
                points_single.unsqueeze(dim=0), gt_boxes[k:k + 1, :, 0:7].contiguous()
            ).long().squeeze(dim=0)
            box_fg_flag = (box_idxs_of_pts >= 0)
            if set_ignore_flag:
                extend_box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
                    points_single.unsqueeze(dim=0), extend_gt_boxes[k:k+1, :, 0:7].contiguous()
                ).long().squeeze(dim=0)
                fg_flag = box_fg_flag
                ignore_flag = fg_flag ^ (extend_box_idxs_of_pts >= 0)
                point_cls_labels_single[ignore_flag] = -1
            elif use_ball_constraint:
                box_centers = gt_boxes[k][box_idxs_of_pts][:, 0:3].clone()
                box_centers[:, 2] += gt_boxes[k][box_idxs_of_pts][:, 5] / 2
                ball_flag = ((box_centers - points_single).norm(dim=1) < central_radius)
                fg_flag = box_fg_flag & ball_flag
            else:
                raise NotImplementedError

            gt_box_of_fg_points = gt_boxes[k][box_idxs_of_pts[fg_flag]]
            point_cls_labels_single[fg_flag] = 1 if self.num_class == 1 else gt_box_of_fg_points[:, -1].long()
            point_cls_labels[bs_mask] = point_cls_labels_single
            gt_boxes_of_fg_points.append(gt_box_of_fg_points)

            if ret_box_labels and gt_box_of_fg_points.shape[0] > 0:
                point_box_labels_single = point_box_labels.new_zeros((bs_mask.sum(), 8))
                fg_point_box_labels = self.box_coder.encode_torch(
                    gt_boxes=gt_box_of_fg_points[:, :-1], points=points_single[fg_flag],
                    gt_classes=gt_box_of_fg_points[:, -1].long()
                )
                point_box_labels_single[fg_flag] = fg_point_box_labels
                point_box_labels[bs_mask] = point_box_labels_single

        gt_boxes_of_fg_points = torch.cat(gt_boxes_of_fg_points, dim=0)
        targets_dict = {
            'point_cls_labels': point_cls_labels,
            'point_box_labels': point_box_labels,
            'box_idxs_of_pts': box_idxs_of_pts,
            'gt_box_of_fg_points': gt_boxes_of_fg_points,
        }
        return targets_dict

    def assign_stack_targets_bev(self, points, gt_boxes, extend_gt_boxes=None, ret_box_labels=False, 
        ret_part_labels=False, set_ignore_flag=True, use_ball_constraint=False, central_radius=2):
            
            assert len(points.shape) == 2 and points.shape[1] == 4, 'points.shape=%s' % str(points.shape)
            assert len(gt_boxes.shape) == 3 and gt_boxes.shape[2] == 8, 'gt_boxes.shape=%s' % str(gt_boxes.shape)
            assert extend_gt_boxes is None or len(extend_gt_boxes.shape) == 3 and extend_gt_boxes.shape[2] == 8, \
                'extend_gt_boxes.shape=%s' % str(extend_gt_boxes.shape)
            assert set_ignore_flag != use_ball_constraint, 'Choose one only!'
            batch_size = gt_boxes.shape[0]
            bs_idx = points[:, 0]
            point_cls_labels = points.new_zeros(points.shape[0]).long()
            point_box_labels = gt_boxes.new_zeros((points.shape[0], 8)) if ret_box_labels else None
            gt_boxes_of_fg_points = []
            gt_boxes_bev = deepcopy(gt_boxes)
            gt_boxes_bev[:, :, 2] = 0.5
            for k in range(batch_size):
                bs_mask = (bs_idx == k)
                points_single = points[bs_mask][:, 1:4]
                points_single_bev = deepcopy(points_single)
                points_single_bev[:,-1] = 0.5 # set all points at z = 0.5m
                point_cls_labels_single = point_cls_labels.new_zeros(bs_mask.sum())
                box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
                    points_single_bev.unsqueeze(dim=0), gt_boxes_bev[k:k + 1, :, 0:7].contiguous()
                ).long().squeeze(dim=0)
                box_fg_flag = (box_idxs_of_pts >= 0)
                if set_ignore_flag:
                    extend_box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
                        points_single_bev.unsqueeze(dim=0), extend_gt_boxes[k:k+1, :, 0:7].contiguous()
                    ).long().squeeze(dim=0)
                    fg_flag = box_fg_flag
                    ignore_flag = fg_flag ^ (extend_box_idxs_of_pts >= 0)
                    point_cls_labels_single[ignore_flag] = -1
                elif use_ball_constraint:
                    box_centers = gt_boxes[k][box_idxs_of_pts][:, 0:3].clone()
                    box_centers[:, 2] += gt_boxes[k][box_idxs_of_pts][:, 5] / 2
                    ball_flag = ((box_centers - points_single_bev).norm(dim=1) < central_radius)
                    fg_flag = box_fg_flag & ball_flag
                else:
                    raise NotImplementedError

                gt_box_of_fg_points = gt_boxes[k][box_idxs_of_pts[fg_flag]]
                point_cls_labels_single[fg_flag] = 1 if self.num_class == 1 else gt_box_of_fg_points[:, -1].long()
                point_cls_labels[bs_mask] = point_cls_labels_single
                gt_boxes_of_fg_points.append(gt_box_of_fg_points)

                if ret_box_labels and gt_box_of_fg_points.shape[0] > 0:
                    point_box_labels_single = point_box_labels.new_zeros((bs_mask.sum(), 8))
                    fg_point_box_labels = self.box_coder.encode_torch(
                        gt_boxes=gt_box_of_fg_points[:, :-1], points=points_single[fg_flag],
                        gt_classes=gt_box_of_fg_points[:, -1].long()
                    )
                    point_box_labels_single[fg_flag] = fg_point_box_labels
                    point_box_labels[bs_mask] = point_box_labels_single

            gt_boxes_of_fg_points = torch.cat(gt_boxes_of_fg_points, dim=0)
            targets_dict = {
                'point_cls_labels': point_cls_labels,
                'point_box_labels': point_box_labels,
                'box_idxs_of_pts': box_idxs_of_pts,
                'gt_box_of_fg_points': gt_boxes_of_fg_points,
            }
            return targets_dict
        

    def get_loss(self, tb_dict=None):

        tb_dict = {} if tb_dict is None else tb_dict

        if self.model_cfg.LOSS_CONFIG.EASY_SAMPLE:
            sample_rate = self.model_cfg.LOSS_CONFIG.PERCENT
            seg_loss, tb_dict_1 = self.get_seg_loss(easy_sample=True, percent=sample_rate.SEG_CLS, pos_num=sample_rate.SEG_POS_NUM)
            center_loss_reg, tb_dict_3 = self.get_center_reg_layer_loss_es(percent=sample_rate.CTR_REG)
            center_loss_cls, tb_dict_4 = self.get_center_cls_layer_loss_es(percent=sample_rate.CTR_CLS)
            center_loss_box, tb_dict_5 = self.get_center_box_binori_layer_loss_es(percent=sample_rate.BOX_REG)
            corner_loss, tb_dict_6 = self.get_corner_layer_loss_es(percent=sample_rate.COR_REG)
        else:
            seg_loss, tb_dict_1 = self.get_seg_loss()
            center_loss_reg, tb_dict_3 = self.get_center_reg_layer_loss()
            center_loss_cls, tb_dict_4 = self.get_center_cls_layer_loss()
            center_loss_box, tb_dict_5 = self.get_center_box_binori_layer_loss()
            corner_loss, tb_dict_6 = self.get_corner_layer_loss()
            
        total_loss = seg_loss + center_loss_reg + center_loss_cls + center_loss_box + corner_loss
        # total_loss =  \
        #     center_loss_reg + center_loss_cls \
        #         + center_loss_box + corner_loss
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print('sth wrong here')
        tb_dict.update(tb_dict_1)
        tb_dict.update(tb_dict_3)
        tb_dict.update(tb_dict_4)
        tb_dict.update(tb_dict_5)
        tb_dict.update(tb_dict_6)
        return total_loss, tb_dict

    def soften_label(self, batch_dict):
        # without cross modal supervision

        # with cross modal supervision
        pass

    def get_seg_loss(self, tb_dict=None, easy_sample=False, percent=0.2, pos_num=160):
        tb_dict = {} if tb_dict is None else tb_dict
        # seg_preds = self.forward_ret_dict['seg_labels'].view(-1, self.num_class)
        seg_preds = self.forward_ret_dict['seg_preds'][...,1:].view(-1, self.num_class)
        seg_labels = self.forward_ret_dict['seg_labels'].view(-1)
        positives = (seg_labels > 0)
        negative_cls_weights = (seg_labels == 0) * 1.0
        cls_weights = (negative_cls_weights + 1.0 * positives).float()
        pos_normalizer = positives.sum(dim=0).float()
        cls_weights /= torch.clamp(pos_normalizer, min=1.0)
        one_hot_targets = seg_preds.new_zeros(*list(seg_labels.shape), self.num_class + 1)
        one_hot_targets.scatter_(-1, (seg_labels * (seg_labels >= 0).long()).unsqueeze(dim=-1).long(), 1.0)
        one_hot_targets = one_hot_targets[..., 1:]
        seg_cls_loss = self.seg_loss_func(seg_preds, one_hot_targets, weights=cls_weights)

        def get_easy_sample(seg_loss, positives, pos_num=160):
            pos_cls_loss = seg_loss * positives
            neg_cls_loss = seg_loss * (~positives)
            pos_cls_loss[pos_cls_loss==0] = 100 # set ignore
            neg_cls_loss[neg_cls_loss==0] = 100
            sample_mask = seg_loss.new_zeros(seg_loss.shape)
            if positives.sum() < pos_num:
                sample_mask[positives==1] = 1
            else:
                _,idx = torch.topk(pos_cls_loss, pos_num, largest=False) # get min loss sample
                sample_mask[idx] = 1
            total_neg = seg_loss.shape[0] - positives.sum()
            neg_num = int(percent*total_neg)
            _, neg_idx = torch.topk(neg_cls_loss, neg_num, largest=False) # get min loss sample
            sample_mask[neg_idx] = 1
            return sample_mask

        if easy_sample:
            sample_mask = get_easy_sample(seg_cls_loss, positives=positives, pos_num=pos_num)
            seg_loss = seg_cls_loss * sample_mask
            seg_loss = seg_loss.sum()
        else:
            seg_loss = seg_cls_loss.sum()
        loss_weights_dict = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS
        seg_loss = seg_loss * loss_weights_dict['seg_weight']
        tb_dict.update({
            'seg_loss': seg_loss.item()
        })
        return seg_loss, tb_dict

    @staticmethod
    def get_easy_sample_general(loss, percent=0.3):
        sample_num = loss.shape[0] * percent
        _, idx = torch.topk(loss, int(sample_num), largest=False)
        mask = loss.new_zeros(loss.shape)
        mask[idx] = 1
        return mask

    def get_center_cls_layer_loss_es(self, tb_dict=None, percent=0.3):
        point_cls_labels = self.forward_ret_dict['center_cls_labels'].view(-1)
        point_cls_preds = self.forward_ret_dict['center_cls_preds'].view(-1, self.num_class)

        positives = (point_cls_labels > 0)
        negative_cls_weights = (point_cls_labels == 0) * 1.0
        cls_weights = (negative_cls_weights + 1.0 * positives).float()
        pos_normalizer = positives.sum(dim=0).float()
        cls_weights /= torch.clamp(pos_normalizer, min=1.0)

        one_hot_targets = point_cls_preds.new_zeros(*list(point_cls_labels.shape), self.num_class + 1)
        one_hot_targets.scatter_(-1, (point_cls_labels * (point_cls_labels >= 0).long()).unsqueeze(dim=-1).long(), 1.0)
        one_hot_targets = one_hot_targets[..., 1:]

        if self.model_cfg.LOSS_CONFIG.CENTERNESS_REGULARIZATION:
            centerness_mask = self.generate_center_ness_mask()
            one_hot_targets = one_hot_targets * centerness_mask.unsqueeze(-1).repeat(1, one_hot_targets.shape[1])
            cls_loss_src = loss_utils.SigmoidFocalClassificationLoss.sigmoid_cross_entropy_with_logits(point_cls_preds, one_hot_targets)
            # cls_loss_src = cls_loss_src * cls_weights.unsqueeze(-1)
            cls_loss_src = cls_loss_src.sum(dim=-1)
            sample_mask = self.get_easy_sample_general(cls_loss_src, percent=percent)
            equal_weight = 1 / (sample_mask.sum() / 1e-9)
            cls_loss_src = cls_loss_src * sample_mask * equal_weight
            pass


        else:
            cls_loss_src = self.cls_loss_func(point_cls_preds, one_hot_targets, weights=cls_weights)
        point_loss_cls = cls_loss_src.sum()

        loss_weights_dict = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS
        point_loss_cls = point_loss_cls * loss_weights_dict['point_cls_weight']
        if tb_dict is None:
            tb_dict = {}
        tb_dict.update({
            'center_loss_cls': point_loss_cls.item(),
            'center_pos_num': pos_normalizer.item()
        })
        return point_loss_cls, tb_dict

    def get_center_reg_layer_loss_es(self, tb_dict=None, percent=0.3):
        pos_mask = self.forward_ret_dict['center_cls_labels'] > 0
        center_box_labels = self.forward_ret_dict['center_gt_box_of_fg_points'][:, 0:3]
        centers_origin = self.forward_ret_dict['centers_origin']
        ctr_offsets = self.forward_ret_dict['ctr_offsets']
        centers_pred = centers_origin + ctr_offsets
        centers_pred = centers_pred[pos_mask][:, 1:4]

        center_loss_box = F.smooth_l1_loss(
            centers_pred, center_box_labels, reduction='none'
        )
        # center_loss_box = center_loss_box.mean()
        center_loss_box = center_loss_box.sum(dim=-1)
        mask = self.get_easy_sample_general(center_loss_box, percent=percent)
        center_loss_box = center_loss_box * mask
        # center_loss_box = center_loss_box.sum()
        center_loss_box = center_loss_box.sum() / (mask.sum() + 1e-9) # get mean loss
        if tb_dict is None:
            tb_dict = {}
        tb_dict.update({'center_loss_reg': center_loss_box.item()})
        return center_loss_box, tb_dict

    def get_center_box_binori_layer_loss_es(self, tb_dict=None, percent=0.3):
        pos_mask = self.forward_ret_dict['center_cls_labels'] > 0
        point_box_labels = self.forward_ret_dict['center_box_labels']
        point_box_preds = self.forward_ret_dict['center_box_preds']

        reg_weights = pos_mask.float()
        pos_normalizer = pos_mask.sum().float()
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)

        pred_box_xyzwhl = point_box_preds[:, :6]
        label_box_xyzwhl = point_box_labels[:, :6]

        point_loss_box_src = self.reg_loss_func(
            pred_box_xyzwhl[None, ...], label_box_xyzwhl[None, ...], weights=reg_weights[None, ...]
        )
        # ====================== select easy sample ======================
        # point_loss_box_src = point_loss_box_src.sum(dim=-1).view(-1)
        # mask = self.get_easy_sample_general(point_loss_box_src, percent=percent)
        # point_loss_box_src = point_loss_box_src * mask
        # point_loss_box_src = point_loss_box_src.sum() / mask.sum()
        # # ====================== select easy sample ======================

        point_loss_xyzwhl = point_loss_box_src.sum()

        pred_ori_bin_id = point_box_preds[:, 6:6+self.box_coder.bin_size]
        pred_ori_bin_res = point_box_preds[:, 6+self.box_coder.bin_size:]

        label_ori_bin_id = point_box_labels[:, 6]
        label_ori_bin_res = point_box_labels[:, 7]
        criterion = torch.nn.CrossEntropyLoss(reduction='none')
        loss_ori_cls = criterion(pred_ori_bin_id.contiguous(), label_ori_bin_id.long().contiguous())
        # # ====================== select easy sample ======================
        # mask = self.get_easy_sample_general(loss_ori_cls, percent=percent)
        # loss_ori_cls = loss_ori_cls * mask
        # loss_ori_cls = loss_ori_cls.sum() / mask.sum()
        # # ====================== select easy sample ======================
        loss_ori_cls = torch.sum(loss_ori_cls * reg_weights)

        label_id_one_hot = F.one_hot(label_ori_bin_id.long().contiguous(), self.box_coder.bin_size)
        pred_ori_bin_res = torch.sum(pred_ori_bin_res * label_id_one_hot.float(), dim=-1)
        loss_ori_reg = F.smooth_l1_loss(pred_ori_bin_res, label_ori_bin_res, reduction='none')
        # # ====================== select easy sample ======================
        # mask = self.get_easy_sample_general(loss_ori_reg, percent=percent)
        # loss_ori_reg = loss_ori_reg * mask
        # loss_ori_reg = loss_ori_reg.sum() / mask.sum()
        # # ====================== select easy sample ======================

        loss_ori_reg = torch.sum(loss_ori_reg * reg_weights)

        point_loss_box = point_loss_xyzwhl + loss_ori_reg + loss_ori_cls
        loss_weights_dict = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS
        point_loss_box = point_loss_box * loss_weights_dict['point_box_weight']
        if tb_dict is None:
            tb_dict = {}
        tb_dict.update({'center_loss_box': point_loss_box.item()})
        return point_loss_box, tb_dict

    def get_corner_layer_loss_es(self, tb_dict=None, percent=0.5):
        pos_mask = self.forward_ret_dict['center_cls_labels'] > 0
        gt_boxes = self.forward_ret_dict['center_gt_box_of_fg_points']
        pred_boxes = self.forward_ret_dict['point_box_preds']
        pred_boxes = pred_boxes[pos_mask]
        loss_corner = loss_utils.get_corner_loss_lidar(
            pred_boxes[:, 0:7],
            gt_boxes[:, 0:7]
        )
        mask = self.get_easy_sample_general(loss_corner, percent=percent)
        loss_corner = loss_corner * mask
        loss_corner = loss_corner.sum() / (mask.sum() + 1e-9)
        loss_corner = loss_corner * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['corner_weight']
        if tb_dict is None:
            tb_dict = {}
        tb_dict.update({'corner_loss_reg': loss_corner.item()})
        return loss_corner, tb_dict
