import torch
import torch.nn.functional as F

from ...utils import box_coder_utils, box_utils, loss_utils, common_utils
from .point_head_template import PointHeadTemplate
from ...ops.roiaware_pool3d import roiaware_pool3d_utils

class PointHeadBox3DSSD(PointHeadTemplate):
    """
    A simple point-based segmentation head, which are used for PointRCNN.
    Reference Paper: https://arxiv.org/abs/1812.04244
    PointRCNN: 3D Object Proposal Generation and Detection from Point Cloud
    """
    def __init__(self, num_class, input_channels, model_cfg, predict_boxes_when_training=False, **kwargs):
        super().__init__(model_cfg=model_cfg, num_class=num_class)
        self.predict_boxes_when_training = predict_boxes_when_training

        target_cfg = self.model_cfg.TARGET_CONFIG
        self.box_coder = getattr(box_coder_utils, target_cfg.BOX_CODER)(
            **target_cfg.BOX_CODER_CONFIG
        )

        self.cls_center_layers = self.make_fc_layers(
            fc_cfg=self.model_cfg.CLS_FC,
            input_channels=512,
            output_channels=num_class
        )
        self.box_center_layers = self.make_fc_layers(
            fc_cfg=self.model_cfg.REG_FC,
            input_channels=512,
            output_channels=self.box_coder.code_size
        )

    def assign_stack_targets(self, points, gt_boxes, extend_gt_boxes=None,
                            ret_box_labels=False, ret_part_labels=False,
                            set_ignore_flag=True, use_ball_constraint=False, central_radius=2.0):
        """
        Args:
            points: (N1 + N2 + N3 + ..., 4) [bs_idx, x, y, z]
            gt_boxes: (B, M, 8)
            extend_gt_boxes: [B, M, 8]
            ret_box_labels:
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
        point_part_labels = gt_boxes.new_zeros((points.shape[0], 3)) if ret_part_labels else None
        gt_boxes_of_fg_points = []
        # print('in assign_stack_targets')
        for k in range(batch_size):
            bs_mask = (bs_idx == k)
            points_single = points[bs_mask][:, 1:4]
            point_cls_labels_single = point_cls_labels.new_zeros(bs_mask.sum())
            box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
                points_single.unsqueeze(dim=0), gt_boxes[k:k + 1, :, 0:7].contiguous()
            ).long().squeeze(dim=0)
            # print(f'gt box shape= {gt_boxes.shape}')
            # print(f'box idx of pts = {box_idxs_of_pts.shape}')
            
            box_fg_flag = (box_idxs_of_pts >= 0)
            # print(f'box fg flag = {box_fg_flag[:20]}')
            if set_ignore_flag:
                extend_box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
                    points_single.unsqueeze(dim=0), extend_gt_boxes[k:k+1, :, 0:7].contiguous()
                ).long().squeeze(dim=0)
                fg_flag = box_fg_flag
                ignore_flag = fg_flag ^ (extend_box_idxs_of_pts >= 0)
                # print(f'extend_box_idxs_of_pts >= 0 {extend_box_idxs_of_pts >= 0}')
                # print(f'ignore flag {ignore_flag[:20]}')
                point_cls_labels_single[ignore_flag] = -1
                # print(f'point_cls_labels_single {point_cls_labels_single[:20]}')
            elif use_ball_constraint:
                box_centers = gt_boxes[k][box_idxs_of_pts][:, 0:3].clone()
                box_centers[:, 2] += gt_boxes[k][box_idxs_of_pts][:, 5] / 2
                ball_flag = ((box_centers - points_single).norm(dim=1) < central_radius)
                fg_flag = box_fg_flag & ball_flag
            else:
                raise NotImplementedError

            gt_box_of_fg_points = gt_boxes[k][box_idxs_of_pts[fg_flag]]
            # print(f'box_idxs_of_pts[fg_flag] {box_idxs_of_pts[fg_flag].shape}')
            # print(f'box_idxs_of_pts[fg_flag] {box_idxs_of_pts[fg_flag]}')
            # print(f'gt_box_of_fg_points {gt_box_of_fg_points.shape}')
            # print(f'gt_box_of_fg_points {gt_box_of_fg_points}')
            point_cls_labels_single[fg_flag] = 1 if self.num_class == 1 else gt_box_of_fg_points[:, -1].long()
            # print(f'point_cls_labels_single[fg_flag] {point_cls_labels_single[fg_flag].shape}')
            # print(f'point_cls_labels_single[fg_flag] {point_cls_labels_single[fg_flag]}')
            point_cls_labels[bs_mask] = point_cls_labels_single
            gt_boxes_of_fg_points.append(gt_box_of_fg_points)
            # print(f'gt_box_of_fg_points {gt_box_of_fg_points.shape}')
            if ret_box_labels and gt_box_of_fg_points.shape[0] > 0:
                point_box_labels_single = point_box_labels.new_zeros((bs_mask.sum(), 8))
                fg_point_box_labels = self.box_coder.encode_torch(
                    gt_boxes=gt_box_of_fg_points[:, :-1], points=points_single[fg_flag],
                    gt_classes=gt_box_of_fg_points[:, -1].long()
                )
                point_box_labels_single[fg_flag] = fg_point_box_labels
                point_box_labels[bs_mask] = point_box_labels_single
                # print(f'bs_mask {bs_mask}')
                # print(f'gt_box_of_fg_points[:, :-1] {gt_box_of_fg_points[:, :-1].shape}')
                # print(f'gt_box_of_fg_points[:, :-1] {gt_box_of_fg_points[:, :-1]}')
                # print(f'points_single[fg_flag] {points_single[fg_flag].shape}')
                # print(f'points_single[fg_flag] {points_single[fg_flag]}')
                # print(f'fg_point_box_labels, {fg_point_box_labels}')
                # print(f'fg_point_box_labels, {fg_point_box_labels.shape}')

            if ret_part_labels:
                point_part_labels_single = point_part_labels.new_zeros((bs_mask.sum(), 3))
                transformed_points = points_single[fg_flag] - gt_box_of_fg_points[:, 0:3]
                transformed_points = common_utils.rotate_points_along_z(
                    transformed_points.view(-1, 1, 3), -gt_box_of_fg_points[:, 6]
                ).view(-1, 3)
                offset = torch.tensor([0.5, 0.5, 0.5]).view(1, 3).type_as(transformed_points)
                point_part_labels_single[fg_flag] = (transformed_points / gt_box_of_fg_points[:, 3:6]) + offset
                point_part_labels[bs_mask] = point_part_labels_single

        gt_boxes_of_fg_points = torch.cat(gt_boxes_of_fg_points, dim=0)
        targets_dict = {
            'point_cls_labels': point_cls_labels,
            'point_box_labels': point_box_labels,
            'point_part_labels': point_part_labels,
            'box_idxs_of_pts': box_idxs_of_pts,
            'gt_box_of_fg_points': gt_boxes_of_fg_points,
        }
        # print('+++'*50)
        # for thing in targets_dict:
        #     # print(thing)
        #     if thing == 'batch_size' or targets_dict[thing] is None:
        #         print(f'{thing}')
        #     else:
        #         print(f'{thing}, {targets_dict[thing].shape}')
        # print('+++'*50)
        return targets_dict

    def assign_targets(self, input_dict):
        """
        Args:
            input_dict:
                point_features: (N1 + N2 + N3 + ..., C)
                batch_size:
                point_coords: (N1 + N2 + N3 + ..., 4) [bs_idx, x, y, z]
                gt_boxes (optional): (B, M, 8)
        Returns:
            point_cls_labels: (N1 + N2 + N3 + ...), long type, 0:background, -1:ignored
            point_part_labels: (N1 + N2 + N3 + ..., 3)
        """
        # point_coords = input_dict['point_coords']
        gt_boxes = input_dict['gt_boxes']
        # assert gt_boxes.shape.__len__() == 3, 'gt_boxes.shape=%s' % str(gt_boxes.shape)
        # assert point_coords.shape.__len__() in [2], 'points.shape=%s' % str(point_coords.shape)

        batch_size = gt_boxes.shape[0]
        extend_gt_boxes = box_utils.enlarge_box3d(
            gt_boxes.view(-1, gt_boxes.shape[-1]), extra_width=self.model_cfg.TARGET_CONFIG.GT_EXTRA_WIDTH
        ).view(batch_size, -1, gt_boxes.shape[-1])

        centers = input_dict['centers'].detach()
        assert gt_boxes.shape.__len__() == 3, 'gt_boxes.shape=%s' % str(gt_boxes.shape)
        assert centers.shape.__len__() in [2], 'points.shape=%s' % str(centers.shape)
        targets_dict_center = self.assign_stack_targets(
            points=centers, gt_boxes=gt_boxes, extend_gt_boxes=extend_gt_boxes,
            set_ignore_flag=True, use_ball_constraint=False,
            ret_part_labels=False, ret_box_labels=True
        )
        targets_dict_center['center_gt_box_of_fg_points'] = targets_dict_center['gt_box_of_fg_points']
        targets_dict_center['center_cls_labels'] = targets_dict_center['point_cls_labels']
        targets_dict_center['center_box_labels'] = targets_dict_center['point_box_labels']

        targets_dict = targets_dict_center

        return targets_dict

    def get_loss(self, tb_dict=None):
        tb_dict = {} if tb_dict is None else tb_dict
        # point_loss_cls, tb_dict_1 = self.get_cls_layer_loss()
        # point_loss_box, tb_dict_2 = self.get_box_layer_loss()
        center_loss_reg, tb_dict_3 = self.get_center_reg_layer_loss()
        center_loss_cls, tb_dict_4 = self.get_center_cls_layer_loss()
        center_loss_box, tb_dict_5 = self.get_center_box_binori_layer_loss()
        corner_loss, tb_dict_6 = self.get_corner_layer_loss()

        # point_loss = point_loss_cls + point_loss_box + center_loss_reg + center_loss_cls + center_loss_box
        point_loss = center_loss_reg + center_loss_cls + center_loss_box + corner_loss
        if torch.isnan(point_loss) or torch.isinf(point_loss):
            print('sth wrong here')
        # tb_dict.update(tb_dict_1)
        # tb_dict.update(tb_dict_2)
        tb_dict.update(tb_dict_3)
        tb_dict.update(tb_dict_4)
        tb_dict.update(tb_dict_5)
        tb_dict.update(tb_dict_6)
        return point_loss, tb_dict

    def get_center_reg_layer_loss(self, tb_dict=None):
        pos_mask = self.forward_ret_dict['center_cls_labels'] > 0
        center_box_labels = self.forward_ret_dict['center_gt_box_of_fg_points'][:, 0:3]
        centers_origin = self.forward_ret_dict['centers_origin']
        ctr_offsets = self.forward_ret_dict['ctr_offsets']
        centers_pred = centers_origin + ctr_offsets
        centers_pred = centers_pred[pos_mask][:, 1:4]

        center_loss_box = F.smooth_l1_loss(
            centers_pred, center_box_labels
        )

        if tb_dict is None:
            tb_dict = {}
        tb_dict.update({'center_loss_reg': center_loss_box.item()})
        return center_loss_box, tb_dict

    def get_center_cls_layer_loss(self, tb_dict=None):
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
            cls_loss_src = cls_loss_src * cls_weights.unsqueeze(-1)
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

    def generate_center_ness_mask(self):
        pos_mask = self.forward_ret_dict['center_cls_labels'] > 0
        gt_boxes = self.forward_ret_dict['center_gt_box_of_fg_points']
        pred_boxes = self.forward_ret_dict['point_box_preds']
        pred_boxes = pred_boxes[pos_mask].clone().detach()

        offset_xyz = pred_boxes[:, 0:3] - gt_boxes[:, 0:3]
        offset_xyz_canical = common_utils.rotate_points_along_z(offset_xyz.unsqueeze(dim=1), -gt_boxes[:, 6]).squeeze(dim=1)

        template = gt_boxes.new_tensor(([1, 1, 1], [-1, -1, -1])) / 2
        margin = gt_boxes[:, None, 3:6].repeat(1, 2, 1) * template[None, :, :]
        distance = margin - offset_xyz_canical[:, None, :].repeat(1, 2, 1)
        distance[:, 1, :] = -1 * distance[:, 1, :]
        distance_min = torch.where(distance[:, 0, :] < distance[:, 1, :], distance[:, 0, :], distance[:, 1, :])
        distance_max = torch.where(distance[:, 0, :] > distance[:, 1, :], distance[:, 0, :], distance[:, 1, :])

        centerness = distance_min / distance_max
        centerness = centerness[:, 0] * centerness[:, 1] * centerness[:, 2]
        centerness = torch.clamp(centerness, min=1e-6)
        centerness = torch.pow(centerness, 1/3)

        centerness_mask = pos_mask.new_zeros(pos_mask.shape).float()
        centerness_mask[pos_mask] = centerness
        return centerness_mask

    def get_center_box_binori_layer_loss(self, tb_dict=None):
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
        point_loss_xyzwhl = point_loss_box_src.sum()

        pred_ori_bin_id = point_box_preds[:, 6:6+self.box_coder.bin_size]
        pred_ori_bin_res = point_box_preds[:, 6+self.box_coder.bin_size:]

        label_ori_bin_id = point_box_labels[:, 6]
        label_ori_bin_res = point_box_labels[:, 7]
        criterion = torch.nn.CrossEntropyLoss(reduction='none')
        loss_ori_cls = criterion(pred_ori_bin_id.contiguous(), label_ori_bin_id.long().contiguous())
        loss_ori_cls = torch.sum(loss_ori_cls * reg_weights)

        label_id_one_hot = F.one_hot(label_ori_bin_id.long().contiguous(), self.box_coder.bin_size)
        pred_ori_bin_res = torch.sum(pred_ori_bin_res * label_id_one_hot.float(), dim=-1)
        loss_ori_reg = F.smooth_l1_loss(pred_ori_bin_res, label_ori_bin_res)
        loss_ori_reg = torch.sum(loss_ori_reg * reg_weights)

        point_loss_box = point_loss_xyzwhl + loss_ori_reg + loss_ori_cls
        loss_weights_dict = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS
        point_loss_box = point_loss_box * loss_weights_dict['point_box_weight']
        if tb_dict is None:
            tb_dict = {}
        tb_dict.update({'center_loss_box': point_loss_box.item()})
        # tb_dict.update({'center_loss_box_xyzwhl': point_loss_xyzwhl.item()})
        # tb_dict.update({'center_loss_box_ori_cls': loss_ori_cls.item()})
        # tb_dict.update({'center_loss_box_ori_reg': loss_ori_reg.item()})
        return point_loss_box, tb_dict

    def get_center_box_layer_loss(self, tb_dict=None):
        pos_mask = self.forward_ret_dict['center_cls_labels'] > 0
        point_box_labels = self.forward_ret_dict['center_box_labels']
        point_box_preds = self.forward_ret_dict['center_box_preds']

        reg_weights = pos_mask.float()
        pos_normalizer = pos_mask.sum().float()
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)

        point_loss_box_src = self.reg_loss_func(
            point_box_preds[None, ...], point_box_labels[None, ...], weights=reg_weights[None, ...]
        )
        point_loss = point_loss_box_src.sum()

        point_loss_box = point_loss
        loss_weights_dict = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS
        point_loss_box = point_loss_box * loss_weights_dict['point_box_weight']
        if tb_dict is None:
            tb_dict = {}
        tb_dict.update({'center_loss_box': point_loss_box.item()})
        return point_loss_box, tb_dict

    def get_corner_layer_loss(self, tb_dict=None):
        pos_mask = self.forward_ret_dict['center_cls_labels'] > 0
        gt_boxes = self.forward_ret_dict['center_gt_box_of_fg_points']
        pred_boxes = self.forward_ret_dict['point_box_preds']
        pred_boxes = pred_boxes[pos_mask]
        loss_corner = loss_utils.get_corner_loss_lidar(
            pred_boxes[:, 0:7],
            gt_boxes[:, 0:7]
        )
        loss_corner = loss_corner.mean()
        loss_corner = loss_corner * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['corner_weight']
        if tb_dict is None:
            tb_dict = {}
        tb_dict.update({'corner_loss_reg': loss_corner.item()})
        return loss_corner, tb_dict

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                point_features: (N1 + N2 + N3 + ..., C) or (B, N, C)
                point_features_before_fusion: (N1 + N2 + N3 + ..., C)
                point_coords: (N1 + N2 + N3 + ..., 4) [bs_idx, x, y, z]
                point_labels (optional): (N1 + N2 + N3 + ...)
                gt_boxes (optional): (B, M, 8)
        Returns:
            batch_dict:
                point_cls_scores: (N1 + N2 + N3 + ..., 1)
                point_part_offset: (N1 + N2 + N3 + ..., 3)
        """
        # if self.model_cfg.get('USE_POINT_FEATURES_BEFORE_FUSION', False):
        #     point_features = batch_dict['point_features_before_fusion']
        # else:
        #     point_features = batch_dict['point_features']
        # point_cls_preds = self.cls_layers(point_features)  # (total_points, num_class)
        # point_box_preds = self.box_layers(point_features)  # (total_points, box_code_size)
        #
        # point_cls_preds_max, _ = point_cls_preds.max(dim=-1)
        # batch_dict['point_cls_scores'] = torch.sigmoid(point_cls_preds_max)

        center_features = batch_dict['centers_features']
        center_cls_preds = self.cls_center_layers(center_features)  # (total_centers, num_class)
        center_box_preds = self.box_center_layers(center_features)  # (total_centers, box_code_size)
        center_cls_preds_max, _ = center_cls_preds.max(dim=-1)
        batch_dict['center_cls_scores'] = torch.sigmoid(center_cls_preds_max)
        # print(self.cls_center_layers)
        # print(center_cls_preds.shape)
        # print(f'center cls pred max {center_cls_preds_max.shape}')
        # print(f'center cls pred max {center_cls_preds_max[:20]}')
        # print(f'center cls pred max+sigmoid {torch.sigmoid(center_cls_preds_max).shape}')
        # print(f'center cls pred max+sigmoid {torch.sigmoid(center_cls_preds_max)[:20]}')
        # print(self.box_center_layers)
        # print(center_box_preds.shape)
        # ret_dict = {'point_cls_preds': point_cls_preds,
        #             'point_box_preds': point_box_preds,
        #             'center_cls_preds': center_cls_preds,
        #             'center_box_preds': center_box_preds,
        #             'ctr_offsets': batch_dict['ctr_offsets'],
        #             'centers': batch_dict['centers'],
        #             'centers_origin': batch_dict['centers_origin']}
        ret_dict = {'center_cls_preds': center_cls_preds,
                    'center_box_preds': center_box_preds,
                    'ctr_offsets': batch_dict['ctr_offsets'],
                    'centers': batch_dict['centers'],
                    'centers_origin': batch_dict['centers_origin']}

        if self.training:
            targets_dict = self.assign_targets(batch_dict)
            # ret_dict['point_cls_labels'] = targets_dict['point_cls_labels']
            # ret_dict['point_box_labels'] = targets_dict['point_box_labels']
            ret_dict['center_cls_labels'] = targets_dict['center_cls_labels']
            ret_dict['center_box_labels'] = targets_dict['center_box_labels']
            ret_dict['center_gt_box_of_fg_points'] = targets_dict['center_gt_box_of_fg_points']

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

            # cls_scores, cls_labels = torch.max(point_cls_preds, dim=1)
            # batch_size = batch_dict['batch_size']
            # batch_dict['batch_pred_labels'] = cls_labels.view(batch_size, -1) + 1
            # batch_dict['batch_cls_preds'] = cls_scores.view(batch_size, -1).unsqueeze(-1)
            # batch_dict['batch_box_preds'] = point_box_preds.view(batch_size, -1, 7)
            # batch_dict['cls_preds_normalized'] = False
            # batch_dict.pop('batch_index', None)

        self.forward_ret_dict = ret_dict

        return batch_dict