from torchvision.ops import focal_loss

from models.utils.track4d_utils import euc_distance
from utils.model_utils import *
import torch.nn as nn
import torch.nn.functional as F

def track_4d_loss(objs1, objs2, mappings_prev, mappings_curr, mappings_inv, lbl1, lbl2, pc1, pc2, pc1_wrap, cls, gt_flow, aff_list, gt_mov_pts, gt_cls, gt_objs, objs_idx, objs_centre, pretrain=False):
    pc2 = pc2.permute(0, 2, 1)
    sf_loss = flow_loss(objs2, pc1, pc2, pc1_wrap, lbl2, mappings_inv, gt_mov_pts, gt_cls, gt_objs, gt_flow)

    trk_loss = affinity_loss(mappings_prev, mappings_curr, aff_list)
    seg_loss = motion_seg_loss(cls, gt_cls)

    if sf_loss.isnan():
        sf_loss = torch.tensor(0, dtype=torch.float32, requires_grad=True)
    if trk_loss.isnan():
        trk_loss = torch.tensor(0, dtype=torch.float32, requires_grad=True)
    if seg_loss.isnan():
        seg_loss = torch.tensor(0, dtype=torch.float32, requires_grad=True)

    total_loss = 0.5 * sf_loss + 0.5 * trk_loss + 1 * seg_loss
    if pretrain:
        total_loss = 0 + 0 + seg_loss
    items = {
        'Loss': total_loss,
        'SceneFlowLoss': sf_loss,
        'TrackingLoss': trk_loss,
        'SegLoss': seg_loss
    }
    return total_loss, items

def tracking_loss(mappings_prev, mappings_curr):
    expected = []
    actual = []
    for key_curr, val_curr in mappings_curr.items():
        if key_curr in mappings_prev:
            if val_curr == mappings_prev[key_curr]:
                expected.append(1)
                actual.append(1)
            else:
                expected.append(1)
                actual.append(0)
    expected = torch.tensor(expected).cuda()
    actual = torch.tensor(actual).cuda()
    return torch.sum(focal_loss.sigmoid_focal_loss(actual.float(), expected.float()))

def affinity_loss(mappings_prev, mappings_curr, aff_mat):
    aff_gt = []
    aff_gt_list = []
    for i in range(len(mappings_prev.keys())):
        row = []
        for j in range(len(mappings_curr.keys())):
            if i >= len(mappings_prev.keys()) or j >= len(mappings_curr.keys()):
                row.append(0)
                aff_gt_list.append(0)
                continue
            m = list(mappings_prev.keys())[i]
            n = list(mappings_curr.keys())[j]
            if m == n:
                row.append(1)
                aff_gt_list.append(1)
            else:
                row.append(0)
                aff_gt_list.append(0)
        aff_gt.append(row)
    aff_gt = torch.tensor(aff_gt).cuda()
    aff_gt_list = torch.tensor(aff_gt_list).cuda()
    if len(mappings_prev) == 0 or len(mappings_curr) == 0:
        return torch.tensor(0)

    return F.binary_cross_entropy(aff_mat.float(), aff_gt_list.float())

def obj_flow_loss(objs_gt, objs, mapping_inv):
    total_sc_loss = 0
    for obj_id, obj in objs.items():
        if obj_id in mapping_inv:
            if mapping_inv[obj_id] in objs_gt:
                obj_gt = objs_gt[mapping_inv[obj_id]]
                dist = obj_points_loss(obj[:, 3:6], obj_gt)
                total_sc_loss += dist
    return total_sc_loss


def flow_loss(pred_objs, pc1, gt_scene, pc1_wrap, labels, mappings_inv, gt_mov_pts, gt_cls, gt_objs, gt_flow):
    # mappings, mappings_inv = map_gt_objects(labels, pred_objs)
    sc_loss = ((pc1_wrap - gt_flow).pow(2).sum(dim=1)).sqrt()
    sc_loss = torch.mean(sc_loss, dim=1)
    return sc_loss[0]


def GIoU_loss(objs, objs_prev):
    pass


def sinkhorn_loss(A):
    pass


def flow_loss_temp(gt_scene, pc1_wrap):
    # mappings, mappings_inv = map_gt_objects(labels, pred_objs)
    sc_loss = 0
    for i in range(gt_scene.size(2)):
        dsts = euc_distance(gt_scene[:, :, i].unsqueeze(2), pc1_wrap[:, :3, :])
        best_i = torch.argmin(dsts, dim=1)
        sc_loss += dsts[:, best_i[0]]

    return sc_loss


def obj_points_loss(obj1, obj2):
    obj_matching_loss = 0
    for i in range(obj2.size(2)):
        dists = []
        # best_dist = float('inf')
        for j in range(obj1.size(2)):
            dist = (obj2[:, :, i] - obj1[:, :, j]).pow(2).sum().sqrt()
            dists.append(dist)
        dists = torch.tensor(dists)
        best_match = torch.argmin(dists)
        obj_matching_loss += dists[best_match]
    return obj_matching_loss

def motion_seg_loss(pred_cls, gt_cls):

    true_mask = gt_cls == True
    false_mask = gt_cls == False

    pred_cls = pred_cls.float().cuda()
    gt_cls = gt_cls.float().cuda().unsqueeze(0)

    bec = nn.BCELoss(reduction='mean')

    loss_pos = bec(pred_cls[:, true_mask],
                                         gt_cls[:, true_mask],
                                        #  alpha=0.25,
                                        #  gamma=2,
                                        #  reduction='mean'
                                         )
    loss_neg = bec(pred_cls[:, false_mask],
                                         gt_cls[:, false_mask],
                                        #  alpha=0.25,
                                        #  gamma=2,
                                        #  reduction='mean'
                                         )
    return 0.4 * loss_pos + 0.6 * loss_neg