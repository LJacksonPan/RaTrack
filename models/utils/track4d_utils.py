import math
import os
import random
from dataset_classes.track_kitti_3d import load_detection

import torch
import torch.nn.functional as F
import numpy as np
from scipy import interpolate
from matplotlib import pyplot as plt

from dataset_classes.kitti.box import Box3D
import open3d as o3d
from scipy.spatial.transform import Rotation as R

from vod.frame.transformations import homogeneous_transformation


def obj_centre(obj):
    """
    3 N
    """
    return torch.mean(obj, dim=2)


def euc_distance(p1, p2):
    return (torch.tensor(p1).cuda() - torch.tensor(p2).cuda()).pow(2).sum(dim=1).sqrt()


def find_nearest_obj_coord(obj_coord, obj_centres):
    if len(obj_centres) == 0:
        return None
    objs_coords = obj_centres
    dist = euc_distance(obj_coord, objs_coords)
    nearest_id = torch.argmin(dist, dim=1)
    return nearest_id


def find_nearest_obj(obj, objects):
    obj_coord = obj_centre(obj)
    nearest_id = 0
    nearest_dist = float('inf')
    for obj_id, obj_prev in objects.items():
        # TODO: use sum of point distances instead of object centre
        obj_prev_coord = obj_centre(obj_prev)
        dist = euc_distance(obj_coord, obj_prev_coord)
        if dist < nearest_dist:
            nearest_id = obj_id
            nearest_dist = dist
    return nearest_id


def iou_points(obj_a, obj_b):
    obj_a = np.array(obj_a[:, 3:6])
    obj_b = np.array(obj_b)
    
    total_p = obj_a.shape[0] + obj_b.shape[0]
    common_p = 0

    for i in range(obj_a.shape[0]):
        p1 = obj_a[i]
        for j in range(obj_b.shape[0]):
            p2 = obj_b[j]
            dist = np.linalg.norm(p1 - p2)
            if dist < 0.00001:
                common_p += 1
                continue
    
    if total_p - common_p == 0:
        return 0

    return common_p / (total_p - common_p)


def map_gt_objects(gt_obj_centres, gt_objs1, objects):
    # label_id: id
    mapping = dict()
    mapping_inv = dict()

    gt_centres = [c.unsqueeze(2) for _, c in gt_obj_centres.items()]
    if len(gt_centres) == 0:
        return dict(), dict()
    # Use IoU instead of distance in the future
    gt_used = []
    for key, obj in objects.items():

        best_iou = 0
        best_gt = -1
        for gt_key, gt in gt_objs1.items():
            iou = iou_points(obj[0].detach().cpu().numpy().T, gt[0].detach().cpu().numpy().T)
            if iou > best_iou:
                best_iou = iou
                best_gt = gt_key
        if best_gt == -1 or best_gt in gt_used:
            mapping[-random.randint(99999, 9999999999999999)] = key
            mapping_inv[key] = -random.randint(99999, 9999999999999999)
            continue

        mapping[best_gt] = key
        mapping_inv[key] = gt_key
        gt_used.append(best_gt)
    return mapping, mapping_inv


def filter_object_points(args, labels, pc, transforms):
    # pc = pc.permute(0, 2, 1)
    boxes = dict()
    boxes_centre = dict()
    for key, label in labels.items():
        box = Box3D(x=label.x,
                    y=label.y,
                    z=label.z,
                    h=label.h,
                    w=label.w,
                    l=label.l,
                    ry=label.ry)
        boxes[label.id] = box
        boxes[label.id] = get_bbx_param([label.h, label.w, label.l, label.x, label.y, label.z, label.ry], transforms, 'radar')
        boxes_centre[label.id] = boxes[label.id].get_center()
    cls = torch.full((pc.shape[2],), False).cuda()
    pc_fil = []
    objs = dict()
    objs_idx = dict()
    objs_centre = dict()
    cls_obj_id = torch.full((pc.shape[2],), -1).cuda()
    for obj_id, box in boxes.items():
        pc_ = o3d.utility.Vector3dVector(pc[0].cpu().T)
        in_box_idx = box.get_point_indices_within_bounding_box(pc_)
        if len(in_box_idx) == 0:
            continue
        cls[in_box_idx] = True
        cls_obj_id[in_box_idx] = obj_id
        objs[obj_id] = torch.tensor(pc[:, :, in_box_idx]).cuda()
        objs_centre[obj_id] = obj_centre(objs[obj_id])
        objs_idx[obj_id] = torch.tensor(in_box_idx).cuda()
        if len(pc_fil) == 0:
            pc_fil = torch.tensor(pc[:, :, in_box_idx]).cuda()
        else:
            pc_fil = torch.cat((pc_fil,  torch.tensor(pc[:, :, in_box_idx]).cuda()), dim=2)

    # Combine riders and their bicycles
    ids_to_pop = list()
    for obj_id, centre1 in objs_centre.items():
        if labels[obj_id].type == 'rider':
            nearest_id = -1
            best_dist = 9999999999999
            for obj_id_search, centre2 in objs_centre.items():
                if obj_id_search == obj_id:
                    continue
                dist = euc_distance(centre1, centre2)
                if dist < best_dist:
                    best_dist = dist
                    nearest_id = obj_id_search
            if nearest_id == -1:
                continue
            ids_to_pop.append(obj_id)
            objs[nearest_id] = torch.cat((objs[obj_id], objs[nearest_id]), dim=2)
            objs[nearest_id] = torch.unique(objs[nearest_id], dim=2)

    # remove GT objects with less than 2 points
    for idx, obj in objs.items():
        if obj.size(2) < args.min_obj_points:
            ids_to_pop.append(idx)
    objs_combined = dict()
    objs_idx_combined = dict()
    objs_centre_combined = dict()
    for obj_id, obj in objs.items():
        if obj_id not in ids_to_pop:
            objs_combined[obj_id] = objs[obj_id]
            objs_idx_combined[obj_id] = objs_idx[obj_id]
            objs_centre_combined[obj_id] = objs_centre[obj_id]

    if len(pc_fil) == 0:
        pc_fil = None

    return pc_fil, torch.tensor(cls), objs, objs_idx, objs_centre, cls_obj_id, boxes, objs_combined, objs_idx_combined, objs_centre_combined


def filter_object_points_box(labels, pc, det_boxes):
    pc = pc.permute(0, 2, 1).detach().cpu().numpy()
    gt_boxes = dict()
    gt_boxes_centre = dict()
    for key, label in labels.items():
        box = Box3D(x=label.x,
                    y=label.y,
                    z=label.z,
                    h=label.h,
                    w=label.w,
                    l=label.l,
                    ry=label.ry)
        gt_boxes[label.id] = box
        rot = o3d.geometry.get_rotation_matrix_from_axis_angle([0, label.ry, 0])
        gt_boxes[label.id] = o3d.geometry.OrientedBoundingBox(np.array(Box3D.box2corners3d_camcoord(box).mean(axis=0)),
                                                           rot,
                                                           np.array([label.l, label.h, label.w]))
        gt_boxes_centre[label.id] = Box3D.box2corners3d_camcoord(box).mean(axis=0)
    boxes = dict()
    for det_box in det_boxes:
        box = Box3D(x=det_box.x,
                    y=det_box.y,
                    z=det_box.z,
                    h=det_box.h,
                    w=det_box.w,
                    l=det_box.l,
                    ry=det_box.ry)
        rot = o3d.geometry.get_rotation_matrix_from_axis_angle([0, box.ry, 0])
        box_centre = Box3D.box2corners3d_camcoord(box).mean(axis=0)
        gt_idx = find_nearest_obj_coord(box_centre, torch.tensor(list(gt_boxes_centre.values())).unsqueeze(0))
        gt_id = list(gt_boxes_centre.keys())[gt_idx]
        boxes[gt_id] = o3d.geometry.OrientedBoundingBox(np.array(Box3D.box2corners3d_camcoord(box).mean(axis=0)),
                                                        rot,
                                                        np.array([box.l, box.h, box.w]))
    cls = torch.full((pc.shape[2],), False).cuda()
    pc_fil = []
    objs = dict()
    objs_idx = dict()
    objs_centre = dict()
    cls_obj_id = torch.full((pc.shape[2],), -1).cuda()
    for obj_id, box in boxes.items():
        pc_ = o3d.utility.Vector3dVector(pc[0].T)
        in_box_idx = box.get_point_indices_within_bounding_box(pc_)
        if len(in_box_idx) == 0:
            continue
        cls[in_box_idx] = True
        cls_obj_id[in_box_idx] = obj_id
        objs[obj_id] = torch.tensor(pc[:, :, in_box_idx]).cuda()
        objs_centre[obj_id] = obj_centre(objs[obj_id])
        objs_idx[obj_id] = torch.tensor(in_box_idx).cuda()
        if len(pc_fil) == 0:
            pc_fil = torch.tensor(pc[:, :, in_box_idx]).cuda()
        else:
            pc_fil = torch.cat((pc_fil,  torch.tensor(pc[:, :, in_box_idx]).cuda()), dim=2)

    if len(pc_fil) == 0:
        pc_fil = None

    return pc_fil, torch.tensor(cls), objs, objs_idx, objs_centre, cls_obj_id


def filter_object_points2(labels, pc):
    pc = pc.permute(0, 2, 1)
    boxes = dict()
    bbb = dict()
    for key, label in labels.items():
        box = Box3D(x=label.x,
                    y=label.y,
                    z=label.z,
                    h=label.h,
                    w=label.w,
                    l=label.l,
                    ry=label.ry)
        boxes[label.id] = box
    pc_fil = []
    cls = []
    objs = dict()
    objs_idx = dict()
    for i in range(pc.size(2)):
        in_box = False
        for box_id, box in boxes.items():
            box_corners = Box3D.box2corners3d_camcoord(box)
            box_corners_tr = [box_corners[7], box_corners[4], box_corners[5], box_corners[6], box_corners[3],
                              box_corners[0], box_corners[1], box_corners[2]]
            box_corners = np.array(box_corners_tr)
            if box_id not in bbb:
                bbb[box_id] = box_corners
            if check_points_in_box(pc[:, :3, i], box_corners):
                in_box = True
                break
        if in_box:
            pc_fil.append(pc[:, :, i].unsqueeze(2))
            cls.append(True)
            if box_id not in objs:
                objs[box_id] = []
                objs_idx[box_id] = []
            objs[box_id].append(pc[:, :, i].unsqueeze(2))
            objs_idx[box_id].append(i)
        else:
            cls.append(False)
    objs_cat = dict()
    objs_centre = dict()
    for obj_id, obj in objs.items():
        objs_cat[obj_id] = torch.cat(obj, dim=2)
        objs_centre[obj_id] = obj_centre(objs_cat[obj_id])

    if len(pc_fil) == 0:
        gt_objs_pt = None
    else:
        gt_objs_pt = torch.cat(pc_fil, dim=2)
    return gt_objs_pt, torch.tensor(cls), objs_cat, bbb, objs_idx, objs_centre


def pc_flow_mapping(pc1_wrap, pc2, f):
    pc2_wrap = []
    for i in range(pc1_wrap.size(2)):
        best_i = torch.argmin(euc_distance(pc1_wrap[:, :, i].unsqueeze(2), pc2[:, :3, :]))
        pc2_wrap.append(pc2[:, :, best_i].unsqueeze(2))
    return torch.cat(pc2_wrap, dim=2).cuda()


def check_points_in_box(points, cube3d):
    """
    cube3d  =  numpy array of the shape (8,3) with coordinates in the clockwise order. first the bottom plane is considered then the top one.
    points = array of points with shape (N, 3).

    Returns the indices of the points array which are outside the cube3d
    """
    b1, b2, b3, b4, t1, t2, t3, t4 = cube3d

    dir1 = (t1 - b1)
    size1 = np.linalg.norm(dir1)
    dir1 = dir1 / size1

    dir2 = (b2 - b1)
    size2 = np.linalg.norm(dir2)
    dir2 = dir2 / size2

    dir3 = (b4 - b1)
    size3 = np.linalg.norm(dir3)
    dir3 = dir3 / size3

    cube3d_center = (b1 + t3) / 2.0

    dir_vec = points.detach().cpu() - cube3d_center

    res1 = np.where((np.absolute(np.dot(dir_vec, dir1)) * 2) > size1)[0]
    res2 = np.where((np.absolute(np.dot(dir_vec, dir2)) * 2) > size2)[0]
    res3 = np.where((np.absolute(np.dot(dir_vec, dir3)) * 2) > size3)[0]

    res = list(set().union(res1, res2, res3))
    if res is None or len(res) == 0:
        return True
    else:
        return False
    # return


def get_gt_flow_new(obj_c1, obj_c2, obj_cls1, obj_cls_id1, obj_cls_id2, pc1, pc1_comp, bboxes1, bboxes2):
    gt_flow_pc = []
    for i in range(pc1.size(2)):
        in_box = False
        if obj_cls1[i]:
            if int(obj_cls_id1[i]) in obj_c2.keys():
                obj_id = int(obj_cls_id1[i])

                t_ego_bbx1 = get_bbx_transformation(bboxes1[obj_id])
                t_ego_bbx2 = get_bbx_transformation(bboxes2[obj_id])
                t_bbx1_bbx2 = np.dot(t_ego_bbx2,np.linalg.inv(t_ego_bbx1))
                pnt_torch = pc1[:, :3, i]
                gt_p_torch = (torch.tensor(t_bbx1_bbx2, dtype=torch.float32).cuda() @ torch.cat((pnt_torch, torch.ones((pnt_torch.shape[0],1)).cuda()), dim=1).T)[:3]

                # check if gt_p result and gt_p_torch are the same
                # assert np.allclose(gt_p, gt_p_torch.detach().cpu().numpy())

                # gt_p = get_gt_flow_point(obj_c1[obj_id][0], obj_c2[obj_id][0], pc1[:, :3, i].unsqueeze(2))
                gt_flow_pc.append(gt_p_torch.unsqueeze(0))
                in_box = True
        if not in_box:
            gt_flow_pc.append(pc1_comp[:, :, i].unsqueeze(2).cuda())
    return torch.cat(gt_flow_pc, dim=2)


def get_gt_flow(lbl1, lbl2, pc1, pc1_comp):
    pc1 = pc1.permute(0, 2, 1)
    boxes = dict()
    gt_flow_pc = []
    for key, label in lbl1.items():
        box = Box3D(x=label.x,
                    y=label.y,
                    z=label.z,
                    h=label.h,
                    w=label.w,
                    l=label.l,
                    ry=label.ry)
        boxes[label.id] = box
    for i in range(pc1.size(2)):
        in_box = False
        for box_id, box in boxes.items():
            box_corners = Box3D.box2corners3d_camcoord(box)
            box_corners_tr = [box_corners[7], box_corners[4], box_corners[5], box_corners[6], box_corners[3],
                              box_corners[0], box_corners[1], box_corners[2]]
            box_corners = np.array(box_corners_tr)
            if check_points_in_box(pc1[:, :3, i], box_corners):
                if box_id in lbl2:
                    coord1 = torch.tensor([box.x, box.y, box.z]).cuda()
                    coord2 = torch.tensor([lbl2[box_id].x, lbl2[box_id].y, lbl2[box_id].z]).cuda()
                    gt_p = get_gt_flow_point(coord1, coord2, pc1[:, :3, i].unsqueeze(2))
                    gt_flow_pc.append(gt_p)
                    in_box = True
                break
        if not in_box:
            gt_flow_pc.append(pc1_comp[:, :, i].unsqueeze(2))
    return torch.cat(gt_flow_pc, dim=2)


def get_gt_flow_point(gt_coord1, gt_coord2, p):
    return p + (gt_coord2.unsqueeze(0).unsqueeze(2) - gt_coord1.unsqueeze(0).unsqueeze(2))


def motion_seg(pc1_wrap, pc1_comp):
    cls = (torch.tensor(pc1_wrap[0]).cuda() - torch.tensor(pc1_comp[0]).cuda()).pow(2).sum(dim=0).sqrt()
    cls = cls > 0.7
    return cls


def log_sinkhorn_iterations(Z: torch.Tensor, log_mu: torch.Tensor, log_nu: torch.Tensor, iters: int) -> torch.Tensor:
    """ Perform Sinkhorn Normalization in Log-space for stability"""
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)
    return Z + u.unsqueeze(2) + v.unsqueeze(1)


def log_optimal_transport(scores: torch.Tensor, alpha: torch.Tensor, iters: int) -> torch.Tensor:
    """ Perform Differentiable Optimal Transport in Log-space for stability"""
    b, m, n = scores.shape
    one = scores.new_tensor(1)
    ms, ns = (m * one).to(scores), (n * one).to(scores)

    bins0 = alpha.expand(b, m, 1)
    bins1 = alpha.expand(b, 1, n)
    alpha = alpha.expand(b, 1, 1)

    couplings = torch.cat([torch.cat([scores, bins0], -1),
                           torch.cat([bins1, alpha], -1)], 1)

    norm = - (ms + ns).log()
    log_mu = torch.cat([norm.expand(m), ns.log()[None] + norm])
    log_nu = torch.cat([norm.expand(n), ms.log()[None] + norm])
    log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1)

    Z = log_sinkhorn_iterations(couplings, log_mu, log_nu, iters)
    Z = Z - norm  # multiply probabilities by M+N
    return Z


def arange_like(x, dim: int):
    return x.new_ones(x.shape[dim]).cumsum(0) - 1  # traceable in 1.1


def mismatches(mappings_prev, mappings_curr, aff_mat, objs, gt_objs):
    fp = 0
    mismat = 0
    if mappings_prev == None:
        return 0
    # for m in mappings_prev.keys():
    for gt, mapped in mappings_curr.items():
        if gt in mappings_prev and gt in gt_objs:
            # skip low confidence objects
            aff_idx_prev = list(mappings_prev.keys()).index(gt)
            aff_idx_curr = list(mappings_curr.keys()).index(gt)
            if aff_mat[0, aff_idx_prev, aff_idx_curr] < 0.5:
                # fp += 1
                continue
            if mapped != mappings_prev[gt]:
                mismat += 1
    return mismat, fp


def eval_tracking(args, objs, gt_objs, mappings_prev, mappings_curr, aff_mat):
    # matched but affinity < thres, false positive
    # gt can't find object, miss
    gt_objs_ = dict()
    for key, gt_obj in gt_objs.items():
        if gt_obj.size(2) < args.min_obj_points:
            continue
        gt_objs_[key] = gt_obj
    gt_objs = gt_objs_

    misses = 0
    if len(gt_objs) > len(objs):
        misses = len(gt_objs) - len(objs)
    mismat, fp = mismatches(mappings_prev, mappings_curr, aff_mat, objs, gt_objs)
    if len(gt_objs) == 0:
        return {'na': 1}

    total_p = 0
    common_p = 0
    for key, gt_obj in gt_objs.items():
        if key in mappings_curr and key in mappings_prev:
            
            if mappings_curr[key] != mappings_prev[key]:
                continue
            # skip low confidence objects
            aff_idx_prev = list(mappings_prev.keys()).index(key)
            aff_idx_curr = list(mappings_curr.keys()).index(key)
            if aff_mat[0, aff_idx_prev, aff_idx_curr] < 0.3:
                continue
            obj = objs[mappings_curr[key]][:, 3:6, :]

            total_p += obj.size(2)
            total_p += gt_obj.size(2)
            for i in range(gt_obj.size(2)):
                point = gt_obj[:, :, i].unsqueeze(2)
                idx = find_nearest_obj_coord(point, obj)
                if euc_distance(point, obj[:, :, idx]) < 0.00001:
                    common_p += 1
        else:
            continue
    if (total_p - common_p) == 0:
        return {'na': 1}

    if common_p / (total_p - common_p) > 1:
        print()
    return {'MOTA': 1 - (misses + fp + mismat) / len(gt_objs), 'MOTP': common_p / (total_p - common_p), 'IDS': mismat}


def get_frame_det(dets_all, frame):
    additional_info = None    

    # get 3D box
    dets = dets_all[dets_all[:, 0] == frame, 7:14]        

    dets_frame = {'dets': dets, 'info': additional_info}
    return dets_frame


def process_dets(dets):
    # convert each detection into the class Box3D 
    # inputs: 
    #     dets - a numpy array of detections in the format [[h,w,l,x,y,z,theta],...]

    dets_new = []
    for det in dets:
        det_tmp = Box3D.array2bbox_raw(det)
        dets_new.append(det_tmp)

    return dets_new


def load_dets(idx, seq, ds_path):

    all_dets_1 = []
    idx = int(idx)

    seq_dets, flag = load_detection(os.path.join(ds_path, 'detection/car'), int(seq))
    dets_frame = get_frame_det(seq_dets, idx)
    dets, info = dets_frame['dets'], dets_frame['info'] 
    dets = process_dets(dets)
    all_dets_1 += dets

    seq_dets, flag = load_detection(os.path.join(ds_path, 'detection/cyclist'), int(seq))
    dets_frame = get_frame_det(seq_dets, idx)
    dets, info = dets_frame['dets'], dets_frame['info'] 
    dets = process_dets(dets)
    all_dets_1 += dets

    seq_dets, flag = load_detection(os.path.join(ds_path, 'detection/pedestrian'), int(seq))
    dets_frame = get_frame_det(seq_dets, idx)
    dets, info = dets_frame['dets'], dets_frame['info'] 
    dets = process_dets(dets)
    all_dets_1 += dets
    
    all_dets_2 = []

    seq_dets, flag = load_detection(os.path.join(ds_path, 'detection/car'), int(seq))
    dets_frame = get_frame_det(seq_dets, idx + 1)
    dets, info = dets_frame['dets'], dets_frame['info'] 
    dets = process_dets(dets)
    all_dets_2 += dets

    seq_dets, flag = load_detection(os.path.join(ds_path, 'detection/cyclist'), int(seq))
    dets_frame = get_frame_det(seq_dets, idx + 1)
    dets, info = dets_frame['dets'], dets_frame['info'] 
    dets = process_dets(dets)
    all_dets_2 += dets

    seq_dets, flag = load_detection(os.path.join(ds_path, 'detection/pedestrian'), int(seq))
    dets_frame = get_frame_det(seq_dets, idx + 1)
    dets, info = dets_frame['dets'], dets_frame['info'] 
    dets = process_dets(dets)
    all_dets_2 += dets

    return all_dets_1, all_dets_2


def get_bbx_param(obj_info, transforms, sensor):
    
    ## get box in the radar/lidar coordinates
    if sensor == 'lidar':
        center = (transforms.t_lidar_camera @ np.array([obj_info[3],obj_info[4], obj_info[5], 1]))[:3]
    if sensor == 'radar':
        center = (transforms.t_radar_camera @ np.array([obj_info[3],obj_info[4], obj_info[5], 1]))[:3]
    # enlarge the box field to include points with meansure errors
    extent = np.array([obj_info[2], obj_info[1], obj_info[0]]) # l w h
    angle = [0, 0, -(obj_info[6] + np.pi / 2)]
    rot_m = R.from_euler('XYZ', angle).as_matrix()
    if sensor == 'lidar':
        rot_m = np.eye(3) @ rot_m
    if sensor == 'radar':
        rot_m = transforms.t_radar_lidar[:3,:3] @ rot_m
        
    obbx = o3d.geometry.OrientedBoundingBox(center.T, rot_m, extent.T)
    
    return obbx


def get_bbx_transformation(bbx):

    t_ego_bbx = np.zeros((4,4))
    t_ego_bbx[:3,:3] = bbx.R
    t_ego_bbx[:3,3] = bbx.center
    t_ego_bbx[3, 3] = 1

    return t_ego_bbx


def get_bbx_param_ego(obj_info, transforms, sensor, ego_motion):
    
    ## get box in the radar/lidar coordinates
    # if sensor == 'lidar':
    #     center = (transforms.t_lidar_camera @ np.array([obj_info[3],obj_info[4], obj_info[5], 1]))[:3]
    if sensor == 'radar':
        center = (transforms.t_radar_camera @ np.array([obj_info[3],obj_info[4], obj_info[5], 1]))
        center_comp = np.dot(center, np.linalg.inv(ego_motion.T))[:3]
    # enlarge the box field to include points with meansure errors
    extent = np.array([obj_info[2], obj_info[1], obj_info[0]]) # l w h
    angle = [0, 0, -(obj_info[6] + np.pi / 2)]
    rot_m = R.from_euler('XYZ', angle).as_matrix()
    if sensor == 'lidar':
        rot_m = np.eye(3) @ rot_m
    if sensor == 'radar':
        rot_m = transforms.t_radar_lidar[:3,:3] @ rot_m
        
    obbx_comp = o3d.geometry.OrientedBoundingBox(center_comp.T, rot_m, extent.T)
    
    return obbx_comp


def filter_moving_boxes_det(raw_detection_labels, lbls):
    mov_lbl = dict()
    for i in range(len(raw_detection_labels)):
        line = raw_detection_labels[i]
        key = list(lbls.keys())[i]
        line = line.split(' ')
        mov = int(line[1])
        if mov == 1:
            mov_lbl[key] = lbls[key]
    return mov_lbl


def filter_moving_boxes(lbls0, lbls1, transforms0, transforms1, mov_thres, pc0, pc1):
    boxes0_comp = dict()
    boxes1 = dict()
    types = dict()
    odom_cam_0 = transforms0.t_odom_camera
    odom_cam_1 = transforms1.t_odom_camera
    cam_radar_0 = transforms0.t_camera_radar
    cam_radar_1 = transforms1.t_camera_radar
    odom_radar_0 = np.dot(odom_cam_0,cam_radar_0)
    odom_radar_2 = np.dot(odom_cam_1,cam_radar_1)
    ego_motion = np.dot(np.linalg.inv(odom_radar_0), odom_radar_2)

    for key, label in lbls0.items():
        
        boxes0_comp[label.id] = get_bbx_param_ego([label.h, label.w, label.l, label.x, label.y, label.z, label.ry], transforms0, 'radar', ego_motion)
        
        # pc_ = o3d.utility.Vector3dVector(pc[0].T)
        # in_box_idx = box.get_point_indices_within_bounding_box(pc_)

    for key, label in lbls1.items():
        boxes1[label.id] = get_bbx_param([label.h, label.w, label.l, label.x, label.y, label.z, label.ry], transforms0, 'radar')
        types[label.id] = label.type
    
    mov_index = dict()
    mov_lbl = dict()

    for key1, box1 in boxes1.items():
        if key1 in boxes0_comp:
            
            centre0 = boxes0_comp[key1].get_center()
            centre1 = boxes1[key1].get_center()

            if types[key1] == 'Pedestrian':
                thres = 0.06
            elif types[key1] == 'Car':
                thres = 0.15
            elif types[key1] == 'truck':
                thres = 0.15
            elif types[key1] == 'Cyclist':
                thres = 0.10
            elif types[key1] == 'rider':
                thres = 0.10
            elif types[key1] == 'bicycle':
                thres = 0.10
            elif types[key1] == 'ride_uncertain':
                thres = 0.10
            # elif types[key1] == 'bicycle_rack':
            #     thres = 0.08
            elif types[key1] == 'ride_other':
                thres = 0.10
            elif types[key1] == 'motor':
                thres = 0.10
            elif types[key1] == 'moped_scooter':
                thres = 0.10
            # elif types[key1] == 'other_vehicle':
            #     thres = 0.11
            elif types[key1] == 'human_depiction':
                thres = 0.06
            else:
                thres = 0.20

            # thres = 0.03
                
            if np.abs(np.linalg.norm(centre0 - centre1)) > thres:
                mov_index[key1] = True
                mov_lbl[key1] = lbls1[key1]
            else:
                mov_index[key1] = False
    
    return mov_index, mov_lbl