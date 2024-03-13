import os
import gc
from turtle import update
import cv2
import yaml
import ujson
import torch
import open3d as o3d
import argparse
import numpy as np
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
from common import get_frame_list
from scipy.spatial.transform import Rotation as R
from utils.show_bev_bbx import show_bev_bbx
from vod.configuration import KittiLocations
from vod.frame import FrameDataLoader
from vod.frame import FrameTransformMatrix
from vod.frame import homogeneous_transformation, project_3d_to_2d
from vod.frame import FrameLabels
from vod.visualization.helpers import get_transformed_3d_label_corners
from vod.visualization.settings import *
from pathos.multiprocessing import ProcessingPool as Pool
from utils.show_bev_pcs import show_bev_lidar, show_bev_radar, filt_points_in_range
from utils.global_param import *
from RAFT.core.raft import RAFT
from RAFT.core.utils.flow_viz import flow_to_image
from utils.optical_flow import *

DEBUG_PATH = '/mnt/12T/fangqiang/vod/debug_labels_vis/'

def get_radar_flow_samples(data_loc,frames,smp_path,opt_path,clip,split, label_path, mode):

    save_path = os.path.join(smp_path, split, clip)
    opt_path = os.path.join(opt_path, split, clip)
    if mode == 'test' or mode == 'val':
        label_path = label_path
    if mode == 'train':
        label_path = os.path.join(label_path, clip)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if not os.path.exists(opt_path):
        os.makedirs(opt_path)
    num_frames = len(frames)

    #pool = Pool(4) 
    
    #zip_data_loc = [data_loc for i in range(num_frames-1)]
    #zip_save_path = [save_path for i in range(num_frames-1)]
    #zip_opt_path = [opt_path for i in range(num_frames-1)]
    #zip_label_path = [label_path for i in range(num_frames-1)]
    
    for i in tqdm(range(num_frames-1),desc="generate scene flow samples for " + clip):
        get_one_sample(frames[i], frames[i+1], data_loc, save_path, opt_path, label_path, mode, clip)

    #with tqdm(range(num_frames-1), desc="showing bev align radar for " + clip) as t:
    #    for i in enumerate(pool.imap(get_one_sample,frames[:-1], frames[1:], \
    #            zip_data_loc,zip_save_path,zip_opt_path,zip_min_pnts,zip_label_path)):
    #        t.update()

def get_one_sample(frame1, frame2, data_loc, save_path, opt_path, label_path, mode, clip):

    raft_model = init_raft()

    data1 = FrameDataLoader(kitti_locations=data_loc,
                    frame_number=frame1)
    data2 = FrameDataLoader(kitti_locations=data_loc,
                    frame_number=frame2)

    # annos 
    annos1 = FrameLabels(data1.raw_labels)

    # transform info
    transforms1 = FrameTransformMatrix(data1)
    transforms2 = FrameTransformMatrix(data2)

    # x y z RCS v_r
    radar_data1 = data1.radar_data[:,0:5]
    radar_data2 = data2.radar_data[:,0:5]
    indices1 = filt_points_in_fov(radar_data1, transforms1,'radar')
    indices2 = filt_points_in_fov(radar_data2, transforms2,'radar')
    radar_data1 = radar_data1[indices1]
    radar_data2 = radar_data2[indices2]
    indices1 = filt_points_by_height(radar_data1, [-3,3])
    indices2 = filt_points_by_height(radar_data2, [-3,3])
    radar_data1 = radar_data1[indices1]
    radar_data2 = radar_data2[indices2]


    # for fast batch preprocess, only keep frames whose points is more than min_pnts 
    if mode == 'train' or mode == 'val':
        min_pnts = 0
    else:
        min_pnts = 0


    if not (radar_data1.shape[0]<min_pnts or radar_data2.shape[0]<min_pnts):

        # coordinate frame transformation from radar1 to radar2
        odom_cam_1 = transforms1.t_odom_camera
        odom_cam_2 = transforms2.t_odom_camera
        cam_radar_1 = transforms1.t_camera_radar
        cam_radar_2 = transforms2.t_camera_radar
        odom_radar_1 = np.dot(odom_cam_1,cam_radar_1)
        odom_radar_2 = np.dot(odom_cam_2,cam_radar_2)
        radar1_radar2 = np.dot(np.linalg.inv(odom_radar_1), odom_radar_2) 

        # estimate and show optical flow from images
        if mode == 'train':
            img1 = cv2.cvtColor(data1.image, cv2.COLOR_RGB2BGR)
            img2 = cv2.cvtColor(data2.image, cv2.COLOR_RGB2BGR)
            opt_flow = estimate_optical_flow(img1, img2, raft_model)
            show_optical_flow(img1, img2, opt_flow, opt_path, frame1)
            opt_info = info_from_opt_flow(radar_data1, transforms1, opt_flow)
        else:
            opt_info =  opt_info = {"radar_u": np.array([]),
                                    "radar_v": np.array([]),
                                    "opt_flow": np.array([]),
                                    }
        ## get LiDAR MOT results for training or real MOT labels for validation/test/train_anno 
        # get foreground info (index, confidence, flow labels) 
        labels1 = load_track_labels(label_path, frame1, mode)
        labels2 = load_track_labels(label_path, frame2, mode)
        fg_idx, fg_confs, fg_labels, fg_bboxes = extract_fg_labels(labels1, labels2, radar_data1, transforms1, transforms2, 'radar')
        
        ## show fg_labels and fg_bboxes for debugging 
        # show_fg_label_bbx(fg_idx, fg_confs, fg_labels, annos1, transforms1, radar_data1, frame1, clip)
        
        gt_mask = np.zeros(radar_data1.shape[0],dtype=np.float32)
        gt_labels = np.zeros((radar_data1.shape[0],3),dtype=np.float32)
        pse_mask = np.zeros(radar_data1.shape[0],dtype=np.float32)
        pse_labels = np.zeros((radar_data1.shape[0],3),dtype=np.float32)

        ## for test or val set, to report scores on different metrics,
        ## we obtain the scene flow and mask GT with the ego-motion info and foreground info
        if mode == 'test' or mode == 'val':
            
            # get rigid flow components induced by ego-motion
            flow_r = get_rigid_flow(radar_data1, radar1_radar2)
            # get non-rigid components for inbox points
            flow_nr = fg_labels[fg_idx] - flow_r[fg_idx]
            # obtain the index for moving points from foreground
            mov_idx = np.array(fg_idx)[np.linalg.norm(flow_nr,axis=1)>0.05]

            if len(mov_idx)>0:
                stat_idx = np.delete(np.arange(0,radar_data1.shape[0]), mov_idx)
            else:
                stat_idx = np.arange(0,radar_data1.shape[0])

            gt_mask[stat_idx] = 1
            gt_labels[stat_idx] = flow_r[stat_idx]
            if len(mov_idx)>0:
                gt_labels[mov_idx] = fg_labels[mov_idx]
                gt_mask[mov_idx] = 1 - fg_confs[mov_idx]

        ## for train set, to provide cross-modal supervision during traininig
        ## we obtain the pseudo scene flow and mask from foreground info obtained from LiDAR
        else:
            if len(fg_idx)>0:
                bg_idx = np.delete(np.arange(0,radar_data1.shape[0]), fg_idx)
            else:
                bg_idx = np.arange(0,radar_data1.shape[0])

            pse_mask[bg_idx] = 1
            if len(fg_idx)>0:
                pse_labels[fg_idx] = fg_labels[fg_idx]
                pse_mask[fg_idx] = 1 - fg_confs[fg_idx]
            
       
        # convert numpy array to list for json serializable
        for key in opt_info:
            opt_info[key] = opt_info[key].tolist()
        radar_data1 = radar_data1.tolist()
        radar_data2 = radar_data2.tolist()
        radar1_radar2 = radar1_radar2.tolist()
        
        gt_mask = gt_mask.tolist()
        gt_labels = gt_labels.tolist()
        pse_mask = pse_mask.tolist()
        pse_labels = pse_labels.tolist()

        # all info 
        sample = {
                "pc1": radar_data1,
                "pc2": radar_data2,
                "trans": radar1_radar2,
                "opt_info": opt_info,
                "gt_mask": gt_mask,
                "gt_labels": gt_labels,
                "pse_mask": pse_mask,
                "pse_labels": pse_labels
                }

        out_path = save_path + '/' + frame1 + '_' + frame2 + '.json'
        ujson.dump(sample, open(out_path, "w"))


def extract_fg_labels(labels1, labels2, pc_data1, transforms1, transforms2, sensor):

    num_pnts = pc_data1.shape[0]
    num_obj = np.size(labels1,0)
    fg_idx = []
    fg_bboxes = []
    fg_confs = np.zeros(num_pnts,dtype=np.float32)
    fg_labels = np.zeros((num_pnts,3),dtype=np.float32)

    if labels1.ndim==2 and labels2.ndim==2:
        for i in range(num_obj):
            track_id1 = labels1[i,-1]
            next_idx = np.where(labels2[:,-1] == track_id1)[0]
            if len(next_idx)!=0:
                # object in the first frame
                obj1 = labels1[i,:]
                bbx1 = get_bbx_param(obj1, transforms1, sensor)
                fg_bboxes.append(bbx1)
                # object in the second frame
                obj2 = labels2[next_idx[0],:]
                bbx2 = get_bbx_param(obj2, transforms2, sensor)
                # select radar points within the bounding box in the first frame
                pc1 = o3d.utility.Vector3dVector(pc_data1[:,0:3])
                in_box_idx = bbx1.get_point_indices_within_bounding_box(pc1)

                if len(in_box_idx)>0:
                    in_box_pnts = pc_data1[in_box_idx,0:3]
                    t_ego_bbx1 = get_bbx_transformation(bbx1)
                    t_ego_bbx2 = get_bbx_transformation(bbx2)
                    in_box_labels = get_inbox_flow(in_box_pnts, t_ego_bbx1, t_ego_bbx2)
                    # avoid wrong labels caused by inaccurate MOT output                    
                    if np.linalg.norm(in_box_labels,axis=1).max()<3:
                        fg_labels[in_box_idx] = in_box_labels
                        fg_confs[in_box_idx] = obj1[-2]
                        fg_idx.extend(in_box_idx)
                    
            else:
                continue
    
    return fg_idx, fg_confs, fg_labels, fg_bboxes



def get_rigid_flow(pc_data1, ego_trans):

    # get rigid flow labels for all points with ego-motion transformation
    pc1_rg = o3d.utility.Vector3dVector(pc_data1[:,0:3])
    pc1_geo = o3d.geometry.PointCloud()
    pc1_geo.points = pc1_rg
    pc1_tran = pc1_geo.transform(np.linalg.inv(ego_trans)).points
    flow_r = np.asarray(pc1_tran)-np.asarray(pc1_rg)
    
    return flow_r


def get_inbox_flow(pnts, t_ego_bbx1, t_ego_bbx2):

    # the transformation of bbx1 to bbx2 under the ego coordinates
    t_bbx1_bbx2 = np.dot(t_ego_bbx2,np.linalg.inv(t_ego_bbx1))
    labels = (t_bbx1_bbx2 @ np.concatenate((pnts, np.ones((pnts.shape[0],1))), axis=1).T)[:3] - pnts.T

    return labels.T

def get_bbx_transformation(bbx):

    t_ego_bbx = np.zeros((4,4))
    t_ego_bbx[:3,:3] = bbx.R
    t_ego_bbx[:3,3] = bbx.center
    t_ego_bbx[3, 3] = 1

    return t_ego_bbx


def load_track_labels(label_path, frame, mode):

    label_file = label_path + '/' + frame + '.txt'
    if os.path.exists(label_file):
        with open(label_file, 'r') as text:
            labels = text.readlines()
        labels = get_track_labels(labels, mode)
    else:
        labels = np.array([])
 
    return labels


def get_track_labels(labels, mode):
    
    labels_ls = []
    for act_line in labels:  # Go line by line to split the keys
        act_line = act_line.split()
        if len(act_line)==17:
            label, id, _, _, _, _, _, _, _, h, w, l, x, y, z, rot, score = act_line
        if len(act_line)==16:
            label, id, _, _, _, _, _, _, h, w, l, x, y, z, rot, score = act_line
        h, w, l, x, y, z, rot, score = float(h), float(w), float(l), float(x), float(y), float(z), float(rot), float(score)
        # for debug 
        # h = 10
        id = int(id)
        labels_ls.append([h, w, l, x, y, z, rot, score, id])

    labels_np = np.array(labels_ls)

    return labels_np



def get_bbx_param(obj_info,transforms,sensor):
    
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


def filt_points_by_height(radar_data, ranges):

    radar_h = radar_data[:,2]
    filt_h = np.logical_and(radar_h>=ranges[0], radar_h<=ranges[1])
    indices = np.argwhere(filt_h).flatten()

    return indices

def show_fg_label_bbx(fg_idx, fg_confs, fg_labels, annos, transforms, radar_data, frame, clip):
    save_path = os.path.join(DEBUG_PATH, clip)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    fig_path = os.path.join(save_path, frame + '.png')
    x_max = int((FWD_RANGE[1] - FWD_RANGE[0]) / RES)
    y_max = int((SIDE_RANGE[1] - SIDE_RANGE[0]) / RES)
    im = np.zeros([y_max, x_max,3], dtype=np.uint8) + 255
    x_points = radar_data[:,0]
    y_points = radar_data[:,1]
    z_points = radar_data[:,2]
    N = radar_data.shape[0]
    x_img = np.floor((x_points) / RES).astype(np.int32)
    y_img = np.floor(-(y_points + SIDE_RANGE[0])/RES).astype(np.int32)
    fg_color = (255,0,0)
    bg_color = (0,0,255)
    for j in range(N):
        if j in fg_idx:
            im=cv2.circle(im,(x_img[j],y_img[j]),2, color=fg_color, thickness=-1)
        else:
            im=cv2.circle(im,(x_img[j],y_img[j]),2, color=bg_color, thickness=-1)
    im = show_bev_bbx(annos, im, transforms)
    cv2.imwrite(fig_path,im)

