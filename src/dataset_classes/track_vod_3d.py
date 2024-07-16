import os.path
import struct
from datetime import time

import numpy as np
from torch.utils.data import Dataset

from dataset_classes.kitti.kitti_calib import Calibration
from vod.frame.transformations import homogeneous_transformation
from .kitti.kitti_trk_vod import Tracklet_3D
from .kitti.kitti_oxts import load_oxts

from vod.configuration import VodTrackLocations
from vod.frame import FrameDataLoader, FrameTransformMatrix

# from kitti.kitti_oxts import

import matplotlib
# matplotlib.use('TkAgg', force=True)
import matplotlib.pyplot as plt

# Load: raw + label + ego

class TrackingDataVOD(Dataset):

    def __init__(self, args, data_dir):
        self.eval = args.eval
        self.dataset_path = args.dataset_path
        # set params
        self.dir = data_dir
        self.index_incre = 0
        self.is_new_seq = True

        test = ['delft_7','delft_8','delft_16','delft_18','delft_20','delft_21','delft_25']
        val = ['delft_1','delft_10','delft_14','delft_22']
        train = ['delft_2','delft_3','delft_4','delft_6','delft_9','delft_11','delft_12','delft_13','delft_19','delft_23','delft_24','delft_26','delft_27']
        self.clips_dir = "./clips"

        if self.eval:
            self.clips = val
        else:
            self.clips = train
        self.current_first = 0
        self.current_last = 0
        self.clip_idx = -1
        self.current_frame = 0


    def __getitem__(self, index):

        new_seq = False

        if self.current_frame + 1 > self.current_last:
            self.clip_idx += 1
            if self.clip_idx >= len(self.clips):
                self.clip_idx = 0
            txt_path = os.path.join(self.clips_dir, self.clips[self.clip_idx] + '.txt')
            with open(txt_path) as f:
                frames = f.read().splitlines()
            self.current_first = int(frames[0])
            self.current_last = int(frames[-1])
            self.current_frame = self.current_first
            new_seq = True

        while True:
            try:
                kitti_locations = VodTrackLocations(root_dir=self.dataset_path,
                                                output_dir=self.dataset_path,
                                                frame_set_path="",
                                                pred_dir="",
                                                )
                
                frame_data_0 = FrameDataLoader(kitti_locations=kitti_locations,
                                            frame_number=str(self.current_frame+1).zfill(5))
                frame_data_1 = FrameDataLoader(kitti_locations=kitti_locations,
                                            frame_number=str(self.current_frame).zfill(5))
                frame_data_last = FrameDataLoader(kitti_locations=kitti_locations,
                                            frame_number=str(self.current_frame-1).zfill(5))

                raw_pc0 = frame_data_0.radar_data[:, :3]
                raw_pc1 = frame_data_1.radar_data[:, :3]

                features0 = frame_data_0.radar_data[:, 3:6]
                features1 = frame_data_1.radar_data[:, 3:6]

                transforms0 = FrameTransformMatrix(frame_data_0)
                transforms1 = FrameTransformMatrix(frame_data_1)
                transforms_last = FrameTransformMatrix(frame_data_last)
                
                raw_pc_last_lidar = frame_data_last.lidar_data[:, :3]
                raw_pc0_lidar = frame_data_0.lidar_data[:, :3]
                raw_pc1_lidar = frame_data_1.lidar_data[:, :3]

                n0_ = raw_pc_last_lidar.shape[0]
                pts_3d_hom0_ = np.hstack((raw_pc_last_lidar, np.ones((n0_, 1))))
                raw_pc_last_lidar = homogeneous_transformation(pts_3d_hom0_, transforms_last.t_lidar_radar)
                
                n1_ = raw_pc0_lidar.shape[0]
                pts_3d_hom1_ = np.hstack((raw_pc0_lidar, np.ones((n1_, 1))))
                raw_pc0_lidar = homogeneous_transformation(pts_3d_hom1_, transforms0.t_lidar_radar)
                
                n2_ = raw_pc1_lidar.shape[0]
                pts_3d_hom2_ = np.hstack((raw_pc1_lidar, np.ones((n2_, 1))))
                raw_pc1_lidar = homogeneous_transformation(pts_3d_hom2_, transforms1.t_lidar_radar)

                odom_cam_0 = transforms0.t_odom_camera
                odom_cam_1 = transforms1.t_odom_camera
                cam_radar_0 = transforms0.t_camera_radar
                cam_radar_1 = transforms1.t_camera_radar
                odom_radar_0 = np.dot(odom_cam_0,cam_radar_0)
                odom_radar_2 = np.dot(odom_cam_1,cam_radar_1)
                ego_motion = np.dot(np.linalg.inv(odom_radar_0), odom_radar_2) 

                comp_hom = np.hstack((raw_pc0, np.ones((raw_pc0.shape[0], 1))))
                raw_pc0_comp = np.dot(comp_hom, np.linalg.inv(ego_motion.T))

                curr_idx = self.current_frame + 1
                self.current_frame += 1
                return raw_pc0, raw_pc1, features0, features1, raw_pc0_comp, curr_idx, self.clips[self.clip_idx], ego_motion, raw_pc_last_lidar, raw_pc0_lidar, raw_pc1_lidar, new_seq

            except:
                self.current_frame += 1

    def __len__(self):
        total = 0
        for clip in self.clips:
            txt_path = os.path.join(self.clips_dir, clip + '.txt')
            with open(txt_path) as f:
                frames = f.read().splitlines()
            total += len(frames)
        return total


def load_poses(oxts_path, seq):
    file_path = os.path.join(oxts_path, str(seq).zfill(4) + '.txt')
    oxts = load_oxts(file_path)
    return oxts


def load_labels(labels, frame):
    labels_trk = Tracklet_3D(labels, frame)
    return labels_trk


def load_calib(calib_path, seq):
    file_path = os.path.join(calib_path, str(seq).zfill(4) + '.txt')
    calib = Calibration(file_path)
    return calib


def load_raw_pc(velodyne_path, seq):
    seq_path = os.path.join(velodyne_path, str(seq).zfill(4))
    _, _, files = next(os.walk(seq_path))
    file_count = len(files)
    raw_pc = []

    for i in range(file_count):
        file_path = os.path.join(seq_path, str(i).zfill(6) + '.bin')

        point_cloud_data = np.fromfile(file_path, '<f4')  # little-endian float32
        point_cloud_data = np.reshape(point_cloud_data, (-1, 4))  # x, y, z, r

        raw_pc.append(point_cloud_data)

    return raw_pc


def load_raw_pc_frame(velodyne_path, frame):
    # seq_path = os.path.join(velodyne_path, str(seq).zfill(4))
    file_path = os.path.join(velodyne_path, str(frame).zfill(5) + '.bin')

    raw_pc = np.fromfile(file_path, '<f4')  # little-endian float32
    raw_pc = np.reshape(raw_pc, (-1, 4))  # x, y, z, r

    return raw_pc