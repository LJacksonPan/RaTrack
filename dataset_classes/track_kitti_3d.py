import os.path

import numpy as np
from torch.utils.data import Dataset, DataLoader

from dataset_classes.gnd.module import lidar_projection
from dataset_classes.ground_removal import Processor
from dataset_classes.kitti.kitti_calib import Calibration
from .kitti.kitti_trk import Tracklet_3D
from .kitti.kitti_oxts import load_oxts
# Load: raw + label + ego

class TrackingData(Dataset):

    def __init__(self, data_dir, args):
        # set params
        self.dir = data_dir
        self.curr_seq = 0
        self.seq_frames = []
        self.prev_seq_frames_sum = 0
        self.index_incre = 0
        self.args = args

        velodyne_path = os.path.join(self.dir, 'velodyne')
        for seq in range(21):
            seq_path = os.path.join(velodyne_path, str(seq).zfill(4))
            _, _, files = next(os.walk(seq_path))
            file_count = len(files)
            self.seq_frames.append(file_count - 1)

        self.curr_seq = args.start_seq

    def __getitem__(self, index):
        is_new_seq = False
        loaded = False

        if index == 110 and self.curr_seq == 16:
            self.prev_seq_frames_sum = index
            self.curr_seq += 1
            true_index = 0
            is_new_seq = True


        while not loaded:
            try:
                true_index = index + self.index_incre - self.prev_seq_frames_sum
                if true_index == self.seq_frames[self.curr_seq]:
                    self.prev_seq_frames_sum += self.seq_frames[self.curr_seq]
                    self.curr_seq += 1
                    true_index = 0
                    is_new_seq = True

                #############################################################################
                labels1 = load_labels(os.path.join(self.dir, 'label_02'), self.curr_seq)
                labels2 = load_labels(os.path.join(self.dir, 'label_02'), self.curr_seq)
                #############################################################################

                raw_pc = load_raw_pc_frame(os.path.join(self.dir, 'velodyne'), self.curr_seq, true_index)
                raw_pc2 = load_raw_pc_frame(os.path.join(self.dir, 'velodyne'), self.curr_seq, true_index + 1)

                raw_pc = np.delete(raw_pc, 3, 1)
                raw_pc2 = np.delete(raw_pc2, 3, 1)

                process = Processor(n_segments=70, n_bins=80, line_search_angle=0.3, max_dist_to_line=0.15,
                                    sensor_height=1.73, max_start_height=0.5, long_threshold=8)

                raw_pc = raw_pc * np.array([1, 1, -1])
                raw_pc = process(raw_pc)
                raw_pc2 = raw_pc2 * np.array([1, 1, -1])
                raw_pc2 = process(raw_pc2)

                raw_pc = raw_pc * np.array([1, 1, -1])
                raw_pc2 = raw_pc2 * np.array([1, 1, -1])

                # [0, -40, -3, 70.4, 40, 1]
                raw_pc_x = np.logical_and((raw_pc[:, 0] > 0), (raw_pc[:, 0] < 70))
                raw_pc_z = np.logical_and((raw_pc[:, 1] > -999), (raw_pc[:, 1] < 999))
                raw_pc_y = np.logical_and((raw_pc[:, 2] > -40), (raw_pc[:, 2] < 40))

                raw_pc_range = np.logical_and(raw_pc_x, raw_pc_y, raw_pc_z)
                raw_pc = raw_pc[raw_pc_range]

                raw_pc_x = np.logical_and((raw_pc2[:, 0] > 0), (raw_pc2[:, 0] < 70))
                raw_pc_z = np.logical_and((raw_pc2[:, 1] > -3), (raw_pc2[:, 1] < 1))
                raw_pc_y = np.logical_and((raw_pc2[:, 2] > -40), (raw_pc2[:, 2] < 40))

                raw_pc_range = np.logical_and(raw_pc_x, raw_pc_y, raw_pc_z)
                raw_pc2 = raw_pc2[raw_pc_range]

                calib = load_calib(os.path.join(self.dir, 'calib'), self.curr_seq)
                raw_pc = calib.project_velo_to_ref(raw_pc)
                raw_pc2 = calib.project_velo_to_ref(raw_pc2)
                loaded = True
            except:
                self.index_incre += 1
                pass

        # TODO: test feature, num of points
        raw_pc_idx = np.random.choice(raw_pc.shape[0], self.args.npoints, replace=False)
        raw_pc2_idx = np.random.choice(raw_pc2.shape[0], self.args.npoints, replace=False)

        raw_pc = raw_pc[raw_pc_idx, :]
        raw_pc2 = raw_pc2[raw_pc2_idx, :]


        return raw_pc, raw_pc2, self.curr_seq, true_index, is_new_seq

    def __len__(self):
        return sum(self.seq_frames[self.args.start_seq:self.args.end_seq])


def load_detection(det_path, seq):

    # load from raw file
    file_path = os.path.join(det_path, str(seq).zfill(4) + '.txt')
    dets = np.loadtxt(file_path, delimiter=',')     # load detections, N x 15

    if len(dets.shape) == 1: dets = np.expand_dims(dets, axis=0)     
    if dets.shape[1] == 0:        # if no detection in a sequence
        return [], False
    else:
        return dets, True


def load_poses(oxts_path, seq):
    file_path = os.path.join(oxts_path, str(seq).zfill(4) + '.txt')
    oxts = load_oxts(file_path)
    return oxts


def load_labels(labels_path, seq):
    file_path = os.path.join(labels_path, str(seq).zfill(4) + '.txt')
    labels_trk = Tracklet_3D(file_path)
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


def load_raw_pc_frame(velodyne_path, seq, frame):
    seq_path = os.path.join(velodyne_path, str(seq).zfill(4))
    file_path = os.path.join(seq_path, str(frame).zfill(6) + '.bin')

    raw_pc = np.fromfile(file_path, '<f4')  # little-endian float32
    raw_pc = np.reshape(raw_pc, (-1, 4))  # x, y, z, r

    return raw_pc
