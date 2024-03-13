import pcl
import cv2
import numpy as np
from module import lidar_projection
from module.ground_removal import Processor

np.set_printoptions(precision=3, suppress=True)

# Load the pcd file
vel_msg = np.asarray(pcl.load('img/kitti_sample.pcd'))
vel_msg = vel_msg * np.array([1,1,-1]) # revert the z axis

# Segment the ground from the local point cloud
process = Processor(n_segments=70, n_bins=80, line_search_angle=0.3, max_dist_to_line = 0.15,
                    sensor_height=1.73, max_start_height=0.5, long_threshold=8)
vel_non_ground = process(vel_msg)

# Generate BEV image
img_raw = lidar_projection.birds_eye_point_cloud(vel_msg,
                                                 side_range=(-50, 50), fwd_range=(-50, 50),
                                                 res=0.25, min_height=-2, max_height=4)
cv2.imwrite('img/kitti_raw.png', img_raw)


img_non_ground = lidar_projection.birds_eye_point_cloud(vel_non_ground,
                                                        side_range=(-50, 50), fwd_range=(-50, 50),
                                                        res=0.25, min_height=-2, max_height=4)
cv2.imwrite('img/kitti_non_ground.png', img_non_ground)