import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation as R

def get_rotation(yaw):
    # x,y,_ = arr[:3]
    # yaw = np.arctan(y/x)
    angle = np.array([0, 0, yaw])
    r = R.from_euler('XYZ', angle)
    return r.as_matrix()

def get_bbx_param(obj_info, scale=1.1):

    center = obj_info[:3]
    extent = obj_info[3:6]
    angle = obj_info[6]

    rot_m = get_rotation(angle)
    
    obbx = o3d.geometry.OrientedBoundingBox(center.T, rot_m, extent.T)
    return obbx