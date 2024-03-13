import matplotlib.pyplot as plt
import pickle 
from pathlib import Path as P
from matplotlib.patches import Rectangle as Rec
import numpy as np
from pcdet.utils import calibration_kitti
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from tqdm import tqdm
from vod.visualization.settings import label_color_palette_2d
cls_name = ['Car','Pedestrian', 'Cyclist', 'Others']
color_dict = {}
for i, v in enumerate(cls_name):
    color_dict[v] = label_color_palette_2d[v]
from collections import Counter

def transform_anno(loc, frame_id):
    x,y, z = loc[0], loc[1], loc[2]
    calib_path = "/root/dj/code/CenterPoint-KITTI/data/vod_radar/training/calib/{0}.txt".format(frame_id)
    calib = calibration_kitti.Calibration(calib_path)
    loc = np.array([[x,y,z]])
    loc_lidar = calib.rect_to_lidar(loc)
    x,y,z = loc_lidar[0]
    return x,y,z


def get_rot_corner(x,y,l,w,a):

    s,c = np.sin(a),np.cos(a)

    corner_x = x - l/2
    corner_y = y - w/2

    corner_x -= x
    corner_y -= y

    new_corner_x = corner_x*c - corner_y*s 
    new_corner_y = corner_x*s + corner_y*c

    new_corner_x += x
    new_corner_y += y

    return new_corner_x,new_corner_y




def anno2plt(anno, color_dict, lw, frame_id, xz=False):
    dim = anno['dimensions']
    loc = anno['location']
    # angle = anno['rotation_y'] * 180 / 3.14
    angle = -(anno['rotation_y']+ np.pi / 2) 
    rec_list = []
    cls = anno['name']
    for idx in range(dim.shape[0]):
        name = cls[idx]
        # print(name)
        if name not in color_dict:
            color = 'gray'
        else:
            color = color_dict[name]
            # print(color)
    
        if xz:

            x, _, y = transform_anno(loc[idx], frame_id)
            # w, _, l = dim[idx]
            l, w, _ = dim[idx]  # 
            ang = -angle[idx]* 0
        else:
            # print(loc[idx])
            x, y, z = transform_anno(loc[idx], frame_id)
            # print(x,y)
            
            ### X -> LENGTH
            ### Y -> WIDTH 
            ### Z -> HEIGHT, not used. 
            # x,y,z = loc[idx]
            # w, l, _ = dim[idx]
            # print(dim[idx]) 
            l, h, w  = dim[idx] # <-- SHOULD BE CORRECT? 
            ang = angle[idx]
            # ang = 0
            # print(l,w,ang)
            # print("="*40)

            ax,ay = get_rot_corner(x,y,l,w,ang)
            ang = ang * 180 / 3.14
            # ax = x - (l/4)
            # ay = y - (w/4)

        rec_list += [Rec((ax, ay), l, w, ang, fill=False, color=color,lw=lw)]
    return rec_list

def boxes2rec(bbox, c_names=None):
    '''
    bbox: [x, y, z, dx, dy, dz, heading, cls], (x, y, z) is the box center with origin at (.5, .5, .5)
            in the shape of [n, 8]
    '''
    num = bbox.shape[0]
    rec_list = []
    if c_names is None:
        c_list = cls_name
    else:
        c_list = c_names
    for i in range(num):
        box = bbox[i, :]
        x, y, z, l, w, h, ang, c = box
        ax, ay = get_rot_corner(x, y, l, w, ang)
        c_name = c_list[int(c)]
        ang = ang * 180 / 3.14
        color = color_dict[c_name]
        rec_list += [Rec((ax, ay), l, w, ang, fill=False, color=color,lw=2)]

    return rec_list

def drawBEV(ax, pts, centers, annos, frame_id, ax_title, set_legend=False):


    # 3. draw bbx
    if annos is not None:
        try:
            rec_list = anno2plt(annos, color_dict, 2, frame_id=frame_id, xz=False)
        except:
            rec_list = anno2plt(annos[0], color_dict, 2, frame_id=frame_id, xz=False)
        
        for rec in rec_list:
            ax.add_patch(rec)
    # 1. draw original points if exist
    if pts is not None:
        x = pts[:, 0]
        y = pts[:, 1]
        ax.scatter(x, y, c='black', s=0.1)
    # 2. overlay centers
    if centers is not None:
        cx = centers[:, 0]
        cy = centers[:, 1]
        ax.scatter(cx, cy, c='red', s=0.1)
    if set_legend:
        legend_elements = [Patch(facecolor='white', edgecolor=v, label=k) for i, (k, v) in enumerate(color_dict.items())]
        legend_elements += [Line2D([0], [0], marker='o', color='w', label='FG points',
                            markerfacecolor='r', markersize=10)]
        ax.legend(handles=legend_elements, loc=1)
    ax.set_title(ax_title)

def drawBEV_match(ax, pts, centers, annos, color_dict, frame_id, ax_title):


    # 3. draw bbx
    try:
        rec_list = anno2plt(annos, color_dict, 2, frame_id=frame_id, xz=False)
    except:
        rec_list = anno2plt(annos[0], color_dict, 2, frame_id=frame_id, xz=False)
    
    for rec in rec_list:
        ax.add_patch(rec)
    # 1. draw lidar center points if exist
    if pts is not None:
        x = pts[:, 0]
        y = pts[:, 1]
        ax.scatter(x, y, c='black', s=0.1)
    # 2. overlay centers
    if centers is not None:
        cx = centers[:, 0]
        cy = centers[:, 1]
        ax.scatter(cx, cy, c='red', s=0.1)

    legend_elements = [Patch(facecolor='white', edgecolor=v, label=k) for i, (k, v) in enumerate(color_dict.items())]
    legend_elements += [Line2D([0], [0], marker='o', color='w', label='radar centers',
                          markerfacecolor='r', markersize=10),
                        Line2D([0], [0], marker='o', color='w', label='lidar centers',
                          markerfacecolor='black', markersize=10)]
    ax.legend(handles=legend_elements, loc=1)
    ax.set_title(ax_title)

def draw_cross_line(xyA, xyB, fig, ax1, ax2):
    transFigure = fig.transFigure.inverted()
    coord1 = transFigure.transform(ax1.transData.transform(xyA))
    coord2 = transFigure.transform(ax2.transData.transform(xyB))
    line = Line2D(
        (coord1[0], coord2[0]),  # xdata
        (coord1[1], coord2[1]),  # ydata
        transform=fig.transFigure,
        color="cyan",
    )
    fig.lines.append(line)

def get_line(coord1, coord2, fig):
    line = Line2D(
        (coord1[0], coord2[0]),  # xdata
        (coord1[1], coord2[1]),  # ydata
        transform=fig.transFigure,
        color="cyan",
    )
    return line

def draw_two_compare(d0, d1, t0, t1, id, bbox):
    fig = plt.figure(figsize=(4*2*2,3*2))

    # First subplot
    ax1 = fig.add_subplot(121)
    # plt.plot([0, 1], [0, 1])
    drawBEV_match(ax1, d0[id], None, bbox[id], color_dict, id, t0)
    ax1.set_xlim(-0, 75)
    ax1.set_ylim(-30, 30)
    # Second subplot
    ax2 = fig.add_subplot(122)
    # plt.plot([0, 1], [0, 1])
    drawBEV_match(ax2, None, d1[id], bbox[id], color_dict, id, t1)
    ax2.set_xlim(-0, 75)
    ax2.set_ylim(-30, 30)
    return fig, ax1, ax2

def draw_two_pointcloud(pts0, pts1, t0, t1, c0='black', c1='red'):
    fig = plt.figure(figsize=(4*2*2,3*2))

    # First subplot
    ax1 = fig.add_subplot(121)
    # plt.plot([0, 1], [0, 1])
    x0 = pts0[:, 0]
    y0 = pts0[:, 1]
    ax1.scatter(x0, y0, c=c0, s=0.1)
    ax1.set_xlim(-0, 75)
    ax1.set_ylim(-30, 30)
    ax1.set_title(t0)

    # draw pts1
    ax2 = fig.add_subplot(122)
    x1 = pts1[:, 0]
    y1 = pts1[:, 1]
    ax2.scatter(x1, y1, c=c1, s=0.1)
    ax2.set_xlim(-0, 75)
    ax2.set_ylim(-30, 30)
    ax2.set_title(t1)
    return fig, ax1, ax2