import numpy as np, cv2, os
# from AB3DMOT_libs.box import Box3D
from dataset_classes.kitti.kitti_obj import Object_3D
from xinshuo_io import load_txt_file


# read tracklet data in the KITTI tracking results format, one file per sequence

class Tracklet_3D(object):
    def __init__(self, lines, frame):
        # lines, num_lines = load_txt_file(label_file)

        # a dictionary containing all data, with frame as the key, value is another dictionary 
        # with the ID as the key and the Object_3D class as the value
        self.data = dict()

        # load each line of data into the dictionary
        self.frame = frame
        for line in lines:
            self.load_line(line)

    def load_line(self, line):
        # load data in each line of the tracklet file

        # get key info 
        line = line.split(' ')
        obj_type = line[0]
        # frame = int(line[0])
        obj_id = int(line[1])

        # extracting the rest of data
        rest = [float(x) for x in line[1:]]
        obj = Object_3D(obj_type=obj_type, occ=rest[1], alpha=rest[2],
                        xmin=rest[3], ymin=rest[4], xmax=rest[5], ymax=rest[6],
                        h=rest[7], w=rest[8], l=rest[9], x=rest[10], y=rest[11], z=rest[12], ry=rest[13], id=obj_id)

        # create entry for this frame and this ID
        if self.frame not in self.data:
            self.data[self.frame] = dict()

        # if item type is DontCare
        # if obj_id == -1:
        #     return
        # assert obj_id not in self.data[frame], 'error! object ID %d already in the frame %d' % (obj_id, frame)
        self.data[self.frame][obj_id] = obj
