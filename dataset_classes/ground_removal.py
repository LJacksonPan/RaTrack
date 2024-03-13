import numpy as np

class Processor:
    '''
    Module: Processor

    Args:
        n_segments(int): The number of fan-shaped regions divided by 360 degrees
        n_bins(int): The number of bins divided in a segment.
        r_max(float): The max boundary of lidar point.(meters)
        r_min(float): The min boundary of lidar point.(meters)
        line_search_angle(float): The angle for relative search in nearby segments.
        max_dist_to_line(float): The distance threshold of the non-ground object to the ground.(meters)

        max_slope(float): Local maximum slope of the ground.
        max_error(float): The max MSE to fit the ground.(meters)
        long_threshold(int): The max threshold of ground wave interval.
        max_start_height(float): The max height difference between hillside and ground.(meters)
        sensor_height(float): The distance from the lidar sensor to the ground.(meters)

    Call:
        Arg:
            vel_msg(numpy.ndarray): The raw local LiDAR cloud points in 3D(x,y,z).

            For example:
                vel_msg shapes [n_point, 3], with `n_point` refers to the number of cloud points,
                    while `3` is the number of 3D(x,y,z) axis.
                vel_msg = array([[0.3, 0.1, 0.7],
                                 [0.6, 0.6, 0.5],
                                 [0.1, 0.4, 0.8],
                                  ...  ...  ...
                                 [0.5, 0.3, 0.6],
                                 [0.6, 0.3, 0.4]]
        Returns:
            vel_non_ground(numpy.ndarray):  The local LiDAR cloud points after filter out ground info.
    '''

    def __init__(self, n_segments=60, n_bins=80, r_max=150, r_min=0.3,
                 line_search_angle=0.2, max_dist_to_line=0.25,
                 max_slope=2.0, max_error=0.1, long_threshold=3,
                 max_start_height=0.2, sensor_height=1.73):
        self.n_segments = n_segments  # number of segments
        self.n_bins = n_bins  # number of bins in a segment
        self.r_max = r_max
        self.r_min = r_min
        self.line_search_angle = line_search_angle
        self.max_dist_to_line = max_dist_to_line

        self.max_slope = max_slope
        self.max_error = max_error
        self.long_threshold = long_threshold  # int
        self.max_start_height = max_start_height
        self.sensor_height = sensor_height

        self.segment_step = 2 * np.pi / self.n_segments
        self.bin_step = (self.r_max - self.r_min) / self.n_bins

        self.segments = []
        self.seg_list = []

    def __call__(self, vel_msg):
        point5D = self.Model_Ground(vel_msg)
        vel_non_ground = self.Segment_Vel(point5D)

        return vel_non_ground

    def Model_Ground(self, vel_msg):
        point5D = self.project_5D(vel_msg)
        point5D = self.filter_out_range(point5D)
        point5D = point5D[np.argsort(point5D[:, 3])]

        self.seg_list = np.int16(np.unique(point5D[:, 3]))

        for seg_idx in self.seg_list:
            segment = Segmentation(self.max_slope, self.max_error, self.long_threshold,
                                   self.max_start_height, self.sensor_height)
            point5D_seg = point5D[point5D[:, 3] == seg_idx]

            min_z = segment.get_min_z(point5D_seg)  # checked
            segment.fitSegmentLines(min_z)  # checked
            self.segments.append(segment)

        return point5D

    def Segment_Vel(self, point5D):
        label = np.zeros([point5D.shape[0]])
        slice_list = np.r_[np.nonzero(np.r_[1, np.diff(point5D[:, 3])])[0], len(point5D)]

        for i, seg_idx in enumerate(self.seg_list):
            segment = self.segments[i]
            point5D_seg = point5D[point5D[:, 3] == seg_idx]

            non_ground = segment.verticalDistanceToLine(point5D_seg[:, [4, 2]])  # x,y -> d,z
            non_ground[non_ground > self.max_dist_to_line] = 0

            step = 1
            idx_search = lambda i, step: (i % len(self.seg_list) + step) % len(self.seg_list)

            while step * self.segment_step < self.line_search_angle:
                segment_f = self.segments[idx_search(i, -step)]
                segment_b = self.segments[idx_search(i, step)]

                non_ground_b = segment_f.verticalDistanceToLine(point5D_seg[:, [4, 2]])
                non_ground_b[non_ground_b > self.max_dist_to_line] = 0
                non_ground_f = segment_b.verticalDistanceToLine(point5D_seg[:, [4, 2]])
                non_ground_f[non_ground_f > self.max_dist_to_line] = 0

                non_ground += non_ground_b + non_ground_f

                step += 1

            label[slice_list[i]:slice_list[i + 1]] = non_ground == 0

        vel_non_ground = point5D[label == 1][:, :3]

        return vel_non_ground

    def project_5D(self, point3D):
        '''
        Args:
            point3D: shapes (n_row, 3), while 3 represent x,y,z axis in order.
        Returns:
            point5D: shapes (n_row, 3+2), while 5 represent x,y,z,seg,bin axis in order.
        '''
        x = point3D[:, 0]
        y = point3D[:, 1]
        z = point3D[:, 2]

        # index mapping
        angle = np.arctan2(y, x)
        segment_index = np.int32(np.floor((angle + np.pi) / self.segment_step))  # segment

        radius = np.sqrt(x ** 2 + y ** 2)
        bin_index = np.int32(np.floor((radius - self.r_min) / self.bin_step))  # bin

        point5D = np.vstack([point3D.T, segment_index, bin_index]).T

        return point5D

    def filter_out_range(self, point5D):
        '''
        Args:
            point5D: shapes (n_row, 3+2), while 5 represent x,y,z,seg,bin axis in order.
        Returns:
            point5D: shapes (n_row_filtered, 5), while 5 represent x,y,z,seg,bin axis in order.
        '''
        radius = point5D[:, 4]  # [x,y,z,seg,bin]
        condition = np.logical_and(radius < self.r_max, radius > self.r_min)
        point5D = point5D[condition]

        return point5D

class Segmentation:
    '''
    Args:
        max_slope(float): Local maximum slope of the ground.
        max_error(float): The max MSE to fit the ground.
        long_threshold(int): The max threshold of ground wave interval.
        max_start_height(float): The max height difference between hillside and ground.
        sensor_height(float): The distance from the lidar sensor to the ground.
    '''

    def __init__(self, max_slope=2.0, max_error=0.1, long_threshold=3,
                 max_start_height=0.2, sensor_height=1.73):
        self.max_slope_ = max_slope
        self.max_error_ = max_error
        self.long_threshold_ = long_threshold  # int
        self.max_start_height_ = max_start_height
        self.sensor_height_ = sensor_height

        self.matrix_new = np.array([[1, 0, 0], [0, 1, 0]])
        self.matrix_one = np.array([[0, 0, 1]])

        self.lines = []

    def get_min_z(self, point5D_seg):
        '''
        Args:
            point5D: shapes (n_row, 5), while 5 represent x,y,z,seg,bin axis in order.
        Returns:
            pointSBZ: shapes (n_row, 2), while 3 represent bin,z axis in order. Bin order sorted.
        '''
        bin_ = point5D_seg[:, 4]
        pointBZ = np.array([point5D_seg[bin_ == bin_idx].min(axis=0)[2:] for bin_idx in np.unique(bin_)])[:, [2, 0]]

        return pointBZ

    def fitLocalLine(self, cur_line_points, error=False):
        xy1 = np.array(cur_line_points) @ self.matrix_new + self.matrix_one
        A = xy1[:, [0, 2]]
        y = xy1[:, [1]]
        [[m], [b]] = np.linalg.lstsq(A, y, rcond=None)[0]
        if error:
            mse = (A @ np.array([[m], [b]]) - y) ** 2
            return [m, b], mse
        else:
            return [m, b]

    def verticalDistanceToLine(self, xy):  # checked
        kMargin = 0.1
        label = np.zeros(len(xy))

        for d_l, d_r, m, b in self.lines:
            distance = np.abs(m * xy[:,0] + b - xy[:,1])
            con = (xy[:, 0] > d_l - kMargin) & (xy[:, 0] < d_r + kMargin)
            label[con] = distance[con]

        return label.flatten()

    def fitSegmentLines(self, min_z):
        cur_line_points = [min_z[0]]
        long_line = False
        cur_ground_height = self.sensor_height_
        d_i = 1
        while d_i < len(min_z):
            lst_point = cur_line_points[-1]
            cur_point = min_z[d_i]

            if cur_point[0] - lst_point[0] > self.long_threshold_:
                long_line = True

            if len(cur_line_points) < 2:
                if (cur_point[0] - lst_point[0] < self.long_threshold_) and abs(
                        lst_point[1] - cur_ground_height) < self.max_start_height_:
                    cur_line_points.append(cur_point)
                else:
                    cur_line_points = [cur_point]
            else:
                cur_line_points.append(cur_point)
                cur_line, mse = self.fitLocalLine(cur_line_points, True)
                if (mse.max() > self.max_error_ or cur_line[0] > self.max_slope_ or long_line):
                    cur_line_points.pop()
                    if len(cur_line_points) >= 3:
                        new_line = self.fitLocalLine(cur_line_points)
                        self.lines.append([cur_line_points[0][0], cur_line_points[-1][0], *new_line])  # b boundary
                        cur_ground_height = new_line[0] * cur_line_points[-1][0] + new_line[1]  # m*x+b
                    long_line = False
                    cur_line_points = [cur_line_points[-1]]
                    d_i -= 1
            d_i += 1
        if len(cur_line_points) > 2:
            new_line = self.fitLocalLine(cur_line_points)
            self.lines.append([cur_line_points[0][0], cur_line_points[-1][0], *new_line])