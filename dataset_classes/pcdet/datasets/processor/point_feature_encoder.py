import numpy as np


class PointFeatureEncoder(object):
    def __init__(self, config, point_cloud_range=None):
        super().__init__()
        self.point_encoding_config = config
        assert list(self.point_encoding_config.src_feature_list[0:3]) == ['x', 'y', 'z']
        self.used_feature_list = self.point_encoding_config.used_feature_list
        self.src_feature_list = self.point_encoding_config.src_feature_list
        self.point_cloud_range = point_cloud_range

    @property
    def num_point_features(self):
        return getattr(self, self.point_encoding_config.encoding_type)(points=None)

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)
                ...
        Returns:
            data_dict:
                points: (N, 3 + C_out),
                use_lead_xyz: whether to use xyz as point-wise features
                ...
        """
        
        data_dict['points'], use_lead_xyz = getattr(self, self.point_encoding_config.encoding_type)(
            data_dict['points']
        )
        data_dict['use_lead_xyz'] = use_lead_xyz
        # if data_dict['points'].shape[0] < 50:
        #     print('sth is wrong here')
        #     print('number of input points: ', data_dict['points'].shape[0])
        return data_dict

    def absolute_coordinates_encoding(self, points=None):
        if points is None:
            num_output_features = len(self.used_feature_list)
            return num_output_features

        point_feature_list = [points[:, 0:3]]
        for x in self.used_feature_list:
            if x in ['x', 'y', 'z']:
                continue
            idx = self.src_feature_list.index(x)
            point_feature_list.append(points[:, idx:idx+1])
        point_features = np.concatenate(point_feature_list, axis=1)
        return point_features, True

    def with_spherical(self, points=None):
        if points is None:
            num_output_features = len(self.used_feature_list)+3
            return num_output_features

        
        eps = 1e-8

        distance = np.linalg.norm(points[:, 0:3], axis=1)
        theta = np.arctan2(points[:, 1],(points[:,0]+eps))
        phi = np.arctan2(np.linalg.norm(points[:, 0:2]),(points[:, 3]+eps))

        # points = np.column_stack((points,distance,theta,phi))

        point_feature_list = [points[:, 0:3]]
        for x in self.used_feature_list:
            if x in ['x', 'y', 'z']:
                continue
            idx = self.src_feature_list.index(x)
            
            point_feature_list.append(points[:, idx:idx+1])
        
        

     
        point_feature_list.append(np.reshape(distance,(-1,1)))
        point_feature_list.append(np.reshape(theta,(-1,1)))
        point_feature_list.append(np.reshape(phi,(-1,1)))
        point_features = np.concatenate(point_feature_list, axis=1)
        # print(point_features.shape)
        return point_features, True



        

                
            
            