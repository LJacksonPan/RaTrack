from collections import defaultdict

from utils.model_utils import *
from utils import *
from sklearn.cluster import DBSCAN
# from cuml import DBSCAN as cumlDBSCAN
# import cupy as cp
import time

from models.utils.track4d_utils import log_optimal_transport, arange_like, obj_centre


class Track4D(nn.Module):

    def __init__(self, args):
        super(Track4D, self).__init__()

        self.rigid_thres = args.rigid_thres
        self.rigid_pcs = 0.25
        self.npoints = args.num_points

        # multi-scale set feature abstraction
        last_dim = [128]
        num_sas = 2
        self.pn_head = PNHead(args.npoints, 5)

        # feature correlation layer (cost volumn)
        fc_inch = num_sas * last_dim[-1]
        fc_mlps = [fc_inch, fc_inch, fc_inch]

        self.fc_layer = FeatureCorrelator(16, in_channel=fc_inch * 2 + 3, mlp=fc_mlps)

        # flow decoder layer (output coarse scene flow)
        self.fd_layer = FlowDecoder(fc_inch=fc_inch, args=args)

        self.dbscan = DBSCAN(eps=1.5, min_samples=args.min_obj_points)
        # self.dbscan = cumlDBSCAN(eps=1.5, min_samples=args.min_obj_points)
        self.affinity = Affinity(141)
        self.sig = nn.Sigmoid()

        bin_score = torch.nn.Parameter(torch.tensor(1.))
        self.register_parameter('bin_score', bin_score)

        self.preset_aff_mat_size = 20
        self.max_id = 0
        self.relu = nn.ReLU()

    # TODO: gt_cls, objs_idx are temporary
    def forward(self, pc1, pc2, feature1, feature2, h, objects_prev):
        output, h, cls, cor_features, pc1_features, pc2_features, prop_features = self.backbone(pc1, pc2, feature1,
                                                                                                feature2, h)

        pc1_warp = pc1 + output
        pc1_features_warp = torch.cat((pc1_warp, pc1, output, feature1, prop_features), dim=1)

        mov_mask = (cls > 0.5).squeeze(0)
        pc1_features_select = pc1_features_warp[:, :, mov_mask]
        objects_curr = self.clustering(pc1_features_select)

        timeout_obj_curr = dict()
        objects = dict()

        aff_list, aff_mat, indices1, confs = self.association_module(objects, objects_curr, objects_prev)

        return h, pc1_warp, cls, aff_list, aff_mat, indices1, confs, objects, timeout_obj_curr, objects_curr

    def backbone(self, pc1, pc2, feature1, feature2, h):

        """
        pc1: B 3 N
        pc2: B 3 N
        feature1: B 3 N
        feature2: B 3 N

        """

        feature1, pc1, pc1_features, pc2, pc2_features, xyz1_new, xyz2_new = self.feature_extraction_head(feature1,
                                                                                                          feature2, pc1,
                                                                                                          pc2)

        cor_features, h, output, cls, pc1_features, pc2_features, prop_features = self.flow_head(feature1, h, pc1,
                                                                                                 pc1_features, pc2,
                                                                                                 pc2_features, xyz1_new,
                                                                                                 xyz2_new)

        return output, h, cls, cor_features, pc1_features, pc2_features, prop_features

    def flow_head(self, feature1, h, pc1, pc1_features, pc2, pc2_features, xyz1_new, xyz2_new):
        gfeat_1 = torch.max(pc1_features, -1)[0].unsqueeze(2).expand(pc1_features.size()[0], pc1_features.size()[1],
                                                                     pc1.size()[2])
        gfeat_2 = torch.max(pc2_features, -1)[0].unsqueeze(2).expand(pc2_features.size()[0], pc2_features.size()[1],
                                                                     pc2.size()[2])
        # concat local and global features
        pc1_features = torch.cat((pc1_features, gfeat_1), dim=1)
        pc2_features = torch.cat((pc2_features, gfeat_2), dim=1)
        # associate data from two sets
        cor_features = self.fc_layer(pc1, pc2, pc1_features, pc2_features)
        # decoding scene flow from embeddings
        output, h, prop_features, cls = self.fd_layer(pc1, feature1, pc1_features, cor_features, h)
        return cor_features, h, output, cls, pc1_features, pc2_features, prop_features

    def feature_extraction_head(self, feature1, feature2, pc1, pc2):
        # extract multi-scale local features for each point
        xyz1_new, pc1_features = self.pn_head(pc1.permute(0, 2, 1).contiguous(), feature1)
        xyz2_new, pc2_features = self.pn_head(pc2.permute(0, 2, 1).contiguous(), feature2)
        return feature1, pc1, pc1_features, pc2, pc2_features, xyz1_new, xyz2_new

    def clustering(self, pc1_features):
        objects = defaultdict(list)

        f_detached = torch.cat((pc1_features[0, 3:9, :], pc1_features[0, 10:12, :]), dim=0).detach().cpu().numpy().T
        # f_detached = torch.cat((pc1_features[0, 3:9, :], pc1_features[0, 10:12, :]), dim=0).detach().T
        if f_detached.shape[0] == 0:
            return []
        cluster_labels = self.dbscan.fit_predict(f_detached)
        # cluster_labels = cp.asnumpy(cluster_labels)

        for i in range(pc1_features.size(2)):
            if cluster_labels[i] != -1:
                objects[cluster_labels[i]].append(pc1_features[:, :, i].unsqueeze(2))

        objects_curr = []
        for key, obj in objects.items():
            objects_curr.append(torch.cat(obj, dim=2).cuda())

        return objects_curr

    def affinity_l(self, net, obj, objs_prev):
        l_aff = []
        for key_prev, obj_prev in objs_prev.items():
            aff = net(obj, obj_prev)
            l_aff.append(aff)
        return torch.tensor(l_aff)

    def association_module(self, objects, objects_curr, objects_prev):
        aff_list, aff_mat, m, n = self.affinity_module(objects_curr, objects_prev)
        indices0 = None
        indices1 = None
        confs = []
        if aff_mat.size(1) > 0 and aff_mat.size(0) > 0:
            try:
                indices1 = self.sinkhorn_module(aff_mat, indices1)
                # Associate with previous frame
                for i in range(n):
                    conf = aff_mat[0, indices1[0, i], i]
                    if indices1[0, i] == -1 or indices1[0, i] >= m or conf < 0.01:
                        objects[self.max_id] = objects_curr[i]
                        self.max_id += 1
                        confs.append(0)
                    else:
                        idx = list(objects_prev.keys())[indices1[0, i]]
                        objects[idx] = objects_curr[i]
                        confs.append(conf)
            except:
                for obj in objects_curr:
                    objects[self.max_id] = obj
                    self.max_id += 1
                    confs.append(0)
        else:
            for obj in objects_curr:
                objects[self.max_id] = obj
                self.max_id += 1
                confs.append(0)
        return aff_list, aff_mat, indices1, confs

    def sinkhorn_module(self, aff_mat_tensor, indices1):
        scores = log_optimal_transport(aff_mat_tensor, torch.tensor(0.9).cuda(), 500)
        max0, max1 = scores[:, :-1, :-1].max(2), scores[:, :-1, :-1].max(1)
        indices0, indices1 = max0.indices, max1.indices
        mutual0 = arange_like(indices0, 1)[None] == indices1.gather(1, indices0)
        mutual1 = arange_like(indices1, 1)[None] == indices0.gather(1, indices1)
        zero = scores.new_tensor(0)
        mscores0 = torch.where(mutual0, max0.values.exp(), zero)
        mscores1 = torch.where(mutual1, mscores0.gather(1, indices1), zero)
        # matching threshold
        valid0 = mutual0 & (mscores0 > 0)
        valid1 = mutual1 & valid0.gather(1, indices1)
        indices0 = torch.where(valid0, indices0, indices0.new_tensor(-1))
        indices1 = torch.where(valid1, indices1, indices1.new_tensor(-1))
        return indices1

    def affinity_module(self, objects_curr, objects_prev):
        # Construct M(old) * N(new) affinity matrix
        aff_mat = []
        aff_list = []
        m = len(objects_prev.keys())
        n = len(objects_curr)
        objects_curr_p = list()
        objects_prev_p = list()
        for i in range(m):
            row = []
            # for o2 in objects_curr:
            for j in range(n):
                if i >= m or j >= n:
                    objects_prev_p.append(torch.zeros(1, 1, 128 + 3 + 128 + 3 + 3, requires_grad=True).cuda())
                    objects_curr_p.append(torch.zeros(1, 1, 128 + 3 + 128 + 3 + 3, requires_grad=True).cuda())
                    continue
                o1_key = list(objects_prev.keys())[i]
                o2 = objects_curr[j]
                obj_feat1 = torch.max(o2[:, 11:(11 + 128), :], dim=2)[0].unsqueeze(1)
                obj_feat2 = torch.max(objects_prev[o1_key][:, 11:(11 + 256), :], dim=2)[0].unsqueeze(1)
                obj_flow1 = torch.mean(o2[:, 6:9, :], dim=2).unsqueeze(1)
                obj_flow2 = torch.mean(objects_prev[o1_key][:, 6:9, :], dim=2).unsqueeze(1)

                obj_pos1 = obj_centre(o2[:, 3:6, :]).unsqueeze(1)
                obj_pos2 = obj_centre(objects_prev[o1_key][:, 3:6, :]).unsqueeze(1)
                obj_rrv1 = torch.mean(o2[:, 9:11, :], dim=2).unsqueeze(1)
                obj_rrv2 = torch.mean(objects_prev[o1_key][:, 9:11, :], dim=2).unsqueeze(1)
                obj_rrv_var1 = torch.var(o2[:, 9:11, :], dim=2, unbiased=False).unsqueeze(1)
                obj_rrv_var2 = torch.var(objects_prev[o1_key][:, 9:11, :], dim=2, unbiased=False).unsqueeze(1)
                obj_var1 = torch.var(o2[:, 3:6, :], dim=2, unbiased=False).unsqueeze(1)
                obj_var2 = torch.var(objects_prev[o1_key][:, 3:6, :], dim=2, unbiased=False).unsqueeze(1)
                obj_curr = torch.cat((obj_pos1, obj_var1, obj_feat1, obj_flow1, obj_rrv1, obj_rrv_var1), dim=2)
                obj_prev = torch.cat((obj_pos2, obj_var2, obj_feat2, obj_flow2, obj_rrv2, obj_rrv_var2), dim=2)
                aff = self.affinity(obj_curr, obj_prev)
                row.append(aff)
                aff_list.append(aff.squeeze(0))
            aff_mat.append(row)
        aff_mat_tensor = torch.tensor(aff_mat).unsqueeze(0).cuda()
        if m != 0 and n != 0:
            aff_list = torch.cat(aff_list, dim=0)
            aff_mat_tensor = torch.reshape(aff_list, (m, n)).unsqueeze(0)
        return aff_list, aff_mat_tensor, m, n


class Affinity(nn.Module):
    def __init__(self, emb_dims=137):
        super(Affinity, self).__init__()
        self.emb_dims = emb_dims

        self.affinity = nn.Sequential(nn.Linear(emb_dims, emb_dims * 4),
                                      nn.ReLU(),
                                      nn.Linear(emb_dims * 4, emb_dims * 2),
                                      nn.ReLU(),
                                      nn.Linear(emb_dims * 2, emb_dims // 2),
                                      nn.ReLU(),
                                      nn.Linear(emb_dims // 2, emb_dims // 4),
                                      nn.ReLU(),
                                      nn.Linear(emb_dims // 4, 1),
                                      nn.Sigmoid())

    def forward(self, *input):
        src_embedding = input[0]
        tgt_embedding = input[1]
        affinity = self.affinity((src_embedding - tgt_embedding)[0])
        return affinity