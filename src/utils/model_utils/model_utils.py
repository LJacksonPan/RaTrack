import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
import numpy as np

from models.utils.flowstep3d import GRU
from models.pointnet2_utils import PointNetSetAbstraction, PointNetSetAbstractionMsg
from models.update import ConvGRU
from lib import pointnet2_utils as pointutils
from lib.pointnet2_modules import PointnetFPModule, PointnetSAModuleMSG
from models.pointnet2_utils import PointNetFeaturePropagation

# from models.track4d import PNHead


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm;
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    dist = torch.maximum(dist, torch.zeros(dist.size()).cuda())
    return dist


def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, C]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    try:
        _, group_idx = torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=False)
    except:
        print()
    return group_idx


class MultiScaleEncoder(nn.Module):
    def __init__(self, radius, nsample, in_channel, mlp, mlp2):
        super(MultiScaleEncoder, self).__init__()

        self.ms_ls = nn.ModuleList()
        num_sas = len(radius)
        for l in range(num_sas):
            self.ms_ls.append(PointLocalFeature(radius[l],
                                                nsample[l], in_channel=in_channel, mlp=mlp, mlp2=mlp2))

    def forward(self, xyz, features):

        new_features = torch.zeros(0).cuda()

        for i, sa in enumerate(self.ms_ls):
            new_features = torch.cat((new_features, sa(xyz, features)), dim=1)

        return new_features


class PointLocalFeature(nn.Module):
    def __init__(self, radius, nsample, in_channel, mlp, mlp2):
        super(PointLocalFeature, self).__init__()
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        self.mlp2_convs = nn.ModuleList()
        self.mlp2_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1, bias=False))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

        last_channel = mlp[-1]
        for out_channel in mlp2:
            self.mlp2_convs.append(nn.Conv2d(last_channel, out_channel, 1, bias=False))
            self.mlp2_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

        self.queryandgroup = pointutils.QueryAndGroup(radius, nsample)

    def forward(self, xyz, points):

        device = xyz.device
        B, C, N = xyz.shape
        xyz_t = xyz.permute(0, 2, 1).contiguous()
        new_points = self.queryandgroup(xyz_t, xyz_t, points)

        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))

        new_points = torch.max(new_points, -1)[0].unsqueeze(2)

        for i, conv in enumerate(self.mlp2_convs):
            bn = self.mlp2_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        new_points = new_points.squeeze(2)

        return new_points


class FeatureCorrelator(nn.Module):
    def __init__(self, nsample, in_channel, mlp, bn=False, use_leaky=True):
        super(FeatureCorrelator, self).__init__()
        self.nsample = nsample
        self.bn = bn
        self.mlp_convs = nn.ModuleList()
        self.mlp2 = nn.ModuleList()
        if bn:
            self.mlp_bns = nn.ModuleList()
            self.mlp_bns2 = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            # add extra channel for motion seg
            # out_channel += 1
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            if bn:
                self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.cls_mlp = nn.Linear(16, 1)

        self.weightnet1 = WeightNet(3, last_channel)
        self.weightnet2 = WeightNet(3, last_channel)

        self.relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(0.1, inplace=True)

        self.sig = nn.Sigmoid()

    def forward(self, pc1, pc2, feature1, feature2):
        """
        Cost Volume layer for Flow Estimation
        Input:
            xyz1: input points position data, [B, C, N1]
            xyz2: input points position data, [B, C, N2]
            points1: input points data, [B, D, N1]
            points2: input points data, [B, D, N2]
        Return:
            new_points: upsample points feature data, [B, D', N1]
        """
        # import ipdb; ipdb.set_trace()

        B, C, N1 = pc1.shape
        _, _, N2 = pc2.shape
        _, D1, _ = feature1.shape
        _, D2, _ = feature2.shape
        pc1 = pc1.permute(0, 2, 1)
        pc2 = pc2.permute(0, 2, 1)
        feature1 = feature1.permute(0, 2, 1)
        feature2 = feature2.permute(0, 2, 1)

        # point-to-patch Volume
        knn_idx = knn_point(self.nsample, pc2, pc1)  # B, N1, nsample
        neighbor_xyz = index_points(pc2, knn_idx)
        direction_xyz = neighbor_xyz - pc1.reshape(B, N1, 1, C)

        grouped_feature2 = index_points(feature2, knn_idx)  # B, N1, nsample, D2
        grouped_feature1 = feature1.reshape(B, N1, 1, D1)
        grouped_feature1 = grouped_feature1.repeat(1, 1, self.nsample, 1)
        new_features = torch.cat([grouped_feature1, grouped_feature2, direction_xyz], dim=-1)  # B, N1, nsample, D1+D2+3
        new_features = new_features.permute(0, 3, 2, 1)  # [B, D1+D2+3, nsample, N1]

        for i, mlp in enumerate(self.mlp_convs):
            if self.bn:
                bn = self.mlp_bns[i]
                new_features = self.relu(bn(mlp(new_features)))
            else:
                new_features = self.relu(mlp(new_features))

        # weighted sum
        weights = self.weightnet1(direction_xyz.permute(0, 3, 2, 1))  # B C nsample N1

        new_features = torch.sum(weights * new_features, dim=2)  # B C N

        # Patch to Patch Cost
        knn_idx = knn_point(self.nsample, pc1, pc1)  # B, N1, nsample
        neighbor_xyz = index_points(pc1, knn_idx)
        direction_xyz = neighbor_xyz - pc1.view(B, N1, 1, C)

        # weights for group cost
        weights = self.weightnet2(direction_xyz.permute(0, 3, 2, 1))  # B C nsample N1
        new_features = index_points(new_features.permute(0, 2, 1),
                                    knn_idx)  # B, N1, nsample, C
        new_features = weights * new_features.permute(0, 3, 2, 1)
        new_features = torch.sum(new_features, dim=2)  # B C N

        return new_features


class FlowDecoder(nn.Module):
    def __init__(self, fc_inch, args):
        super(FlowDecoder, self).__init__()
        # multi-scale flow embeddings propogation
        # different scale share the same mlps hyper-parameters
        ep_radius = [2.0, 4.0, 8.0, 16.0]
        ep_inch = fc_inch * 2 + 5
        ep_mlp2s = [int(fc_inch / 8), int(fc_inch / 8), int(fc_inch / 8)]
        num_eps = len(ep_radius)
        self.mse = PNHead(args.npoints,  ep_inch)
        # scene flow predictor
        sf_inch = num_eps * ep_mlp2s[-1] * 2
        sf_mlps = [int(sf_inch / 2), int(sf_inch / 4), int(sf_inch / 8)]
        self.fp = FlowPredictor(in_channel=sf_inch, mlp=sf_mlps)
        self.cp = ClsPredictor(in_channel=sf_inch, mlp=sf_mlps)

        self.mlp2 = nn.ModuleList()
        last_channel2 = 3
        for out_channel in [1]:
            # add extra channel for motion seg
            self.mlp2.append(nn.Linear(last_channel2, out_channel))
            last_channel2 = out_channel

        self.gru2 = nn.GRU(input_size=fc_inch, hidden_size=fc_inch)
        self.sig = nn.Sigmoid()
        self.pnnGru = GRU(1024, fc_inch // 2, fc_inch // 2, False)
        self.torchGRU = nn.GRU(fc_inch // 2, fc_inch // 2, 5)

    def forward(self, pc1, feature1, pc1_features, cor_features, h):

        cls = self.cp(cor_features)

        if feature1 is not None:
            embeddings = torch.cat((feature1, pc1_features, cor_features), dim=1)
        else:
            embeddings = torch.cat((pc1_features, cor_features), dim=1)
        ## multi-scale flow embeddings propogation
        new_xyz, prop_features = self.mse(pc1.permute(0, 2, 1).contiguous(), embeddings)
        gfeat = torch.max(prop_features, -1)[0].unsqueeze(2)

        # No-GRU test 
        if h is None:
            h = torch.zeros(5, 1, 128).cuda()
        gfeat, h = self.torchGRU(gfeat.permute(2, 0, 1), h)
        gfeat = gfeat.permute(1, 2, 0)
        
        gfeat = gfeat.expand(prop_features.size()[0], prop_features.size()[1], pc1.size()[2])
        new_features = torch.cat((prop_features, gfeat), dim=1)

        ## initial scene flow prediction
        output = self.fp(new_features)

        return output, h, prop_features, cls


class FlowPredictor(nn.Module):
    def __init__(self, in_channel, mlp):
        super(FlowPredictor, self).__init__()
        self.sf_mlp = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.sf_mlp.append(nn.Sequential(nn.Conv2d(last_channel, out_channel, 1, bias=False),
                                             nn.BatchNorm2d(out_channel),
                                             nn.ReLU(inplace=False)))
            last_channel = out_channel

        self.conv2 = nn.Conv2d(mlp[-1], 3, 1, bias=False)

    def forward(self, feat):

        feat = feat.unsqueeze(3)
        for conv in self.sf_mlp:
            feat = conv(feat)

        output = self.conv2(feat)

        return output.squeeze(3)


class ClsPredictor(nn.Module):
    def __init__(self, in_channel, mlp):
        super(ClsPredictor, self).__init__()
        self.sf_mlp = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.sf_mlp.append(nn.Sequential(nn.Conv2d(last_channel, out_channel, 1, bias=False),
                                             nn.BatchNorm2d(out_channel),
                                             nn.ReLU(inplace=False)))
            last_channel = out_channel

        self.conv2 = nn.Conv2d(mlp[-1], 3, 1, bias=False)
        self.linear = nn.Linear(3, 1)
        self.sig = nn.Sigmoid()

    def forward(self, feat):

        feat = feat.unsqueeze(3)
        for conv in self.sf_mlp:
            feat = conv(feat)

        output = self.conv2(feat)
        output = self.linear(output.squeeze(3).permute(0, 2, 1))
        output = self.sig(output)

        return output.squeeze(2)

class WeightNet(nn.Module):

    def __init__(self, in_channel, out_channel, hidden_unit=[8, 8], bn=False):
        super(WeightNet, self).__init__()

        self.bn = bn
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        if hidden_unit is None or len(hidden_unit) == 0:
            self.mlp_convs.append(nn.Conv2d(in_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
        else:
            self.mlp_convs.append(nn.Conv2d(in_channel, hidden_unit[0], 1))
            self.mlp_bns.append(nn.BatchNorm2d(hidden_unit[0]))
            for i in range(1, len(hidden_unit)):
                self.mlp_convs.append(nn.Conv2d(hidden_unit[i - 1], hidden_unit[i], 1))
                self.mlp_bns.append(nn.BatchNorm2d(hidden_unit[i]))
            self.mlp_convs.append(nn.Conv2d(hidden_unit[-1], out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))

    def forward(self, localized_xyz):
        # xyz : BxCxKxN

        weights = localized_xyz
        for i, conv in enumerate(self.mlp_convs):
            if self.bn:
                bn = self.mlp_bns[i]
                weights = F.relu(bn(conv(weights)))
            else:
                weights = F.relu(conv(weights))

        return weights


class PNHead(nn.Module):
    def __init__(self, sample_point_num, in_channels):
        super(PNHead, self).__init__()

        self.sa1 = PointnetSAModuleMSG(npoint=sample_point_num, radii=[2, 4], nsamples=[4, 8], mlps=[[in_channels, 16, 16, 32], [in_channels, 16, 16, 32]])
        self.sa2 = PointnetSAModuleMSG(npoint=sample_point_num, radii=[4, 8], nsamples=[8, 16], mlps=[[3+32, 32, 32], [3+32, 32, 64]])
        self.sa3 = PointnetSAModuleMSG(npoint=sample_point_num, radii=[8, 16], nsamples=[16, 32], mlps=[[3+64, 64, 64], [3+64, 64, 64]])

        self.fp3 = PointnetFPModule(mlp=[128, 128])
        self.fp2 = PointnetFPModule(mlp=[160, 128])
        self.fp1 = PointnetFPModule(mlp=[128, 128])
        
        self.linear1 = nn.Linear(64, 32)
        self.linear2 = nn.Linear(96, 64)
        self.linear3 = nn.Linear(128, 64)

    def forward(self, pc, features):
        l0_points = features.contiguous()
        l0_xyz = pc.contiguous()

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l1_points = self.linear1(l1_points.permute(0, 2, 1)).permute(0, 2, 1).contiguous()
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l2_points = self.linear2(l2_points.permute(0, 2, 1)).permute(0, 2, 1).contiguous()
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l3_points = self.linear3(l3_points.permute(0, 2, 1)).permute(0, 2, 1).contiguous()
        
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)

        return l3_xyz, l0_points

class PNHead_UpScale_seg(nn.Module):
    def __init__(self, sample_point_num, in_channels):
        super(PNHead_UpScale_seg, self).__init__()
        self.fp3 = PointnetFPModule(mlp=[64+128+256+256, 128, 128])
        self.sa1 = PointnetSAModuleMSG(npoint=sample_point_num, 
                                       radii=[0.1, 0.4, 0.8, 1.6],
                                       nsamples=[4096, 1024, 256, 64],
                                       mlps=[[in_channels, 128, 128, 256], [in_channels, 128, 128, 256], [in_channels, 128, 128, 256], [in_channels, 128, 128, 256]])
        self.conv1 = nn.Conv1d(64, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(64, 1, 1)
        self.sig = nn.Sigmoid()
        self.lin = nn.Linear(32, 1)
        self.max_pool = nn.MaxPool1d(2)

    def forward(self, pc, features):
        l0_points = features
        l0_xyz = pc

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)

        x = self.drop1(F.leaky_relu(self.bn1(self.conv1(l0_points))))
        x = self.conv2(x)
        x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        x = x.permute(0, 2, 1)
        x = self.sig(x)

        return x
