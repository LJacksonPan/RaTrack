import torch
import torch.nn as nn
import torch.nn.functional as F
from .common import PointNetSetAbstraction, FlowEmbedding, PointNetFeaturePropogation


class Flow0Regressor(nn.Module):
    def __init__(self, npoint, use_instance_norm):
        super(Flow0Regressor, self).__init__()
        self.sa1 = PointNetSetAbstraction(npoint=int(npoint / 4), radius=None, nsample=32, in_channel=128, mlp=[128, 128, 128],
                                          group_all=False, use_instance_norm=use_instance_norm)
        self.fc = torch.nn.Linear(128, 3)

    def forward(self, pc1_l_loc, corr_feats):
        _, x = self.sa1(pc1_l_loc[2], corr_feats)
        x = x.permute(0, 2, 1).contiguous()
        x = self.fc(x)
        flow0 = x.permute(0, 2, 1).contiguous()
        return flow0


class FlowRegressor(nn.Module):
    def __init__(self, npoint, use_instance_norm):
        super(FlowRegressor, self).__init__()
        self.sa1 = PointNetSetAbstraction(npoint=int(npoint / 4), radius=None, nsample=32, in_channel=128, mlp=[128, 128, 128],
                                          group_all=False, use_instance_norm=use_instance_norm)
        self.sa2 = PointNetSetAbstraction(npoint=int(npoint / 4), radius=None, nsample=32, in_channel=128, mlp=[128, 128, 128],
                                          group_all=False, use_instance_norm=use_instance_norm)
        self.fc = torch.nn.Linear(128, 3)


    def forward(self, pc1_l_loc, corr_feats):
        _, x = self.sa1(pc1_l_loc[2], corr_feats)
        _, x = self.sa2(pc1_l_loc[2], x)
        x = x.permute(0, 2, 1).contiguous()
        x = self.fc(x)
        flow = x.permute(0, 2, 1).contiguous()
        return flow


class GlobalCorrLayer(nn.Module):
    def __init__(self, npoint, use_instance_norm):
        super(GlobalCorrLayer, self).__init__()
        self.support_th = 10 ** 2  # 10 m
        self.epsilon = torch.nn.Parameter(torch.zeros(1))
        self.fp0 = PointNetFeaturePropogation(in_channel=3, mlp=[])
        self.sa1 = PointNetSetAbstraction(npoint=int(npoint / 16), radius=None, nsample=16, in_channel=3, mlp=[32, 32, 64],
                                          group_all=False, use_instance_norm=use_instance_norm)
        self.fp1 = PointNetFeaturePropogation(in_channel=64, mlp=[])
        self.sa2 = PointNetSetAbstraction(npoint=int(npoint / 8), radius=None, nsample=16, in_channel=64, mlp=[64, 64, 128],
                                          group_all=False, use_instance_norm=use_instance_norm)
        self.fp2 = PointNetFeaturePropogation(in_channel=128, mlp=[])

    def calc_corr_mat(self, pcloud1, pcloud2, feature1, feature2):
        eps = torch.exp(self.epsilon) + 0.03
        distance_matrix = torch.sum(pcloud1 ** 2, -1, keepdim=True)
        distance_matrix = distance_matrix + torch.sum(
            pcloud2 ** 2, -1, keepdim=True
        ).transpose(1, 2)
        distance_matrix = distance_matrix - 2 * torch.bmm(pcloud1, pcloud2.transpose(1, 2))
        support = (distance_matrix < self.support_th).float()
        feature1 = feature1 / torch.sqrt(torch.sum(feature1 ** 2, -1, keepdim=True) + 1e-8)
        feature2 = feature2 / torch.sqrt(torch.sum(feature2 ** 2, -1, keepdim=True) + 1e-8)
        C = 1.0 - torch.bmm(feature1, feature2.transpose(1, 2))
        corr_mat = torch.exp(-C / eps) * support
        return corr_mat

    def forward(self, pc1_l_glob, pc2_l_glob, feats1_glob, feats2_glob):
        pcloud1 = pc1_l_glob[3]
        pcloud2 = pc2_l_glob[3]
        corr_mat = self.calc_corr_mat(pcloud1.permute(0, 2, 1), pcloud2.permute(0, 2, 1),
                                      feats1_glob.permute(0, 2, 1), feats2_glob.permute(0, 2, 1))
        row_sum = corr_mat.sum(-1, keepdim=True)
        flow0 = (corr_mat @ pcloud2.permute(0, 2, 1).contiguous()) / (row_sum + 1e-8) - pcloud1.permute(0, 2, 1).contiguous()

        flow0_us = self.fp0(pc1_l_glob[2], pc1_l_glob[3], None, flow0.permute(0, 2, 1).contiguous())
        _, corr_feats_l2 = self.sa1(pc1_l_glob[2], flow0_us)
        corr_feats_l1 = self.fp1(pc1_l_glob[1], pc1_l_glob[2], None, corr_feats_l2)
        _, corr_feats_l1 = self.sa2(pc1_l_glob[1], corr_feats_l1)
        corr_feats = self.fp2(pc1_l_glob[0], pc1_l_glob[1], None, corr_feats_l1)

        return corr_feats


class EncoderLoc(nn.Module):
    def __init__(self, npoint, use_instance_norm):
        super(EncoderLoc, self).__init__()
        self.sa1 = PointNetSetAbstraction(npoint=int(npoint / 2), radius=None, nsample=32, in_channel=3, mlp=[32, 32, 32],
                                          group_all=False, return_fps=True, use_instance_norm=use_instance_norm)
        self.sa2 = PointNetSetAbstraction(npoint=int(npoint / 4), radius=None, nsample=32, in_channel=32, mlp=[64, 64, 64],
                                          group_all=False, return_fps=True, use_instance_norm=use_instance_norm)

    def forward(self, pc, feature, fps_idx=None):
        fps_idx1 = fps_idx[0] if fps_idx is not None else None
        pc_l1, feat_l1, fps_idx1 = self.sa1(pc, feature, fps_idx=fps_idx1)
        fps_idx2 = fps_idx[1] if fps_idx is not None else None
        pc_l2, feat_l2, fps_idx2 = self.sa2(pc_l1, feat_l1, fps_idx=fps_idx2)
        pc_l = [pc, pc_l1, pc_l2]
        fps_idx = [fps_idx1, fps_idx2]
        return pc_l, feat_l2, fps_idx


class EncoderGlob(nn.Module):
    def __init__(self, npoint, use_instance_norm):
        super(EncoderGlob, self).__init__()
        self.sa1 = PointNetSetAbstraction(npoint=int(npoint / 8), radius=None, nsample=32, in_channel=64, mlp=[128, 128, 128],
                                                 group_all=False, use_instance_norm=use_instance_norm)
        self.sa2 = PointNetSetAbstraction(npoint=int(npoint / 16), radius=None, nsample=24, in_channel=128, mlp=[128, 128, 128],
                                                 group_all=False, use_instance_norm=use_instance_norm)
        self.sa3 = PointNetSetAbstraction(npoint=int(npoint / 32), radius=None, nsample=16, in_channel=128, mlp=[256, 256, 256],
                                                 group_all=False, use_instance_norm=use_instance_norm)

    def forward(self, pc, feature):
        pc_l1, feat_l1 = self.sa1(pc, feature)
        pc_l2, feat_l2 = self.sa2(pc_l1, feat_l1)
        pc_l3, feat_l3 = self.sa3(pc_l2, feat_l2)
        pc_l = [pc, pc_l1, pc_l2, pc_l3]
        return pc_l, feat_l3


class H0Net(nn.Module):
    def __init__(self, npoint, use_instance_norm):
        super(H0Net, self).__init__()
        self.sa1 = PointNetSetAbstraction(npoint=int(npoint / 4), radius=None, nsample=8, in_channel=64, mlp=[128, 128, 128],
                                        group_all=False, use_instance_norm=use_instance_norm)
        self.sa2 = PointNetSetAbstraction(npoint=int(npoint / 4), radius=None, nsample=8, in_channel=128, mlp=[128],
                                          group_all=False, use_act=False, use_instance_norm=use_instance_norm)

    def forward(self, pc, feature):
        _, feat_l1 = self.sa1(pc, feature)
        _, feat_l2 = self.sa2(pc, feat_l1)
        return feat_l2


class GRU(nn.Module):
    def __init__(self, npoint, hidden_dim, input_dim, use_instance_norm):
        super(GRU, self).__init__()
        in_ch = hidden_dim + input_dim
        self.convz = PointNetSetAbstraction(npoint=int(npoint), radius=None, nsample=4, in_channel=in_ch,
                                            mlp=[hidden_dim], group_all=False, use_act=False, use_instance_norm=use_instance_norm)
        self.convr = PointNetSetAbstraction(npoint=int(npoint), radius=None, nsample=4, in_channel=in_ch,
                                            mlp=[hidden_dim], group_all=False, use_act=False, use_instance_norm=use_instance_norm)
        self.convq = PointNetSetAbstraction(npoint=int(npoint), radius=None, nsample=4, in_channel=in_ch,
                                            mlp=[hidden_dim], group_all=False, use_act=False, use_instance_norm=use_instance_norm)

    def forward(self, h, x, pc):
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz(pc, hx)[1])
        r = torch.sigmoid(self.convr(pc, hx)[1])
        q = torch.tanh(self.convq(pc, torch.cat([r * h, x], dim=1))[1])
        h = (1 - z) * h + z * q
        return h


class FlowStep3D(nn.Module):
    def __init__(self, npoint=8192, use_instance_norm=False, loc_flow_nn=32, loc_flow_rad=1.5, k_decay_fact=1.0, **kwargs):
        super(FlowStep3D, self).__init__()
        self.k_decay_fact = k_decay_fact
        self.encoder_loc = EncoderLoc(npoint, use_instance_norm)
        self.encoder_glob = EncoderGlob(npoint, use_instance_norm)
        self.global_corr_layer = GlobalCorrLayer(npoint, use_instance_norm)
        self.h0_net = H0Net(npoint, use_instance_norm)
        self.flow0_regressor = Flow0Regressor(npoint, use_instance_norm)
        self.flow_regressor = FlowRegressor(npoint, use_instance_norm)
        self.local_corr_layer = FlowEmbedding(radius=loc_flow_rad, nsample=loc_flow_nn, in_channel=64, mlp=[128, 128, 128],
                                              pooling='max', corr_func='concat', use_instance_norm=use_instance_norm)
        self.gru = GRU(npoint, hidden_dim=128, input_dim=128+64+16+3, use_instance_norm=use_instance_norm)

        self.flow_conv1 = PointNetSetAbstraction(npoint=int(npoint / 4), radius=None, nsample=16, in_channel=3, mlp=[32, 32, 32],
                                          group_all=False, use_instance_norm=use_instance_norm)
        self.flow_conv2 = PointNetSetAbstraction(npoint=int(npoint / 4), radius=None, nsample=8, in_channel=32, mlp=[16, 16, 16],
                                          group_all=False, use_instance_norm=use_instance_norm)      

        self.flow_up_sample = PointNetFeaturePropogation(in_channel=3, mlp=[])


    def calc_glob_corr(self, pc1_loc, feats1_loc, pc2_loc, feats2_loc):
        pc1_l_glob, feats1_glob = self.encoder_glob(pc1_loc, feats1_loc)
        pc2_l_glob, feats2_glob = self.encoder_glob(pc2_loc, feats2_loc)
        corr_feats = self.global_corr_layer(pc1_l_glob, pc2_l_glob, feats1_glob, feats2_glob)
        return corr_feats

    def calc_h0(self, feats1_loc, pc):
        h = self.h0_net(pc, feats1_loc)
        h = torch.tanh(h)
        return h

    def get_x(self, feats1_loc_new, corr_feats, flow, pc):
        _, flow_feats = self.flow_conv1(pc, flow)
        _, flow_feats = self.flow_conv2(pc, flow_feats)
        x = torch.cat([feats1_loc_new, corr_feats, flow_feats, flow], dim=1) # [64, 128, 16, 3]

        return x

    def forward(self, pc1, pc2, feature1, feature2, iters=1):
        # ---------------------- init ----------------------------
        flow_predictions = []
        pc1 = pc1.permute(0, 2, 1).contiguous()
        pc2 = pc2.permute(0, 2, 1).contiguous()
        feature1 = feature1.permute(0, 2, 1).contiguous()  # B 3 N
        feature2 = feature2.permute(0, 2, 1).contiguous()  # B 3 N
        # --------------------------------------------------------

        pc1_l_loc, feats1_loc, fps_idx1 = self.encoder_loc(pc1, feature1)
        pc2_l_loc, feats2_loc, _ = self.encoder_loc(pc2, feature2)

        corr_feats = self.calc_glob_corr(pc1_l_loc[-1], feats1_loc, pc2_l_loc[-1], feats2_loc)
        flow0_lr = self.flow0_regressor(pc1_l_loc, corr_feats)

        flow0 = self.flow_up_sample(pc1_l_loc[0], pc1_l_loc[2], None, flow0_lr)
        flow_predictions.append(flow0.permute(0, 2, 1))

        h = self.calc_h0(feats1_loc, pc1_l_loc[-1])

        pc1_new = pc1 + flow0.detach()
        pc1_new_lr = pc1_l_loc[2] + flow0_lr.detach()
        for iter in range(iters-1):
            pc1_new = pc1_new.detach()
            pc1_new_lr = pc1_new_lr.detach()
            flow_lr = pc1_new_lr - pc1_l_loc[2]

            pc1_new_l_loc, feats1_loc_new, _ = self.encoder_loc(pc1_new, pc1_new, fps_idx1)
            _, corr_feats = self.local_corr_layer(pc1_new_l_loc[-1], pc2_l_loc[-1], feats1_loc_new, feats2_loc)
            
            x = self.get_x(feats1_loc_new, corr_feats, flow_lr, pc=pc1_l_loc[2])
            h = self.gru(h=h, x=x, pc=pc1_l_loc[-1])
            delta_flow_lr = self.flow_regressor(pc1_l_loc, h)
            delta_flow_lr = delta_flow_lr / (self.k_decay_fact*iter + 1)
            pc1_new_lr = pc1_new_lr + delta_flow_lr
            
            delta_flow = self.flow_up_sample(pc1_l_loc[0], pc1_l_loc[2], None, delta_flow_lr)
            pc1_new = pc1_new + delta_flow
            flow = pc1_new - pc1
            flow_predictions.append(flow.permute(0, 2, 1))
        return flow_predictions


if __name__ == '__main__':
    import torch

    npoint = 512
    device = torch.device("cuda")
    model = FlowStep3D(npoint)
    model = model.to(device)
    print(f'num_params:{sum(p.numel() for p in model.parameters() if p.requires_grad)}')
    input = torch.randn((1, npoint, 3))
    input = input.to(torch.float32).to(device)

    output = model(input, input, input, input, iters=2)
    print(r'outputs shape:')
    for output_i in output:
        print(f'{output_i.shape}')
