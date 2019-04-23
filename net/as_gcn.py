import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from net.utils.graph import Graph
from net.utils.operate import *
from net.utils.module import *



class Model(nn.Module):

    def __init__(self, n_in_enc, n_hid_enc, edge_type_act, n_in_dec, n_hid_dec,
                       in_channels, num_class, graph_args, edge_importance_weighting, **kwargs):
        super().__init__()

        self.aim = AIM(n_in_enc, n_hid_enc, edge_type_act, n_in_dec, n_hid_dec, node_num=25)
        self.main_net = Main_net(edge_type_act, in_channels, num_class, graph_args, edge_importance_weighting, **kwargs)

    def forward(self, x, x_target, x_last, x_dnsp):
        A_act, alink_prob, aim_pred, aim_input = self.aim(x_dnsp)
        label_rec, data_pred, target = self.main_net(x, x_target, x_last, A_act)

        return label_rec, data_pred, target, alink_prob, aim_pred, aim_input



class Main_net(nn.Module):

    def __init__(self, edge_type_act, in_channels, num_class, graph_args, edge_importance_weighting, **kwargs):
        super().__init__()

        self.graph = Graph(**graph_args)
        A_skl = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A_skl', A_skl)
        self.edge_type_act = edge_type_act-1

        temporal_kernel_size = 7
        spatial_kernel_size = A_skl.size(0) + self.edge_type_act
        st_kernel_size = (temporal_kernel_size, spatial_kernel_size)
        
        self.data_bn = nn.BatchNorm1d(in_channels * A_skl.size(1))
       
        self.class_layer_0 = StgcnBlock(in_channels, 64, st_kernel_size, self.edge_type_act, stride=1, residual=False, **kwargs)
        self.class_layer_1 = StgcnBlock(64,  64,  st_kernel_size, self.edge_type_act, stride=1, **kwargs)
        self.class_layer_2 = StgcnBlock(64,  64,  st_kernel_size, self.edge_type_act, stride=1, **kwargs)
        self.class_layer_3 = StgcnBlock(64,  128, st_kernel_size, self.edge_type_act, stride=2, **kwargs)
        self.class_layer_4 = StgcnBlock(128, 128, st_kernel_size, self.edge_type_act, stride=1, **kwargs)
        self.class_layer_5 = StgcnBlock(128, 128, st_kernel_size, self.edge_type_act, stride=1, **kwargs)
        self.class_layer_6 = StgcnBlock(128, 256, st_kernel_size, self.edge_type_act, stride=2, **kwargs)
        self.class_layer_7 = StgcnBlock(256, 256, st_kernel_size, self.edge_type_act, stride=1, **kwargs)
        self.class_layer_8 = StgcnBlock(256, 256, st_kernel_size, self.edge_type_act, stride=1, **kwargs)

        self.recon_layer_0 = StgcnBlock(256, 128, st_kernel_size, self.edge_type_act, stride=1, **kwargs)
        self.recon_layer_1 = StgcnBlock(128, 128, st_kernel_size, self.edge_type_act, stride=2, **kwargs)     
        self.recon_layer_2 = StgcnBlock(128, 128, st_kernel_size, self.edge_type_act, stride=2, **kwargs) 
        self.recon_layer_3 = StgcnBlock(128, 128, st_kernel_size, self.edge_type_act, stride=2, **kwargs)
        self.recon_layer_4 = StgcnBlock(128, 128, (3, spatial_kernel_size), self.edge_type_act, stride=2, **kwargs) 
        self.recon_layer_5 = StgcnBlock(128, 128, (5, spatial_kernel_size), self.edge_type_act, stride=1, padding=False, residual=False, **kwargs)
        self.recon_layer_6 = StgcnReconBlock(128+3, 64, (1, spatial_kernel_size), self.edge_type_act, stride=1, padding=False, **kwargs)
        self.recon_layer_7 = StgcnReconBlock(64+3,  32, (1, spatial_kernel_size), self.edge_type_act, stride=1, padding=False, **kwargs)
        self.recon_layer_8 = StgcnReconBlock(32+3,  30, (1, spatial_kernel_size), self.edge_type_act, stride=1, padding=False, residual=False, activation=None, **kwargs)
        
        
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([nn.Parameter(torch.ones(self.A_skl.size())) for i in range(9)])
            self.edge_importance_recon = nn.ParameterList([nn.Parameter(torch.ones(self.A_skl.size())) for i in range(9)])
        else:
            self.edge_importance = [1] * (len(self.st_gcn_networks)+len(self.st_gcn_recon))
        self.fcn = nn.Conv2d(256, num_class, kernel_size=1)

    def forward(self, x, x_target, x_last, A_act):

        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N*M, V*C, T)
        x_bn = self.data_bn(x)
        x_bn = x_bn.view(N, M, V, C, T)
        x_bn = x_bn.permute(0, 1, 3, 4, 2).contiguous()
        x_bn = x_bn.view(N*M, C, T, V)

        # x_last = torch.cat((x_last,x_last,x_last,x_last,x_last), dim=2
        x_last = x_last.permute(0,4,1,2,3).contiguous()
        x_last = x_last.view(N * M, C, 1, V)
        x_target = x_target.permute(0,4,1,2,3).contiguous().view(N*M,C*10,V)
        # x_target = x_target.permute(0,4,3,1,2).contiguous().view(N*M, V*C, 10)
        # x_target_bn = self.data_bn(x_target)
        # x_target_bn = x_target_bn.view(N, M, V, C, 10)
        # x_target_bn = x_target_bn.permute(0, 1, 3, 4, 2).contiguous()
        # x_target_bn = x_target_bn.view(N * M, C, 10, V)

        h0, _ = self.class_layer_0(x_bn, self.A_skl*self.edge_importance[0], A_act)                   # [N, 64, 300, 25]
        h1, _ = self.class_layer_1(h0,   self.A_skl*self.edge_importance[1], A_act)                   # [N, 64, 300, 25]
        h2, _ = self.class_layer_2(h1,   self.A_skl*self.edge_importance[2], A_act)                   # [N, 64, 300, 25]
        h3, _ = self.class_layer_3(h2,   self.A_skl*self.edge_importance[3], A_act)                   # [N, 128, 150, 25]
        h4, _ = self.class_layer_4(h3,   self.A_skl*self.edge_importance[4], A_act)                   # [N, 128, 150, 25]
        h5, _ = self.class_layer_5(h4,   self.A_skl*self.edge_importance[5], A_act)                   # [N, 128, 150, 25]
        h6, _ = self.class_layer_6(h5,   self.A_skl*self.edge_importance[6], A_act)                   # [N, 256, 75, 25]
        h7, _ = self.class_layer_7(h6,   self.A_skl*self.edge_importance[7], A_act)                   # [N, 256, 75, 25]
        h8, _ = self.class_layer_8(h7,   self.A_skl*self.edge_importance[8], A_act)                   # [N, 256, 75, 25]

        x_class = F.avg_pool2d(h8, h8.size()[2:])
        x_class = x_class.view(N, M, -1, 1, 1).mean(dim=1)
        x_class = self.fcn(x_class)
        x_class = x_class.view(x_class.size(0), -1)

        r0, _ = self.recon_layer_0(h8, self.A_skl*self.edge_importance_recon[0], A_act)                          # [N, 128, 75, 25]
        r1, _ = self.recon_layer_1(r0, self.A_skl*self.edge_importance_recon[1], A_act)                          # [N, 128, 38, 25]
        r2, _ = self.recon_layer_2(r1, self.A_skl*self.edge_importance_recon[2], A_act)                          # [N, 128, 19, 25]
        r3, _ = self.recon_layer_3(r2, self.A_skl*self.edge_importance_recon[3], A_act)                          # [N, 128, 10, 25]
        r4, _ = self.recon_layer_4(r3, self.A_skl*self.edge_importance_recon[4], A_act)                          # [N, 128, 5, 25]
        r5, _ = self.recon_layer_5(r4, self.A_skl*self.edge_importance_recon[5], A_act)                          # [N, 128, 1, 25]
        r6, _ = self.recon_layer_6(torch.cat((r5, x_last),1), self.A_skl*self.edge_importance_recon[6], A_act)   # [N, 64, 1, 25]
        r7, _ = self.recon_layer_7(torch.cat((r6, x_last),1), self.A_skl*self.edge_importance_recon[7], A_act)   # [N, 32, 1, 25]
        r8, _ = self.recon_layer_8(torch.cat((r7, x_last),1), self.A_skl*self.edge_importance_recon[8], A_act)   # [N, 30, 1, 25]
        pred = x_last.squeeze().repeat(1,10,1) + r8.squeeze()                                                    # [N, 3, 25]

        return x_class, pred[::2, :, :], x_target[::2, :, :]