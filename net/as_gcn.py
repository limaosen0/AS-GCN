import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from net.utils.graph import Graph


class Model(nn.Module):

    def __init__(self, in_channels, num_class, graph_args,
                 edge_importance_weighting, **kwargs):
        super().__init__()

        self.graph = Graph(**graph_args)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)
        self.edge_type = 2

        temporal_kernel_size = 9
        spatial_kernel_size = A.size(0) + self.edge_type
        st_kernel_size = (temporal_kernel_size, spatial_kernel_size)
        
        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))

        self.class_layer_0 = StgcnBlock(in_channels, 64, st_kernel_size, self.edge_type, stride=1, residual=False, **kwargs)
        self.class_layer_1 = StgcnBlock(64, 64, st_kernel_size, self.edge_type, stride=1, **kwargs)
        self.class_layer_2 = StgcnBlock(64, 64, st_kernel_size, self.edge_type, stride=1, **kwargs)
        self.class_layer_3 = StgcnBlock(64, 128, st_kernel_size, self.edge_type, stride=2, **kwargs)
        self.class_layer_4 = StgcnBlock(128, 128, st_kernel_size, self.edge_type, stride=1, **kwargs)
        self.class_layer_5 = StgcnBlock(128, 128, st_kernel_size, self.edge_type, stride=1, **kwargs)
        self.class_layer_6 = StgcnBlock(128, 256, st_kernel_size, self.edge_type, stride=2, **kwargs)
        self.class_layer_7 = StgcnBlock(256, 256, st_kernel_size, self.edge_type, stride=1, **kwargs)
        self.class_layer_8 = StgcnBlock(256, 256, st_kernel_size, self.edge_type, stride=1, **kwargs)

        self.recon_layer_0 = StgcnBlock(256, 128, st_kernel_size, self.edge_type, stride=1, **kwargs)
        self.recon_layer_1 = StgcnBlock(128, 128, st_kernel_size, self.edge_type, stride=2, **kwargs)     
        self.recon_layer_2 = StgcnBlock(128, 128, st_kernel_size, self.edge_type, stride=2, **kwargs) 
        self.recon_layer_3 = StgcnBlock(128, 128, st_kernel_size, self.edge_type, stride=2, **kwargs)
        self.recon_layer_4 = StgcnBlock(128, 128, (3, spatial_kernel_size), self.edge_type, stride=2, **kwargs) 
        self.recon_layer_5 = StgcnBlock(128, 128, (5, spatial_kernel_size), self.edge_type, stride=1, padding=False, residual=False, **kwargs)
        self.recon_layer_6 = StgcnReconBlock(128+3, 30, (1, spatial_kernel_size), self.edge_type, stride=1, padding=False, residual=False, activation=None, **kwargs)


        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([nn.Parameter(torch.ones(self.A.size())) for i in range(9)])
            self.edge_importance_recon = nn.ParameterList([nn.Parameter(torch.ones(self.A.size())) for i in range(9)])
        else:
            self.edge_importance = [1] * (len(self.st_gcn_networks)+len(self.st_gcn_recon))
        self.fcn = nn.Conv2d(256, num_class, kernel_size=1)

    def forward(self, x, x_target, x_last, A_act, lamda_act):
        N, C, T, V, M = x.size()
        x_recon = x[:,:,:,:,0]                                  # [2N, 3, 300, 25]  wsx: x_recon(4,3,290,25) select the first person data?
        x = x.permute(0, 4, 3, 1, 2).contiguous()               # [N, 2, 25, 3, 300] wsx: x(4,2,25,3,290)
        x = x.view(N * M, V * C, T)                             # [2N, 75, 300]m wsx: x(8,75,290)

        x_last = x_last.permute(0,4,1,2,3).contiguous().view(-1,3,1,25)  #(2N,3,1,25)
        
        x_bn = self.data_bn(x)
        x_bn = x_bn.view(N, M, V, C, T)
        x_bn = x_bn.permute(0, 1, 3, 4, 2).contiguous()
        x_bn = x_bn.view(N * M, C, T, V) #2N,3,290,25

        h0, _ = self.class_layer_0(x_bn, self.A * self.edge_importance[0], A_act, lamda_act)       # [N, 64, 300, 25]
        h1, _ = self.class_layer_1(h0, self.A * self.edge_importance[1], A_act, lamda_act)         # [N, 64, 300, 25]
        h1, _ = self.class_layer_1(h0, self.A * self.edge_importance[1], A_act, lamda_act)         # [N, 64, 300, 25]
        h2, _ = self.class_layer_2(h1, self.A * self.edge_importance[2], A_act, lamda_act)         # [N, 64, 300, 25]
        h3, _ = self.class_layer_3(h2, self.A * self.edge_importance[3], A_act, lamda_act)         # [N, 128, 150, 25]
        h4, _ = self.class_layer_4(h3, self.A * self.edge_importance[4], A_act, lamda_act)         # [N, 128, 150, 25]
        h5, _ = self.class_layer_5(h4, self.A * self.edge_importance[5], A_act, lamda_act)         # [N, 128, 150, 25]
        h6, _ = self.class_layer_6(h5, self.A * self.edge_importance[6], A_act, lamda_act)         # [N, 256, 75, 25]
        h7, _ = self.class_layer_7(h6, self.A * self.edge_importance[7], A_act, lamda_act)         # [N, 256, 75, 25]
        h8, _ = self.class_layer_8(h7, self.A * self.edge_importance[8], A_act, lamda_act)         # [N, 256, 75, 25]

        x_class = F.avg_pool2d(h8, h8.size()[2:])  #(8,256,1,1)
        x_class = x_class.view(N, M, -1, 1, 1).mean(dim=1) #(4,256,1,1)
        x_class = self.fcn(x_class) #(4,60,1,1)  Conv2d(256, 60, kernel_size=(1, 1), stride=(1, 1))
        x_class = x_class.view(x_class.size(0), -1) #(4,60)

        r0, _ = self.recon_layer_0(h8, self.A*self.edge_importance_recon[0], A_act, lamda_act)                          # [N, 128, 75, 25]
        r1, _ = self.recon_layer_1(r0, self.A*self.edge_importance_recon[1], A_act, lamda_act)                          # [N, 128, 38, 25]
        r2, _ = self.recon_layer_2(r1, self.A*self.edge_importance_recon[2], A_act, lamda_act)                          # [N, 128, 19, 25]
        r3, _ = self.recon_layer_3(r2, self.A*self.edge_importance_recon[3], A_act, lamda_act)                          # [N, 128, 10, 25]
        r4, _ = self.recon_layer_4(r3, self.A*self.edge_importance_recon[4], A_act, lamda_act)                          # [N, 128, 5, 25]
        r5, _ = self.recon_layer_5(r4, self.A*self.edge_importance_recon[5], A_act, lamda_act)                          # [N, 128, 1, 25]
        r6, _ = self.recon_layer_6(torch.cat((r5, x_last),1), self.A*self.edge_importance_recon[6], A_act, lamda_act)   # [N, 64, 1, 25]  wsx:(8,30,1,25)
        pred = x_last.squeeze().repeat(1,10,1) + r6.squeeze()                                                  # [N, 3, 25] wsx:(8,30,25)

        pred = pred.contiguous().view(-1, 3, 10, 25)
        x_target = x_target.permute(0,4,1,2,3).contiguous().view(-1,3,10,25)

        return x_class, pred[::2], x_target[::2]

    def extract_feature(self, x):

        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        _, c, t, v = x.size()
        feature = x.view(N, M, c, t, v).permute(0, 2, 3, 4, 1)

        x = self.fcn(x)
        output = x.view(N, M, -1, t, v).permute(0, 2, 3, 4, 1)

        return output, feature


class StgcnBlock(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 edge_type=2,
                 t_kernel_size=1,
                 stride=1,
                 padding=True,
                 dropout=0,
                 residual=True):
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        if padding == True:
            padding = ((kernel_size[0] - 1) // 2, 0)
        else:
            padding = (0,0)

        self.gcn = SpatialGcn(in_channels=in_channels,
                              out_channels=out_channels,
                              k_num=kernel_size[1],
                              edge_type=edge_type,
                              t_kernel_size=t_kernel_size)
        self.tcn = nn.Sequential(nn.BatchNorm2d(out_channels),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(out_channels,
                                           out_channels,
                                           (kernel_size[0], 1),
                                           (stride, 1),
                                           padding),
                                 nn.BatchNorm2d(out_channels),
                                 nn.Dropout(dropout, inplace=True))

        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=(stride, 1)),
                                          nn.BatchNorm2d(out_channels))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A, B, lamda_act):

        res = self.residual(x)
        x, A = self.gcn(x, A, B, lamda_act)
        x = self.tcn(x) + res

        return self.relu(x), A


class StgcnReconBlock(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 edge_type=2,
                 t_kernel_size=1,
                 stride=1,
                 padding=True,
                 dropout=0,
                 residual=True,
                 activation='relu'):
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1

        if padding == True:
            padding = ((kernel_size[0] - 1) // 2, 0)
        else:
            padding = (0,0)

        self.gcn_recon = SpatialGcnRecon(in_channels=in_channels,
                                         out_channels=out_channels,
                                         k_num=kernel_size[1],
                                         edge_type=edge_type,
                                         t_kernel_size=t_kernel_size)
        self.tcn_recon = nn.Sequential(nn.BatchNorm2d(out_channels),
                                       nn.ReLU(inplace=True),
                                       nn.ConvTranspose2d(in_channels=out_channels,
                                                          out_channels=out_channels,
                                                          kernel_size=(kernel_size[0], 1),
                                                          stride=(stride, 1),
                                                          padding=padding,
                                                          output_padding=(stride-1,0)),
                                       nn.BatchNorm2d(out_channels),
                                       nn.Dropout(dropout, inplace=True))

        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(nn.ConvTranspose2d(in_channels=in_channels,
                                                             out_channels=out_channels,
                                                             kernel_size=1,
                                                             stride=(stride, 1),
                                                             output_padding=(stride-1,0)),
                                          nn.BatchNorm2d(out_channels))
        self.relu = nn.ReLU(inplace=True)
        self.activation = activation

    def forward(self, x, A, B, lamda_act):

        res = self.residual(x)
        x, A = self.gcn_recon(x, A, B, lamda_act)
        x = self.tcn_recon(x) + res
        if self.activation == 'relu':
            x = self.relu(x)
        else:
            x = x

        return x, A


class SpatialGcn(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 k_num,
                 edge_type=2,
                 t_kernel_size=1,
                 t_stride=1,
                 t_padding=0,
                 t_dilation=1,
                 bias=True):
        super().__init__()

        self.k_num = k_num
        self.edge_type = edge_type
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels*k_num,
                              kernel_size=(t_kernel_size, 1),
                              padding=(t_padding, 0),
                              stride=(t_stride, 1),
                              dilation=(t_dilation, 1),
                              bias=bias)

    def forward(self, x, A, B, lamda_act):

        x = self.conv(x)
        n, kc, t, v = x.size()
        x = x.view(n, self.k_num,  kc//self.k_num, t, v)
        x1 = x[:,:self.k_num-self.edge_type,:,:,:]
        x2 = x[:,-self.edge_type:,:,:,:]
        x1 = torch.einsum('nkctv,kvw->nctw', (x1, A))
        x2 = torch.einsum('nkctv,nkvw->nctw', (x2, B))
        x_sum = x1+x2*lamda_act

        return x_sum.contiguous(), A


class SpatialGcnRecon(nn.Module):

    def __init__(self, in_channels, out_channels, k_num, edge_type=3,
                 t_kernel_size=1, t_stride=1, t_padding=0, t_outpadding=0, t_dilation=1,
                 bias=True):
        super().__init__()

        self.k_num = k_num
        self.edge_type = edge_type
        self.deconv = nn.ConvTranspose2d(in_channels=in_channels,
                                         out_channels=out_channels*k_num,
                                         kernel_size=(t_kernel_size, 1),
                                         padding=(t_padding, 0),
                                         output_padding=(t_outpadding, 0),
                                         stride=(t_stride, 1),
                                         dilation=(t_dilation, 1),
                                         bias=bias)

    def forward(self, x, A, B, lamda_act):

        x = self.deconv(x)
        n, kc, t, v = x.size()
        x = x.view(n, self.k_num,  kc//self.k_num, t, v)
        x1 = x[:,:self.k_num-self.edge_type,:,:,:]
        x2 = x[:,-self.edge_type:,:,:,:]
        x1 = torch.einsum('nkctv,kvw->nctw', (x1, A))
        x2 = torch.einsum('nkctv,nkvw->nctw', (x2, B))
        x_sum = x1+x2*lamda_act

        return x_sum.contiguous(), A
