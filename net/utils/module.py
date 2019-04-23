import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from net.utils.graph import Graph
from net.utils.operate import *


class StgcnBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, edge_type=3,
                 t_kernel_size=1, stride=1, padding=True, dropout=0,
                 residual=True, activation='relu'):
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        if padding == True:
            padding = ((kernel_size[0] - 1) // 2, 0)
        else:
            padding = (0,0)

        self.gcn = SpatialGcn(in_channels, out_channels, kernel_size[1], edge_type, t_kernel_size=t_kernel_size)
        self.tcn = nn.Sequential(nn.BatchNorm2d(out_channels),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(out_channels, out_channels, (kernel_size[0], 1), (stride, 1), padding),
                                 nn.BatchNorm2d(out_channels),
                                 nn.Dropout(dropout, inplace=True))

        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                                          nn.BatchNorm2d(out_channels))
        self.relu = nn.ReLU(inplace=True)
        self.activation = activation

    def forward(self, x, A, B):
        res = self.residual(x)
        x, A = self.gcn(x, A, B)
        x = self.tcn(x) + res
        if self.activation == 'relu':
            x = self.relu(x)
        else:
            x = x
        return x, A


class SpatialGcn(nn.Module):

    def __init__(self, in_channels, out_channels, k_num, edge_type=3, lamda=0.5,
                 t_kernel_size=1, t_stride=1, t_padding=0, bias=True):
        super().__init__()

        self.k_num = k_num
        self.edge_type = edge_type
        self.lamda = lamda
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels*k_num,
                              kernel_size=(t_kernel_size, 1),
                              padding=(t_padding, 0),
                              stride=(t_stride, 1),
                              bias=bias)

    def forward(self, x, A, B):
        x = self.conv(x)
        n, kc, t, v = x.size()
        x = x.view(n, self.k_num,  kc//self.k_num, t, v)
        x1 = x[:,:self.k_num-self.edge_type,:,:,:]
        x2 = x[:,-self.edge_type:,:,:,:]
        x1 = torch.einsum('nkctv,kvw->nctw', (x1, A))
        x2 = torch.einsum('nkctv,nkvw->nctw', (x2, B))
        x = x1+x2*self.lamda
        return x.contiguous(), A


class StgcnReconBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, edge_type=3,
                 t_kernel_size=1, stride=1, padding=True, dropout=0,
                 residual=True, activation='relu'):
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        if padding == True:
            padding = ((kernel_size[0]-1)//2, 0)
        else:
            padding = (0,0)
        
        self.gcn_recon = SpatialGcnPred(in_channels, out_channels, kernel_size[1], edge_type, t_kernel_size, )
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

    def forward(self, x, A, B):

        res = self.residual(x)
        x, A = self.gcn_recon(x, A, B)
        x = self.tcn_recon(x) + res
        if self.activation == 'relu':
            x = self.relu(x)
        else:
            x = x
        return x, A


class SpatialGcnPred(nn.Module):

    def __init__(self, in_channels, out_channels, k_num, edge_type=3, lamda=0.5,
                 t_kernel_size=1, t_stride=1, t_padding=0, t_outpadding=0, bias=True):
        super().__init__()

        self.k_num = k_num
        self.edge_type = edge_type
        self.lamda = lamda
        self.deconv = nn.ConvTranspose2d(in_channels=in_channels,
                                         out_channels=out_channels*k_num,
                                         kernel_size=(t_kernel_size, 1),
                                         padding=(t_padding, 0),
                                         output_padding=(t_outpadding, 0),
                                         stride=(t_stride, 1),
                                         bias=bias)

    def forward(self, x, A, B):
        x = self.deconv(x)
        n, kc, t, v = x.size()
        x = x.view(n, self.k_num,  kc//self.k_num, t, v)
        x1 = x[:,:self.k_num-self.edge_type,:,:,:]
        x2 = x[:,-self.edge_type:,:,:,:]
        x1 = torch.einsum('nkctv,kvw->nctw', (x1, A))
        x2 = torch.einsum('nkctv,nkvw->nctw', (x2, B))
        x = x1+x2*self.lamda
        return x.contiguous(), A


class MLP(nn.Module):

    def __init__(self, n_in, n_hid, n_out, do_prob=0.):
        super().__init__()
        self.fc1 = nn.Linear(n_in, n_hid)
        self.fc2 = nn.Linear(n_hid, n_out)
        self.bn = nn.BatchNorm1d(n_out)
        self.dropout = nn.Dropout(p=do_prob)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def batch_norm(self, inputs):
        x = inputs.view(inputs.size(0) * inputs.size(1), -1)
        x = self.bn(x)
        return x.view(inputs.size(0), inputs.size(1), -1)
        
    def forward(self, inputs):
        x = F.elu(self.fc1(inputs))
        x = self.dropout(x)
        x = F.elu(self.fc2(x))
        return self.batch_norm(x)


class Encoder(nn.Module):

    def __init__(self, n_in, n_hid, n_out, do_prob=0., factor=True):
        super().__init__()
        self.factor = factor
        self.mlp1 = MLP(n_in, n_hid, n_hid, do_prob)
        self.mlp2 = MLP(n_hid*2, n_hid, n_hid, do_prob)
        self.mlp3 = MLP(n_hid, n_hid, n_hid, do_prob)
        self.mlp4 = MLP(n_hid*3, n_hid, n_hid, do_prob) if self.factor else MLP(n_hid*2, n_hid, n_hid, do_prob)
        self.fc_out = nn.Linear(n_hid, n_out)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)

    def node2edge(self, x, rel_rec, rel_send):
        receivers = torch.matmul(rel_rec, x)
        senders = torch.matmul(rel_send, x)
        edges = torch.cat([receivers, senders], dim=2)
        return edges

    def edge2node(self, x, rel_rec, rel_send):
        incoming = torch.matmul(rel_rec.t(), x)
        nodes = incoming / incoming.size(1)
        return nodes
    
    def forward(self, inputs, rel_rec, rel_send):              # input: [N, v, t, c] = [N, 25, 50, 3]
        x = inputs.contiguous()                                # x: [N, 25, 50, 4]
        x = x.view(inputs.size(0), inputs.size(1), -1)         # [N, 25, 50, 3] -> [N, 25, 50*3=150]
        x = self.mlp1(x)                                       # [N, 25, 150] -> [N, 25, n_hid=256] -> [N, 25, n_out=256]
        x = self.node2edge(x, rel_rec, rel_send)               # [N, 25, 256] -> [N, 600, 256]|[N, 600, 256]=[N, 600, 512]
        x = self.mlp2(x)                                       # [N, 600, 512] -> [N, 600, n_hid=256] -> [N, 600, n_out=256]
        x_skip = x
        if self.factor:
            x = self.edge2node(x, rel_rec, rel_send)           # [N, 600, 256] -> [N, 25, 256]
            x = self.mlp3(x)                                   # [N, 25, 256] -> [N, 25, n_hid=256] -> [N, 25, n_out=256]
            x = self.node2edge(x, rel_rec, rel_send)           # [N, 25, 256] -> [N, 600, 256]|[N, 600, 256]=[N, 600, 512]
            x = torch.cat((x, x_skip), dim=2)                  # [N, 600, 512] -> [N, 600, 512]|[N, 600, 256]=[N, 600, 768]
            x = self.mlp4(x)                                   # [N, 600, 768] -> [N, 600, n_hid=256] -> [N, 600, n_out=256]
        else:
            x = self.mlp3(x)
            x = torch.cat((x, x_skip), dim=2)
            x = self.mlp4(x)
        return self.fc_out(x)                                  # [N, 600, 256] -> [N, 600, 4]


class Decoder(nn.Module):
    
    def __init__(self, n_in_node, edge_types, n_hid, do_prob=0., skip_first=True):
        super().__init__()

        self.msg_fc1 = nn.ModuleList([nn.Linear(2*n_hid, n_hid) for _ in range(edge_types)])
        self.msg_fc2 = nn.ModuleList([nn.Linear(n_hid, n_hid) for _ in range(edge_types)])
        self.msg_out_shape = n_hid
        self.skip_first_edge_type = skip_first

        self.hidden_r = nn.Linear(n_hid, n_hid, bias=False)
        self.hidden_i = nn.Linear(n_hid, n_hid, bias=False)
        self.hidden_n = nn.Linear(n_hid, n_hid, bias=False)

        self.input_r = nn.Linear(n_in_node, n_hid, bias=True)  # 3 x 256
        self.input_i = nn.Linear(n_in_node, n_hid, bias=True)
        self.input_n = nn.Linear(n_in_node, n_hid, bias=True)

        self.out_fc1 = nn.Linear(n_hid, n_hid)
        self.out_fc2 = nn.Linear(n_hid, n_hid)
        self.out_fc3 = nn.Linear(n_hid, n_in_node)

        self.dropout1 = nn.Dropout(p=do_prob)
        self.dropout2 = nn.Dropout(p=do_prob)
        self.dropout3 = nn.Dropout(p=do_prob)

    def single_step_forward(self, inputs, rel_rec, rel_send, rel_type, hidden):
        receivers = torch.matmul(rel_rec, hidden)
        senders = torch.matmul(rel_send, hidden)
        pre_msg = torch.cat([receivers, senders], dim=-1)
        all_msgs = torch.zeros(pre_msg.size(0), pre_msg.size(1), self.msg_out_shape)
        gpu_id = rel_rec.get_device()
        all_msgs = all_msgs.cuda(gpu_id)
        if self.skip_first_edge_type:
            start_idx = 1
            norm = len(self.msg_fc2)-1.
        else:
            start_idx = 0
            norm = len(self.msg_fc2)
        for k in range(start_idx, len(self.msg_fc2)):
            msg = torch.tanh(self.msg_fc1[k](pre_msg))
            msg = self.dropout1(msg)
            msg = torch.tanh(self.msg_fc2[k](msg))
            msg = msg * rel_type[:, :, k:k + 1]                               # rel_type: [N, 600, 4]
            all_msgs += msg / norm
        agg_msgs = all_msgs.transpose(-2, -1).matmul(rel_rec).transpose(-2, -1)
        agg_msgs = agg_msgs.contiguous()/inputs.size(2)

        r = torch.sigmoid(self.input_r(inputs) + self.hidden_r(agg_msgs))
        i = torch.sigmoid(self.input_i(inputs) + self.hidden_i(agg_msgs))
        n = torch.tanh(self.input_n(inputs) + r * self.hidden_n(agg_msgs))
        hidden = (1-i)*n + i*hidden

        pred = self.dropout2(F.relu(self.out_fc1(hidden)))
        pred = self.dropout2(F.relu(self.out_fc2(pred)))
        pred = self.out_fc3(pred)
        pred = inputs + pred

        return pred, hidden

    def forward(self, data, rel_type, rel_rec, rel_send, pred_steps=1):
        inputs = data.transpose(1, 2).contiguous()
        time_steps = inputs.size(1)
        hidden = torch.zeros(inputs.size(0), inputs.size(2), self.msg_out_shape)
        gpu_id = rel_rec.get_device()
        hidden = hidden.cuda(gpu_id)
        pred_all = []
        for step in range(0, inputs.size(1) - 1):
            if not step % pred_steps:
                ins = inputs[:, step, :, :]
            else:
                ins = pred_all[step - 1]
            pred, hidden = self.single_step_forward(ins, rel_rec, rel_send, rel_type, hidden)
            pred_all.append(pred)

        preds = torch.stack(pred_all, dim=1)
        return preds.transpose(1, 2).contiguous()


class AIM(nn.Module):

    def __init__(self, n_in_enc, n_hid_enc, edge_types, n_in_dec, n_hid_dec, node_num=25):
        super().__init__()

        self.encoder = Encoder(n_in=n_in_enc, n_hid=n_hid_enc, n_out=edge_types, do_prob=0.5, factor=True)
        self.decoder = Decoder(n_in_node=n_in_dec, edge_types=edge_types, n_hid=n_hid_dec, do_prob=0.5, skip_first=True)
        self.off_diag = np.ones([node_num, node_num])-np.eye(node_num, node_num)
        self.rel_rec = torch.FloatTensor(np.array(encode_onehot(np.where(self.off_diag)[1]), dtype=np.float32))
        self.rel_send = torch.FloatTensor(np.array(encode_onehot(np.where(self.off_diag)[0]), dtype=np.float32))
        self.offdiag_indices, _ = get_offdiag_indices(node_num)
        
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, inputs):                         # [N, 3, 50, 25, 2]
        gpu_id = inputs.get_device()
        N, C, T, V, M = inputs.size()
        x = inputs.permute(0, 4, 3, 1, 2).contiguous()       # [N, 2, 25, 3, 50]
        x = x.view(N * M, V, C, T)                           # [2N, 25, 3, 50]
        x = x.permute(0, 1, 3, 2)
        self.rel_rec = self.rel_rec.cuda(gpu_id)
        self.rel_send = self.rel_send.cuda(gpu_id)

        self.logits = self.encoder(x, self.rel_rec, self.rel_send)
        self.N, self.e, self.c = self.logits.shape           # [N, 300, 4]
        self.edges = gumbel_softmax(self.logits, tau=0.5, hard=True)
        self.prob = my_softmax(self.logits, -1)
        self.outputs = self.decoder(x, self.edges, self.rel_rec, self.rel_send)
        self.offdiag_indices = self.offdiag_indices.cuda(gpu_id)
        print(self.edges)
        
        A_batch = []
        for i in range(self.N):
            A_types = []
            for j in range(1, self.c):
                A = torch.sparse.FloatTensor(self.offdiag_indices, self.edges[i,:,j], torch.Size([25, 25])).to_dense().cuda(gpu_id)
                A = A + torch.eye(V, V).cuda(gpu_id)
                D = torch.sum(A, dim=0).squeeze().pow(-1)
                D = torch.diag(D).cuda(gpu_id)
                A_ = torch.matmul(A, D)
                A_types.append(A_)
            A_types = torch.stack(A_types)
            A_batch.append(A_types)
        self.A_batch = torch.stack(A_batch).cuda(gpu_id) # [N, 3, 25, 25]

        return self.A_batch, self.prob, self.outputs, x