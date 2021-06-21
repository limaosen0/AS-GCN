import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from torch.autograd import Variable


def my_softmax(input, axis=1):
	trans_input = input.transpose(axis, 0).contiguous()
	soft_max_1d = F.softmax(trans_input)
	return soft_max_1d.transpose(axis, 0)


def get_offdiag_indices(num_nodes):
	ones = torch.ones(num_nodes, num_nodes)
	eye = torch.eye(num_nodes, num_nodes)
	offdiag_indices = (ones - eye).nonzero().t()
	offdiag_indices_ = offdiag_indices[0] * num_nodes + offdiag_indices[1]
	return offdiag_indices, offdiag_indices_


def gumbel_softmax(logits, tau=1, hard=False, eps=1e-10):
	y_soft = gumbel_softmax_sample(logits, tau=tau, eps=eps)
	if hard:
		shape = logits.size()
		_, k = y_soft.data.max(-1)
		y_hard = torch.zeros(*shape)
		if y_soft.is_cuda:
			y_hard = y_hard.cuda()
		y_hard = y_hard.zero_().scatter_(-1, k.view(shape[:-1] + (1,)), 1.0)
		y = Variable(y_hard - y_soft.data) + y_soft
	else:
		y = y_soft
	return y


def gumbel_softmax_sample(logits, tau=1, eps=1e-10):
	gumbel_noise = sample_gumbel(logits.size(), eps=eps)
	if logits.is_cuda:
		gumbel_noise = gumbel_noise.cuda()
	y = logits + Variable(gumbel_noise)
	return my_softmax(y / tau, axis=-1)


def sample_gumbel(shape, eps=1e-10):
	uniform = torch.rand(shape).float()
	return - torch.log(eps - torch.log(uniform + eps))


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot


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


class InteractionNet(nn.Module):

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
		x = inputs.contiguous()
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
		return self.fc_out(x)                                  # [N, 600, 256] -> [N, 600, 3]


class InteractionDecoderRecurrent(nn.Module):
	
	def __init__(self, n_in_node, edge_types, n_hid, do_prob=0., skip_first=True):
		super().__init__()

		self.msg_fc1 = nn.ModuleList([nn.Linear(2 * n_hid, n_hid) for _ in range(edge_types)])
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
			norm = float(len(self.msg_fc2)) - 1.
		else:
			start_idx = 0
			norm = float(len(self.msg_fc2))
		for k in range(start_idx, len(self.msg_fc2)):
			msg = torch.tanh(self.msg_fc1[k](pre_msg))
			msg = self.dropout1(msg)
			msg = torch.tanh(self.msg_fc2[k](msg))
			msg = msg * rel_type[:, :, k:k + 1]
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

	def forward(self, data, rel_type, rel_rec, rel_send, pred_steps=1, 
		        burn_in=False, burn_in_steps=1, dynamic_graph=False,
		        encoder=None, temp=None):
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


class AdjacencyLearn(nn.Module):

	def __init__(self, n_in_enc, n_hid_enc, edge_types, n_in_dec, n_hid_dec, node_num=25):
		super().__init__()

		self.encoder = InteractionNet(n_in=n_in_enc,                        # 150
			                          n_hid=n_hid_enc,                      # 256
			                          n_out=edge_types,                     # 3
			                          do_prob=0.5,
			                          factor=True)
		self.decoder = InteractionDecoderRecurrent(n_in_node=n_in_dec,      # 256
			                                       edge_types=edge_types,   # 3
			                                       n_hid=n_hid_dec,         # 256
			                                       do_prob=0.5,
			                                       skip_first=True)
		self.offdiag_indices, _ = get_offdiag_indices(node_num)

		off_diag = np.ones([node_num, node_num])-np.eye(node_num, node_num)
		self.rel_rec = torch.FloatTensor(np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float32))
		self.rel_send = torch.FloatTensor(np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float32))
		self.dcy = 0.1

		self.init_weights()

	def init_weights(self):
		for m in self.modules():
			if isinstance(m, nn.BatchNorm1d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()

	def forward(self, inputs): # [N, 3, 50, 25, 2]
		print("enter AdjacencyLearn")

		N, C, T, V, M = inputs.size()
		x = inputs.permute(0, 4, 3, 1, 2).contiguous() # [N, 2, 25, 3, 50]
		x = x.contiguous().view(N*M, V, C, T).permute(0,1,3,2)  # [2N, 25, 50, 3]

		gpu_id = x.get_device()
		rel_rec = self.rel_rec.cuda(gpu_id)
		rel_send = self.rel_send.cuda(gpu_id)

		self.logits = self.encoder(x, rel_rec, rel_send)
		self.N, self.v, self.c = self.logits.size()
		self.edges = gumbel_softmax(self.logits, tau=0.5, hard=True)
		self.prob = my_softmax(self.logits, -1)
		self.outputs = self.decoder(x, self.edges, rel_rec, rel_send, burn_in=False, burn_in_steps=40)
		self.offdiag_indices = self.offdiag_indices.cuda(gpu_id)

		A_batch = []
		for i in range(self.N):
			A_types = []
			for j in range(1, self.c):
				A = torch.sparse.FloatTensor(self.offdiag_indices, self.edges[i,:,j], torch.Size([25, 25])).to_dense().cuda(gpu_id)
				A = A + torch.eye(25, 25).cuda(gpu_id)
				D = torch.sum(A, dim=0).squeeze().pow(-1)+1e-10
				D = torch.diag(D)
				A_ = torch.matmul(A, D)*self.dcy
				A_types.append(A_)
			A_types = torch.stack(A_types)
			A_batch.append(A_types)
		self.A_batch = torch.stack(A_batch).cuda(gpu_id) # [N, 2, 25, 25]

		return self.A_batch, self.prob, self.outputs, x
