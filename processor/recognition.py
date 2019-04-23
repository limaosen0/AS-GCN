import sys
import os
import argparse
import yaml
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

import torchlight
from torchlight import str2bool
from torchlight import DictAction
from torchlight import import_class

from .processor import Processor


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv1d') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class REC_Processor(Processor):

    def load_model(self):
        self.model = self.io.load_model(self.arg.model, **(self.arg.model_args))

        self.loss_recog = nn.CrossEntropyLoss()
        self.loss_predt = nn.MSELoss()
        self.w_pred = self.arg.w_pred

        p0 = 0.95
        prior = np.array([p0, (1-p0)/3, (1-p0)/3, (1-p0)/3])
        self.log_prior = torch.FloatTensor(np.log(prior)).to(self.dev)
        self.log_prior = torch.unsqueeze(torch.unsqueeze(self.log_prior, 0), 0)

        
    def load_optimizer(self):
        if self.arg.optimizer == 'SGD':
            self.optimizer_rec = optim.SGD(params=self.model.module.main_net.parameters(),
                                           lr=self.arg.base_lr_rec,
                                           momentum=0.9,
                                           nesterov=self.arg.nesterov,
                                           weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'Adam':
            self.optimizer_rec = optim.Adam(params=self.model.module.main_net.parameters(),
                                            lr=self.arg.base_lr_rec,
                                            weight_decay=self.arg.weight_decay)
        self.optimizer_aim = optim.Adam(params=self.model.module.aim.parameters(), lr=self.arg.base_lr_aim)


    def adjust_lr(self):
        if self.arg.optimizer == 'SGD' and self.arg.step:
            lr = self.arg.base_lr_rec * (0.1**np.sum(self.meta_info['epoch']>= np.array(self.arg.step)))
            for param_group in self.optimizer_rec.param_groups:
                param_group['lr'] = lr
            self.lr = lr
        else:
            self.lr = self.arg.base_lr_rec


    def show_topk(self, k):
        rank = self.result.argsort()
        hit_top_k = [l in rank[i, -k:] for i, l in enumerate(self.label)]
        accuracy = sum(hit_top_k) * 1.0 / len(hit_top_k)
        self.io.print_log('\tTop{}: {:.2f}%'.format(k, 100 * accuracy))


    def nll_gaussian(self, preds, target, variance, add_const=False):
        neg_log_p = ((preds-target)**2/(2*variance))
        if add_const:
            const = 0.5*np.log(2*np.pi*variance)
            neg_log_p += const
        return neg_log_p.sum() / (target.size(0) * target.size(1))


    def kl_categorical(self, preds, log_prior, num_atoms, eps=1e-16):
        kl_div = preds*(torch.log(preds+eps)-log_prior)
        print('preds', preds)
        print('prior', torch.exp(log_prior))
        return kl_div.sum()/(num_atoms*preds.size(0))


    def encode_onehot(self, labels):
        classes = set(labels)
        classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
        labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
        return labels_onehot


    def get_offdiag_indices(self, num_nodes):
        ones = torch.ones(num_nodes, num_nodes)
        eye = torch.eye(num_nodes, num_nodes)
        offdiag_indices = (ones - eye).nonzero().t()
        offdiag_indices_ = offdiag_indices[0] * num_nodes + offdiag_indices[1]
        return offdiag_indices, offdiag_indices_


    def train(self, training_aim=False):

        self.model.train()
        self.adjust_lr()
        loader = self.data_loader['train']
        loss_model_value = []
        loss_recog_value = []
        loss_predt_value = []
        loss_aim_value = []
        loss_nll_value = []
        loss_kld_value = []

        if training_aim==True:
            for name, param in self.model.named_parameters():
                if 'aim' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

            self.iter_info.clear()
            self.epoch_info.clear()

            for input_data, input_data_dnsp, target_data, data_last, label in loader:
                input_data_dnsp = input_data_dnsp.float().to(self.dev)
                gpu_id = input_data_dnsp.get_device()
                A_batch, prob, outputs, data = self.model.module.aim(input_data_dnsp)
                loss_nll = self.nll_gaussian(outputs, data[:,:,1:,:], variance=5e-4)
                loss_kld = self.kl_categorical(prob, self.log_prior, num_atoms=25)
                loss_aim = loss_nll + loss_kld

                self.optimizer_aim.zero_grad()
                loss_aim.backward()
                self.optimizer_aim.step()

                self.iter_info['loss_aim'] = loss_aim.data.item()
                loss_aim_value.append(self.iter_info['loss_aim'])
                self.iter_info['loss_nll'] = loss_nll.data.item()
                loss_nll_value.append(self.iter_info['loss_nll'])
                self.iter_info['loss_kld'] = loss_kld.data.item()
                loss_kld_value.append(self.iter_info['loss_kld'])
                self.show_iter_info()
                self.meta_info['iter'] += 1

            self.epoch_info['mean_loss_aim'] = np.mean(loss_aim_value)
            self.epoch_info['mean_loss_nll'] = np.mean(loss_nll_value)
            self.epoch_info['mean_loss_kld'] = np.mean(loss_kld_value)
            self.show_epoch_info()
            self.io.print_timer()

        else:
            for name, param in self.model.named_parameters():
                if 'aim' in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True

            self.iter_info.clear()
            self.epoch_info.clear()

            for input_data, input_data_dnsp, target_data, data_last, label in loader:
                input_data = input_data.float().to(self.dev)
                input_data_dnsp = input_data_dnsp.float().to(self.dev)
                target_data = target_data.float().to(self.dev)
                data_last = data_last.float().to(self.dev)
                label = label.long().to(self.dev)
                gpu_id = input_data_dnsp.get_device()

                label_rec, data_pred, target, alink_prob, aim_pred, aim_input = self.model(input_data, target_data, data_last, input_data_dnsp)
                loss_recog = self.loss_recog(label_rec, label)
                loss_predt = self.loss_predt(data_pred, target) 
                loss_model = loss_recog + self.w_pred*loss_predt

                self.optimizer_rec.zero_grad()
                loss_model.backward()
                self.optimizer_rec.step()

                self.iter_info['loss_model'] = loss_model.data.item()
                loss_model_value.append(self.iter_info['loss_model'])
                self.iter_info['loss_recog'] = loss_recog.data.item()
                loss_recog_value.append(self.iter_info['loss_recog'])
                self.iter_info['loss_predt'] = loss_predt.data.item()
                loss_predt_value.append(self.iter_info['loss_predt'])
                self.iter_info['lr'] = '{:.6f}'.format(self.lr)
                self.show_iter_info()
                self.meta_info['iter'] += 1

            self.epoch_info['mean_loss_model'] = np.mean(loss_model_value)
            self.epoch_info['mean_loss_recog'] = np.mean(loss_recog_value)
            self.epoch_info['mean_loss_predt'] = np.mean(loss_predt_value)
            self.show_epoch_info()
            self.io.print_timer()


    def test(self, evaluation=True, testing_aim=False, save=False):

        self.model.eval()
        loader = self.data_loader['test']

        loss_rec_value = []
        loss_recog_value = []
        loss_predt_value = []
        loss_aim_value = []
        loss_nll_value = []
        loss_kld_value = []
        loss_err_value = []
        result_frag = []
        label_frag = []

        node_num = self.arg.node_num

        if testing_aim==True:
            self.epoch_info.clear()
            for input_data, input_data_dnsp, target_data, data_last, label in loader:
                input_data_dnsp = input_data_dnsp.float().to(self.dev)

                with torch.no_grad():
                    A_batch, prob, outputs, data = self.model.module.aim(input_data_dnsp)
                if evaluation:
                    loss_nll = self.nll_gaussian(outputs, data[:,:,1:,:], variance=5e-4)
                    loss_kld = self.kl_categorical(prob, self.log_prior, num_atoms=25)
                    loss_aim = loss_nll + loss_kld
                    loss_err = torch.mean(torch.abs(outputs-data[:,:,1:,:]))

                    loss_aim_value.append(loss_aim.item())
                    loss_nll_value.append(loss_nll.item())
                    loss_kld_value.append(loss_kld.item())
                    loss_err_value.append(loss_err.item())

            if evaluation:
                self.epoch_info['mean_loss_aim'] = np.mean(loss_aim_value)
                self.epoch_info['mean_loss_nll'] = np.mean(loss_nll_value)
                self.epoch_info['mean_loss_kld'] = np.mean(loss_kld_value)
                self.epoch_info['mean_pred_err'] = np.mean(loss_err_value)
                self.show_epoch_info()

        else:
            self.epoch_info.clear()
            for input_data, input_data_dnsp, target_data, data_last, label in loader:
                input_data = input_data.float().to(self.dev)
                input_data_dnsp = input_data_dnsp.float().to(self.dev)
                target_data = target_data.float().to(self.dev)
                data_last = data_last.float().to(self.dev)
                label = label.long().to(self.dev)

                with torch.no_grad():
                    label_rec, data_pred, target, alink_prob, aim_pred, aim_input = self.model(input_data, target_data, data_last, input_data_dnsp)
                result_frag.append(label_rec.data.cpu().numpy())

                if evaluation:
                    loss_recog = self.loss_recog(output, label)
                    loss_predt = self.loss_predt(pred, data_bn)
                    loss_rec = loss_recog + self.w_pred*loss_predt

                    loss_rec_value.append(loss_rec.item())
                    loss_recog_value.append(loss_recog.item())
                    loss_predt_value.append(loss_predt.item())
                    label_frag.append(label.data.cpu().numpy())

            self.result = np.concatenate(result_frag)
            if evaluation:
                self.label = np.concatenate(label_frag)
                self.epoch_info['mean_loss_rec'] = np.mean(loss_rec_value)
                self.epoch_info['mean_loss_recog'] = np.mean(loss_recog_value)
                self.epoch_info['mean_loss_predt'] = np.mean(loss_predt_value)
                self.show_epoch_info()
                for k in self.arg.show_topk:
                    self.show_topk(k)


    @staticmethod
    def get_parser(add_help=False):

        parent_parser = Processor.get_parser(add_help=False)
        parser = argparse.ArgumentParser(add_help=add_help,
                                         parents=[parent_parser],
                                         description='Spatial Temporal Graph Convolution Network')


        parser.add_argument('--show_topk', type=int, default=[1, 5], nargs='+', help='which Top K accuracy will be shown')
        # optim
        parser.add_argument('--w_pred', type=float, default=1., help='weights of prediction head')
        parser.add_argument('--base_lr_rec', type=float, default=0.01, help='initial learning rate')
        parser.add_argument('--base_lr_aim', type=float, default=0.01, help='initial learning rate')
        parser.add_argument('--node_num', type=int, default=25, help='node number')
        parser.add_argument('--step', type=int, default=[], nargs='+', help='the epoch where optimizer reduce the learning rate')
        parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
        parser.add_argument('--nesterov', type=str2bool, default=True, help='use nesterov or not')
        parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay for optimizer')
        # endregion yapf: enable

        return parser
