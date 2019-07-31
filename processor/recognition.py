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
        self.model1 = self.io.load_model(self.arg.model1, **(self.arg.model1_args))
        self.model1.apply(weights_init)
        self.model2 = self.io.load_model(self.arg.model2, **(self.arg.model2_args))

        self.loss_class = nn.CrossEntropyLoss()
        self.loss_pred = nn.MSELoss()
        self.w_pred = 0.01

        prior = np.array([0.95, 0.05/2, 0.05/2])
        self.log_prior = torch.FloatTensor(np.log(prior))
        self.log_prior = torch.unsqueeze(torch.unsqueeze(self.log_prior, 0), 0)

        self.load_optimizer()
        
    def load_optimizer(self):
        if self.arg.optimizer == 'SGD':
            self.optimizer1 = optim.SGD(params=self.model1.parameters(),
                                        lr=self.arg.base_lr1,
                                        momentum=0.9,
                                        nesterov=self.arg.nesterov,
                                        weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'Adam':
            self.optimizer1 = optim.Adam(params=self.model1.parameters(),
                                         lr=self.arg.base_lr1,
                                         weight_decay=self.arg.weight_decay)
        else:
            raise ValueError()
        self.optimizer2 = optim.Adam(params=self.model2.parameters(),
                                     lr=self.arg.base_lr2)

    def adjust_lr(self):
        if self.arg.optimizer == 'SGD' and self.arg.step:
            lr = self.arg.base_lr1 * (0.1**np.sum(self.meta_info['epoch']>= np.array(self.arg.step)))
            for param_group in self.optimizer1.param_groups:
                param_group['lr'] = lr
            self.lr = lr
        else:
            self.lr = self.arg.base_lr1
        self.lr2 = self.arg.base_lr2

    def nll_gaussian(self, preds, target, variance, add_const=False):
        neg_log_p = ((preds-target)**2/(2*variance))
        if add_const:
            const = 0.5*np.log(2*np.pi*variance)
            neg_log_p += const
        return neg_log_p.sum() / (target.size(0) * target.size(1))

    def kl_categorical(self, preds, log_prior, num_node, eps=1e-16):
        kl_div = preds*(torch.log(preds+eps)-log_prior)
        return kl_div.sum()/(num_node*preds.size(0))


    def train(self, training_A=False):

        self.model1.train()
        self.model2.train()
        self.adjust_lr()
        loader = self.data_loader['train']
        loss1_value = []
        loss_class_value = []
        loss_recon_value = []
        loss2_value = []
        loss_nll_value = []
        loss_kl_value = []

        if training_A:
            for param1 in self.model1.parameters():
                param1.requires_grad = False
            for param2 in self.model2.parameters():
                param2.requires_grad = True
            self.iter_info.clear()
            self.epoch_info.clear()

            for data, data_downsample, target_data, data_last, label in loader:
                data = data.float().to(self.dev)
                data_downsample = data_downsample.float().to(self.dev)
                label = label.long().to(self.dev)

                gpu_id = data.get_device()
                self.log_prior = self.log_prior.cuda(gpu_id)
                A_batch, prob, outputs, data_target = self.model2(data_downsample)
                loss_nll = self.nll_gaussian(outputs, data_target[:,:,1:,:], variance=5e-4)
                loss_kl = self.kl_categorical(prob, self.log_prior, num_node=25)
                loss2 = loss_nll + loss_kl

                self.optimizer2.zero_grad()
                loss2.backward()
                self.optimizer2.step()

                self.iter_info['loss2'] = loss2.data.item()
                self.iter_info['loss_nll'] = loss_nll.data.item()
                self.iter_info['loss_kl'] = loss_kl.data.item()
                self.iter_info['lr'] = '{:.6f}'.format(self.lr2)

                loss2_value.append(self.iter_info['loss2'])
                loss_nll_value.append(self.iter_info['loss_nll'])
                loss_kl_value.append(self.iter_info['loss_kl'])
                self.show_iter_info()
                self.meta_info['iter'] += 1
            self.epoch_info['mean_loss2'] = np.mean(loss2_value)
            self.epoch_info['mean_loss_nll'] = np.mean(loss_nll_value)
            self.epoch_info['mean_loss_kl'] = np.mean(loss_kl_value)
            self.show_epoch_info()
            self.io.print_timer()

        else:
            for param1 in self.model1.parameters():
                param1.requires_grad = True
            for param2 in self.model2.parameters():
                param2.requires_grad = True
            self.iter_info.clear()
            self.epoch_info.clear()
            for data, data_downsample, target_data, data_last, label in loader:
                data = data.float().to(self.dev)
                data_downsample = data_downsample.float().to(self.dev)
                target_data = target_data.float().to(self.dev)
                data_last = data_last.float().to(self.dev)
                label = label.long().to(self.dev)

                A_batch, prob, outputs, _ = self.model2(data_downsample)
                x_class, pred, target = self.model1(data, target_data, data_last, A_batch, self.arg.lamda_act)
                loss_class = self.loss_class(x_class, label)
                loss_recon = self.loss_pred(pred, target)
                loss1 = loss_class + self.w_pred*loss_recon

                self.optimizer1.zero_grad()
                loss1.backward()
                self.optimizer1.step()

                self.iter_info['loss1'] = loss1.data.item()
                self.iter_info['loss_class'] = loss_class.data.item()
                self.iter_info['loss_recon'] = loss_recon.data.item()*self.w_pred
                self.iter_info['lr'] = '{:.6f}'.format(self.lr)

                loss1_value.append(self.iter_info['loss1'])
                loss_class_value.append(self.iter_info['loss_class'])
                loss_recon_value.append(self.iter_info['loss_recon'])
                self.show_iter_info()
                self.meta_info['iter'] += 1

            self.epoch_info['mean_loss1']= np.mean(loss1_value)
            self.epoch_info['mean_loss_class'] = np.mean(loss_class_value)
            self.epoch_info['mean_loss_recon'] = np.mean(loss_recon_value)
            self.show_epoch_info()
            self.io.print_timer()


    def test(self, evaluation=True, testing_A=False, save=False, save_feature=False):

        self.model1.eval()
        self.model2.eval()
        loader = self.data_loader['test']
        loss1_value = []
        loss_class_value = []
        loss_recon_value = []
        loss2_value = []
        loss_nll_value = []
        loss_kl_value = []
        result_frag = []
        label_frag = []

        if testing_A:
            A_all = []
            self.epoch_info.clear()
            for data, data_downsample, target_data, data_last, label in loader:
                data = data.float().to(self.dev)
                data_downsample = data_downsample.float().to(self.dev)
                label = label.long().to(self.dev)

                with torch.no_grad():
                    A_batch, prob, outputs, data_bn = self.model2(data_downsample)

                if save:
                    n = A_batch.size(0)
                    a = A_batch[:int(n/2),:,:,:].cpu().numpy()
                    A_all.extend(a)

                if evaluation:
                    gpu_id = data.get_device()
                    self.log_prior = self.log_prior.cuda(gpu_id)
                    loss_nll = self.nll_gaussian(outputs, data_bn[:,:,1:,:], variance=5e-4)
                    loss_kl = self.kl_categorical(prob, self.log_prior, num_node=25)
                    loss2 = loss_nll + loss_kl

                    loss2_value.append(loss2.item())
                    loss_nll_value.append(loss_nll.item())
                    loss_kl_value.append(loss_kl.item())

            if save:
                A_all = np.array(A_all)
                np.save(os.path.join(self.arg.work_dir, 'test_adj.npy'), A_all)

            if evaluation:
                self.epoch_info['mean_loss2'] = np.mean(loss2_value)
                self.epoch_info['mean_loss_nll'] = np.mean(loss_nll_value)
                self.epoch_info['mean_loss_kl'] = np.mean(loss_kl_value)
                self.show_epoch_info()

        else:
            recon_data = []
            feature_map = []
            self.epoch_info.clear()
            for data, data_downsample, target_data, data_last, label in loader:
                data = data.float().to(self.dev)
                data_downsample = data_downsample.float().to(self.dev)
                target_data = target_data.float().to(self.dev)
                data_last = data_last.float().to(self.dev)
                label = label.long().to(self.dev)

                with torch.no_grad():
                    A_batch, prob, outputs, _ = self.model2(data_downsample)
                    x_class, pred, target = self.model1(data, target_data, data_last, A_batch, self.arg.lamda_act)
                result_frag.append(x_class.data.cpu().numpy())

                if save:
                    n = pred.size(0)                      
                    p = pred[::2,:,:,:].cpu().numpy()      
                    recon_data.extend(p)

                if evaluation:
                    loss_class = self.loss_class(x_class, label)
                    loss_recon = self.loss_pred(pred, target)
                    loss1 = loss_class + self.w_pred*loss_recon

                    loss1_value.append(loss1.item())
                    loss_class_value.append(loss_class.item())
                    loss_recon_value.append(loss_recon.item())
                    label_frag.append(label.data.cpu().numpy())

            if save:
                recon_data = np.array(recon_data)
                np.save(os.path.join(self.arg.work_dir, 'recon_data.npy'), recon_data)


            self.result = np.concatenate(result_frag)
            if evaluation:
                self.label = np.concatenate(label_frag)
                self.epoch_info['mean_loss1'] = np.mean(loss1_value)
                self.epoch_info['mean_loss_class'] = np.mean(loss_class_value)
                self.epoch_info['mean_loss_recon'] = np.mean(loss_recon_value)
                self.show_epoch_info()

                for k in self.arg.show_topk:
                    hit_top_k = []
                    rank = self.result.argsort()
                    for i,l in enumerate(self.label):
                        hit_top_k.append(l in rank[i, -k:])
                    self.io.print_log('\n')
                    accuracy = sum(hit_top_k)*1.0/len(hit_top_k)
                    self.io.print_log('\tTop{}: {:.2f}%'.format(k, 100 * accuracy))



    @staticmethod
    def get_parser(add_help=False):

        parent_parser = Processor.get_parser(add_help=False)
        parser = argparse.ArgumentParser(
            add_help=add_help,
            parents=[parent_parser],
            description='Spatial Temporal Graph Convolution Network')

        parser.add_argument('--show_topk', type=int, default=[1, 5], nargs='+', help='which Top K accuracy will be shown')
        parser.add_argument('--base_lr1', type=float, default=0.1, help='initial learning rate')
        parser.add_argument('--base_lr2', type=float, default=0.0005, help='initial learning rate')
        parser.add_argument('--step', type=int, default=[], nargs='+', help='the epoch where optimizer reduce the learning rate')
        parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
        parser.add_argument('--nesterov', type=str2bool, default=True, help='use nesterov or not')
        parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay for optimizer')

        parser.add_argument('--max_hop_dir', type=str, default='max_hop_4')
        parser.add_argument('--lamda_act', type=float, default=0.5)
        parser.add_argument('--lamda_act_dir', type=str, default='lamda_05')

        return parser
