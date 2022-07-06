import sys
import argparse
import yaml
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

import torchlight
from torchlight.io import str2bool
from torchlight.io import DictAction
from torchlight.io import import_class

from .io import IO


class Processor(IO):

    def __init__(self, argv=None):

        self.load_arg(argv)
        self.init_environment()
        self.load_model()
        self.load_weights()
        self.gpu()
        self.load_data()

    def init_environment(self):

        super().init_environment()
        self.result = dict()
        self.iter_info = dict()
        self.epoch_info = dict()
        self.meta_info = dict(epoch=0, iter=0)


    def load_data(self):
        Feeder = import_class(self.arg.feeder)
        if 'debug' not in self.arg.train_feeder_args:
            self.arg.train_feeder_args['debug'] = self.arg.debug
        self.data_loader = dict()
        if self.arg.phase == 'train':
            self.data_loader['train'] = torch.utils.data.DataLoader(dataset=Feeder(**self.arg.train_feeder_args),
                                                                    batch_size=self.arg.batch_size,
                                                                    shuffle=True,
                                                                    num_workers=self.arg.num_worker,
                                                                    drop_last=True)
        if self.arg.test_feeder_args:
            self.data_loader['test'] = torch.utils.data.DataLoader(dataset=Feeder(**self.arg.test_feeder_args),
                                                                   batch_size=self.arg.test_batch_size,
                                                                   shuffle=False,
                                                                   num_workers=self.arg.num_worker)

    def show_epoch_info(self):
        for k, v in self.epoch_info.items():
            self.io.print_log('\t{}: {}'.format(k, v))
        if self.arg.pavi_log:
            self.io.log('train', self.meta_info['iter'], self.epoch_info)

    def show_iter_info(self):
        if self.meta_info['iter'] % self.arg.log_interval == 0:
            info ='\tIter {} Done.'.format(self.meta_info['iter'])
            for k, v in self.iter_info.items():
                if isinstance(v, float):
                    info = info + ' | {}: {:.4f}'.format(k, v)
                else:
                    info = info + ' | {}: {}'.format(k, v)

            self.io.print_log(info)

            if self.arg.pavi_log:
                self.io.log('train', self.meta_info['iter'], self.iter_info)

    def train(self):
        for _ in range(100):
            self.iter_info['loss'] = 0
            self.iter_info['loss_class'] = 0
            self.iter_info['loss_recon'] = 0
            self.show_iter_info()
            self.meta_info['iter'] += 1
        self.epoch_info['mean_loss'] = 0
        self.epoch_info['mean_loss_class'] = 0
        self.epoch_info['mean_loss_recon'] = 0
        self.show_epoch_info()

    def test(self):
        for _ in range(100):
            self.iter_info['loss'] = 1
            self.iter_info['loss_class'] = 1
            self.iter_info['loss_recon'] = 1
            self.show_iter_info()
        self.epoch_info['mean_loss'] = 1
        self.epoch_info['mean_loss_class'] = 1
        self.epoch_info['mean_loss_recon'] = 1
        self.show_epoch_info()

    def start(self):
        self.io.print_log('Parameters:\n{}\n'.format(str(vars(self.arg))))

        if self.arg.phase == 'train':
            for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                self.meta_info['epoch'] = epoch

                if epoch < 10:
                    self.io.print_log('Training epoch: {}'.format(epoch))
                    self.train(training_A=True)
                    self.io.print_log('Done.')
                else:
                    self.io.print_log('Training epoch: {}'.format(epoch))
                    self.train(training_A=False)
                    self.io.print_log('Done.')

                # save model
                if ((epoch + 1) % self.arg.save_interval == 0) or (epoch + 1 == self.arg.num_epoch):
                    filename1 = 'epoch{}_model1.pt'.format(epoch)
                    self.io.save_model(self.model1, filename1)
                    filename2 = 'epoch{}_model2.pt'.format(epoch)
                    self.io.save_model(self.model2, filename2)

                # evaluation
                if ((epoch + 1) % self.arg.eval_interval == 0) or (epoch + 1 == self.arg.num_epoch):
                    self.io.print_log('Eval epoch: {}'.format(epoch))
                    if epoch <= 10:
                        self.test(testing_A=True)
                    else:
                        self.test(testing_A=False)
                    self.io.print_log('Done.') 


        elif self.arg.phase == 'test':
            if self.arg.weights2 is None:
                raise ValueError('Please appoint --weights.')
            self.io.print_log('Model:   {}.'.format(self.arg.model2))
            self.io.print_log('Weights: {}.'.format(self.arg.weights2))

            self.io.print_log('Evaluation Start:')
            self.test(testing_A=False, save_feature=True)
            self.io.print_log('Done.\n')

            if self.arg.save_result:
                result_dict = dict(
                    zip(self.data_loader['test'].dataset.sample_name,
                        self.result))
                self.io.save_pkl(result_dict, 'test_result.pkl')


    @staticmethod
    def get_parser(add_help=False):

        parser = argparse.ArgumentParser( add_help=add_help, description='Base Processor')

        parser.add_argument('-w', '--work_dir', default='./work_dir/tmp', help='the work folder for storing results')
        parser.add_argument('-c', '--config', default=None, help='path to the configuration file')

        parser.add_argument('--phase', default='train', help='must be train or test')
        parser.add_argument('--save_result', type=str2bool, default=False, help='if ture, the output of the model will be stored')
        parser.add_argument('--start_epoch', type=int, default=0, help='start training from which epoch')
        parser.add_argument('--num_epoch', type=int, default=80, help='stop training in which epoch')
        parser.add_argument('--use_gpu', type=str2bool, default=True, help='use GPUs or not')
        parser.add_argument('--device', type=int, default=0, nargs='+', help='the indexes of GPUs for training or testing')

        parser.add_argument('--log_interval', type=int, default=100, help='the interval for printing messages (#iteration)')
        parser.add_argument('--save_interval', type=int, default=1, help='the interval for storing models (#iteration)')
        parser.add_argument('--eval_interval', type=int, default=5, help='the interval for evaluating models (#iteration)')
        parser.add_argument('--save_log', type=str2bool, default=True, help='save logging or not')
        parser.add_argument('--print_log', type=str2bool, default=True, help='print logging or not')
        parser.add_argument('--pavi_log', type=str2bool, default=False, help='logging on pavi or not')

        parser.add_argument('--feeder', default='feeder.feeder', help='data loader will be used')
        parser.add_argument('--num_worker', type=int, default=4, help='the number of worker per gpu for data loader')
        parser.add_argument('--train_feeder_args', action=DictAction, default=dict(), help='the arguments of data loader for training')
        parser.add_argument('--test_feeder_args', action=DictAction, default=dict(), help='the arguments of data loader for test')
        parser.add_argument('--batch_size', type=int, default=256, help='training batch size')
        parser.add_argument('--test_batch_size', type=int, default=256, help='test batch size')
        parser.add_argument('--debug', action="store_true", help='less data, faster loading')

        parser.add_argument('--model1', default=None, help='the model will be used')
        parser.add_argument('--model2', default=None, help='the model will be used')
        parser.add_argument('--model1_args', action=DictAction, default=dict(), help='the arguments of model')
        parser.add_argument('--model2_args', action=DictAction, default=dict(), help='the arguments of model')
        parser.add_argument('--weights1', default=None, help='the weights for network initialization')
        parser.add_argument('--weights2', default=None, help='the weights for network initialization')
        parser.add_argument('--ignore_weights', type=str, default=[], nargs='+', help='the name of weights which will be ignored in the initialization')

        return parser
