import sys
import argparse
import yaml
import numpy as np

import torch
import torch.nn as nn

import torchlight
from torchlight import str2bool
from torchlight import DictAction
from torchlight import import_class


class IO():

    def __init__(self, argv=None):

        self.load_arg(argv)
        self.init_environment()
        self.load_model()
        self.load_weights()
        self.gpu()

    def load_arg(self, argv=None):
        parser = self.get_parser()

        # load arg form config file
        p = parser.parse_args(argv)
        if p.config is not None:
            # load config file
            with open(p.config, 'r') as f:
                default_arg = yaml.load(f)

            # update parser from config file
            key = vars(p).keys()
            for k in default_arg.keys():
                if k not in key:
                    print('Unknown Arguments: {}'.format(k))
                    assert k in key

            parser.set_defaults(**default_arg)

        self.arg = parser.parse_args(argv)

    def init_environment(self):
        self.io = torchlight.IO(
            self.arg.work_dir,
            save_log=self.arg.save_log,
            print_log=self.arg.print_log)
        self.io.save_arg(self.arg)

        # gpu
        if self.arg.use_gpu:
            gpus = torchlight.visible_gpu(self.arg.device)
            torchlight.occupy_gpu(gpus)
            self.gpus = gpus
            self.dev = "cuda:0"
        else:
            self.dev = "cpu"

    def load_model(self):
        self.model1 = self.io.load_model(self.arg.model1, **(self.arg.model1_args))
        self.model2 = self.io.load_model(self.arg.model2, **(self.arg.model2_args))

    def load_weights(self):
        if self.arg.weights1:
            self.model1 = self.io.load_weights(self.model1, self.arg.weights1, self.arg.ignore_weights)
            self.model2 = self.io.load_weights(self.model2, self.arg.weights2, self.arg.ignore_weights)

    def gpu(self):
        # move modules to gpu
        self.model1 = self.model1.to(self.dev)
        self.model2 = self.model2.to(self.dev)
        for name, value in vars(self).items():
            cls_name = str(value.__class__)
            if cls_name.find('torch.nn.modules') != -1:
                setattr(self, name, value.to(self.dev))

        # model parallel
        if self.arg.use_gpu and len(self.gpus) > 1:
            self.model1 = nn.DataParallel(self.model1, device_ids=self.gpus)
            self.model2 = nn.DataParallel(self.model2, device_ids=self.gpus)

    def start(self):
        self.io.print_log('Parameters:\n{}\n'.format(str(vars(self.arg))))

    @staticmethod
    def get_parser(add_help=False):

        #region arguments yapf: disable
        # parameter priority: command line > config > default
        parser = argparse.ArgumentParser( add_help=add_help, description='IO Processor')

        parser.add_argument('-w', '--work_dir', default='./work_dir/tmp', help='the work folder for storing results')
        parser.add_argument('-c', '--config', default=None, help='path to the configuration file')

        # processor
        parser.add_argument('--use_gpu', type=str2bool, default=True, help='use GPUs or not')
        parser.add_argument('--device', type=int, default=0, nargs='+', help='the indexes of GPUs for training or testing')

        # visulize and debug
        parser.add_argument('--print_log', type=str2bool, default=True, help='print logging or not')
        parser.add_argument('--save_log', type=str2bool, default=True, help='save logging or not')

        # model
        parser.add_argument('--model1', default=None, help='the model will be used')
        parser.add_argument('--model2', default=None, help='the model will be used')
        parser.add_argument('--model1_args', action=DictAction, default=dict(), help='the arguments of model')
        parser.add_argument('--model2_args', action=DictAction, default=dict(), help='the arguments of model')
        parser.add_argument('--weights', default=None, help='the weights for network initialization')
        parser.add_argument('--ignore_weights', type=str, default=[], nargs='+', help='the name of weights which will be ignored in the initialization')
        #endregion yapf: enable

        return parser
