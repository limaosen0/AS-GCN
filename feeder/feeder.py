import os
import sys
import numpy as np
import random
import pickle
import time
import copy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms

from . import tools

class Feeder(torch.utils.data.Dataset):

    def __init__(self,
                 data_path, label_path,
                 repeat_pad=False,
                 random_choose=False,
                 random_move=False,
                 window_size=-1,
                 debug=False,
                 down_sample = False,
                 mmap=True):
        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.repeat_pad = repeat_pad
        self.random_choose = random_choose
        self.random_move = random_move
        self.window_size = window_size
        self.down_sample = down_sample

        self.load_data(mmap)

    def load_data(self, mmap):

        with open(self.label_path, 'rb') as f:
            self.sample_name, self.label = pickle.load(f)

        if mmap:
            self.data = np.load(self.data_path, mmap_mode='r')
        else:
            self.data = np.load(self.data_path)

        if self.debug:
            self.label = self.label[0:100]
            self.data = self.data[0:100]
            self.sample_name = self.sample_name[0:100]

        self.N, self.C, self.T, self.V, self.M = self.data.shape # (40091, 3, 300, 25, 2)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):

        data_numpy = np.array(self.data[index]).astype(np.float32)
        label = self.label[index]

        valid_frame = (data_numpy!=0).sum(axis=3).sum(axis=2).sum(axis=0)>0
        begin, end = valid_frame.argmax(), len(valid_frame)-valid_frame[::-1].argmax()
        length = end-begin

        if self.repeat_pad:
            data_numpy = tools.repeat_pading(data_numpy)
        if self.random_choose:
            data_numpy = tools.random_choose(data_numpy, self.window_size)
        elif self.window_size > 0:
            data_numpy = tools.auto_pading(data_numpy, self.window_size)
        if self.random_move:
            data_numpy = tools.random_move(data_numpy)

        data_last = copy.copy(data_numpy[:,-11:-10,:,:])
        target_data = copy.copy(data_numpy[:,-10:,:,:])
        input_data = copy.copy(data_numpy[:,:-10,:,:])

        if self.down_sample:
            if length<=60:
                input_data_dnsp = input_data[:,:50,:,:]
            else:
                rs = int(np.random.uniform(low=0, high=np.ceil((length-10)/50)))
                input_data_dnsp = [input_data[:,int(i)+rs,:,:] for i in [np.floor(j*((length-10)/50)) for j in range(50)]]
                input_data_dnsp = np.array(input_data_dnsp).astype(np.float32)
                input_data_dnsp = np.transpose(input_data_dnsp, axes=(1,0,2,3))
                
        return input_data, input_data_dnsp, target_data, data_last, label