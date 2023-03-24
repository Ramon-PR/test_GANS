# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 11:37:42 2023

@author: Ramón Pozuelo
"""

import torch
import torch.nn as nn


def count_parameters(model):
    # print(count_parameters(model))
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def dim_after_filter(dim_in, dim_kernel, pad, stripe):
    return int((dim_in + 2*pad - dim_kernel)/stripe) + 1

def blockRelUMaxP(chan_in, chan_out, kernel_size=3, pad1=1, str1=1, kernel_maxpool=2, str_maxpool=2):
    return torch.nn.Sequential(
        torch.nn.Conv2d(chan_in, chan_out, kernel_size, padding=pad1, stride=str1),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(kernel_maxpool, stride=str_maxpool)
    )

def blockTanhMaxP(chan_in, chan_out, kernel_size=3, pad1=1, str1=1, kernel_maxpool=2, str_maxpool=2):
    return torch.nn.Sequential(
        torch.nn.Conv2d(chan_in, chan_out, kernel_size, padding=pad1, stride=str1),
        torch.nn.Tanh(),
        torch.nn.MaxPool2d(kernel_maxpool, stride=str_maxpool)
    )

def block2(c_in, c_out):
    return torch.nn.Sequential(
        torch.nn.Linear(c_in, c_out),
        torch.nn.ReLU()
    )

def blockRelUMaxP_BN(chan_in, chan_out, kernel_size=3, pad1=1, str1=1, kernel_maxpool=2, str_maxpool=2):
    return torch.nn.Sequential(
        torch.nn.Conv2d(chan_in, chan_out, kernel_size, padding=pad1, stride=str1),
        torch.nn.BatchNorm2d(chan_out),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(kernel_maxpool, stride=str_maxpool)
    )

class CNN(torch.nn.Module):
  def __init__(self, n_channels=1, Hin=32, Win=32, Hout=32, Wout=32):
    super().__init__()

    chan_in = n_channels 

    n_filters1, n_filters2 = 5, 5
    kernel_size1, kernel_size2 = 3, 5
    pad1, pad2 = 1, 2
    str1, str2 = 1, 1

    kernel_maxpool, str_maxpool = 2, 2

    self.Hin = Hin
    self.Win = Win
    self.Hout = Hout
    self.Wout = Wout
    
    n_outputs = self.Hout* self.Wout

    self.conv1 = blockRelUMaxP(chan_in, n_filters1, kernel_size1, pad1, str1)
    H = dim_after_filter(Hin, kernel_size1, pad1, str1)
    W = dim_after_filter(Win, kernel_size1, pad1, str1)
    # if maxpool2d
    H, W = dim_after_filter(H, kernel_maxpool, 0, str_maxpool), dim_after_filter(W, kernel_maxpool, 0, str_maxpool) 

    self.conv2 = blockRelUMaxP(n_filters1, n_filters2, kernel_size2, pad2, str2)
    H = dim_after_filter(H, kernel_size2, pad2, str2)
    W = dim_after_filter(W, kernel_size2, pad2, str2)
    # if maxpool2d
    H, W = dim_after_filter(H, kernel_maxpool, 0, str_maxpool), dim_after_filter(W, kernel_maxpool, 0, str_maxpool)
    
    self.fc = torch.nn.Linear(n_filters2*H*W, n_outputs)

  def forward(self, x):
    x = self.conv1(x)
    x = self.conv2(x)
    x = x.view(x.shape[0], -1)
    x = self.fc(x)
    return x


class CNN_TanH(torch.nn.Module):
  def __init__(self, n_channels=1, Hin=32, Win=32, Hout=32, Wout=32):
    super().__init__()

    chan_in = n_channels 

    n_filters1, n_filters2 = 5, 5
    kernel_size1, kernel_size2 = 3, 5
    pad1, pad2 = 1, 2
    str1, str2 = 1, 1

    kernel_maxpool, str_maxpool = 2, 2

    self.Hin = Hin
    self.Win = Win
    self.Hout = Hout
    self.Wout = Wout
    
    n_outputs = self.Hout* self.Wout

    # self.conv1 = block(chan_in, n_filters1, kernel_size1, pad1, str1)
    self.conv1 = blockTanhMaxP(chan_in, n_filters1, kernel_size1, pad1, str1)
    H = dim_after_filter(Hin, kernel_size1, pad1, str1)
    W = dim_after_filter(Win, kernel_size1, pad1, str1)
    # if maxpool2d
    H, W = dim_after_filter(H, kernel_maxpool, 0, str_maxpool), dim_after_filter(W, kernel_maxpool, 0, str_maxpool) 

    # self.conv2 = block(n_filters1, n_filters2, kernel_size2, pad2, str2)
    self.conv2 = blockTanhMaxP(n_filters1, n_filters2, kernel_size2, pad2, str2)
    H = dim_after_filter(H, kernel_size2, pad2, str2)
    W = dim_after_filter(W, kernel_size2, pad2, str2)
    # if maxpool2d
    H, W = dim_after_filter(H, kernel_maxpool, 0, str_maxpool), dim_after_filter(W, kernel_maxpool, 0, str_maxpool)
    
    self.fc = torch.nn.Linear(n_filters2*H*W, n_outputs)

  def forward(self, x):
    # if self.training:        
    x = self.conv1(x)
    x = self.conv2(x)
    x = x.view(x.shape[0], -1)
    x = self.fc(x)
    # print("training", self.training)
    return x
    # else:
        # x = self.conv1(x)
        # x = self.conv2(x)
        # x = x.view(x.shape[0], -1)
        # x = self.fc(x)
        # print("not training", self.training)
        # return x


class CNN_batchnorm(torch.nn.Module):
  def __init__(self, n_channels=1, Hin=32, Win=32, Hout=32, Wout=32):
    super().__init__()

    chan_in = n_channels 

    n_filters1, n_filters2 = 5, 5
    kernel_size1, kernel_size2 = 3, 5
    pad1, pad2 = 1, 2
    str1, str2 = 1, 1

    kernel_maxpool, str_maxpool = 2, 2

    self.Hin = Hin
    self.Win = Win
    self.Hout = Hout
    self.Wout = Wout
    
    n_outputs = self.Hout* self.Wout

    self.conv1 = blockRelUMaxP_BN(chan_in, n_filters1, kernel_size1, pad1, str1)

    H = dim_after_filter(Hin, kernel_size1, pad1, str1)
    W = dim_after_filter(Win, kernel_size1, pad1, str1)
    # if maxpool2d
    H, W = dim_after_filter(H, kernel_maxpool, 0, str_maxpool), dim_after_filter(W, kernel_maxpool, 0, str_maxpool) 

    self.conv2 = blockRelUMaxP_BN(n_filters1, n_filters2, kernel_size2, pad2, str2)
    H = dim_after_filter(H, kernel_size2, pad2, str2)
    W = dim_after_filter(W, kernel_size2, pad2, str2)
    # if maxpool2d
    H, W = dim_after_filter(H, kernel_maxpool, 0, str_maxpool), dim_after_filter(W, kernel_maxpool, 0, str_maxpool)
    
    self.fc = torch.nn.Linear(n_filters2*H*W, n_outputs)

  def forward(self, x):
    x = self.conv1(x)
    x = self.conv2(x)
    x = x.view(x.shape[0], -1)
    x = self.fc(x)
    return x