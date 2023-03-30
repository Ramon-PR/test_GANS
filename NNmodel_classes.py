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

class MinPool2d(torch.nn.Module):
    def __init__(self, kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False):
        super().__init__()
  
        self.kernel_size, self.stride = kernel_size, stride 
        self.padding, self.dilation = padding, dilation
        self.ceil_mode = ceil_mode
  
    def forward(self, x):
        x = -torch.nn.MaxPool2d(self.kernel_size, self.stride, self.padding, self.dilation, self.ceil_mode)(-x)
        return x
    
    def extra_repr(self):
        # (Optional)Set the extra information about this module. You can test
        # it by printing an object of this class.
        return 'kernel_size={}, stride={}, padding={}, dilation={}, ceil_mode={})'.format(
            self.kernel_size, self.stride, self.padding, self.dilation, self.ceil_mode is not None)


def blockRelU(chan_in, chan_out, kernel_size=3, pad1=1, str1=1):
    return torch.nn.Sequential(
        torch.nn.Conv2d(chan_in, chan_out, kernel_size, padding=pad1, stride=str1),
        torch.nn.ReLU()
    )

def blockRelUMaxP(chan_in, chan_out, kernel_size=3, pad1=1, str1=1, kernel_maxpool=2, str_maxpool=2):
    return torch.nn.Sequential(
        torch.nn.Conv2d(chan_in, chan_out, kernel_size, padding=pad1, stride=str1),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(kernel_maxpool, stride=str_maxpool)
    )

def blockRelUMinP(chan_in, chan_out, kernel_size=3, pad1=1, str1=1, kernel_minpool=2, str_minpool=2):
    return torch.nn.Sequential(
        torch.nn.Conv2d(chan_in, chan_out, kernel_size, padding=pad1, stride=str1),
        torch.nn.ReLU(),
        MinPool2d(kernel_minpool, stride=str_minpool)
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

def block_fc_tanh(c_in, c_out):
    return torch.nn.Sequential(
        torch.nn.Linear(c_in, c_out),
        torch.nn.Tanh()
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

class CNNReLU_Tanh(torch.nn.Module):
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
    
    self.fc = block_fc_tanh(n_filters2*H*W, n_outputs)

  def forward(self, x):
    x = self.conv1(x)
    x = self.conv2(x)
    x = x.view(x.shape[0], -1)
    x = self.fc(x)
    return x


class CNNTanh_Tanh(torch.nn.Module):
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

    self.conv1 = blockTanhMaxP(chan_in, n_filters1, kernel_size1, pad1, str1)
    H = dim_after_filter(Hin, kernel_size1, pad1, str1)
    W = dim_after_filter(Win, kernel_size1, pad1, str1)
    # if maxpool2d
    H, W = dim_after_filter(H, kernel_maxpool, 0, str_maxpool), dim_after_filter(W, kernel_maxpool, 0, str_maxpool) 

    self.conv2 = blockTanhMaxP(n_filters1, n_filters2, kernel_size2, pad2, str2)
    H = dim_after_filter(H, kernel_size2, pad2, str2)
    W = dim_after_filter(W, kernel_size2, pad2, str2)
    # if maxpool2d
    H, W = dim_after_filter(H, kernel_maxpool, 0, str_maxpool), dim_after_filter(W, kernel_maxpool, 0, str_maxpool)
    
    self.fc = block_fc_tanh(n_filters2*H*W, n_outputs)

  def forward(self, x):
    x = self.conv1(x)
    x = self.conv2(x)
    x = x.view(x.shape[0], -1)
    x = self.fc(x)
    return x



class CNN_2branch(torch.nn.Module):
  def __init__(self, n_channels=1, Hin=32, Win=32, Hout=32, Wout=32):
    super().__init__()

    chan_in = n_channels 

    n_filters1, n_filters2 = 5, 5
    kernel_size1, kernel_size2 = 3, 3
    pad1, pad2 = 1, 1
    str1, str2 = 1, 1

    kernel_maxpool, str_maxpool = 2, 2
    kernel_minpool, str_minpool = 2, 2

    self.Hin = Hin
    self.Win = Win
    self.Hout = Hout
    self.Wout = Wout
    
    n_outputs = self.Hout* self.Wout

    self.conv1 = blockRelUMaxP(chan_in, n_filters1, kernel_size1, pad1, str1)
    H1 = dim_after_filter(Hin, kernel_size1, pad1, str1)
    W1 = dim_after_filter(Win, kernel_size1, pad1, str1)
    # if maxpool2d
    H1, W1 = dim_after_filter(H1, kernel_maxpool, 0, str_maxpool), dim_after_filter(W1, kernel_maxpool, 0, str_maxpool) 

    self.conv2 = blockRelUMinP(chan_in, n_filters2, kernel_size2, pad2, str2)
    H2 = dim_after_filter(Hin, kernel_size2, pad2, str2)
    W2 = dim_after_filter(Win, kernel_size2, pad2, str2)
    # if minpool2d
    H2, W2 = dim_after_filter(H2, kernel_minpool, 0, str_minpool), dim_after_filter(W2, kernel_minpool, 0, str_minpool) 
    
    self.fc = torch.nn.Linear(n_filters1*H1*W1 + n_filters2*H2*W2, n_outputs)

  def forward(self, x):
    x1 = self.conv1(x)
    x2 = self.conv2(x)

    y = torch.cat((x1.view(x1.shape[0], -1), x2.view(x2.shape[0], -1)), -1)    

    y = self.fc(y)
    return y









