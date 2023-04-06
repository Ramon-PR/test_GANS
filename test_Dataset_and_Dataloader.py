# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 14:10:01 2023

@author: Ramón Pozuelo
"""

from pathlib import Path
from module_DatabaseOperations import load_DB_ZEA, subsamp_RIR

path_script = Path(r"C:\Users\keris\Desktop\Postdoc")
# path_script = Path(r"/scratch/ramonpr/3NoiseModelling")
# path_script = Path(r"C:\Users\keris\Desktop")

folder_Data = "DataBase_Zea"
files = ["BalderRIR.mat", "FrejaRIR.mat", "MuninRIR.mat"]

# Paths of each RIR database
path=[]
for i_file in range(len(files)):
    path.append(Path(path_script).joinpath(folder_Data, files[i_file]))

nx = 32
nt = 32

# %% Load image
i_file=0
RIR = load_DB_ZEA(path[i_file])
Y1 = subsamp_RIR(nt, nx, RIR)

i_file=1
RIR = load_DB_ZEA(path[i_file])
Y2 = subsamp_RIR(nt, nx, RIR)

i_file=2
RIR = load_DB_ZEA(path[i_file])
Y3 = subsamp_RIR(nt, nx, RIR)

import matplotlib.pyplot as plt

idx = 0

fig = plt.figure(dpi=200, figsize=(1,1))
ax = plt.subplot(131)
plt.imshow(Y1[idx].squeeze(), cmap="gray")
plt.axis("off")
plt.tight_layout()
ax = plt.subplot(132)
plt.imshow(Y2[idx].squeeze(), cmap="gray")
plt.axis("off")
plt.tight_layout()
ax = plt.subplot(133)
plt.imshow(Y3[idx].squeeze(), cmap="gray")
plt.axis("off")
plt.tight_layout()


# %% Try Database with several RIR
import numpy as np
import torch
import torchvision
from module_DatabaseOperations import RirDataset, unif_downsamp_RIR, rand_downsamp_RIR

std=0.1
transformaciones = torchvision.transforms.Compose([
        torchvision.transforms.RandomHorizontalFlip(p=0.5),
        torchvision.transforms.Normalize((0.0), (std)) # ( img - mean)/std
    ])

f_downsamp_RIR = unif_downsamp_RIR
target_transform = transformaciones
input_transform = True
param_downsampling = {'ratio_t':1, 'ratio_x':0.6, 
                      'kernel':(32, 32), 'stride':(32,32)}

database = RirDataset(path[0:1], param_downsampling, 
                      f_downsamp_RIR, target_transform, input_transform)


idx = 114
image, target, im_mask = database.__getitem__(idx)

print("Shape image: ", image.size())
print("Shape target: ", target.size())
print("Shape im_mask: ", im_mask.size())

fig = plt.figure(dpi=200, figsize=(1,1))
ax = plt.subplot(221)
plt.imshow(image.squeeze(), cmap="gray")
plt.axis("off")
plt.tight_layout()
ax = plt.subplot(222)
plt.imshow(target.squeeze(), cmap="gray")
plt.axis("off")
plt.tight_layout()
ax = plt.subplot(223)
plt.imshow(im_mask.squeeze(0), cmap="gray")
plt.axis("off")
plt.tight_layout()


# %% Test dataset train and validation

param_downsampling = {'ratio_t':1, 'ratio_x':0.5, 
                      'kernel':(32, 32), 'stride':(32,32)}

std=1
transformaciones = torchvision.transforms.Compose([
        torchvision.transforms.RandomHorizontalFlip(p=0.5),
        torchvision.transforms.Normalize((0.0), (std)) # ( img - mean)/std
    ])

f_downsamp_RIR = unif_downsamp_RIR
target_transform = transformaciones
input_transform = True

# path[0:1]=='Balder', path[1:2]=='Freja', path[2:3]=='Munin'
# path[0:2]=='Balder'&'Freja' 

dataset = {
    'train': RirDataset(path[0:2], param_downsampling, 
                      unif_downsamp_RIR, target_transform, "crop"),
    'val': RirDataset(path[2:3], param_downsampling, 
                      unif_downsamp_RIR, None, "crop")
}

idx = 0

# image1, target1, submask1 = dataset['train'][idx]
image1, target1, submask1 = dataset['val'][idx]


fig = plt.figure(dpi=200, figsize=(3,2))
ax = plt.subplot(131)
plt.imshow(image1.squeeze(), cmap="gray")
plt.axis("off")
ax = plt.subplot(132)
plt.imshow(target1.squeeze(), cmap="gray")
plt.axis("off")
plt.tight_layout()







