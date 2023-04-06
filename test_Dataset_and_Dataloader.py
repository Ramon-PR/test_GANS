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





# %%



# --- Uniform downsample of full RIR image -----------------------------------------
ratioT = 1
ratioX = 0.5

i_file = 0
RIR = load_DB_ZEA(path[i_file])
maskX, maskT, id_X, id_T = unif_downsamp_RIR(RIR, ratio_t=ratioT, ratio_x=ratioX)
Y0, submask0 = divide_RIR(RIR, maskX, kt=32, kx=32, strideT=32, strideX=32)

i_file = 1
RIR = load_DB_ZEA(path[i_file])
maskX, maskT, id_X, id_T = unif_downsamp_RIR(RIR, ratio_t=ratioT, ratio_x=ratioX)
Y1, submask1 = divide_RIR(RIR, maskX, kt=32, kx=32, strideT=32, strideX=32)

i_file = 2
RIR = load_DB_ZEA(path[i_file])
maskX, maskT, id_X, id_T = unif_downsamp_RIR(RIR, ratio_t=ratioT, ratio_x=ratioX)
Y2, submask2 = divide_RIR(RIR, maskX, kt=32, kx=32, strideT=32, strideX=32)

Y = np.concatenate((Y0,Y1), axis=0)
submask = np.concatenate((submask0,submask1), axis=0)

dataset = {
    'train': RirDataset(Y, submask), #(X_train, y_train),
    'val': RirDataset(Y2, submask2) #(X_val, y_val)
}

idx = 0

image1, target1, submask1 = dataset['train'][idx]
fig = plt.figure(dpi=200, figsize=(3,2))
ax = plt.subplot(131)
plt.imshow(image1.squeeze(), cmap="gray")
plt.axis("off")
ax = plt.subplot(132)
plt.imshow(target1.squeeze(), cmap="gray")
plt.axis("off")
ax = plt.subplot(133)
plt.imshow((image1*submask1).squeeze(0), cmap="gray")
plt.axis("off")



plt.tight_layout()




# %%
import numpy as np
Y = np.concatenate((Y0,Y1), axis=0)

print(Y0.shape)
print(Y1.shape)
print(Y.shape)

X0 = torch.from_numpy(Y0).float().unsqueeze(1)
X1 = torch.from_numpy(Y1).float().unsqueeze(1)
# X = torch.stack((X0, X1), dim=0) # does not work
X = torch.cat((X0, X1), dim=0) # it works

print(X0.shape)
print(X1.shape)
print(X.shape)




