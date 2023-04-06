# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 16:16:34 2023

@author: Ramón Pozuelo
"""

from pathlib import Path
from module_DatabaseOperations import load_DB_ZEA

path_script = Path(r"C:\Users\keris\Desktop\Postdoc")
# path_script = Path(r"/scratch/ramonpr/3NoiseModelling")
# path_script = Path(r"C:\Users\keris\Desktop")

folder_Data = "DataBase_Zea"
files = ["BalderRIR.mat", "FrejaRIR.mat", "MuninRIR.mat"]

i_file=0
nx = 32
nt = 32

path = Path(path_script).joinpath(folder_Data, files[i_file])
RIR = load_DB_ZEA(path)

# %% Extract images and submasks for downsampling

from module_DatabaseOperations import rand_downsamp_RIR, divide_RIR

mask_X, mask_T, id_X, id_T = rand_downsamp_RIR(RIR, ratio_t=1, ratio_x=0.5)
Y, submask = divide_RIR(RIR, maskX=mask_X, kt=32, kx=32, strideT=32, strideX=16)


# %% PLOT example of original and subsampled image
import matplotlib.pyplot as plt

idx = 0
image1 = Y[idx]
image2 = Y[idx]*submask[idx]

fig = plt.figure(dpi=200, figsize=(1,2))
ax = plt.subplot(1,2,1)
plt.imshow(image1.squeeze(), cmap="gray")
plt.axis("off")
plt.title("Subimage \n of RIR", fontsize=3)
plt.tight_layout()

ax = plt.subplot(1,2,2)
plt.imshow(image2.squeeze(), cmap="gray")
plt.axis("off")
plt.title("Downsampled \n subimage", fontsize=3)
plt.tight_layout()

# %% Definir Dataset
import torch
import torchvision
from module_CNN_ZEA import down_samp_transf, ZeaDataset

RIR_mean, RIR_min, RIR_max = Y.mean(), Y.min(), Y.max()
RIR_amp = RIR_max - RIR_min
std = max(RIR_max, abs(RIR_min))

transformaciones = torchvision.transforms.Compose([
        torchvision.transforms.RandomHorizontalFlip(p=0.5),
        torchvision.transforms.Normalize((0.0), (std)) # ( img - mean)/std
    ])

transformaciones_crop = torchvision.transforms.Compose([
        down_samp_transf(stride=2, val=0)
    ])


dataset = {
    'train': ZeaDataset(Y, transformaciones, transformaciones_crop), #(X_train, y_train),
    'val': ZeaDataset(Y) #(X_val, y_val)
}

dataloader = {
    'train': torch.utils.data.DataLoader(dataset['train'], batch_size=1000, shuffle=True),
    'val': torch.utils.data.DataLoader(dataset['val'], batch_size=1000, shuffle=False)
}




