# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 16:16:34 2023

@author: Ramón Pozuelo
"""

from pathlib import Path
from module_DatabaseOperations import load_DB_ZEA, subsamp_RIR

path_script = Path(r"C:\Users\keris\Desktop\Postdoc")
# path_script = Path(r"/scratch/ramonpr/3NoiseModelling")
# path_script = Path(r"C:\Users\keris\Desktop")

folder_Data = "DataBase_Zea"
files = ["BalderRIR.mat", "FrejaRIR.mat", "MuninRIR.mat"]

i_file=0
nx = 32
nt = 32

# %% Load image
path = Path(path_script).joinpath(folder_Data, files[i_file])
RIR = load_DB_ZEA(path)
Y = subsamp_RIR(nt, nx, RIR)

import matplotlib.pyplot as plt

idx = 0
image = Y[idx]

fig = plt.figure(dpi=200, figsize=(1,1))
ax = plt.subplot()
# plt.title("Input image")
plt.imshow(image.squeeze(), cmap="gray")
plt.axis("off")
plt.tight_layout()


# %% Tests downsamples in RIR. And divide RIR with the original no-overlap function: subsamp_RIR
from module_DatabaseOperations import subsamp_RIR, rand_downsamp_RIR, unif_downsamp_RIR
import numpy as np

ratioX = 0.5
ratioT = 0.5

# --- Random downsample of full RIR image -----------------------------------------
mask_X, mask_T, id_X, id_T = rand_downsamp_RIR(RIR, ratio_t=ratioT, ratio_x=ratioX)

# --- Apply downsample masks to RIR image -----------------------------------------
RIR_inp = RIR*mask_X 
RIR_inp = RIR_inp*mask_T 
Y = subsamp_RIR(nt, nx, RIR_inp)

# --- Plot downsample subimage -----------------------------------------------
idx = 0
image = Y[idx]
fig = plt.figure(dpi=200, figsize=(1,1))
ax = plt.subplot()
plt.title("Random downsample \n in t and x", fontsize=4)
plt.imshow(image.squeeze(), cmap="gray")
plt.axis("off")
plt.tight_layout()
plt.show()

# --- Uniform downsample of full RIR image -----------------------------------------
mask_X, mask_T, id_X, id_T = unif_downsamp_RIR(RIR, ratio_t=ratioT, ratio_x=ratioX)

# --- Apply downsample masks to RIR image -----------------------------------------
RIR_inp = RIR*mask_X 
RIR_inp = RIR_inp*mask_T 
Y = subsamp_RIR(nt, nx, RIR_inp)

# --- Plot downsample subimage -----------------------------------------------
image = Y[idx]
fig = plt.figure(dpi=200, figsize=(1,1))
ax = plt.subplot()
plt.title("Uniform downsample \n in t and x", fontsize=4)
plt.imshow(image.squeeze(), cmap="gray")
plt.axis("off")
plt.tight_layout()
plt.show()


# %% Test the DIVISION of RIR image in subsamples with the posibility of overlap
# through the election of a stride and the corresponding submask for each subimage.

from module_DatabaseOperations import divide_RIR

# --- Random downsample of full RIR image -----------------------------------------
mask_X, mask_T, id_X, id_T = rand_downsamp_RIR(RIR, ratio_t=1, ratio_x=0.5)

# --- Apply downsample masks to RIR image -----------------------------------------
RIR_inp = RIR*mask_X 
RIR_inp = RIR_inp*mask_T 
Y1 = subsamp_RIR(nt, nx, RIR_inp)

# --- OR send RIR and mask to be divided without applying downsampling -----------------------------------------
Y2, submask = divide_RIR(RIR, maskX=mask_X, kt=32, kx=32, strideT=32, strideX=32)

# --- Plot downsample subimage with both methods -----------------------------------------
idx = 0
image1 = Y1[idx]
image2 = Y2[idx]
image2 = image2*submask[idx]

fig = plt.figure(dpi=200, figsize=(1,1))
ax = plt.subplot(1,2,1)
plt.imshow(image1.squeeze(), cmap="gray")
plt.axis("off")
plt.title("Downsampling in RIR \n and subsample", fontsize=3)
plt.tight_layout()

ax = plt.subplot(1,2,2)
plt.imshow(image2.squeeze(), cmap="gray")
plt.axis("off")
plt.title("Division of RIR \n and mask", fontsize=3)
plt.tight_layout()



