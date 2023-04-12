# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 16:16:34 2023

@author: Ramón Pozuelo
"""

from pathlib import Path

path_script = Path(r"C:\Users\keris\Desktop\Postdoc")
# path_script = Path(r"/scratch/ramonpr/3NoiseModelling")
# path_script = Path(r"C:\Users\keris\Desktop")

folder_Data = "DataBase_Zea"
files = ["BalderRIR.mat", "FrejaRIR.mat", "MuninRIR.mat"]

# Paths of each RIR database
path=[]
for i_file in range(len(files)):
    path.append(Path(path_script).joinpath(folder_Data, files[i_file]))

# %% RIR Database in subsamples
# Downsample the RIR images obtaining a masks
# Divide the RIR images into subimages and submasks
# Inputs, are the subimages cropped through the submasks
# Target are the original subimages
# Inputs and targets are randomly transformed (HorizontalFlip)
import torch
import torchvision
from module_DatabaseOperations import RirDataset, unif_downsamp_RIR, rand_downsamp_RIR

std=1 # Careful with std. With std=0.1 it does not converge. std=1 OK
transformaciones = torchvision.transforms.Compose([
        torchvision.transforms.RandomHorizontalFlip(p=0.5),
        torchvision.transforms.Normalize((0.0), (std)) # ( img - mean)/std
    ])

f_downsamp_RIR = unif_downsamp_RIR # Uniform or random downsampling of RIR?
target_transform = transformaciones # Transform the target?
input_transform = True # Inputs are downsampled?

# Parameters to divide RIR in subimages and ratios of microphones we use (downsampling)
param_downsampling = {'ratio_t':1, 'ratio_x':0.6, 
                      'kernel':(32, 32), 'stride':(32,32)}

# Dataset
# Train: Balder & Freja. Uniform downsampling, DataAugmentation, Input downsampled
# Validation: Munin. Uniform downsampling, No DataAugmentation, Input downsampled
dataset = {
    'train': RirDataset(path[0:2], param_downsampling, 
                      f_downsamp_RIR, target_transform, input_transform),
    'val': RirDataset(path[2:3], param_downsampling, 
                      f_downsamp_RIR, None, input_transform)
}

# DataLoader
# batchsizes:
dataloader = {
    'train': torch.utils.data.DataLoader(dataset['train'],       # datos
                                         batch_size=1000,         # tamaño del batch, número de imágenes por iteración
                                         shuffle=True,           # barajamos los datos antes de cada epoch
                                         num_workers=0,          # número de procesos que se lanzan para cargar los datos (número de cores de la CPU para carga en paralelo)
                                         pin_memory=True,        # si tenemos una GPU, los datos se cargan en la memoria de la GPU
                                         collate_fn=None,        # función para combinar los datos de cada batch                                         
                                         ),
    'val': torch.utils.data.DataLoader(dataset['val'],         # datos
                                       batch_size=1000,         # tamaño del batch, número de imágenes por iteración
                                       shuffle=False,           # barajamos los datos antes de cada epoch
                                       num_workers=0,          # número de procesos que se lanzan para cargar los datos (número de cores de la CPU para carga en paralelo)
                                       pin_memory=True,        # si tenemos una GPU, los datos se cargan en la memoria de la GPU
                                       collate_fn=None,        # función para combinar los datos de cada batch
                                       )
}



# %% Definir NN

from NNmodel_classes import CNN
import timeit

n_channels, Hin, Win, Hout, Wout = 1, 32, 32, 32, 32
model = CNN(n_channels, Hin, Win, Hout, Wout)

from NNmodel_classes import count_parameters
print("\nNumber of trainable parameters: %i " % (count_parameters(model)))

hist=dict(epoch=[0], loss=[0], val_loss=[0], lr=[0])

# %% Train the model
from module_Model_operations import wrapper_fit

nepochs=10
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0)

# multiplica el lr por 0.1 cada 10 epochs
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, 0.1)
# aumenta el lr por 5 epochs, luego decrece
# scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-3, max_lr=1e-1, step_size_up=10, step_size_down=40, cycle_momentum=False)
scheduler=None

tStart = timeit.default_timer()
hist = wrapper_fit(model, dataloader, optimizer, scheduler, epochs=nepochs, log_each=10, weight_decay=0, early_stopping=100, verbose=2, h0=hist)
tStop = timeit.default_timer()

print("\nThe training time is %f sec" % (tStop - tStart))

import pandas as pd
import matplotlib.pyplot as plt

fig = plt.figure(dpi=200, figsize=(3,3))
ax = plt.subplot(211)
pd.DataFrame(hist).plot(x='epoch', y=['loss', 'val_loss'], grid=True, ax=ax)
ax = plt.subplot(212)
pd.DataFrame(hist).plot(x='epoch', y=['lr'], grid=True, ax=ax)
plt.show()

# %%
from module_Model_operations import evaluate
import matplotlib.pyplot as plt

idx=0
image, target, submask = dataset['val'].__getitem__(idx)
# image, target, submask = dataset['train'].__getitem__(idx)

image_predicted = evaluate(model, image) 

fig = plt.figure(dpi=200, figsize=(3,2))
ax = plt.subplot(221)
plt.imshow(image.squeeze(), cmap="gray")
plt.axis("off")
ax = plt.subplot(222)
plt.imshow(target.squeeze(), cmap="gray")
plt.axis("off")
ax = plt.subplot(223)
plt.imshow(image_predicted.squeeze(), cmap="gray")
plt.axis("off")
plt.tight_layout()

















