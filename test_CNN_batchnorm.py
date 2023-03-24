# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 18:20:20 2023

@author: Ramón Pozuelo
"""
from pathlib import Path
# import sys
from module_CNN_ZEA import load_DB_ZEA, down_samp_transf, ZeaDataset, fit

# path_script = Path(r"C:\Users\keris\Desktop\Postdoc")
# path_script = Path(r"/scratch/ramonpr/3NoiseModelling")
path_script = Path(r"C:\Users\keris\Desktop")

folder_Data = "DataBase_Zea"
files = ["BalderRIR.mat", "FrejaRIR.mat", "MuninRIR.mat"]

i_file=0
nx = 32
nt = 32

path = Path(path_script).joinpath(folder_Data, files[i_file])
Y, _ = load_DB_ZEA(path, nx, nt)


# %% Definir Dataset
import torch
import torchvision
RIR_mean, RIR_min, RIR_max = Y.mean(), Y.min(), Y.max()
RIR_amp = RIR_max - RIR_min
std = max(RIR_max, abs(RIR_min))

transformaciones = torchvision.transforms.Compose([
        torchvision.transforms.RandomVerticalFlip(p=0.75),
        torchvision.transforms.RandomHorizontalFlip(p=0.75),
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



# %% Definir NN

# resnet = torchvision.models.resnet18(pretrained=True)
# num_classes = nt*nx
# resnet.fc = torch.nn.Linear(resnet.fc.in_features, num_classes)
# resnet
from NNmodel_classes import CNN_batchnorm
import timeit

# nepochs=500

n_channels, Hin, Win, Hout, Wout = 1, 32, 32, 32, 32
# model = CNN(n_channels, Hin, Win, Hout, Wout)
model = CNN_batchnorm(n_channels, Hin, Win, Hout, Wout)

from NNmodel_classes import count_parameters
# print(count_parameters(model))
print("\nNumber of trainable parameters: %i " % (count_parameters(model)))


# %%

nepochs=500

# optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0)

# multiplica el lr por 0.1 cada 10 epochs
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, 0.1)
# aumenta el lr por 5 epochs, luego decrece
# scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-3, max_lr=1e-1, step_size_up=10, step_size_down=40, cycle_momentum=False)
scheduler=None

tStart = timeit.default_timer()
hist = fit(model, dataloader, optimizer, scheduler, epochs=nepochs, log_each=10, weight_decay=0)
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


# %% PLOT EXAMPLE IMAGE

idx=0
image_in, label_in = dataset['val'][idx]
# Add batch dimension before inputting into the model
image_in = image_in[None,:,:,:]
model.eval()
image_out = model(image_in)
image_out = image_out.reshape(Hout, Wout)

image_out = image_out.cpu().detach().numpy()
image_in = image_in.cpu().detach().numpy()

i=1
plt.subplot(2,1,i)
plt.tight_layout()
plt.title("Input image")
plt.imshow(image_in.squeeze(), cmap="gray")

i=2
plt.subplot(2,1,i)
plt.tight_layout()
plt.title("Output image from the trained model")
plt.imshow(image_out.squeeze(), cmap="gray")

