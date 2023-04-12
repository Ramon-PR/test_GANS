# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 23:26:04 2023

@author: Ramón Pozuelo
https://github.com/pytorch/pytorch/issues/12831

0.03778862953186035
682.9661931991577
"""

import time
import torch
import torch.utils.data as Data


#Step 2: time it
if __name__ == '__main__':
    train_dataset = torch.FloatTensor((100000, 32))

    batch_size = 32

    train_loader = Data.DataLoader(dataset=train_dataset,
    batch_size=batch_size, shuffle=True)
    train_loader2 = Data.DataLoader(dataset=train_dataset,
    batch_size=batch_size, shuffle=True, num_workers=8)

    start = time.time()
    for _ in range(200):
        for x in train_loader:
            pass
    end = time.time()
    print(end - start)

    start = time.time()
    for _ in range(200):
        for x in train_loader2:
            pass
    end = time.time()
    print(end - start)
    
    
    
    