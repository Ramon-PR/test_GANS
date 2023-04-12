# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 23:02:45 2023

@author: Ramón Pozuelo
"""

# Fit and evaluate operations

import torch
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

def wrapper_fit(model, dataloader, optimizer, scheduler, epochs=10, log_each=10, weight_decay=0, early_stopping=100, verbose=2, h0=dict(epoch=[0], loss=[0], val_loss=[0], lr=[0])):
    #Wrapper around fit to concatenate the results of different runs
    hist1 = fit(model, dataloader, optimizer, scheduler, epochs, log_each, weight_decay, early_stopping, verbose)
    
    temp = [ x + h0['epoch'][-1] for x in hist1['epoch'] ]
    hist=dict( epoch = h0['epoch'] + temp ,
               loss = h0['loss'] + hist1['loss'],
               val_loss = h0['val_loss'] + hist1['val_loss'],
               lr = h0['lr'] + hist1['lr']               
               )
    
    return hist

def evaluate(model, x, shape=[32,32]):
    # evaluate model on an image x reshaping the result
    # to a numpy array of size shape
	model.eval()
	x.to(device)
	model.to(device)
	x = torch.unsqueeze(x,0)
	y_pred = model(x)
	y_pred = torch.reshape(y_pred, shape)
	y_pred = y_pred.cpu().detach().numpy()
	return y_pred

# Fit:
# Send model to device
# The criterion is MSELoss
# Initialize list of loss, accuracy, etc that will be recorded
# Start epoch loop
# 	RECORD the learning rate in the Optimizer
# 	-> Training
# 	Set the model in training mode
# 	Load a batch of images, targets and submaks from the dataloader
# 	Send the batches of images, targets and submaks to the device
# 	---- optional rearrange the target as a vector?? ----
# 	Forward: Obtain the prediction of the model
# 	Loss: compare prediction and target with the chosen criterion
# 	RECORD
# 	Optimization:	The optimizer has a pointer to the model parameters to optimize
# 	 				zero_grad: to not accumulate gradients
# 					backward: Computes the gradient of current tensor w.r.t. graph leaves. 
# 					step: The optimizer gives a step towards the direction of minimum loss 
# 	-> Validation
# 	Set the model in evaluation mode
# 	Do not record anything in the gradients
# 	Load a batch of images, targets and submaks from the dataloader
# 	Send the batches of images, targets and submaks to the device
# 	---- optional rearrange the target as a vector?? ----
# 	Forward: Obtain the prediction of the model
# 	Loss: compare prediction and target with the chosen criterion
# 	RECORD
# 	-> Save Best model (Checkpoints)
# 	If the Validation loss is smaller than the val_loss in the previous epochs
# 		Save model (state dictionary) as a checkpoint "ckpt.pt" 
# 	-> Obtain the new learning rate (If there is a scheduler)
# 	-> Early Stopping (If the model does not improve after a certain number of epochs)



# def fit(model, dataloader, optimizer, scheduler=None, epochs=100, log_each=10, weight_decay=0, early_stopping=0):
def fit(model, dataloader, optimizer, scheduler=None, epochs=10, log_each=1, weight_decay=0, early_stopping=0, verbose=1):
	# device = "cuda" if torch.cuda.is_available() else "cpu"

	saved_checkpoint = False

	model.to(device)
	criterion = torch.nn.MSELoss() # sum( (xi-yi)**2 )/n
	l, acc, lr = [], [], []
	val_l, val_acc = [], []
	best_acc, step = 10, 0
	for e in range(1, epochs+1): 
		_l, _acc = [], []

# ------- Optimizer learning rate scheduler --------------------------
		for param_group in optimizer.param_groups:
			lr.append(param_group['lr'])
# --------------------------------------------------------------------

		model.train()
		for image, target, submask in dataloader['train']:
			image, target, submask = image.to(device), target.to(device), submask.to(device)
			target = target.view(target.shape[0],-1)
			y_pred = model(image)
			loss = criterion(y_pred, target)
			_l.append(loss.item())
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
		l.append(np.mean(_l))
		# acc.append(np.mean(_acc))
		model.eval()
		_l, _acc = [], []
		with torch.no_grad():
			for image, target, submask in dataloader['val']:
				image, target, submask = image.to(device), target.to(device), submask.to(device)
				target = target.view(target.shape[0],-1)
				y_pred = model(image)
				loss = criterion(y_pred, target)
				_l.append(loss.item())
				# y_probas = torch.argmax(softmax(y_pred), axis=1)            
				# _acc.append(accuracy_score(target.cpu().numpy(), y_probas.cpu().numpy()))
		val_l.append(np.mean(_l))
		# val_acc.append(np.mean(_acc))

# ---------------   Early stopping & best model ----------------------
		# Save best model
		if val_l[-1] < best_acc:
			best_acc = val_l[-1]
			torch.save(model.state_dict(), 'ckpt.pt')
			saved_checkpoint = True
			step = 0
			if verbose == 2:
				print(f"Best model saved with validation loss {val_l[-1]:.5f} in epoch {e}")
		step += 1

# ------- Optimizer learning rate scheduler --------------------------
		if scheduler:
			scheduler.step()

# -------- Early Stopping: Steps without improving -------------------
		if early_stopping and step > early_stopping:
			print(f"Training stopped in epoch {e}. It did not improve in the last {early_stopping} epochs")
			break
		if not e % log_each and verbose:
			print(f"Epoch {e}/{epochs} loss {l[-1]:.5f} val_loss {val_l[-1]:.5f}")


# -------- Load Best model -------------------------------------------
	if saved_checkpoint:
		model.load_state_dict(torch.load('ckpt.pt'))

	return {'epoch': list(range(1, len(l)+1)), 'loss': l, 'val_loss': val_l, 'lr': lr}

