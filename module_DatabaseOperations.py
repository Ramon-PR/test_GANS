import torch
import numpy as np
import pymatreader as pymat
import random

device = "cuda" if torch.cuda.is_available() else "cpu"
# device="cpu"

#-----------------------------------------------------------------------
#---------- LOAD DATABASE ------------------------------------------------
#-----------------------------------------------------------------------

def load_DB_ZEA(path):
    #Load data
    RIR=[]
    RIR = pymat.read_mat(path)["out"]["image"]
    # T, M = RIR.shape

    return RIR

#-----------------------------------------------------------------------
#---------- DIVIDE RIR IMAGE IN SUBSAMPLES -----------------------------
#-----------------------------------------------------------------------
def subsamp_RIR(nt, nx, RIR):
    # Divide RIR in small imagenes no-overlap
    # Num of imag in x and time
    T, M = RIR.shape
    tImags = T//nt
    xImags = M//nx
    nImags = (xImags)*(tImags)

    # Imags without modification
    Y = np.zeros([nImags,nt,nx])

    for imx in range(xImags):
        for imt in range(tImags):
            im = imx*tImags + imt
            # Y[idx] is (nx,nt) if
            Y[im, :, :] = RIR[imt*nt : (imt+1)*nt, imx*nx : (imx+1)*nx]            
    return Y


def divide_RIR(RIR, maskX, kt=32, kx=32, strideT=32, strideX=32):
    # Divide RIR in small imagenes 
    # the image is a kernel of size (kt, kx) 
    # the kernel moves in strides strideX and strideT
    # if kt=strideT and kx=strideX then "no-overlap"
    # it also gives the downsampling submask for each subimage.
    
    T, M = RIR.shape
    
    # stride = s
    # kernel = k
    # total: M
    xImags = (M - kx)//strideX + 1
    tImags = (T - kt)//strideT + 1

    nImags = (xImags)*(tImags)

    # Imags without modification
    Y = np.zeros([nImags,kt,kx])
    subMask = np.ones([nImags,1,kx])

    # Scaling
    # max_Image = RIR.max()
    # min_Image = RIR.min()

    for imx in range(xImags):
        for imt in range(tImags):
            im = imx*tImags + imt
            # Y[idx] is (nx,nt) if
            Y[im, :, :] = RIR[strideT*imt  : (strideT*imt)+kt, strideX*imx  : (strideX*imx)+kx]
            subMask[im, 0,:] = maskX[0,strideX*imx  : (strideX*imx)+kx]            
    return Y, subMask

#-----------------------------------------------------------------------
#---------- DOWNSAMPLE RIR IMAGE and GENERATE the General Masks --------
#-----------------------------------------------------------------------


def rand_downsamp_RIR(RIR, ratio_t=1, ratio_x=0.5):
    # choose a ratio of samples in time/space from RIR
    # random choice
    T, M = RIR.shape
    tsamples = int(T*ratio_t)
    xMics  = int(M*ratio_x)

    id_T = np.sort(random.sample(range(0,T), tsamples)) # rows to take
    id_X = np.sort(random.sample(range(0,M), xMics)) # cols to take

    # comp_mask_T = list(set(range(0,T)) - set(mask_T))
    # comp_mask_X = list(set(range(0,M)) - set(mask_X))

    mask_T = np.zeros([RIR.shape[0], 1])
    mask_X = np.zeros([1, RIR.shape[1]])

    mask_T[id_T,0] = 1
    mask_X[0,id_X] = 1

    return mask_X, mask_T, id_X, id_T


def unif_downsamp_RIR(RIR, ratio_t=1, ratio_x=0.5):
    # choose a ratio of samples in time/space from RIR
    # random choice
    T, M = RIR.shape
    tsamples = int(T*ratio_t)
    xMics  = int(M*ratio_x)

    deltaT = T/tsamples
    deltaX = M/xMics

    id_T = np.rint(np.arange(0,T,deltaT)).astype(int) # rows to take
    id_X = np.rint(np.arange(0,M,deltaX)).astype(int) # cols to take

    mask_T = np.zeros([RIR.shape[0], 1], dtype=bool)
    mask_X = np.zeros([1, RIR.shape[1]], dtype=bool)

    mask_T[id_T,0] = True
    mask_X[0,id_X] = True

    return mask_X, mask_T, id_X, id_T

#-----------------------------------------------------------------------
#---------- DOWNSAMPLE RIR IMAGE and GENERATE the General Masks --------
#-----------------------------------------------------------------------



default_downsampling = {'ratio_t':1, 'ratio_x':0.5, 'kernel':(32, 32), 'stride':(32,32)}

class RirDataset(torch.utils.data.Dataset):
    # Constructor
    def __init__(self, paths, param_downsampling=default_downsampling, f_downsamp_RIR=unif_downsamp_RIR,  target_transform=None, input_transform=None):
    # def __init__(self, X, transform=None):
        # change to torch tensor, convert to float32 
        # and add a dimension at the beginning (1,H,W) 
        # to indicate the number of channels (batch,C,H,W)
        # device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Add dimension for batch
        self.paths = paths
        self.target_transform = target_transform
        self.input_transform = input_transform
        
        # Parameters downsampling and division of RIR image
        pd = param_downsampling

        RIR = load_DB_ZEA(paths[0])
        maskX, maskT, id_X, id_T = f_downsamp_RIR(RIR, pd['ratio_t'], pd['ratio_x'])
        Y0, submask0 = divide_RIR(RIR, maskX, pd['kernel'][0], pd['kernel'][1], pd['stride'][0], pd['stride'][1])
        Y0 = torch.from_numpy(Y0).float().unsqueeze(1)
        submask0 = torch.from_numpy(submask0).float().unsqueeze(1)            
        
        self.X = Y0
        self.mask_col = submask0

        for file in paths[1:]:
            RIR = load_DB_ZEA(paths[0])
            maskX, maskT, id_X, id_T = f_downsamp_RIR(RIR, pd['ratio_t'], pd['ratio_x'])
            Y0, submask0 = divide_RIR(RIR, maskX, pd['kernel'][0], pd['kernel'][1], pd['stride'][0], pd['stride'][1])
            Y0 = torch.from_numpy(Y0).float().unsqueeze(1)
            submask0 = torch.from_numpy(submask0).float().unsqueeze(1) 
            
            self.X = torch.cat((self.X, Y0), dim=0)
            self.mask_col = torch.cat((self.mask_col, submask0), dim=0)
        
    
    # Return the number of data in our dataset
    def __len__(self):
        return len(self.X)
    
    
    # Return the element idx in the dataset
    def __getitem__(self, idx):
        image = self.X[idx]
        im_mask = self.mask_col[idx]
        target = self.X[idx]
        
        if self.target_transform:
            C, H, W = target.size()
            # random Horizontal Flip of mask and image at the same time
            temp  = torch.cat((image, im_mask),dim=-2)
            temp = self.target_transform(temp)
            
            # Recuperate the transformed target and mask
            target, im_mask = torch.split(temp, H, dim=-2)
            image = target

        if self.input_transform:
            # The input is the target image but downsampled with the mask
            image = target*im_mask
            
        return image, target, im_mask