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

    deltaT = T//tsamples
    deltaX = M//xMics

    id_T = np.arange(0,T,deltaT) # rows to take
    id_X = np.arange(0,M,deltaX) # cols to take

    mask_T = np.zeros([RIR.shape[0], 1])
    mask_X = np.zeros([1, RIR.shape[1]])

    mask_T[id_T,0] = 1
    mask_X[0,id_X] = 1

    return mask_X, mask_T, id_X, id_T

#-----------------------------------------------------------------------
