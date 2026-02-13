#!/usr/bin/env python3
"""/***************************************************************************
 *
 * Authors:    Erney Ramirez-Aportela
 *
  ***************************************************************************/
"""


import mrcfile
import argparse
import sys, os
import numpy as np
import torch
from xmippPyModules.pcaAlignFunction.bnb_gpu import *
from xmippPyModules.pcaAlignFunction.pca_gpu import *
import torchvision.transforms as T
import time

def precalculateBands(nBand, dim, sampling, maxRes, minRes):

    vectorFreq = torch.fft.fftfreq(dim)
    freq_band = torch.full((dim,int(np.floor(dim/2))), 50)   

    maxFreq = sampling/maxRes
    minFreq = sampling/minRes
    factor = nBand * (1/maxFreq)  

    for x in range(dim):
        
        if vectorFreq[x] >= 0:
            wx = vectorFreq[x]

        for y in range(dim):
            wy = vectorFreq[y]

            w = torch.sqrt(wx**2 + wy**2)
            
            if (w > minFreq) and (w < maxFreq):
                freq_band[y][x] = torch.floor(w*factor)
    
    return freq_band


def augmented(images, num_tr):
    batch_size, size, size = images.size()

    degrees_range=(0, 360)
    translate_range=(0.1, 0.1)
    transform = T.RandomAffine(degrees=degrees_range, translate=translate_range)
    transformed_images = transform(images.clone().repeat(num_tr, 1, 1, 1))
    augmented_data = transformed_images.reshape(batch_size*num_tr, size, size)

    return augmented_data


if __name__=="__main__":
       
    parser = argparse.ArgumentParser(description="Train PCA")
    parser.add_argument("-i", "--mrc", help="input mrc file for references images)", required=True)
    parser.add_argument("-s", "--sampling", type=float, help="pixel size of the images", required=True)
    parser.add_argument("-hr", "--highres", type=float, help="highest resolution to consider", required=True)
    parser.add_argument("-lr", "--lowres", type=float, help="lowest resolution to consider", required=True)
    parser.add_argument("-p", "--perc", type=float, help="PCA percentage (between 0-1)", required=True)
    parser.add_argument("-t", "--training", type=int, help="number of image for training. Default is the number of images")
    parser.add_argument("-o", "--output", help="Root directory for the output files, for example: train_pca", required=True)
    parser.add_argument("--batchPCA",  action="store_true", help="BatchPCA only and not onlinePCA")
    # parser.add_argument("-n", "--bands", type=int, default=1, help="number of bands, default = 1")
    
    args = parser.parse_args()

    expFile = args.mrc   
    sampling = args.sampling
    maxRes = args.highres
    minRes = args.lowres
    per_eig = args.perc
    train = args.training
    output = args.output
    batchPCA = args.batchPCA
    # nBand = args.bands
    nBand = 1
    
    torch.cuda.is_available()
    torch.cuda.current_device()
    cuda = torch.device('cuda:0')
    # cuda = torch.device('cpu')        
    
    print("Reading Images")
    mexp = mrcfile.mmap(expFile, permissive=True)
    dim = mexp.data.shape[1]
    nExp = mexp.data.shape[0]
    # nExp = 5000

    if train:
        nExp = train 
  
    # Precalculate frequency bands  
    freq_band = precalculateBands(nBand, dim, sampling, maxRes, minRes) 
    torch.save(freq_band, output + "_bands.pt")

    coef = torch.zeros(nBand, dtype=int)
    for n in range(nBand):
        coef[n] = 2*torch.sum(freq_band==n)  
        
    bnb = BnBgpu(nBand)    
    expBatchSize = 400  
 
    band = [torch.empty(0, coef[n], device = cuda) for n in range(nBand)]     
     
    print("Select bands of images")            
    for initBatch in range(0, nExp, expBatchSize):
        
        endBatch = initBatch+expBatchSize 
        if (endBatch > nExp):
            endBatch = nExp
        
        expImages = mexp.data[initBatch:endBatch].astype(np.float32)#.copy()
        Texp = torch.from_numpy(expImages).float().to(cuda)
        del(expImages)
        
        for augm in range(10):
            Texp_augmented = augmented(Texp, 5)

            ft = torch.fft.rfft2(Texp_augmented, norm="forward")
            bandBatch = bnb.selectBandsRefs(ft, freq_band, coef)
            del(ft)

            for n in range(nBand):
                band[n] = torch.cat((band[n], bandBatch[n]), dim=0)
        del(Texp, bandBatch)
       
        
    #Training PCA 
    print("PCA")
    pca = PCAgpu(nBand)    
    mean, vals, vecs = pca.trainingPCAonline(band, coef, per_eig, batchPCA) 
    torch.save(mean, output + "_mean.pt")
    torch.save(vals, output + "_vals.pt")
    torch.save(vecs, output + "_vecs.pt")      
            
 