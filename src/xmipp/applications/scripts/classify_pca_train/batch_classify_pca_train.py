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
from xmippPyModules.classifyPcaFuntion.bnb_gpu import BnBgpu
from xmippPyModules.classifyPcaFuntion.pca_gpu import PCAgpu
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

def normalize_and_rescale_batch_images( batch_images, desired_min=0, desired_max=1):

    mean = torch.mean(batch_images)
    std = torch.std(batch_images)

    normalized_images = (batch_images - mean) / std
    normalized_images = normalized_images * (desired_max - desired_min) + desired_min
    normalized_images = normalized_images * std + mean

    return normalized_images


if __name__=="__main__":
       
    parser = argparse.ArgumentParser(description="Train PCA")
    parser.add_argument("-i", "--mrc", help="input mrc file for experimental images)", required=True)
    # parser.add_argument("-n", "--bands", help="number of bands", required=True)
    # parser.add_argument("-n", "--bands", type=int, default=1, help="number of bands, default = 1")
    parser.add_argument("-s", "--sampling", help="pixel size of the images", required=True)
    parser.add_argument("-hr", "--highres", help="highest resolution to consider", required=True)
    parser.add_argument("-lr", "--lowres", help="lowest resolution to consider", required=True)
    parser.add_argument("-p", "--perc", help="PCA percentage (between 0-1)", required=True)
    parser.add_argument("-t", "--training", help="number of image for training", required=True)
    parser.add_argument("-o", "--output", help="Root directory for the output files, for example: train_pca", required=True)
    parser.add_argument("--batchPCA",  action="store_true", help="BatchPCA only and not onlinePCA")
    parser.add_argument("-g", "--gpu",  help="GPU ID set. Just one GPU is allowed", required=False)

    args = parser.parse_args()

    expFile = args.mrc  
    # nBand = int(args.bands) 
    sampling = float(args.sampling)
    maxRes = float(args.highres)
    minRes = float(args.lowres)
    per_eig = float(args.perc)
    train = int(args.training)
    gpuId = str(args.gpu)
    output = args.output
    batchPCA = args.batchPCA
    nBand = 1

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpuId


    torch.cuda.is_available()
    torch.cuda.current_device()
    cuda = torch.device('cuda:0')
    # cuda = torch.device('cpu')        
    
    print("Reading Images")
    mexp = mrcfile.mmap(expFile, permissive=True)
    dim = mexp.data.shape[1]
    nExp = train 

    
    # Precalculate frequency bands  
    freq_band = precalculateBands(nBand, dim, sampling, maxRes, minRes) 
    torch.save(freq_band, output + "_bands.pt")

    coef = torch.zeros(nBand, dtype=int)
    for n in range(nBand):
        coef[n] = 2*torch.sum(freq_band==n)  
        
    bnb = BnBgpu(nBand)    
    expBatchSize = 5000  
    band = [torch.zeros(nExp, coef[n], device = cuda) for n in range(nBand)]      
     
    print("Select bands of experimental images")            
    for initBatch in range(0, nExp, expBatchSize):
        
        endBatch = initBatch+expBatchSize 
        if (endBatch > nExp):
            endBatch = nExp
        
        expImages = mexp.data[initBatch:endBatch].astype(np.float32)#.copy()
        Texp = torch.from_numpy(expImages).float().to(cuda)
        Texp = Texp * bnb.create_circular_mask(Texp)

        del(expImages)
        ft = torch.fft.rfft2(Texp, norm="forward")
        del(Texp)
        bandBatch = bnb.selectBandsRefs(ft, freq_band, coef)
        del(ft)
        for n in range(nBand):
            band[n][initBatch:endBatch] = bandBatch[n]
        del(bandBatch)
       
        
    #Training PCA 
    pca = PCAgpu(nBand)    
    mean, vals, vecs = pca.trainingPCAonline(band, coef, per_eig, batchPCA) 
    torch.save(mean, output + "_mean.pt")
    torch.save(vals, output + "_vals.pt")
    torch.save(vecs, output + "_vecs.pt")        
            
 