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
from xmippPyModules.classifyPcaFuntion.assessment import evaluation


def read_images(mrcfilename):

    with mrcfile.open(mrcfilename, permissive=True) as f:
         emImages = f.data.astype(np.float32).copy()
    return emImages


def save_images(data, voxel, outfilename):
    data = data.astype('float32')
    
    if data.ndim == 2:
        data = np.expand_dims(data, axis=0)
        
    with mrcfile.new(outfilename, overwrite=True) as mrc:
        mrc.set_data(data)
        mrc.voxel_size = (voxel, voxel, 1)
        mrc.update_header_stats()
        


def flatGrid(freq_band, coef, nBand):
    
    grid_flat = [torch.zeros((2, int(coef[n]/2)), dtype=torch.float32, device = cuda) for n in range(nBand)]
    
    dim = freq_band.size(dim=0) 
    dimfreq = freq_band.size(dim=1)
    
    freq_x = torch.fft.rfftfreq(dim, d=0.5/np.pi, device=cuda)
    freq_y = torch.fft.fftfreq(dim, d=0.5/np.pi, device=cuda)

    grid = torch.meshgrid(freq_x, freq_y, indexing='xy')
    
    for n in range(nBand):
        grid_flat[n][0] = grid[0][:,:dimfreq][freq_band == n]      
        grid_flat[n][1] = grid[1][:,:dimfreq][freq_band == n]
        
    return(grid_flat)  

       
if __name__=="__main__":
      
    parser = argparse.ArgumentParser(description="align images")
    parser.add_argument("-i", "--exp", help="input mrc file for experimental images)", required=True)
    parser.add_argument("-s", "--sampling", help="pixel size of the images", required=True)
    parser.add_argument("-c", "--classes", help="number of 2D classes", required=True)
    parser.add_argument("-r", "--ref", help="2D classes of external method")   
    parser.add_argument("--mask",  action="store_true", help="A Gaussian mask is used.")
    parser.add_argument("--sigma", type=float, help="value of sigma for the Gaussian mask. "
                                                    "It is only used if the --mask option is applied.")
    parser.add_argument("-o", "--output", help="Root directory for the output files", required=True)
    parser.add_argument("-stExp", "--starExp", help="star file for images")
    #For training
    parser.add_argument("-t", "--training", help="number of image for training", required=True)
    parser.add_argument("-hr", "--highres", help="highest resolution to consider", required=True)
    parser.add_argument("-p", "--perc", help="PCA percentage (between 0-1)", required=True)

    
    args = parser.parse_args()
    
    expFile = args.exp
    sampling = float(args.sampling)
    classes = int(args.classes)
    final_classes = classes  
    refImages = args.ref
    niter = 18
    mask = args.mask
    sigma = args.sigma
    output = args.output
    expStar = args.starExp
    Ntrain = int(args.training)
    highRes = float(args.highres)
    per_eig_value = float(args.perc)
           
    torch.cuda.is_available()
    torch.cuda.current_device()
    cuda = torch.device('cuda:0')
    
    #Determining GPU free memory
    gpu = torch.cuda.get_device_properties(0)
    total_memory = gpu.total_memory 
    allocated_memory = torch.cuda.memory_allocated(0)
    free_memory = (total_memory - allocated_memory) / (1024 ** 3)    # free memory GB
    print("Free memory %s" %free_memory)

    #Read Images
    mmap = mrcfile.mmap(expFile, permissive=True)
    nExp = mmap.data.shape[0]
    dim = mmap.data.shape[1]
    
    if mask and (sigma is None):
        sigma = dim/3
    

    initSubset = min(100000, nExp)
    refClas = torch.zeros(nExp)
    dist = torch.zeros(nExp)
    translation_vector = torch.zeros(nExp, 2)
    angles_deg = np.zeros(nExp)
    
    #PCA function
    nBand = 1
    pca = PCAgpu(nBand)
    
    if refImages:
        maxRes = highRes
    else: 
        maxRes = 18.0    
    freqBn, cvecs, coef = pca.calculatePCAbasis(mmap, Ntrain, nBand, dim, sampling, maxRes, 
                                                minRes=530, per_eig=per_eig_value, batchPCA=True)

    grid_flat = flatGrid(freqBn, coef, nBand)

    bnb = BnBgpu(nBand)
       
    expBatchSize, expBatchSize2, numFirstBatch, initClBatch = bnb.determine_batches(free_memory, dim) 
    print("batches: %s, %s, %s, %s" %(expBatchSize, expBatchSize2, numFirstBatch, initClBatch))   


    #Initial classes
    if refImages: 
        initStep = -1
        clIm = read_images(refImages)
        cl = torch.from_numpy(clIm).float().to(cuda)
    else:
        initStep = int(min(numFirstBatch, np.ceil(nExp/expBatchSize)))
        
        # indices = np.random.choice(nExp, size=min(50000, nExp), replace=False)
        indices = np.random.choice(nExp, size=min(initClBatch, nExp), replace=False)
        Im_zero = mmap.data[indices].astype(np.float32)
        Texp_zero = torch.from_numpy(Im_zero).float().to(cuda)
        del(Im_zero)
        
        # PCA vectors 
        pca_zero = bnb.create_batchExp(Texp_zero, freqBn, coef, cvecs)
        
        cl = bnb.kmeans_pytorch_for_averages(Texp_zero, pca_zero, cvecs, num_clusters = final_classes)
        del Texp_zero, pca_zero
        torch.cuda.empty_cache()

        
    
    if refImages:
        num_batches = int(np.ceil(nExp / expBatchSize2))
    else:       
        num_batches = min(int(np.ceil(nExp / expBatchSize)), 
                          int(numFirstBatch + np.ceil( (nExp - (numFirstBatch * expBatchSize))/(expBatchSize2) )))
    
    
    ### Start initial cycles
    num_cycles = 1 
    for cycles in range (num_cycles):
        batch_projExp_cpu = []
        endBatch = 0
        if cycles < num_cycles-1:
            num_batches_in_iter = initStep 
        else:
            num_batches_in_iter = num_batches
            
        ### Start iterations per batches
        for i in range(num_batches_in_iter):
    
            mode = False
            
            if i < initStep:
                initBatch = i * expBatchSize
                endBatch =  min( (i+1) * expBatchSize, nExp)       
            else:
                initBatch = endBatch
                endBatch = min( endBatch + expBatchSize2, nExp)
            
            expImages = mmap.data[initBatch:endBatch].astype(np.float32)
            Texp = torch.from_numpy(expImages).float().to(cuda)
                  
            if i < initStep:          
                batch_projExp_cpu.append( bnb.batchExpToCpu(Texp, freqBn, coef, cvecs) )           
                if i == initStep-1:
                    mode = "create_classes"
            
            else:            
                batch_projExp_cpu = bnb.create_batchExp(Texp, freqBn, coef, cvecs)
                mode = "align_classes"
            del(Texp)
            
            if mode:    
          
                #Initialization Transformation Matrix
                if mode == "create_classes":
                    subset = endBatch
                else:
                    subset = endBatch - initBatch
    
                     
                tMatrix = torch.eye(2, 3, device = cuda).repeat(subset, 1, 1)
                
                if mode == "align_classes":
                    niter = 3
                    
                for iter in range(niter):
                    # print("-----Iteration %s for updating classes-------"%(iter+1))                
                                              
                    matches = torch.full((subset, 5), float("Inf"), device = cuda)
                    
                    vectorRot, vectorshift = bnb.determine_ROTandSHIFT(iter, mode, dim)                
                    nShift = len(vectorshift)  
            
                    for rot in vectorRot:            
            
                        # print("---Precomputing the projections of the reference images---")          
                        batch_projRef = bnb.precalculate_projection(cl, freqBn, grid_flat, 
                                                            coef, cvecs, float(rot), vectorshift)
                
                        count = 0  
                        steps = initStep if mode == "create_classes" else 1 
                                    
                        for i in range(steps):
                
                            if mode == "create_classes":
                                init = i*expBatchSize
                                batch_projExp = batch_projExp_cpu[count].to('cuda', non_blocking=True)
                            else:
                                init = 0
                                batch_projExp = batch_projExp_cpu
                                
                            matches = bnb.match_batch(batch_projExp, batch_projRef, init, matches, rot, nShift)    
                            del(batch_projExp)
                            count+=1    
                    del(batch_projRef)  
                    
                    if mode == "create_classes":
                        res_map = {4: 15, 7: 12, 10: 10, 13: highRes}
                        
                        if iter in res_map:
                            del (freqBn, coef, grid_flat, cvecs)
                            
                            maxRes = max(res_map[iter], highRes)
                            
                            freqBn, cvecs, coef = pca.calculatePCAbasis(
                                mmap, Ntrain, nBand, dim, sampling, maxRes,
                                minRes=530, per_eig=per_eig_value, batchPCA=True
                            )
                            grid_flat = flatGrid(freqBn, coef, nBand)
                            # print(iter , maxRes , coef)    
     
                    #update classes        
                    classes = len(cl)
                    
                    if mode == "create_classes":
                        cl, tMatrix, batch_projExp_cpu = bnb.create_classes(
                            mmap, tMatrix, iter, subset, expBatchSize, matches, vectorshift, 
                            classes, freqBn, coef, cvecs, mask, sigma, sampling, cycles)
                    else:
                        torch.cuda.empty_cache()
                        cl, tMatrix, batch_projExp_cpu = bnb.align_particles_to_classes(expImages, 
                                        cl, tMatrix, iter, subset, matches, vectorshift, classes,
                                         freqBn, coef, cvecs, mask, sigma, sampling)
    
                    # save classes
                    file = output+"_%s_%s_%s.mrcs"%(initBatch,iter+1,cycles)
                    save_images(cl.cpu().detach().numpy(), sampling, file)
    
    
                    if cycles == num_cycles-1 and mode == "create_classes" and iter == niter-1:
                        
                        refClas[:endBatch] = matches[:, 1]
                        dist[:endBatch] = matches[:, 2].cpu()
                                                              
                        #Applying TMT(inv). 
                        #This is done because the rotation is performed from the center of the image.
                        initial_shift = torch.tensor([[1.0, 0.0, -dim/2],
                                                      [0.0, 1.0, -dim/2],
                                                      [0.0, 0.0, 1.0]], device = tMatrix.device)
                        initial_shift = initial_shift.unsqueeze(0).expand(tMatrix.size(0), -1, -1)
    
                        tMatrix = torch.cat((tMatrix, torch.zeros((tMatrix.size(0), 1, 3), device=tMatrix.device)), dim=1)
                        tMatrix[:, 2, 2] = 1.0
                        tMatrix = torch.matmul(initial_shift, tMatrix)
                        tMatrix = torch.matmul(tMatrix, torch.inverse(initial_shift))
                        
                        #extract final angular and shift transformations
                        rotation_matrix = tMatrix[:, :2, :2]
                        translation_vector[:endBatch] = tMatrix[:, :2, 2]
                        angles_rad = torch.atan2(rotation_matrix[:, 1, 0], rotation_matrix[:, 0, 0])
                        angles_deg[:endBatch] = np.degrees(angles_rad.cpu().numpy())
                        
                    elif mode == "align_classes" and iter == 2:
                        
                        refClas[initBatch:endBatch] = matches[:, 1]
                        dist[initBatch:endBatch] = matches[:, 2].cpu()
                        
                        initial_shift = torch.tensor([[1.0, 0.0, -dim/2],
                                                      [0.0, 1.0, -dim/2],
                                                      [0.0, 0.0, 1.0]], device = tMatrix.device)
                        initial_shift = initial_shift.unsqueeze(0).expand(tMatrix.size(0), -1, -1)
    
                        tMatrix = torch.cat((tMatrix, torch.zeros((tMatrix.size(0), 1, 3), device=tMatrix.device)), dim=1)
                        tMatrix[:, 2, 2] = 1.0
                        tMatrix = torch.matmul(initial_shift, tMatrix)
                        tMatrix = torch.matmul(tMatrix, torch.inverse(initial_shift))
                        
                        rotation_matrix = tMatrix[:, :2, :2]
                        translation_vector[initBatch:endBatch] = tMatrix[:, :2, 2]
                        angles_rad = torch.atan2(rotation_matrix[:, 1, 0], rotation_matrix[:, 0, 0])
                        angles_deg[initBatch:endBatch] = np.degrees(angles_rad.cpu().numpy())
            del(expImages)
        
        
    counts = torch.bincount(refClas.to(torch.int64), minlength=classes)
    
        #save classes        
    file_final = output+".mrcs"
    save_images(cl.cpu().detach().numpy(), sampling, file_final)    

    # print(counts.int())
    
    assess = evaluation()
    assess.updateExpStar(expStar, refClas, -angles_deg, translation_vector, output, dist)
    assess.createClassesStar(classes, file_final, counts, output)



