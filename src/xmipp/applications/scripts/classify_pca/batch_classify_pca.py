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
from xmippPyModules.classifyPcaFuntion.assessment import evaluation


def read_images(mrcfilename):

    with mrcfile.open(mrcfilename, permissive=True) as f:
         emImages = f.data.astype(np.float32).copy()
    return emImages


def save_images(data, outfilename):
    data = data.astype('float32')
    with mrcfile.new(outfilename, overwrite=True) as mrc:
        mrc.set_data(data)


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
    parser.add_argument("-c", "--classes", help="number of 2D classes", required=True)
    parser.add_argument("-r", "--ref", help="2D classes of external method")   
    parser.add_argument("-b", "--bands", help="file with frequency bands", required=True)
    parser.add_argument("-v", "--vecs", help="file with pretrain eigenvectors", required=True)
    parser.add_argument("--mask",  action="store_true", help="A Gaussian mask is used.")
    parser.add_argument("--sigma", type=float, help="value of sigma for the Gaussian mask. "
                                                    "It is only used if the --mask option is applied.")
    parser.add_argument("-o", "--output", help="Root directory for the output files", required=True)
    parser.add_argument("-stExp", "--sartExp", help="star file for images")
    parser.add_argument("-g", "--gpu",  help="GPU ID set. Just one GPU is allowed", required=False)

    
    args = parser.parse_args()
    
    expFile = args.exp
    classes = int(args.classes)
    final_classes = classes  
    refImages = args.ref
    niter = 14
    bands = args.bands
    vecs = args.vecs
    mask = args.mask
    sigma = args.sigma
    output = args.output
    expStar = args.sartExp
    gpuId = str(args.gpu)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpuId

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
    
    if mask and sigma is None:
        sigma = dim/3
    
    initSubset = min(100000, nExp)
    refClas = torch.zeros(nExp)
    translation_vector = torch.zeros(nExp, 2)
    angles_deg = np.zeros(nExp)
    
    freqBn = torch.load(bands) 
    cvecs = torch.load(vecs)
    nBand = freqBn.unique().size(dim=0) - 1
    
    coef = torch.zeros(nBand, dtype=int)
    for n in range(nBand):
        coef[n] = 2*torch.sum(freqBn==n)

    grid_flat = flatGrid(freqBn, coef, nBand)

    bnb = BnBgpu(nBand)
       
    expBatchSize, expBatchSize2, numFirstBatch = bnb.determine_batches(free_memory, dim) 
    print("batches: %s, %s, %s" %(expBatchSize, expBatchSize2, numFirstBatch))   


    #Initial classes
    if refImages: 
        initStep = -1
        clIm = read_images(refImages)
        cl = torch.from_numpy(clIm).float().to(cuda)
    else:
        initStep = int(min(numFirstBatch, np.ceil(nExp/expBatchSize)))
        cl = bnb.init_ramdon_classes(final_classes, mmap, initSubset) 
        
    
    if refImages:
        num_batches = int(np.ceil(nExp / expBatchSize2))
    else:       
        num_batches = min(int(np.ceil(nExp / expBatchSize)), 
                          int(numFirstBatch + np.ceil( (nExp - (numFirstBatch * expBatchSize))/(expBatchSize2) )))
    
    batch_projExp_cpu = []
    endBatch = 0
    for i in range(num_batches):
        mode = False
        
        if i < initStep:
            initBatch = i * expBatchSize
            endBatch =  min( (i+1) * expBatchSize, nExp)       
        else:
            initBatch = endBatch
            endBatch = min( endBatch + expBatchSize2, nExp)
        
        expImages = mmap.data[initBatch:endBatch].astype(np.float32)
        Texp = torch.from_numpy(expImages).float().to(cuda)
        if mask:
            Texp = Texp * bnb.create_circular_mask(Texp)
              
        if i < initStep:          
            batch_projExp_cpu.append( bnb.batchExpToCpu(Texp, freqBn, coef, cvecs) )           
            if i == initStep-1:
                mode = "create_classes"
        
        else:            
            batch_projExp_cpu = bnb.create_batchExp(Texp, freqBn, coef, cvecs)
            mode = "align_classes"
        
        if mode:
            # print(mode)

      
            #Initialization Transformation Matrix
            if mode == "create_classes":
                subset = endBatch
            else:
                subset = endBatch - initBatch

                 
            tMatrix = torch.eye(2, 3, device = cuda).repeat(subset, 1, 1)
            
            if mode == "align_classes":
                niter = 4
                
            for iter in range(niter):
                # print("-----Iteration %s for updating classes-------"%(iter+1))
                           
                matches = torch.full((subset, 5), float("Inf"), device = cuda)
                
                vectorRot, vectorshift = bnb.determine_ROTandSHIFT(iter, mode, dim)                
                nShift = len(vectorshift)  
        
                for rot in vectorRot:            
        
                    # print("---Precomputing the projections of the reference images---")          
                    batch_projRef = bnb.precalculate_projection(cl, freqBn, grid_flat, coef, cvecs, float(rot), vectorshift)
            
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
                
                #update classes        
                classes = len(cl)
        
                if mode == "create_classes":
                    cl, tMatrix, batch_projExp_cpu = bnb.create_classes_version0(mmap, tMatrix, iter, subset, expBatchSize, matches, vectorshift, classes, freqBn, coef, cvecs, mask, sigma)
                else:
                    cl, tMatrix, batch_projExp_cpu = bnb.align_particles_to_classes(expImages, cl, tMatrix, iter, subset, matches, vectorshift, classes, freqBn, coef, cvecs, mask, sigma)

                if mode == "create_classes" and iter == 13:
                    
                    refClas[:endBatch] = matches[:, 1]
                                                          
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
                    
                elif mode == "align_classes" and iter == 3:
                    
                    refClas[initBatch:endBatch] = matches[:, 1]
                    
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
    counts = torch.bincount(refClas.int(), minlength=classes) 
    
        #save classes
        
    file = output+".mrcs"
    save_images(cl.cpu().detach().numpy(), file)
    
    print("Adjust contrast")
    cl = bnb.increase_contrast_sigmoid(cl, 8, 0.6)
    file_contrast = output+"_contrast.mrcs"
    save_images(cl.cpu().detach().numpy(), file_contrast)           

    # print(counts.int())
    
    assess = evaluation()
    assess.updateExpStar(expStar, refClas, -angles_deg, translation_vector, output)
    assess.createClassesStar(classes, file, counts, output)
    assess.createClassesStar(classes, file_contrast, counts, output+"_contrast")




