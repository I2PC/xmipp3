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
from xmippPyModules.alignPcaFunctions.pca_gpu import *
from xmippPyModules.alignPcaFunctions.bnb_gpu import *
from xmippPyModules.alignPcaFunctions.reconstruct_gpu import *
from xmippPyModules.alignPcaFunctions.assessment import *
from builtins import iter
from _weakref import ref


def read_images(mrcfilename):

    with mrcfile.open(mrcfilename, permissive=True) as f:
        return f.data.astype(np.float32)
    
def load_vol(file_path, device='cuda'):

    with mrcfile.open(file_path, mode='r') as mrc:
        data = mrc.data.astype(np.float32)
        vol = torch.from_numpy(data).to(device)
        vol = (vol - vol.mean()) / vol.std()
        
    return vol

def save_vol(volume, filename, psize):
    vol_np = volume.detach().cpu().numpy().astype('float32')
    with mrcfile.new(filename, overwrite=True) as mrc:
        mrc.set_data(vol_np)
        mrc.voxel_size = psize
        
def save_proj(projections, filename, psize):
    proj_np = projections.detach().cpu().numpy().astype("float32")
    with mrcfile.new(filename, overwrite=True) as mrc:
        mrc.set_data(proj_np)
        mrc.voxel_size = psize


def flatGrid(freq_band, nBand):
    
    dim, _ = freq_band.shape

    fx = torch.fft.rfftfreq(dim, d=0.5/np.pi, device=cuda)  
    fy = torch.fft.fftfreq(dim, d=0.5/np.pi, device=cuda)   

    grid_x, grid_y = torch.meshgrid(fx, fy, indexing='xy')
    del fx, fy 

    grid_flat = []

    for n in range(nBand):
        mask = (freq_band == n)

        fx_n = grid_x[mask]
        fy_n = grid_y[mask]
        
        grid_flat.append(torch.stack([fx_n, fy_n], dim=0))
        del mask, fx_n, fy_n 

    del grid_x, grid_y
 
    return grid_flat
  
       
if __name__=="__main__":
      
    parser = argparse.ArgumentParser(description="align images")
    parser.add_argument("-i", "--exp", help="input mrc file for experimental images", required=True)
    parser.add_argument("-r", "--refVol", help="input reference volume")
    parser.add_argument("-s", "--sampling", type=float, help="pixel size of the images", required=True)
    parser.add_argument("-a", "--ang", type=float, help="rotation angle (in degree)", required=True)
    parser.add_argument("-amax", "--angmax", type=float, default=180.0, help="maximum rotation angle (in degree, default = 180)")
    parser.add_argument("-sh", "--shift", type=float, help="shift (px)", required=True)
    parser.add_argument("-msh", "--maxshift", type=float,help="maximum shift (px)", required=True)
    parser.add_argument("-o", "--output", help="Root directory for the output files", required=True)
    parser.add_argument("-stExp", "--sartExp", help="star file for experimental images", required=True)
    # parser.add_argument("-stRef", "--starRef", help="star file for reference images", required=True)
    parser.add_argument("-radius", type=int, help="radius for circular mask (in pixels)")  
    parser.add_argument("-vr", "--volRes", type=float, default=8.0, help="Final volume resolution (A)")     
    parser.add_argument("--apply_shifts",  action="store_true", help="Apply starfile shifts to experimental images")
    parser.add_argument("--posit",  action="store_true", help="Apply relu function")
    parser.add_argument("-nCl", "--numCl", type=int, default=1, help="number of classes for initial model")
    parser.add_argument("--save_class",  action="store_true", help="Save the corresponding class in output xmd")
    #For training
    parser.add_argument("-t", "--training", help="number of image for training", required=True)
    parser.add_argument("-hr", "--highres", help="highest resolution to consider", required=True)
    parser.add_argument("-p", "--perc", help="PCA percentage (between 0-1)", required=True)
    
    
    args = parser.parse_args()
    
    expFile = args.exp  
    # prjFile = args.ref
    initVol = args.refVol
    sampling = args.sampling
    ang = args.ang
    amax = args.angmax
    shiftMove = args.shift
    maxshift = args.maxshift
    output = args.output
    expStar = args.sartExp
    # prjStar = args.starRef 
    radius = args.radius
    volRes = args.volRes
    posit = args.posit
    apply_shifts = args.apply_shifts
    numCl = args.numCl
    save_class = args.save_class
    #PCA training
    Ntrain = int(args.training)
    highRes = float(args.highres)
    per_eig_value = float(args.perc)
           
    torch.cuda.is_available()
    torch.cuda.current_device()
    cuda = torch.device('cuda') 
    
    
    if volRes <= 1 / (2*sampling):
        volRes = 1 / (2*sampling) + 0.5
    
    #Read Experimental Images
    mmap = mrcfile.mmap(expFile, permissive=True)
    nExp = mmap.data.shape[0]
    dim = mmap.data.shape[1]
    Ntrain = nExp
    nIter = 20
    
    
    #Create initial references
    R = reconstruct()
    angular_step = 12
    transf, angle_triplet = R.generate_library(angular_step)
    
    all_refs_cpu = [None] * numCl
    
    if initVol:
        for i in range(numCl):
            zeroVol = load_vol(initVol)
            if posit:
                zeroVol = torch.relu(zeroVol)
            ref = R.generate_projections(zeroVol, transf)
            all_refs_cpu[i] = ref.detach().cpu()#.pin_memory()
            # zeroVol = R.reconstruct_volume(ref, "C1", 10, sampling, dim, transf)
            del zeroVol, ref
    else:    
        for i in range(numCl):
            random_angles = R.generate_random_angles(nExp)
            zeroVol = R.reconstruct_volume(mmap, "C1", 20, sampling, dim, random_angles)
            if posit:
                zeroVol = torch.relu(zeroVol)
            ref = R.generate_projections(zeroVol, transf)
            all_refs_cpu[i] = ref.detach().cpu()#.pin_memory()
            del zeroVol, ref
        
    
    # file = output+"_zerovol.mrc"
    # save_vol(zeroVol.cpu(), file, sampling) 
    # file = output+"_zeroref.mrcs" 
    # save_proj(all_refs_cpu[0], file, sampling) 
    # exit()
    
    nBand = 1
    bnb = BnBgpu(nBand)
    assess = evaluation()
    
    
    #Reading experimental images (solo una vez)
    expImages = read_images(expFile)
    # texp = torch.from_numpy(expImages).pin_memory().to(cuda, non_blocking=True)
    texp = torch.from_numpy(expImages).to(cuda)#.pin_memory().to(cuda)
    del expImages
    if radius:
        texp *= bnb.create_mask(texp, radius)
    texp = bnb.zscore_normalization(texp)
    #posit
    if posit:
        texp = torch.relu(texp)
    
    resultado_tensor = torch.cat([texp, all_refs_cpu[0].to(cuda)], dim=0)
    Ntrain = resultado_tensor.shape[0]
    
    #pca
    # nBand = 1
    pca = PCAgpu(nBand)
    maxRes = 20
    freqBn, cvecs, coef = pca.calculatePCAbasis(resultado_tensor, Ntrain, nBand, dim, sampling, maxRes, 
                                                minRes=530, per_eig=per_eig_value, batchPCA=True)

    grid_flat = flatGrid(freqBn, nBand)
    del(resultado_tensor)

    
    
    # #Precomputed rotation and shift   
    angSet = (-amax, amax, ang)
    shiftSet = (-maxshift, maxshift+shiftMove, shiftMove)
    vectorRot, vectorshift = bnb.setRotAndShift(angSet, shiftSet)
    vectorRot.sort()         
    nShift = len(vectorshift)
    
    
    #Precalculate whitening 
    Im_whitening = mmap.data[:nExp].astype(np.float32)
    Texp_whitening = torch.from_numpy(Im_whitening).float().to(cuda)
    whitening = bnb.compute_radial_whitening_filter(Texp_whitening)
    del Im_whitening, Texp_whitening
    whitening = 1
            
    next_angle_triplet = None
        
    for current_iter in range(nIter):
        print("----------Iter %s------------" %current_iter, flush=True)
        
        #Precomputed rotation and shift  
        if current_iter in (0, 8, 13, 16): 
            pcaRes, filtRes, angular_step = bnb.reconstruct_parameters(current_iter, highRes, volRes)
            ang, shiftMove, maxshift = bnb.search_space(current_iter, ang, shiftMove, maxshift) 
            angSet = (-amax, amax, ang)
            shiftSet = (-maxshift, maxshift+shiftMove, shiftMove)
            vectorRot, vectorshift = bnb.setRotAndShift(angSet, shiftSet)
            vectorRot.sort()         
            nShift = len(vectorshift)
            
    
        matches = [None] * numCl
        
        for i in range(numCl):    
            #Reading references particles 
                   
            tref = all_refs_cpu[i].to(cuda)#, non_blocking=True)
            if radius:
                tref = tref * bnb.create_mask(tref, radius)
            tref = bnb.zscore_normalization(tref) 
            # if posit:
            #     tref = torch.relu(tref)
       
            batch_projRef = bnb.create_batchExp(tref, whitening, freqBn, coef, cvecs) 
            
            matches[i] = torch.full((nExp, 5), float("Inf"), device = cuda)
            
            for rot in vectorRot:
        
                # print("---Computing the projections of the experimental images---")      
                batch_projExp = bnb.precalculate_projection(texp, whitening, freqBn, grid_flat, coef, cvecs, -rot, vectorshift)
                # print("matches")
    
                matches[i] = bnb.match_batch_initVol(batch_projExp, batch_projRef, 0, matches[i], -rot, nShift)
                del(batch_projExp) 
                # del(batch_projRef)
            
            matches[i] = bnb.match_batch_label_minScore(matches[i])
            # print(matches[i])
            # exit()
            
            score = matches[i][:, 2].mean()
            print("mean score = %s" %score.item())
            
            if not save_class: 
                valid_indices, rotM, shiftM = assess.estimatePose(
                    angle_triplet, expStar, matches[i], vectorshift, 
                    nExp, apply_shifts, filter_matches=(current_iter > 4)
                )

            mmap_filtrado = mmap.data[valid_indices.cpu().numpy()].astype('float32')

            vol = R.reconstruct_volume(mmap_filtrado, "C1", filtRes, sampling, dim, rotM, shifts=shiftM)
            # vol = R.reconstruct_volume(mmap_filtrado, "C1", filtRes, sampling, dim, rotM)
            if current_iter < 7:
                vol = R.mask_otsu(vol)
                
            vol = R.apply_spherical_mask(vol, radius)
            #posit
            if posit:
                vol = torch.relu(vol)
            
            if i == 0 and current_iter in (8, 13, 16):
                transf, next_angle_triplet = R.generate_library(angular_step)
                
            ref = R.generate_projections(vol, transf)
            all_refs_cpu[i] = ref.detach().cpu()#.pin_memory()
            file = output+"_iter%s_class%s.mrc"%(current_iter+1,i)
            save_vol(vol.cpu(), file, sampling)
            del tref, vol, ref
            # fileProj = output+"ref_%s_%s.mrcs"%(iter+1,i)
            # save_vol(all_refs_cpu[0], fileProj, sampling)
        
        if next_angle_triplet is not None:
            angle_triplet = next_angle_triplet 
            next_angle_triplet = None 
            #Actualizo PCA
            del freqBn, coef, grid_flat, cvecs 
            resultado_tensor = torch.cat([texp, all_refs_cpu[0].to(cuda)], dim=0)
            Ntrain = resultado_tensor.shape[0]  
            freqBn, cvecs, coef = pca.calculatePCAbasis(
                resultado_tensor, Ntrain, nBand, dim, sampling, pcaRes,
                minRes=530, per_eig=per_eig_value, batchPCA=True
            )
            grid_flat = flatGrid(freqBn, nBand)
            del resultado_tensor
    exit()
    
    
    
    
        # file = output+"_zeroref.mrcs" 
        # save_proj(all_refs_cpu[0], file, sampling) 
    
            #Write new starfile
            # if not save_class:   
            #     # assess.writeExpStar(prjStar, expStar, matches[i], vectorshift, nExp, apply_shifts, output)
            #     assess.writeExpStar_minScore(prjStar, expStar, matches[i], vectorshift, nExp, apply_shifts, output)
            #     exit()
        #
        # if save_class:
        #     matches_min = bnb.match_batch_with_class(matches)
        #     assess.writeExpStarClass(prjStar, expStar, matches_min, vectorshift, nExp, apply_shifts, output)




        








