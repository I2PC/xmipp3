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
    parser.add_argument("--apply_shifts",  action="store_true", help="Apply starfile shifts to experimental images")
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

    
    #Basename for multiple references
    # if numCl > 1:
    #     prjFile_base = os.path.join(os.path.dirname(prjFile), os.path.splitext(os.path.basename(prjFile))[0].split('_class')[0])
    #     output_base = os.path.join(os.path.dirname(output), os.path.splitext(os.path.basename(output))[0].split("_class")[0])

    
    #Read Experimental Images
    mmap = mrcfile.mmap(expFile, permissive=True)
    nExp = mmap.data.shape[0]
    dim = mmap.data.shape[1]
    Ntrain = nExp
    nIter = 10
    
    
    #Create initial references
    # assess = evaluation()
    R = reconstruct()
    angular_step = 12
    transf, angle_triplet = R.generate_library(angular_step)
    # print(angle_triplet)
    # transf2 = assess.determineR(angle_triplet)
    # err = torch.max(torch.abs(transf - transf2))
    # print(err)
    # exit()
    # print("------------------")
    
    # n_proj = transf.shape[0] 
    # all_refs_cpu = torch.zeros((numCl, n_proj, dim, dim), device='cpu', dtype=torch.float32, pin_memory=True)
    all_refs_cpu = [None] * numCl
    
    if initVol:
        for i in range(numCl):
            zeroVol = load_vol(initVol)
            ref = R.generate_projections(zeroVol, transf)
            all_refs_cpu[i] = ref.detach().cpu().pin_memory()
            # zeroVol = R.reconstruct_volume(ref, "C1", 10, sampling, dim, transf)
            del zeroVol, ref
    else:    
        for i in range(numCl):
            random_angles = R.generate_random_angles(nExp)
            zeroVol = R.reconstruct_volume(mmap, "C1", 20, sampling, dim, random_angles)
            ref = R.generate_projections(zeroVol, transf)
            all_refs_cpu[i] = ref.detach().cpu().pin_memory()
            del zeroVol, ref
        
    
    # file = output+"_zerovol.mrc"
    # save_vol(zeroVol.cpu(), file, sampling) 
    file = output+"_zeroref.mrcs" 
    save_proj(all_refs_cpu[0], file, sampling) 
    # exit()
    
    #pca
    nBand = 1
    pca = PCAgpu(nBand)
    maxRes = 12
    freqBn, cvecs, coef = pca.calculatePCAbasis(mmap, Ntrain, nBand, dim, sampling, maxRes, 
                                                minRes=530, per_eig=per_eig_value, batchPCA=True)

    grid_flat = flatGrid(freqBn, nBand)

        
    bnb = BnBgpu(nBand)
    assess = evaluation()
    
    
    #Precomputed rotation and shift   
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
        


    #Reading experimental images (solo una vez)
    expImages = read_images(expFile)
    texp = torch.from_numpy(expImages).pin_memory().to(cuda, non_blocking=True)
    texp = bnb.zscore_normalization(texp)
    del expImages
    if radius:
        texp *= bnb.create_mask(texp, radius)
    
    
    #Precalculate projection angles
    # angular_step = 5
    # transf, angle_triplet = R.generate_library(angular_step)
    # n_proj = transf.shape[0] 
    # all_refs_cpu = torch.zeros((numCl, n_proj, dim, dim), device='cpu', dtype=torch.float32, pin_memory=True)
    
        
    for iter in range(nIter):
        print("----------Iter %s------------" %iter, flush=True)
    
        matches = [None] * numCl
        for i in range(numCl):    
            #Reading references particles 
                   
            tref = all_refs_cpu[i].to(cuda, non_blocking=True)
            if radius:
                tref = tref * bnb.create_mask(tref, radius)
            # del(prjImages)
        
            batch_projRef = bnb.create_batchExp(tref, whitening, freqBn, coef, cvecs) 
            
            
            matches[i] = torch.full((nExp, 5), float("Inf"), device = cuda)
            
            for rot in vectorRot:
        
                # print("---Computing the projections of the experimental images---")      
                batch_projExp = bnb.precalculate_projection(texp, whitening, freqBn, grid_flat, coef, cvecs, -rot, vectorshift)
                # print("matches")
    
                matches[i] = bnb.match_batch_initVol(batch_projExp, batch_projRef, 0, matches[i], -rot, nShift)
                del(batch_projExp) 
                # del(batch_projRef)
            
            # print(matches[i])
            matches[i] = bnb.match_batch_label_minScore(matches[i])
            print(matches[i])
            # exit()
            
            score = matches[i][:, 2].mean()
            print("mean score = %s" %score.item())
            
            if not save_class: 
                valid_indices, rotM, shiftM = assess.estimatePose(angle_triplet, expStar, matches[i], vectorshift, nExp, apply_shifts)
            print(shiftM)
            mmap_filtrado = mmap.data[valid_indices.cpu().numpy()].astype('float32')
            vol = R.reconstruct_volume(mmap_filtrado, "C1", 10, sampling, dim, rotM, shifts=shiftM)
            # vol = R.reconstruct_volume(mmap_filtrado, "C1", 10, sampling, dim, rotM)
            ref = R.generate_projections(vol, transf)
            all_refs_cpu[i] = ref.detach().cpu().pin_memory()
            file = output+"_%s_%s.mrc"%(iter+1,i)
            save_vol(vol.cpu(), file, sampling)
            # fileProj = output+"ref_%s_%s.mrcs"%(iter+1,i)
            # save_vol(all_refs_cpu[0], fileProj, sampling)
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














