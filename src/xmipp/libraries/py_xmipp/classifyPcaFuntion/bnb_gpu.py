#!/usr/bin/env python3
"""/***************************************************************************
 *
 * Authors:    Erney Ramirez-Aportela
 *
  ***************************************************************************/
"""
import numpy as np
import torch
import torch.optim as optim
import time
import torchvision.transforms.functional as T
import torch.nn.functional as F
import kornia
import mrcfile


class BnBgpu:
    
    def __init__(self, nBand):

        self.nBand = nBand 
        
        torch.cuda.is_available()
        torch.cuda.current_device()
        self.cuda = torch.device('cuda:0')    
    
    def setRotAndShift2(self, angle, shift, shiftTotal):

        self.vectorRot = []
        self.vectorShift = []
        
        for rot in range (0, 360, angle):
            self.vectorRot.append(rot)
         
        self.vectorShift = [[0,0]]                   
        for tx in range (-shiftTotal, shiftTotal+1, shift):
            for ty in range (-shiftTotal, shiftTotal+1, shift):
                if (tx | ty != 0):
                    self.vectorShift.append( [tx,ty] )                  

        return self.vectorRot, self.vectorShift
    
    
    #the angle is a triplet
    def setRotAndShift(self, angle, shift):
        
        self.vectorRot = [0]
        for rot in np.arange(*angle):
            if rot < 0:
                rot = 360 + rot
            if rot != 0:
                self.vectorRot.append(rot)
         
        self.vectorShift = [[0,0]]                   
        for tx in np.arange(*shift):
            for ty in np.arange(*shift):
                if (tx or ty != 0):
                    self.vectorShift.append( [float(tx),float(ty)] )                  

        return self.vectorRot, self.vectorShift
    
       
    
    def precShiftBand(self, ft, freq_band, grid_flat, coef, shift):
        
        fourier_band = self.selectFourierBands(ft, freq_band, coef)
        nRef = fourier_band[0].size(dim=0)
        nShift = shift.size(dim=0)
        
        band_shifted = [torch.zeros((nRef*nShift, coef[n]), device = self.cuda) for n in range(self.nBand)]
                     
        ONE = torch.tensor(1, dtype=torch.float32, device=self.cuda)
        
        for n in range (self.nBand):           
            angles = torch.mm(shift, grid_flat[n])     
            filter = torch.polar(ONE, angles)
            
            for i in range(nRef):
                temp = fourier_band[n][i].repeat(nShift,1)
                               
                band_shifted_complex = torch.mul(temp, filter)
                band_shifted_complex[:, int(coef[n] / 2):] = 0.0                
                band_shifted[n][i*nShift : (i*nShift)+nShift] = torch.cat((band_shifted_complex.real, band_shifted_complex.imag), dim=1)
                     
        return(band_shifted)

    
  
    def selectFourierBands(self, ft, freq_band, coef):

        dimFreq = freq_band.shape[1]

        fourier_band = [torch.zeros(int(coef[n]/2), dtype = ft.dtype, device = self.cuda) for n in range(self.nBand)]
        
        freq_band = freq_band.expand(ft.size(dim=0) ,freq_band.size(dim=0), freq_band.size(dim=1))
           
        for n in range(self.nBand):
            fourier_band[n] = ft[:,:,:dimFreq][freq_band == n]
            fourier_band[n] = fourier_band[n].reshape(ft.size(dim=0),int(coef[n]/2)) 
                      
        return fourier_band        
            

    def selectBandsRefs(self, ft, freq_band, coef): 
    
        dimfreq = freq_band.size(dim=1)
        batch_size = ft.size(dim=0)
   
        freq_band = freq_band.to(self.cuda)
        band = [torch.zeros(batch_size, coef[n], device = self.cuda) for n in range(self.nBand)]    
        
        freq_band = freq_band.expand(ft.size(dim=0) ,freq_band.size(dim=0), freq_band.size(dim=1))
        for n in range(self.nBand): 
        
            band_real = ft[:,:,:dimfreq][freq_band == n].real
            band_imag = ft[:,:,:dimfreq][freq_band == n].imag
            band_real = band_real.reshape(batch_size,int(coef[n]/2))
            band_imag = band_imag.reshape(batch_size,int(coef[n]/2))
            band[n] =  torch.cat((band_real, band_imag), dim=1)
    
        return band
    
    
    def phiProjRefs(self, band, vecs):
      
        proj = [torch.matmul(band[n], vecs[n]) for n in range(self.nBand)]
        return proj
        
       
    #Applying rotation and shift
    def precalculate_projection(self, prjTensorCpu, freqBn, grid_flat, coef, cvecs, rot, shift):
                    
        shift_tensor = torch.Tensor(shift).to(self.cuda)       
        prjTensor = prjTensorCpu.to(self.cuda)
   
        rotFFT = torch.fft.rfft2(T.rotate(prjTensor, rot), norm="forward")
        del prjTensor
        band_shifted = self.precShiftBand(rotFFT, freqBn, grid_flat, coef, shift_tensor) 
        del(rotFFT)  
        projBatch = self.phiProjRefs(band_shifted, cvecs)
        del(band_shifted)

        return(projBatch)
    
    
    def create_batchExp(self, Texp, freqBn, coef, vecs):
             
        self.batch_projExp = [torch.zeros((Texp.size(dim=0), vecs[n].size(dim=1)), device = self.cuda) for n in range(self.nBand)]
        expFFT = torch.fft.rfft2(Texp, norm="forward")
        del(Texp)
        bandExp = self.selectBandsRefs(expFFT, freqBn, coef)
        self.batch_projExp = self.phiProjRefs(bandExp, vecs)
        
        torch.cuda.empty_cache()
        return(self.batch_projExp)
    
    
    def match_batch(self, batchExp, batchRef, initBatch, matches, rot, nShift):
        
        nExp = batchExp[0].size(dim=0) 
        nShift = torch.tensor(nShift, device=self.cuda)
                                  
        for n in range(self.nBand):
            score = torch.cdist(batchRef[n], batchExp[n])
            
        min_score, ref = torch.min(score,0)
        del(score)
            
        sel = (torch.floor(ref/nShift)).type(torch.int64)
        shift_location = (ref - (sel*nShift)).type(torch.int64)
        rotation = torch.full((nExp,1), rot, device = self.cuda)
        exp = torch.arange(initBatch, initBatch+nExp, 1, device = self.cuda).view(nExp,1)
        
        iter_matches = torch.cat((exp, sel.reshape(nExp,1), min_score.reshape(nExp,1), 
                                  rotation, shift_location.reshape(nExp,1)), dim=1)  

        cond = iter_matches[:, 2] < matches[initBatch:initBatch + nExp, 2]
        matches[initBatch:initBatch + nExp] = torch.where(cond.view(nExp, 1), iter_matches, matches[initBatch:initBatch + nExp])      
                
        return(matches)
    
    
    def match_batch_correlation(self, batchExp, batchRef, initBatch, matches, rot, nShift):
        
        nExp = batchExp[0].size(dim=0) 
        nShift = torch.tensor(nShift, device=self.cuda)
                                  
        for n in range(self.nBand):
                      
            Ref_bar = batchRef[n] - batchRef[n].mean(axis=1).view(batchRef[n].shape[0],1)
            Exp_bar = batchExp[n] - batchExp[n].mean(axis=1).view(batchExp[n].shape[0],1)
            N = Ref_bar.shape[1]
            cov = (Ref_bar @ Exp_bar.t()) / (N - 1)
            
            normRef = torch.std(batchRef[n], dim=1).view(batchRef[n].shape[0],1)
            normExp = torch.std(batchExp[n], dim=1).view(batchExp[n].shape[0],1)
            den = torch.matmul(normRef,normExp.T)
        
            score = cov/den
           
        min_score, ref = torch.min(-score, 0)
        del(score)
        
        sel = (torch.floor(ref/nShift)).type(torch.int64)
        shift_location = (ref - (sel*nShift)).type(torch.int64)
        rotation = torch.full((nExp,1), rot, device = self.cuda)
        exp = torch.arange(initBatch, initBatch+nExp, 1, device = self.cuda).view(nExp,1)
        
        iter_matches = torch.cat((exp, sel.reshape(nExp,1), min_score.reshape(nExp,1), 
                                  rotation, shift_location.reshape(nExp,1)), dim=1)  

        cond = iter_matches[:, 2] < matches[initBatch:initBatch + nExp, 2]
        matches[initBatch:initBatch + nExp] = torch.where(cond.view(nExp, 1), iter_matches, matches[initBatch:initBatch + nExp])
        
        # torch.cuda.empty_cache()        
        return(matches)
    
    
    def batchExpToCpu(self, Timage, freqBn, coef, cvecs):        

        self.create_batchExp(Timage, freqBn, coef, cvecs)        
        self.batch_projExp = torch.stack(self.batch_projExp)
        batch_projExp_cpu = self.batch_projExp.to("cpu")
        
        return(batch_projExp_cpu)
    
    
    def init_ramdon_classes(self, classes, mmap, initSubset):
        
        cl = torch.zeros((classes, mmap.data.shape[1], mmap.data.shape[2]), device = self.cuda) 
 
        #create initial classes 
        div = int(initSubset/classes)
        resto = int(initSubset%classes)
    
        expBatchSizeClas = div+resto 
        
        count = 0
        for initBatch in range(0, initSubset, expBatchSizeClas):
            expImages = mmap.data[initBatch:initBatch+expBatchSizeClas].astype(np.float32)
            Texp = torch.from_numpy(expImages).float().to(self.cuda)
            Texp = Texp * self.create_circular_mask(Texp)
    
            #Averages classes
            cl[count] = torch.mean(Texp, 0)
            count+=1
            del(Texp) 
        return(cl)
    
    
    def get_robust_zscore_thresholds(self, classes, matches, threshold=2.0):

        thr_low = torch.full((classes,), float('-inf'))
        thr_high = torch.full((classes,), float('inf'))
    
        for n in range(classes):
            class_scores = matches[matches[:, 1] == n, 2]
            if len(class_scores) > 2:
                median = class_scores.median()
                mad = torch.median(torch.abs(class_scores - median)) + 1e-8  # evitar división por cero
                thr_low[n] = median - threshold * mad
                thr_high[n] = median + threshold * mad
                
                vmin = torch.min(matches[matches[:, 1] == n, 2])
                vmax = torch.max(matches[matches[:, 1] == n, 2])
                # print("dist", vmin, vmax)
                # print("thr",   thr_low[n], thr_high[n])
    
        return thr_low, thr_high
    
    
    def split_classes_for_range(self, classes, matches, percent=0.3):
        thr = torch.zeros(classes)
        for n in range(classes):
            if len(matches[matches[:, 1] == n, 2]) > 2: 
                vmin = torch.min(matches[matches[:, 1] == n, 2])
                vmax = torch.max(matches[matches[:, 1] == n, 2])
                
                percentile = (vmax - vmin) * percent
                thr[n] = vmax - percentile
        
            else:
               thr[n] = 0 
            
        return(thr)        
    
    
    def create_classes(self, mmap, tMatrix, iter, nExp, expBatchSize, matches, vectorshift, classes, final_classes, freqBn, coef, cvecs, mask, sigma):
        
        # print("----------create-classes-------------")      
        
        class_split = 0
        # if iter >= 1 and iter < 5:
        if iter >= 5 and iter < 7:

            thr = self.split_classes_for_range(classes, matches)
            # class_split = int(final_classes/((iter-4)*4))
            class_split = int(final_classes/4)

            # if iter == 4:
            # if iter == 7:
            #     class_split = final_classes - classes
            
        newCL = [[] for i in range(classes + class_split)]


        step = int(np.ceil(nExp/expBatchSize))
        batch_projExp_cpu = [0 for i in range(step)]
        
        #rotate and translations
        rotBatch = -matches[:,3].view(nExp,1)
        translations = list(map(lambda i: vectorshift[i], matches[:, 4].int()))
        translations = torch.tensor(translations, device = self.cuda).view(nExp,2)
        
        centerIm = mmap.data.shape[1]/2 
        centerxy = torch.tensor([centerIm,centerIm], device = self.cuda)
        
        count = 0
        for initBatch in range(0, nExp, expBatchSize):
            
            endBatch = min(initBatch+expBatchSize, nExp)
                        
            transforIm, matrixIm = self.center_particles_inverse_save_matrix(mmap.data[initBatch:endBatch], tMatrix[initBatch:endBatch], 
                                                                             rotBatch[initBatch:endBatch], translations[initBatch:endBatch], centerxy)
                                    
            if mask: 
                transforIm = transforIm * self.create_gaussian_mask(transforIm, sigma)
            else:   
                transforIm = transforIm * self.create_circular_mask(transforIm)
            # if mask: 
            #     if iter < 13:
            #         transforIm = transforIm * self.create_gaussian_mask(transforIm, sigma)
            #     else:
            #         transforIm = transforIm * self.create_circular_mask(transforIm)
                                
            tMatrix[initBatch:endBatch] = matrixIm
            
            batch_projExp_cpu[count] = self.batchExpToCpu(transforIm, freqBn, coef, cvecs)
            count+=1
                          
             
            # if iter >= 1 and iter < 5:
            if iter >= 5 and iter < 9: 
                for n in range(classes):
                    
                    if n < class_split:
                        class_images = transforIm[(matches[initBatch:endBatch, 1] == n) & (matches[initBatch:endBatch, 2] < thr[n])]
                        newCL[n].append(class_images)
                        
                        non_class_images = transforIm[(matches[initBatch:endBatch, 1] == n) & (matches[initBatch:endBatch, 2] >= thr[n])] 
                        newCL[n + classes].append(non_class_images)
                        
                    else:
                        class_images = transforIm[matches[initBatch:endBatch, 1] == n]
                        newCL[n].append(class_images)
            
            else:  
      
                for n in range(classes):
                    class_images = transforIm[matches[initBatch:endBatch, 1] == n]
                    newCL[n].append(class_images)
                    # maskSel = matches[initBatch:endBatch, 1] == n  
                    # sorted_indices = torch.argsort(matches[initBatch:endBatch, 2][maskSel])  
                    # class_images = transforIm[maskSel][sorted_indices[:max(1, len(sorted_indices) // 2)]]  
                    # newCL[n].append(class_images)
                         
                    
            del(transforIm)    
                    
   
        newCL = [torch.cat(class_images_list, dim=0) for class_images_list in newCL]    
                     
        clk = self.averages_createClasses(mmap, iter, newCL)
        
        
        if iter < 5:
            clk = clk * self.approximate_otsu_threshold(clk, percentile=10)
        elif iter < 10:
            clk = clk * self.approximate_otsu_threshold(clk, percentile=20)

        clk = clk * self.create_circular_mask(clk)    
        
        # if iter in [2, 3]:
        if iter > 2 and iter < 10:
            clk = self.center_by_com(clk) 
         
        # if mask:
        #     if iter < 13:
        #         clk = clk * self.create_gaussian_mask(clk, sigma)
        #     else:
        #         clk = clk * self.create_circular_mask(clk)
                
        
        return(clk, tMatrix, batch_projExp_cpu)
    
    
    
    def create_classes_version00(self, mmap, tMatrix, iter, nExp, expBatchSize, matches, vectorshift, classes, freqBn, coef, cvecs, mask, sigma, sampling, cycles):
        
        # print("----------create-classes-------------")      
            
        
        # if iter > 3 and iter < 10: # and cycles == 0:
        if iter > 3 and iter < 7 and cycles == 0:
            print("--------", iter, "-----------")
            thr_low, thr_high = self.get_robust_zscore_thresholds(classes, matches, threshold=2.0)
        # elif iter >= 10:
        #     print("--------", iter, "-----------")
        #     thr_low, thr_high = self.get_robust_zscore_thresholds(classes, matches, threshold=2.0)
            

        # if iter > 3 and iter < 10: # and cycles == 0:
        if iter > 3 and iter < 7 and cycles == 0:
            num = int(classes/2)
            newCL = [[] for i in range(classes)]
        else:
            num = classes
            newCL = [[] for i in range(classes)]


        step = int(np.ceil(nExp/expBatchSize))
        batch_projExp_cpu = [0 for i in range(step)]
        
        #rotate and translations
        rotBatch = -matches[:,3].view(nExp,1)
        translations = list(map(lambda i: vectorshift[i], matches[:, 4].int()))
        translations = torch.tensor(translations, device = self.cuda).view(nExp,2)
        
        centerIm = mmap.data.shape[1]/2 
        centerxy = torch.tensor([centerIm,centerIm], device = self.cuda)
        
        count = 0
        for initBatch in range(0, nExp, expBatchSize):
            
            endBatch = min(initBatch+expBatchSize, nExp)
                        
            transforIm, matrixIm = self.center_particles_inverse_save_matrix(mmap.data[initBatch:endBatch], tMatrix[initBatch:endBatch], 
                                                                             rotBatch[initBatch:endBatch], translations[initBatch:endBatch], centerxy)
            
            
            if mask:
                # if iter < 15:
                transforIm = transforIm * self.create_gaussian_mask(transforIm, sigma)
            else:
                transforIm = transforIm * self.create_circular_mask(transforIm)
                
                    
            
            tMatrix[initBatch:endBatch] = matrixIm
            
            batch_projExp_cpu[count] = self.batchExpToCpu(transforIm, freqBn, coef, cvecs)
            count+=1

            
            # if iter > 3 and iter < 10:# and cycles == 0:
            if iter > 3 and iter < 7 and cycles == 0:
                
                for n in range(num):
                    
                    class_images = transforIm[
                                            (matches[initBatch:endBatch, 1] == n) &
                                            (matches[initBatch:endBatch, 2] > thr_low[n]) &
                                            (matches[initBatch:endBatch, 2] < thr_high[n])
                                        ]
                    newCL[n].append(class_images)
                    
                    non_class_images = transforIm[
                                            (matches[initBatch:endBatch, 1] == n) &
                                            (
                                                (matches[initBatch:endBatch, 2] <= thr_low[n]) |
                                                (matches[initBatch:endBatch, 2] >= thr_high[n])
                                            )
                                        ]
                    newCL[n + num].append(non_class_images)


            else:  
      
                for n in range(num):
                    class_images = transforIm[matches[initBatch:endBatch, 1] == n]
                    newCL[n].append(class_images)
                    
            del(transforIm)
                    
        
        newCL = [torch.cat(class_images_list, dim=0) for class_images_list in newCL] 
        
        clk = self.averages_createClasses(mmap, iter, newCL)
        
        # clk = self.filter_classes_relion_style(newCL, clk, sampling, 6.0)
        

        # if iter > 10: 
        if iter > 7:
            res_classes = self.frc_resolution_tensor(newCL, sampling)
            print(res_classes)
            bfactor = self.estimate_bfactor_batch(clk, sampling, res_classes)
            # print(bfactor)
            # clk = self.enhance_averages_butterworth_adaptive(clk, res_classes, sampling)
            # clk = self.highpass_butterworth_soft_batch(clk, res_classes, sampling)
            clk = self.gaussian_lowpass_filter_2D_adaptive(clk, res_classes, sampling)
            clk = self.sharpen_averages_batch(clk, sampling, bfactor, res_classes)
            # clk = self.sharpen_averages_batch_nq(clk, sampling, bfactor)
            # clk = self.enhance_averages_butterworth(clk, sampling)
            # clk = self.enhance_averages_butterworth_combined_cos(clk, res_classes, sampling)
            # clk = self.enhance_averages_attenuate_lowfrequencies(clk, res_classes, sampling)
            # clk = self.unsharp_mask_norm(clk)
    

            # clk = self.unsharp_mask_adaptive_gaussian(clk)
            # mask_C = self.compute_class_consistency_masks(newCL) #Apply consistency mask           
            # clk = self.apply_consistency_masks_vector(clk, mask_C) 
        
        # clk = self.gaussian_lowpass_filter_2D(clk, 6.0, sampling)
        

        # if iter in [13, 16]:
        if iter in [10, 13]:
            # clk = clk * self.approximate_otsu_threshold(clk, percentile=10)
            clk = clk * self.contrast_dominant_mask(clk, window=3, contrast_percentile=80,
                                intensity_percentile=50, contrast_weight=1.5, intensity_weight=1.0)
        # if 3 < iter < 10 and iter % 2 == 0:
        if 3 < iter < 7 and iter % 2 == 0:
            # clk = clk * self.approximate_otsu_threshold(clk, percentile=10)
            clk = clk * self.contrast_dominant_mask(clk, window=3, contrast_percentile=80,
                                intensity_percentile=50, contrast_weight=1.5, intensity_weight=1.0)

        
        # if iter < 17:
        #     clk = self.auto_generate_masks(clk)    
        clk = clk * self.create_circular_mask(clk)
        
        # if iter > 2 and iter < 15:
        if iter > 2 and iter < 12:
            for _ in range(2):
                clk = self.center_by_com(clk)                  
        
        return(clk, tMatrix, batch_projExp_cpu)
    
    
    
    def create_classes_version0(self, mmap, tMatrix, iter, nExp, expBatchSize, matches, vectorshift, classes, freqBn, coef, cvecs, mask, sigma):
        
        # print("----------create-classes-------------")      
            
        newCL = [[] for i in range(classes)]


        step = int(np.ceil(nExp/expBatchSize))
        batch_projExp_cpu = [0 for i in range(step)]
        
        #rotate and translations
        rotBatch = -matches[:,3].view(nExp,1)
        translations = list(map(lambda i: vectorshift[i], matches[:, 4].int()))
        translations = torch.tensor(translations, device = self.cuda).view(nExp,2)
        
        centerIm = mmap.data.shape[1]/2 
        centerxy = torch.tensor([centerIm,centerIm], device = self.cuda)
        
        count = 0
        for initBatch in range(0, nExp, expBatchSize):
            
            endBatch = min(initBatch+expBatchSize, nExp)
                        
            transforIm, matrixIm = self.center_particles_inverse_save_matrix(mmap.data[initBatch:endBatch], tMatrix[initBatch:endBatch], 
                                                                             rotBatch[initBatch:endBatch], translations[initBatch:endBatch], centerxy)
            
            # if iter < 13:
            #     transforIm = transforIm * self.approximate_otsu_threshold(transforIm, percentile=20)
            
            if mask:
                transforIm = transforIm * self.create_gaussian_mask(transforIm, sigma)
            else:
                transforIm = transforIm * self.create_circular_mask(transforIm)
                
            # if mask: 
            #     if iter < 13:
            #         transforIm = transforIm * self.create_gaussian_mask(transforIm, sigma)
            #     else:
            #         transforIm = transforIm * self.create_circular_mask(transforIm)
                    
            
            tMatrix[initBatch:endBatch] = matrixIm
            
            batch_projExp_cpu[count] = self.batchExpToCpu(transforIm, freqBn, coef, cvecs)
            count+=1

 
            for n in range(classes):
                    class_images = transforIm[matches[initBatch:endBatch, 1] == n]
                    newCL[n].append(class_images)
                    # maskSel = matches[initBatch:endBatch, 1] == n  
                    # sorted_indices = torch.argsort(matches[initBatch:endBatch, 2][maskSel])  
                    # class_images = transforIm[maskSel][sorted_indices[:max(1, len(sorted_indices) // 2)]]  
                    # newCL[n].append(class_images)
                
            del(transforIm)    
                    
   
        newCL = [torch.cat(class_images_list, dim=0) for class_images_list in newCL]    
        clk = self.averages_increaseClas(mmap, iter, newCL, classes)
        
        # if iter > 3 and iter < 13:
        if iter in [2, 4]:
            clk = clk * self.approximate_otsu_threshold(clk, percentile=10)
        elif iter in [6, 8, 10]:
            clk = clk * self.approximate_otsu_threshold(clk, percentile=20) 

            
        clk = clk * self.create_circular_mask(clk)
        
        # if iter in [2, 3]:
        if iter > 2 and iter < 10:
            clk = self.center_by_com(clk)     
        # if mask:
        #     if iter < 13:
        #         clk = clk * self.create_gaussian_mask(clk, sigma)
        #     else:
        #         clk = clk * self.create_circular_mask(clk)
                
        
        return(clk, tMatrix, batch_projExp_cpu)
    
    
    
    def align_particles_to_classes(self, data, cl, tMatrix, iter, expBatchSize, matches, vectorshift, classes, freqBn, coef, cvecs, mask, sigma, sampling):
        
        # print("----------align-to-classes-------------")
        
        # if iter == 3:
        #     thr_low, thr_high = self.get_robust_zscore_thresholds(classes, matches, threshold=2.0)
        
        #rotate and translations
        rotBatch = -matches[:,3].view(expBatchSize,1)
        translations = list(map(lambda i: vectorshift[i], matches[:, 4].int()))
        translations = torch.tensor(translations, device = self.cuda).view(expBatchSize,2)
        
        centerIm = data.shape[1]/2 
        centerxy = torch.tensor([centerIm,centerIm], device = self.cuda)
                            
        transforIm, matrixIm = self.center_particles_inverse_save_matrix(data, tMatrix, 
                                                                         rotBatch, translations, centerxy)
        
        if mask:
            # if iter < 2:
            transforIm = transforIm * self.create_gaussian_mask(transforIm, sigma)
        else: 
            transforIm = transforIm * self.create_circular_mask(transforIm)
        # if mask:
        #     if iter < 3:
        #         transforIm = transforIm * self.create_gaussian_mask(transforIm, sigma)
        #     else:
        #         transforIm = transforIm * self.create_circular_mask(transforIm)
                               
        
        tMatrix = matrixIm
        
        batch_projExp_cpu = self.create_batchExp(transforIm, freqBn, coef, cvecs)
        
        # if iter == 3:
        if iter == 2:
            newCL = [[] for i in range(classes)]              
                    
            for n in range(classes):
                class_images = transforIm[matches[:, 1] == n]
                newCL[n].append(class_images)
                          
                # class_images = transforIm[
                #                         (matches[:, 1] == n) &
                #                         (matches[:, 2] > thr_low[n]) &
                #                         (matches[:, 2] < thr_high[n])
                #                     ]
                # newCL[n].append(class_images)
                
                # maskSel = matches[:, 1] == n  
                # sorted_indices = torch.argsort(matches[:, 2][maskSel])  
                # class_images = transforIm[maskSel][sorted_indices[:max(1, len(sorted_indices) // 2)]] 
                # newCL[n].append(class_images)
                         
            del(transforIm)
            
            newCL = [torch.cat(class_images_list, dim=0) for class_images_list in newCL] 
            clk = self.averages(data, newCL, classes)
            
            res_classes = self.frc_resolution_tensor(newCL, sampling)
            bfactor = self.estimate_bfactor_batch(clk, sampling, res_classes)
            # clk = self.enhance_averages_butterworth_adaptive(clk, res_classes, sampling)
            clk = self.gaussian_lowpass_filter_2D_adaptive(clk, res_classes, sampling)
            clk = self.sharpen_averages_batch(clk, sampling, bfactor, res_classes)
            # clk = self.highpass_butterworth_soft_batch(clk, res_classes, sampling)
            # clk = self.sharpen_averages_batch_nq(clk, sampling, bfactor)
            # clk = self.enhance_averages_butterworth(clk, sampling) 
            # clk = self.enhance_averages_butterworth_combined_cos(clk, res_classes, sampling)
            # clk = self.enhance_averages_attenuate_lowfrequencies(clk, res_classes, sampling)
            # clk = self.unsharp_mask_norm(clk)
            # clk = self.gaussian_lowpass_filter_2D(clk, maxRes, sampling)
        
            
            # clk = self.unsharp_mask_adaptive_gaussian(clk)
            # mask_C = self.compute_class_consistency_masks(newCL) #Apply consistency mask           
            # clk = self.apply_consistency_masks_vector(clk, mask_C)
                        
        
            if not hasattr(self, 'grad_squared'):
                self.grad_squared = torch.zeros_like(cl)
            clk, self.grad_squared = self.update_classes_rmsprop(cl, clk, 0.001, 0.9, 1e-8, self.grad_squared)         
            
                
            clk = clk * self.create_circular_mask(clk)
            # clk = clk * self.create_gaussian_masks_different_sigma(clk)
      
        else: 
            del(transforIm)
            clk = cl  
            
        return (clk, tMatrix, batch_projExp_cpu) 
           
    
    def center_particles_inverse_save_matrix(self, data, tMatrix, update_rot, update_shifts, centerxy):
          
        
        rotBatch = update_rot.view(-1)
        batchsize = rotBatch.size(dim=0)

        scale = torch.tensor([[1.0, 1.0]], device=self.cuda).expand(batchsize, -1)      
        
        translations = update_shifts.view(batchsize,2,1)
        
        translation_matrix = torch.eye(3, device=self.cuda).unsqueeze(0).repeat(batchsize, 1, 1)
        translation_matrix[:, :2, 2] = translations.squeeze(-1)

        rotation_matrix = kornia.geometry.get_rotation_matrix2d(centerxy.expand(batchsize, -1), rotBatch, scale)
        del(scale)
        
        M = torch.matmul(rotation_matrix, translation_matrix)
        del(rotation_matrix, translation_matrix)       
        
        M = torch.cat((M, torch.zeros((batchsize, 1, 3), device=self.cuda)), dim=1)
        M[:, 2, 2] = 1.0      

                         
        #combined matrix
        tMatrixLocal = torch.cat((tMatrix, torch.zeros((batchsize, 1, 3), device=self.cuda)), dim=1)
        tMatrixLocal[:, 2, 2] = 1.0
        
        M = torch.matmul(M, tMatrixLocal)
        M = M[:, :2, :] 
        del(tMatrixLocal)  
    
        Texp = torch.from_numpy(data.astype(np.float32)).to(self.cuda).unsqueeze(1)

        transforIm = kornia.geometry.warp_affine(Texp, M, dsize=(data.shape[1], data.shape[2]), mode='bilinear', padding_mode='zeros')
        transforIm = transforIm.view(batchsize, data.shape[1], data.shape[2])
        del(Texp)
        
        return(transforIm, M)
    
    
    def averages_increaseClas(self, mmap, iter, newCL, classes): 
        
        if iter < 10:
            newCL = sorted(newCL, key=len, reverse=True)    
        element = list(map(len, newCL))

        # if iter > 0 and iter < 4:
        if iter > 0 and iter < 5:
            numClas = int(classes/2)
        else:
            numClas = classes
  
        clk_list = []
        for n in range(numClas):
            current_length = len(newCL[n])
            # if iter < 3 and current_length > 2:
            if iter < 4 and current_length > 2:
                split1, split2 = torch.split(newCL[n], current_length // 2 + 1, dim=0)
                # clk_list.append(torch.mean(split1, dim=0))
                # insert = torch.mean(split2, dim=0).view(mmap.data.shape[1], mmap.data.shape[2])
                # clk_list.append(insert)
                sum1 = torch.mean(split1, dim=0)
                sum2 = torch.mean(split2, dim=0)
                clk_list.append(sum1)
                clk_list.append(sum2)
            
            else:
                if current_length:
                    clk_list.append(torch.mean(newCL[n], dim=0))
        
        clk = torch.stack(clk_list)
        return(clk)
    
    
    def averages_increaseClas2(self, mmap, iter, newCL, classes, final_classes): 
        
        if iter < 10:
            newCL = sorted(newCL, key=len, reverse=True)    
        
        #The classes start with half of the total number of classes and are divided into three rounds.
        class_split = int(final_classes/(2*3))
        if iter == 3:
            class_split = final_classes - classes
            
  
        clk_list = []
        for n in range(classes):
            current_length = len(newCL[n])
  
            if iter > 0 and iter < 4 and n < class_split and current_length > 2:
                split1, split2 = torch.split(newCL[n], current_length // 2 + 1, dim=0)
                clk_list.append(torch.mean(split1, dim=0))
                insert = torch.mean(split2, dim=0).view(mmap.data.shape[1], mmap.data.shape[2])
                clk_list.append(insert)
            
            else:
                if current_length:
                    clk_list.append(torch.mean(newCL[n], dim=0))

        clk = torch.stack(clk_list)                           
        return(clk)
    
    
    
    def averages_createClasses(self, mmap, iter, newCL): 
        
        if iter < 10:
            newCL = sorted(newCL, key=len, reverse=True)    
        # element = list(map(len, newCL))
        # print(element)    
        classes = len(newCL)       
  
        clk = []
        for n in range(classes):
            if len(newCL[n]) > 0:
                clk.append(torch.mean(newCL[n], dim=0))
            else:
                clk.append(torch.zeros((mmap.data.shape[1], mmap.data.shape[2]), device=newCL[0].device))
        clk = torch.stack(clk)
        return clk
    
    
    def averages(self, data, newCL, classes): 
        
        # element = list(map(len, newCL))
        # print(element)      
        clk = []
        for n in range(classes):
            if len(newCL[n]) > 0:
                clk.append(torch.mean(newCL[n], dim=0))
            else:
                clk.append(torch.zeros((data.shape[1], data.shape[2]), device=newCL[0].device))
        clk = torch.stack(clk)
        return clk
    
    
    def create_gaussian_mask(self, images, sigma):
        dim = images.size(dim=1)
        center = dim // 2
        y, x = torch.meshgrid(torch.arange(dim) - center, torch.arange(dim) - center, indexing='ij')
        dist = torch.sqrt(x**2 + y**2).float().to(images.device)  
        
        sigma2 = sigma**2
        K = 1. / (torch.sqrt(2 * torch.tensor(np.pi)) * sigma)**2
    
        mask = K * torch.exp(-0.5 * (dist**2 / sigma2))
        mask = mask / mask[center, center].clone()
        
        return mask  
    
    
    def create_circular_mask(self, images):
        dim = images.size(dim=1)
        center = dim // 2
        y, x = torch.meshgrid(torch.arange(dim) - center, torch.arange(dim) - center, indexing='ij')
        dist = torch.sqrt(x**2 + y**2).float().to(images.device)
        
        # Creamos una máscara circular
        circular_mask = torch.zeros_like(dist)
        circular_mask[dist <= center] = 1.0
        
        return circular_mask
    
    
    def center_by_com(self, batch: torch.Tensor, use_abs: bool = True, eps: float = 1e-8):
        B, H, W = batch.shape
        device = batch.device
    
        weights = batch.abs() if use_abs else batch
        weights = weights.unsqueeze(1)
    
        y = torch.arange(H, device=device) - H // 2
        x = torch.arange(W, device=device) - W // 2
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        xx = xx[None, None, ...].float()
        yy = yy[None, None, ...].float()
    
        mass = weights.sum(dim=(2, 3), keepdim=True) + eps
        x_com = (weights * xx).sum(dim=(2, 3), keepdim=True) / mass
        y_com = (weights * yy).sum(dim=(2, 3), keepdim=True) / mass
    
        shift = torch.cat([-x_com, -y_com], dim=1).squeeze(-1).squeeze(-1)
        batch_input = batch.unsqueeze(1)
        centered = kornia.geometry.transform.translate(batch_input, shift, mode='bilinear', padding_mode='zeros', align_corners=True)
    
        return centered.squeeze(1)
    
    def center_by_com2(self, batch: torch.Tensor, use_abs: bool = True, eps: float = 1e-8):
        B, H, W = batch.shape
        device = batch.device
    
        # === 1. Niveles de gris originales
        mean0 = batch.mean(dim=(1, 2), keepdim=True)
        std0  = batch.std (dim=(1, 2), keepdim=True)
    
        # === 2. Centro de masa
        weights = batch.abs() if use_abs else batch
        weights = weights.unsqueeze(1)
    
        y = torch.arange(H, device=device) - H // 2
        x = torch.arange(W, device=device) - W // 2
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        xx = xx[None, None, ...].float()
        yy = yy[None, None, ...].float()
    
        mass = weights.sum(dim=(2, 3), keepdim=True) + eps
        x_com = (weights * xx).sum(dim=(2, 3), keepdim=True) / mass
        y_com = (weights * yy).sum(dim=(2, 3), keepdim=True) / mass
    
        # === 3. Aplicar desplazamiento subpixel
        shift = torch.cat([-x_com, -y_com], dim=1).squeeze(-1).squeeze(-1)
        batch_input = batch.unsqueeze(1)
        centered = kornia.geometry.transform.translate(
            batch_input, shift, mode='bilinear', padding_mode='zeros', align_corners=True).squeeze(1)
    
        # === 4. Restaurar media y contraste originales
        mean1 = centered.mean(dim=(1, 2), keepdim=True)
        std1  = centered.std (dim=(1, 2), keepdim=True)
    
        centered = (centered - mean1) / (std1 + eps) * std0 + mean0
    
        return centered
    
    
    def apply_leaky_relu(self, images, relu = 0.5):
        images = torch.where(images > 0, images, relu * images)
        return images       
        
    
    def gaussian_lowpass_filter_2D(self, imgs, resolution_angstrom, pixel_size, clamp_exp = 80.0, hard_cut: bool = False):
    
        N, H, W = imgs.shape
        device = imgs.device
    
        # Guardamos estadísticos originales
        mean0 = imgs.mean(dim=(1, 2), keepdim=True)
        std0 = imgs.std(dim=(1, 2), keepdim=True)
    
        # Malla de frecuencias
        fy = torch.fft.fftfreq(H, d=pixel_size).to(device)
        fx = torch.fft.fftfreq(W, d=pixel_size).to(device)
        grid_y, grid_x = torch.meshgrid(fy, fx, indexing='ij')
        freq_squared = grid_x ** 2 + grid_y ** 2   
    
        # Filtro gaussiano en frecuencia
        D0_freq = 1.0 / resolution_angstrom
        sigma_freq = D0_freq / np.sqrt(2 * np.log(2))
        exponent = -freq_squared / (2.0 * sigma_freq ** 2)
        exponent = exponent.clamp(max=clamp_exp)
        filter_map = torch.exp(exponent)
    
        if hard_cut:
            filter_map[freq_squared > D0_freq**2] = 0.0
    
        # Broadcasting del filtro
        filter_map = filter_map.to(device).unsqueeze(0).expand(N, -1, -1)
    
        # FFT y filtrado
        fft_imgs = torch.fft.fft2(imgs)
        fft_filtered_imgs = fft_imgs * filter_map
        filtered_imgs = torch.fft.ifft2(fft_filtered_imgs).real
        filtered_imgs = torch.nan_to_num(filtered_imgs)
    
        mean = filtered_imgs.mean(dim=(1, 2), keepdim=True)
        std = filtered_imgs.std(dim=(1, 2), keepdim=True)
        valid = std > 1e-6
    
        normalized = (filtered_imgs - mean) / (std + 1e-8) * std0 + mean0
        output = torch.where(valid, normalized, imgs)
    
        return output
    
    
    @torch.no_grad()
    def gaussian_lowpass_filter_2D_adaptive(self, imgs, res_angstrom, pixel_size,
                                            floor_res=25.0, clamp_exp=80.0,
                                            hard_cut=False, nyquist_margin=0.95, normalize = True):
        B, H, W = imgs.shape
        device, eps = imgs.device, 1e-8
    
        # === Limitar resolución efectiva según Nyquist
        nyquist_res = 2.0 * pixel_size           
        safe_res = nyquist_res / nyquist_margin  
    
        res_eff = torch.nan_to_num(res_angstrom, nan=floor_res,
                                   posinf=floor_res, neginf=floor_res)
        res_eff = torch.minimum(res_eff, torch.full_like(res_eff, floor_res))
        res_eff = torch.clamp(res_eff, min=safe_res)  # Prevenimos aliasing
    
        # === Coordenadas de frecuencia
        fy, fx = (torch.fft.fftfreq(H, d=pixel_size, device=device),
                  torch.fft.fftfreq(W, d=pixel_size, device=device))
        gy, gx = torch.meshgrid(fy, fx, indexing='ij')
        freq2  = (gx**2 + gy**2).unsqueeze(0)  # [1, H, W]
    
        # === Filtro Gaussiano adaptativo
        ln2    = torch.log(torch.tensor(2.0, device=device))
        D0     = (1.0 / res_eff).view(B,1,1)                            
        sigma2 = (D0 / torch.sqrt(2*ln2))**2             
    
        exponent = (-freq2) / (2*sigma2 + eps)
        filt     = torch.exp(exponent.clamp(max=clamp_exp))
    
        if hard_cut:
            filt = torch.where(freq2 > D0**2, 0.0, filt)
    
        # === Aplicar filtro y transformar inversa
        img_filt = torch.fft.ifft2(torch.fft.fft2(imgs, norm="forward") * filt, norm="forward").real
        img_filt = torch.nan_to_num(img_filt)
    
        # === Restaurar contraste original
        if normalize:
            mean0 = imgs.mean(dim=(1,2), keepdim=True)
            std0  = imgs.std (dim=(1,2), keepdim=True)
            
            mean_f = img_filt.mean(dim=(1,2), keepdim=True)
            std_f  = img_filt.std (dim=(1,2), keepdim=True)
            valid  = std_f > 1e-6
            img_filt = torch.where(valid,
                                   (img_filt - mean_f)/(std_f+eps)*std0 + mean0,
                                   imgs)
    
        return img_filt

    

    def update_classes_rmsprop(self, cl, clk, learning_rate, decay_rate, epsilon, grad_squared):
        
        grad = clk - cl
        
        grad_squared = decay_rate * grad_squared + (1 - decay_rate) * grad**2        
        update = learning_rate * grad / (torch.sqrt(grad_squared) + epsilon)
        cl = torch.add(cl, update)
        # print(grad_squared.shape)
        # file = "grad.mrcs"
        # self.save_images(grad_squared.cpu().numpy(), file)
        
        return cl, grad_squared
    
    
    def save_images(self, data, outfilename):
        data = data.astype('float32')
        with mrcfile.new(outfilename, overwrite=True) as mrc:
            mrc.set_data(data)
    

    def gamma_contrast(self, images, gamma=0.5):
        epsilon = 1e-8  #avoid div/0
        normalized_images = (images + 1) / 2.0
        normalized_images = torch.clamp(normalized_images, epsilon, 1.0 - epsilon) 
        corrected_images = torch.pow(normalized_images, 1.0 / gamma)
        corrected_images = corrected_images * 2.0 - 1.0
        
        return corrected_images
    
    
    def increase_contrast_sigmoid(self, images, alpha=10, beta=0.6):
   
        normalized_images = (images + 1) / 2.0 
        # sigmoid function
        adjusted_images = 1 / (1 + torch.exp(-alpha * (normalized_images - beta)))
        adjusted_images = adjusted_images * 2.0 - 1.0

        return adjusted_images
    
    
    def normalize_particles_batch(self, images,  eps: float = 1e-8):
        
        mean = images.mean(dim=(1, 2), keepdim=True)  
        std = images.std(dim=(1, 2), keepdim=True) + eps  
        
        normalized_batch = (images - mean) / std
        
        return normalized_batch
    
    
    def normalize_particles_global(self, images, eps=1e-8):
        
        mean = images.mean()  
        std = images.std()  
        images = (images - mean) / (std + eps)  
        return images
    
    
    def process_images_iteratively(self, batch, num_iterations):
        batch = batch.float()
        for _ in range(num_iterations):
            img_means = batch.mean(dim=(1, 2), keepdim=True)
            lower_values_mask = batch < img_means
            lower_values_sum = (batch * lower_values_mask.float()).sum(dim=(1, 2), keepdim=True)
            lower_values_count = lower_values_mask.sum(dim=(1, 2), keepdim=True)
            lower_values_mean = lower_values_sum / (lower_values_count + 1e-8)
            batch = batch + torch.abs(lower_values_mean)
        return batch
    
    def contrast_dominant_mask2(self, imgs,
                                window=3,
                                contrast_percentile=80,
                                intensity_percentile=50,
                                contrast_weight=1.5,
                                intensity_weight=1.0):
    
        N, H, W = imgs.shape
        imgs = imgs.float().unsqueeze(1)  # [N, 1, H, W]
        
        mean_local = F.avg_pool2d(imgs, window, stride=1, padding=window // 2)
        mean_sq_local = F.avg_pool2d(imgs**2, window, stride=1, padding=window // 2)
        std_local = torch.sqrt((mean_sq_local - mean_local**2).clamp(min=0))  # [N, 1, H, W]
    
        contrast_thresh = torch.quantile(std_local.view(N, -1), contrast_percentile / 100.0, dim=1).view(N, 1, 1, 1)
        intensity_thresh = torch.quantile(imgs.view(N, -1), intensity_percentile / 100.0, dim=1).view(N, 1, 1, 1)
    
        score = (contrast_weight * std_local + intensity_weight * imgs)
        mask = (std_local > contrast_thresh) & (imgs > intensity_thresh)
        
        return mask.float().squeeze(1)
    
    
    def contrast_dominant_mask(self, imgs,
                            window=3,
                            contrast_percentile=80,
                            intensity_percentile=50,
                            contrast_weight=1.5,
                            intensity_weight=1.0,
                            smooth_sigma=1.0):
        N, H, W = imgs.shape
        imgs = imgs.float().unsqueeze(1)  # [N, 1, H, W]
        
        mean_local = F.avg_pool2d(imgs, window, stride=1, padding=window // 2)
        mean_sq_local = F.avg_pool2d(imgs**2, window, stride=1, padding=window // 2)
        std_local = torch.sqrt((mean_sq_local - mean_local**2).clamp(min=0))  # [N, 1, H, W]
    
        contrast_thresh = torch.quantile(std_local.view(N, -1), contrast_percentile / 100.0, dim=1).view(N, 1, 1, 1)
        intensity_thresh = torch.quantile(imgs.view(N, -1), intensity_percentile / 100.0, dim=1).view(N, 1, 1, 1)
    
        mask = ((std_local > contrast_thresh) & (imgs > intensity_thresh)).float()  # [N, 1, H, W]
    
        # === Suavizado con gaussiana ===
        if smooth_sigma > 0:
            kernel_size = int(2 * round(2 * smooth_sigma) + 1)
            padding = kernel_size // 2
    
            x = torch.arange(-padding, padding + 1, device=imgs.device).float()
            gauss = torch.exp(-0.5 * (x / smooth_sigma)**2)
            gauss = gauss / gauss.sum()
    
            gauss_2d = gauss[:, None] * gauss[None, :]
            gauss_2d = gauss_2d.unsqueeze(0).unsqueeze(0)  # [1, 1, K, K]
    
            mask = F.conv2d(mask, gauss_2d, padding=padding, groups=1)
    
        return mask.squeeze(1)  # [N, H, W]
    
    
    def approximate_otsu_threshold(self, imgs, percentile=20):

        N, H, W = imgs.shape
        flat = imgs.view(N, -1)
        k = int(flat.shape[1] * (percentile / 100.0))
    
        topk_vals, _ = torch.topk(flat, k=k, dim=1)
        thresholds = topk_vals[:, -1].clamp(min=0.0).view(N, 1, 1)
    
        self.binary_masks = (imgs > thresholds).float()
        return self.binary_masks
    
    def compute_particle_radius(self, imgs, percentile: float = 100):
        
        masks= self.approximate_otsu_threshold(imgs)
        
        B, H, W = masks.shape
        device = masks.device
    
        y_coords = torch.arange(H, device=device).float() - H / 2
        x_coords = torch.arange(W, device=device).float() - W / 2
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
    
        dist_sq = xx**2 + yy**2  # shape (H, W)
        self.max_distances = torch.zeros(B, device=device)
    
        for i in range(B):
            foreground = masks[i] > 0.5
            foreground_distances = dist_sq[foreground]
            if foreground_distances.numel() > 0:
                percentile_value_sq = torch.quantile(foreground_distances, percentile / 100.0)
                self.max_distances[i] = torch.sqrt(percentile_value_sq)
    
        return self.max_distances
    
    
    def auto_generate_masks(
        self,
        averages: torch.Tensor,          # [B, H, W]
        percentile_threshold: float = 30,  # para máscara binaria inicial
        percentile_radius: float = 90,     # para calcular el radio interno
        margin: float = 8.0,               # píxeles extra para radio externo
        transition: str = "sigmoid",        # tipo de transición: "cosine" o "sigmoid"
    ) -> torch.Tensor:
    
        # Paso 1: Crear máscara binaria para estimar forma
        binary_masks = self.approximate_otsu_threshold(averages, percentile=percentile_threshold)
    
        # Paso 2: Calcular radio interno por imagen (percentil sobre distancias)
        B, H, W = binary_masks.shape
        device = averages.device
    
        y_coords = torch.arange(H, device=device).float() - H / 2
        x_coords = torch.arange(W, device=device).float() - W / 2
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
        dist_sq = xx**2 + yy**2  # shape (H, W)
    
        r_inner = torch.zeros(B, device=device)
    
        for i in range(B):
            mask = binary_masks[i] > 0.5
            d = dist_sq[mask]
            if d.numel() > 0:
                r2 = torch.quantile(d, percentile_radius / 100.0)
                r_inner[i] = torch.sqrt(r2)
    
        r_outer = r_inner + margin  # [B]
    
        # Paso 3: Crear máscara suave tipo coseno
        yy = yy.expand(B, -1, -1)
        xx = xx.expand(B, -1, -1)
        r = torch.sqrt(xx**2 + yy**2)  # [B, H, W]
    
        r_inner = r_inner.view(B, 1, 1)
        r_outer = r_outer.view(B, 1, 1)
    
        if transition == "cosine":
            mask = 0.5 * (1 + torch.cos(torch.pi * (r - r_inner) / (r_outer - r_inner)))
            mask = torch.where(r <= r_inner, torch.ones_like(mask), mask)
            mask = torch.where(r >= r_outer, torch.zeros_like(mask), mask)
        elif transition == "sigmoid":
            # Suavizado tipo sigmoide (más gradual)
            width = r_outer - r_inner + 1e-6
            mask = 1.0 / (1 + torch.exp((r - r_inner) / (width / 6)))  # suave
        else:
            raise ValueError("transition must be 'cosine' or 'sigmoid'")
    
        # Paso 4: Aplicar máscara
        masked = averages * mask

        return masked
    
    
    def create_gaussian_masks_different_sigma(self, images):
        
        sigmas = self.compute_particle_radius(images)
        
        B = images.size(0)
        dim = images.size(-1)
        center = dim // 2
    
        y, x = torch.meshgrid(
            torch.arange(dim, device=images.device) - center,
            torch.arange(dim, device=images.device) - center,
            indexing='ij'
        )
        dist2 = (x**2 + y**2).float()
        dist2 = dist2.unsqueeze(0).expand(B, -1, -1)
        sigma2 = sigmas.view(-1, 1, 1)**2
        K = 1. / (torch.sqrt(2 * torch.tensor(np.pi, device=images.device)) * sigmas).view(-1, 1, 1)**2
        masks = K * torch.exp(-0.5 * dist2 / sigma2)
        center_val = masks[:, center, center].clone().view(-1, 1, 1)
        masks = masks / center_val
        return masks
    
    @torch.no_grad()
    def unsharp_mask_norm(self, imgs, kernel_size=5, strength=2.0):
        N, H, W = imgs.shape
        
        mean0 = imgs.mean(dim=(1, 2), keepdim=True)
        std0 = imgs.std(dim=(1, 2), keepdim=True)
        
        pad = kernel_size // 2
        kernel = torch.ones(1, 1, kernel_size, kernel_size, device=imgs.device) / (kernel_size ** 2)
    
        imgs_ = imgs.unsqueeze(1)
        blurred = F.conv2d(imgs_, kernel, padding=pad)
        sharpened = imgs_ + strength * (imgs_ - blurred)
        sharpened = sharpened.squeeze(1)
    
        mean = sharpened.mean(dim=(1, 2), keepdim=True)
        std = sharpened.std(dim=(1, 2), keepdim=True)
    
        valid = std > 1e-6
        normalized = (sharpened - mean) / (std + 1e-8)*std0+mean0
        output = torch.where(valid, normalized, imgs)
    
        return output
    
    def gaussian_kernel(self, kernel_size, sigma, device):
        ax = torch.arange(kernel_size, device=device) - kernel_size // 2
        xx, yy = torch.meshgrid(ax, ax, indexing='ij')
        kernel = torch.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
        kernel = kernel / kernel.sum()
        return kernel.view(1, 1, kernel_size, kernel_size)


    def unsharp_mask_adaptive_gaussian(self, imgs, kernel_size=5, base_strength=1.0,
                                        contrast_window=7, sigma=None):
        N, H, W = imgs.shape
        imgs = imgs.float()
        mean0 = imgs.mean(dim=(1, 2), keepdim=True)
        std0 = imgs.std(dim=(1, 2), keepdim=True)
        pad = kernel_size // 2
        pad_c = contrast_window // 2
        device = imgs.device
    
        if sigma is None:
            sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8
        gkernel = self.gaussian_kernel(kernel_size, sigma, device)
        imgs_ = imgs[:, None]
        blurred = F.conv2d(imgs_, gkernel, padding=pad)
    
        mean_local = F.avg_pool2d(imgs_, kernel_size=contrast_window, stride=1, padding=pad_c)
        mean_sq_local = F.avg_pool2d(imgs_ ** 2, kernel_size=contrast_window, stride=1, padding=pad_c)
        local_var = (mean_sq_local - mean_local**2).clamp(min=0)
        local_std = torch.sqrt(local_var)
    
        global_std = std0.view(N, 1, 1, 1)
        strength_map = base_strength * (local_std / (global_std + 1e-8))
        strength_map = strength_map.clamp(0.0, base_strength * 3.0)
    
        sharpened = imgs_ + strength_map * (imgs_ - blurred)
        sharpened = sharpened.squeeze(1)
    
        mean = sharpened.mean(dim=(1, 2), keepdim=True)
        std = sharpened.std(dim=(1, 2), keepdim=True)
        valid = std > 1e-6
        normalized = (sharpened - mean) / (std + 1e-8) * std0 + mean0
        output = torch.where(valid, normalized, imgs)
    
        return output
    
    
    def compute_class_consistency_masks(self, newCL, eps=1e-8):

        masks = []
        for class_images in newCL:
            if class_images is None or class_images.shape[0] == 0:
                
                for ref in newCL:
                    if ref is not None and ref.shape[0] > 0:
                        H, W = ref.shape[1:]
                        break

                mask = torch.ones(H, W, device=class_images.device if class_images is not None else "cpu")
                masks.append(mask)
                continue
    
            fft_imgs = torch.fft.fft2(class_images)  # [N, H, W]
            mag_sq_sum = (fft_imgs.abs() ** 2).sum(dim=0)  # [H, W]
            complex_sum = fft_imgs.sum(dim=0)             # [H, W]
            mag_of_sum_sq = (complex_sum.abs()) ** 2      # [H, W]
    
            mask = mag_of_sum_sq / (class_images.shape[0] * mag_sq_sum + eps)
            mask = mask.clamp(min=0)
            mask = (mask - mask.min()) / (mask.max() - mask.min() + eps)  # Normalizar entre 0 y 1
            alpha = 0.5 #para suavizar el efecto
            mask = alpha * torch.ones_like(mask) + (1 - alpha) * mask
            # mask = mask ** 0.8
            masks.append(mask)
        
        return masks
    
    
    def apply_consistency_masks_vector(self, clk, mask_C):
        
        mask_C_tensor = torch.stack(mask_C)

        fft_clk = torch.fft.fft2(clk)                     # (N, H, W), complejo
        masked_fft = fft_clk * mask_C_tensor              # broadcasting: (N, H, W) * (N, H, W)
        filtered_clk = torch.fft.ifft2(masked_fft).real   # volver al dominio espacial
    
        return filtered_clk
    
    #Filtro de power spectrum segun relion
    def compute_radial_profile(self, imgs_fft):
        """
        Calcula perfil radial promedio del espectro de potencia (vectorizado).
        imgs_fft: [N, H, W] complejo
        Retorna: [R] promedio sobre imágenes y píxeles con igual radio
        """
        N, H, W = imgs_fft.shape
        power = imgs_fft.real**2 + imgs_fft.imag**2  # más rápido que abs() ** 2
    
        y, x = torch.meshgrid(
            torch.arange(H, device=imgs_fft.device),
            torch.arange(W, device=imgs_fft.device),
            indexing='ij'
        )
        r = ((x - W//2)**2 + (y - H//2)**2).sqrt().long()
        max_r = min(H, W) // 2
        r = r.clamp(0, max_r - 1)
    
        # Reorganiza para usar scatter_add (más rápido que bucles)
        r_flat = r.view(-1)
        power_flat = power.view(N, -1)
    
        radial = torch.zeros((N, max_r), device=imgs_fft.device)
        radial.scatter_add_(1, r_flat.unsqueeze(0).expand(N, -1), power_flat)
    
        count = torch.bincount(r_flat, minlength=max_r).clamp(min=1e-8)
        mean_radial = radial.sum(0) / count  # promedio sobre N
        return mean_radial  # [R]
    

    def relion_filter_from_image_list(self, images_list, class_avg,
                                       sampling, resolution_angstrom, eps=1e-8):
        """
        Aplica un filtro tipo RELION a class_avg en Fourier, adaptado al espectro
        radial de las imágenes en images_list, con límite de resolución.
        """
        if isinstance(images_list, torch.Tensor):
            if images_list.ndim == 2:
                images_list = images_list[None]  # [1, H, W]
            elif images_list.ndim == 3:
                pass
            else:
                raise ValueError("images_list tiene dimensiones no válidas.")
        elif isinstance(images_list, list):
            images_list = torch.stack(images_list)
        else:
            raise TypeError("images_list debe ser lista o tensor")
    
        if images_list.numel() == 0:
            return class_avg
    
        device = class_avg.device
        images_tensor = images_list.float().to(device)
        class_avg = class_avg.float().to(device)
    
        H, W = class_avg.shape
    
        # FFTs sin shift
        fft_imgs = torch.fft.fft2(images_tensor)  # [N, H, W]
        fft_avg = torch.fft.fft2(class_avg)       # [H, W]
    
        # Magnitudes al cuadrado (espectro de potencia)
        pspec_imgs = torch.abs(fft_imgs) ** 2
        pspec_avg = torch.abs(fft_avg) ** 2
    
        # Crea mapa radial solo una vez
        y = torch.arange(H, device=device) - H // 2
        x = torch.arange(W, device=device) - W // 2
        r = torch.sqrt((x[None, :]**2 + y[:, None]**2))
        r = r.long().clamp(0, H // 2)
    
        # Perfil radial eficiente (sin shift): acumula por anillos
        def radial_profile(pspec):
            radial_sum = torch.zeros(H // 2 + 1, device=device)
            radial_count = torch.zeros(H // 2 + 1, device=device)
            for i in range(pspec.shape[0]):
                flat_r = r.flatten()
                flat_val = pspec[i].flatten() if pspec.ndim == 3 else pspec.flatten()
                radial_sum.index_add_(0, flat_r, flat_val)
                radial_count.index_add_(0, flat_r, torch.ones_like(flat_val))
            return radial_sum / (radial_count + eps)
    
        pspec_target = radial_profile(pspec_imgs)
        pspec_avg_1 = radial_profile(pspec_avg[None])
    
        filt = torch.sqrt((pspec_target + eps) / (pspec_avg_1 + eps))
    
        # Límite de resolución
        if resolution_angstrom and resolution_angstrom > 0:
            nyquist = 1.0 / (2.0 * sampling)
            freq_cutoff = 1.0 / resolution_angstrom
            radius_cutoff = int((freq_cutoff / nyquist) * (H // 2))
            radius_cutoff = min(radius_cutoff, len(filt))
            filt[radius_cutoff:] = 1.0
    
        # Crear mapa 2D del filtro (solo una vez)
        filt_map = filt[r.clamp(0, len(filt)-1)]
    
        # Aplicar filtro en Fourier
        fft_avg_shift = torch.fft.fftshift(fft_avg)
        fft_filtered = filt_map * fft_avg_shift
        fft_filtered = torch.fft.ifftshift(fft_filtered)
    
        filtered = torch.fft.ifft2(fft_filtered).real
    
        # Normalización
        mean_orig, std_orig = class_avg.mean(), class_avg.std()
        mean_filt, std_filt = filtered.mean(), filtered.std()
        normalized = (filtered - mean_filt) / (std_filt + eps) * std_orig + mean_orig
    
        # Limpieza explícita
        del fft_imgs, pspec_imgs, fft_avg, pspec_avg, fft_filtered
        torch.cuda.empty_cache()
    
        return normalized
    
    @torch.no_grad()
    def filter_classes_relion_style(self, newCL, clk, sampling, resolution_angstrom):
        """
        Aplica filtro RELION-like a cada clase usando imágenes que la componen.
        """
        filtered_classes = []
        for class_imgs, class_avg in zip(newCL, clk):
            filtered = self.relion_filter_from_image_list(
                class_imgs, class_avg,
                sampling=sampling,
                resolution_angstrom=resolution_angstrom
            )
            filtered_classes.append(filtered)
            torch.cuda.empty_cache()  # Asegura limpieza entre iteraciones
    
        return torch.stack(filtered_classes)
    
    
    @torch.no_grad()
    def frc_resolution_tensor2(
            self,
            newCL,                       # lista de tensores [N_i,H,W]
            pixel_size: float,           # Å/px
            frc_threshold: float = 0.143,
            fallback_res: float = 40.0
    ) -> torch.Tensor:
        """Devuelve tensor [n_classes] con la resolución FRC por clase (Å)."""
        n_classes = len(newCL)
        device    = newCL[0].device
        res_out   = torch.full((n_classes,), float('nan'), device=device)
    
        for c, imgs in enumerate(newCL):
            n, h, w = imgs.shape
            if n < 2:
                continue  # no se puede partir en mitades
    
            # ---------------- half maps -----------------
            perm          = torch.randperm(n, device=device)
            half1, half2  = torch.chunk(imgs[perm], 2, dim=0)
            avg1, avg2    = half1.mean(0), half2.mean(0)
    
            # ---------------- FRC -----------------------
            fft1 = torch.fft.fftshift(torch.fft.fft2(avg1))
            fft2 = torch.fft.fftshift(torch.fft.fft2(avg2))
    
            p1   = (fft1.real**2 + fft1.imag**2)
            p2   = (fft2.real**2 + fft2.imag**2)
            prod = (fft1 * fft2.conj()).real
    
            y, x = torch.meshgrid(torch.arange(h, device=device),
                                  torch.arange(w, device=device),
                                  indexing='ij')
            r     = ((x - w//2)**2 + (y - h//2)**2).sqrt().long()
            Rmax  = min(h, w) // 2
            r.clamp_(0, Rmax-1)
            r_flat = r.view(-1)
    
            frc_num = torch.zeros(Rmax, device=device).scatter_add_(0, r_flat, prod.view(-1))
            frc_d1  = torch.zeros(Rmax, device=device).scatter_add_(0, r_flat, p1.view(-1))
            frc_d2  = torch.zeros(Rmax, device=device).scatter_add_(0, r_flat, p2.view(-1))
            frc     = frc_num / (torch.sqrt(frc_d1*frc_d2) + 1e-12)
    
            freqs   = torch.linspace(0, 0.5/pixel_size, Rmax, device=device)
            idx     = torch.where(frc < frc_threshold)[0]
    
            if len(idx) and idx[0] > 0:                # evita f_cut=0
                f_cut        = freqs[idx[0]]
                res_out[c]   = 1.0 / f_cut
    
        # ---------- sustituye NaN e Inf una sola vez ------------
        res_out = torch.nan_to_num(res_out,
                                   nan   = fallback_res,
                                   posinf= fallback_res,
                                   neginf= fallback_res)
        return res_out
    
  
    @torch.no_grad()
    def frc_resolution_tensor(
            self,
            newCL,                       # lista de tensores [N_i,H,W]
            pixel_size: float,           # Å/px
            frc_threshold: float = 0.143,
            fallback_res: float = 40.0
    ) -> torch.Tensor:
        
        """Devuelve tensor [n_classes] con la resolución FRC por clase (Å)."""
        n_classes = len(newCL)
        h, w = next((imgs.shape[-2], imgs.shape[-1]) for imgs in newCL if imgs.numel() > 0)
        device    = newCL[0].device
        res_out   = torch.full((n_classes,), float('nan'), device=device)
        
        y, x = torch.meshgrid(torch.arange(h, device=device),
                              torch.arange(w, device=device),
                              indexing='ij')
        r     = ((x - w//2)**2 + (y - h//2)**2).sqrt().long()
        Rmax  = min(h, w) // 2
        r.clamp_(0, Rmax-1)
        r_flat = r.view(-1)
        freqs   = torch.linspace(0, 0.5/pixel_size, Rmax, device=device)
    
        for c, imgs in enumerate(newCL):
            n = imgs.shape[0]
            if n < 2:
                continue  # no se puede partir en mitades
    
            # ---------------- half maps -----------------
            perm          = torch.randperm(n, device=device)
            half1, half2  = torch.chunk(imgs[perm], 2, dim=0)
            avg1, avg2    = half1.mean(0), half2.mean(0)
    
            # ---------------- FRC -----------------------
            fft1 = torch.fft.fftshift(torch.fft.fft2(avg1, norm="forward"))
            fft2 = torch.fft.fftshift(torch.fft.fft2(avg2, norm="forward"))
    
            p1   = (fft1.real**2 + fft1.imag**2)
            p2   = (fft2.real**2 + fft2.imag**2)
            prod = (fft1 * fft2.conj()).real
    
            frc_num = torch.zeros(Rmax, device=device).scatter_add_(0, r_flat, prod.view(-1))
            frc_d1  = torch.zeros(Rmax, device=device).scatter_add_(0, r_flat, p1.view(-1))
            frc_d2  = torch.zeros(Rmax, device=device).scatter_add_(0, r_flat, p2.view(-1))
            frc     = frc_num / (torch.sqrt(frc_d1*frc_d2) + 1e-12)
            
            idx     = torch.where(frc < frc_threshold)[0]
    
            if len(idx) and idx[0] > 0:
                res_out[c]   = 1.0 / freqs[idx[0]]
    
        # ---------- sustituye NaN e Inf una sola vez ------------
        res_out = torch.nan_to_num(res_out, nan=fallback_res, posinf=fallback_res, neginf=fallback_res)
        return res_out  

    
    @torch.no_grad()
    def estimate_bfactor_batch(self, averages, pixel_size, res_cutoff, freq_min=0.05, min_points=5):
        N, H, W = averages.shape
        device = averages.device
    
        fft = torch.fft.fftshift(torch.fft.fft2(averages, norm="forward"), dim=(-2, -1))
        amplitude = torch.abs(fft)
    
        fy = torch.fft.fftfreq(H, d=pixel_size).to(device)
        fx = torch.fft.fftfreq(W, d=pixel_size).to(device)
        gy, gx = torch.meshgrid(fy, fx, indexing='ij')
        freq_r = torch.sqrt(gx ** 2 + gy ** 2)
    
        num_bins = 200
        freq_linspace = torch.linspace(0, freq_r.max(), num_bins + 1, device=device)
        freq_r_flat = freq_r.flatten()
        bin_idx = torch.bucketize(freq_r_flat, freq_linspace)
    
        amplitude_flat = amplitude.view(N, -1)
        radial_profile = torch.zeros(N, num_bins, device=device)
    
        for b in range(N):
            for i_bin in range(num_bins):
                mask_bin = (bin_idx == i_bin)
                if mask_bin.any():
                    radial_profile[b, i_bin] = torch.median(amplitude_flat[b, mask_bin])
    
        if torch.is_tensor(res_cutoff):
            if res_cutoff.numel() == 1:
                cutoff_freq = (1.0 / res_cutoff).repeat(N)
            elif res_cutoff.numel() == N:
                cutoff_freq = 1.0 / res_cutoff
            else:
                raise ValueError(f"res_cutoff debe ser escalar o tamaño {N}, tiene {res_cutoff.numel()}")
        else:
            cutoff_freq = torch.full((N,), 1.0 / float(res_cutoff), device=device)
    
        freq_centers = (freq_linspace[:-1] + freq_linspace[1:]) / 2
        freq_centers_exp = freq_centers.unsqueeze(0).expand(N, -1)
        cutoff_freq_exp = cutoff_freq.unsqueeze(1).expand(-1, freq_centers.size(0))
    
        valid_mask = (freq_centers_exp > freq_min) & (freq_centers_exp <= cutoff_freq_exp)
        amplitude_threshold = 1e-6
        valid_mask &= (radial_profile > amplitude_threshold)
    
        x = freq_centers_exp ** 2
        y = torch.log(radial_profile + 1e-10)
    
        b_factors = torch.full((N,), float('nan'), device=device)
    
        for i in range(N):
            xi = x[i][valid_mask[i]]
            yi = y[i][valid_mask[i]]
    
            if xi.numel() < min_points or torch.any(torch.isnan(yi)) or torch.any(torch.isinf(yi)):
                continue
    
            mean_x = xi.mean()
            mean_y = yi.mean()
            mean_xy = (xi * yi).mean()
            mean_x2 = (xi ** 2).mean()
    
            denom = mean_x2 - mean_x ** 2
            if abs(denom) < 1e-12:
                continue
    
            slope = (mean_xy - mean_x * mean_y) / denom
            b_factors[i] = -4 * slope
    
        b_factors = torch.nan_to_num(b_factors, nan=0.0, posinf=0.0, neginf=0.0)
        return b_factors

    @torch.no_grad()
    def sharpen_averages_batch_nq(self, averages, pixel_size, B_factors, eps=1e-6, normalize: bool = True):
        N, H, W = averages.shape
        device = averages.device
    
        # Limpieza de valores inválidos en B
        B_factors = torch.nan_to_num(B_factors, nan=0.0, posinf=0.0, neginf=0.0)
        B_exp = B_factors.view(N, 1, 1).clamp(min=-400.0, max=0.0)
    
        # FFT
        fft = torch.fft.fft2(averages, norm="forward")
    
        # Malla de frecuencias
        fy = torch.fft.fftfreq(H, d=pixel_size).to(device)
        fx = torch.fft.fftfreq(W, d=pixel_size).to(device)
        gy, gx = torch.meshgrid(fy, fx, indexing='ij')
        freq_r = torch.sqrt(gx**2 + gy**2).unsqueeze(0).expand(N, -1, -1)
    
        # --- B-factor completo ---
        filt = torch.exp((-B_exp / 4) * (freq_r ** 2))  # [N, H, W]
    
        # Aplicación del filtro
        fft_sharp = fft * filt
        sharp_imgs = torch.fft.ifft2(fft_sharp, norm="forward").real
    
        # Restaurar nivel de grises
        if normalize:
            mean_orig = averages.mean(dim=(-2, -1), keepdim=True)
            std_orig = averages.std(dim=(-2, -1), keepdim=True)
            mean_filt = sharp_imgs.mean(dim=(-2, -1), keepdim=True)
            std_filt = sharp_imgs.std(dim=(-2, -1), keepdim=True)
            sharp_imgs = (sharp_imgs - mean_filt) / (std_filt + eps) * std_orig + mean_orig
    
        return sharp_imgs
 

    
    @torch.no_grad()
    def sharpen_averages_batch(self, averages, pixel_size, B_factors, res_cutoffs, eps=1e-6, normalize: bool = True):
        N, H, W = averages.shape
        device = averages.device
        
        def create_taper(freq_r, f_cutoff, v0=0.0, vc=1.0):
            f_cutoff_exp = f_cutoff.expand_as(freq_r)
            taper = torch.zeros_like(freq_r)
        
            a = 0.5 * (v0 + vc)
            b = 0.5 * (vc - v0)
        
            mask = (freq_r >= 0) & (freq_r <= f_cutoff_exp)
            taper[mask] = a - b * torch.cos(torch.pi * freq_r[mask] / f_cutoff_exp[mask])
            taper[freq_r > f_cutoff_exp] = 0.0
        
            return taper
    
        B_factors = torch.nan_to_num(B_factors, nan=0.0, posinf=0.0, neginf=0.0)
        B_exp = B_factors.unsqueeze(1).unsqueeze(2).clamp(min=-300.0, max=100.0)
    
        # FFT
        fft = torch.fft.fft2(averages, norm="forward")
    
        # Malla de frecuencias
        fy = torch.fft.fftfreq(H, d=pixel_size).to(device)
        fx = torch.fft.fftfreq(W, d=pixel_size).to(device)
        gy, gx = torch.meshgrid(fy, fx, indexing='ij')
        freq_r = torch.sqrt(gx**2 + gy**2).unsqueeze(0).expand(N, -1, -1)
    
        # Frecuencia de corte y Nyquist
        f_cutoff = (1.0 / res_cutoffs).unsqueeze(1).unsqueeze(2)  # [N,1,1]
        f_nyquist = 1.0 / (2.0 * pixel_size)
        

        #Transición coseno  
        taper = create_taper(freq_r, f_cutoff, v0=0.8, vc=1.0)
        f_cutoff_exp = f_cutoff.expand_as(freq_r)
        # taper = torch.ones_like(freq_r) 
        # taper[freq_r > f_cutoff_exp] = 0.0  
        
        # taper = (freq_r <= f_cutoff_exp).float()                
    
        # Filtro de realce con B y taper hasta f_cutoff
        #Para hacer sharp hay que poner (-B_exp / 4) 
        filt = torch.exp((-B_exp / 4) * (freq_r ** 2)) * taper 
    
        fft_sharp = fft * filt
        fft_sharp = torch.where(freq_r <= f_cutoff_exp, fft_sharp, fft)
        sharp_imgs = torch.fft.ifft2(fft_sharp, norm="forward").real
        
        if normalize:
            mean_orig = averages.mean(dim=(-2, -1), keepdim=True)
            std_orig = averages.std(dim=(-2, -1), keepdim=True)
            # Restaurar nivel de grises
            mean_filt = sharp_imgs.mean(dim=(-2, -1), keepdim=True)
            std_filt = sharp_imgs.std(dim=(-2, -1), keepdim=True)
            sharp_imgs = (sharp_imgs - mean_filt) / (std_filt + eps) * std_orig + mean_orig
    
        return sharp_imgs
    
    
    @torch.no_grad()
    def enhance_averages_butterworth_combined(
        self,
        averages: torch.Tensor,            # [B, H, W]
        resolutions: torch.Tensor,         # [B]
        pixel_size: float,                 # Å/pixel
        low_res_angstrom: float = 20.0,
        order: int = 2,
        blend_factor: float = 0.5,
        normalize: bool = True
    ) -> torch.Tensor:
        """
        Aplica paso bajo (por FRC) al original y realce sobre ese resultado,
        combinando ambos en espacio real.
        """
        device = averages.device
        B, H, W = averages.shape
        eps = 1e-8
    
        # === Malla radial ===
        yy, xx = torch.meshgrid(
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing='ij'
        )
        r = torch.sqrt((xx - W // 2) ** 2 + (yy - H // 2) ** 2)
        r_norm = r / r.max()
        r_norm_exp = r_norm.unsqueeze(0)  # [1, H, W]
    
        # === Frecuencias normalizadas ===
        nyquist = 1.0 / (2.0 * pixel_size)
        MAX_CUTOFF = 0.475
    
        # -- Paso bajo individual (por resolución FRC) --
        res_clamped = torch.clamp(resolutions, max=25.0)
        frc_cutoffs = (1.0 / res_clamped.clamp(min=1e-3)) / nyquist / 2
        frc_cutoffs = torch.clamp(frc_cutoffs, 0.0, MAX_CUTOFF).view(B, 1, 1)
        lp_filter = 1.0 / (1.0 + (r_norm_exp / (frc_cutoffs + eps)) ** (2 * order))  # [B, H, W]
    
        # -- Pasa-banda global --
        low_cutoff = (1.0 / low_res_angstrom) / nyquist / 2
        high_cutoff = (1.0 / ((2 * pixel_size) / 0.95)) / nyquist / 2
    
        low_cutoff = min(max(0.0, low_cutoff), MAX_CUTOFF)
        high_cutoff = min(max(0.0, high_cutoff), MAX_CUTOFF)
    
        low = 1.0 / (1.0 + (r_norm / (low_cutoff + eps)) ** (2 * order))
        high = 1.0 / (1.0 + (high_cutoff / (r_norm + eps)) ** (2 * order))
        bp_filter = (low * high).unsqueeze(0)  # [1, H, W]
    
        # === FFT única del original ===
        fft = torch.fft.fft2(averages)
        fft_shift = torch.fft.fftshift(fft, dim=(-2, -1))
    
        # Filtro paso bajo
        fft_lp = fft_shift * lp_filter
        lowpass = torch.fft.ifft2(torch.fft.ifftshift(fft_lp, dim=(-2, -1))).real
    
        # Filtro realce (pasa banda aplicado sobre el paso bajo en freq)
        fft_enhanced = fft_lp * bp_filter
        
        enhanced = torch.fft.ifft2(torch.fft.ifftshift(fft_enhanced, dim=(-2, -1)), norm="forward").real
    
        # === Normalización (si se desea) ===
        if normalize:
            mean = averages.mean(dim=(-2, -1), keepdim=True)
            std  = averages.std(dim=(-2, -1), keepdim=True)
    
            def norm(x):
                return (x - x.mean(dim=(-2, -1), keepdim=True)) / (x.std(dim=(-2, -1), keepdim=True) + eps) * std + mean
    
            lowpass = norm(lowpass)
            enhanced = norm(enhanced)
    
        # === Mezcla final ===
        return blend_factor * lowpass + (1 - blend_factor) * enhanced
    
    
    @torch.no_grad()
    def enhance_averages_butterworth_combined_cos(
        self,
        averages: torch.Tensor,            # [B, H, W]
        resolutions: torch.Tensor,         # [B]
        pixel_size: float,                 # Å/pixel
        order: int = 2,
        blend_factor: float = 0.5,
        normalize: bool = True
    ) -> torch.Tensor:
        """
        Aplica paso bajo (por FRC) al original y realce sobre ese resultado,
        combinando ambos en espacio real. El filtro de realce sube con forma de coseno
        desde 0 hasta 1 en la frecuencia de corte (basada en resolución) y se mantiene en 1 después.
        """
        device = averages.device
        B, H, W = averages.shape
        eps = 1e-8
    
        # === Malla radial normalizada ===
        yy, xx = torch.meshgrid(
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing='ij'
        )
        r = torch.sqrt((xx - W // 2) ** 2 + (yy - H // 2) ** 2)
        r_norm = r / r.max()
        r_norm_exp = r_norm.unsqueeze(0).expand(B, -1, -1)
    
        # === Frecuencias normalizadas ===
        nyquist = 1.0 / (2.0 * pixel_size)
        MAX_CUTOFF = 0.475
    
        # --- Paso bajo por resolución ---
        res_clamped = torch.clamp(resolutions, max=25.0)
        frc_cutoffs = (1.0 / res_clamped.clamp(min=1e-3)) / nyquist / 2
        frc_cutoffs = torch.clamp(frc_cutoffs, 0.0, MAX_CUTOFF).view(B, 1, 1)
    
        # Butterworth paso bajo
        lp_filter = 1.0 / (1.0 + (r_norm_exp / (frc_cutoffs + eps)) ** (2 * order))  # [B, H, W]
    
        # --- Filtro de realce tipo coseno ---
        enhance_filter = torch.where(
            r_norm_exp <= frc_cutoffs,
            0.5 * (1 - torch.cos(torch.pi * r_norm_exp / (frc_cutoffs + eps))),
            torch.ones_like(r_norm_exp)
        )
        bp_filter = enhance_filter  # [B, H, W]
    
        # === FFT del original ===
        fft = torch.fft.fft2(averages, norm="forward")
        fft_shift = torch.fft.fftshift(fft, dim=(-2, -1))
    
        # Paso bajo y realce en Fourier
        fft_lp = fft_shift * lp_filter
        fft_enhanced = fft_lp * bp_filter
    
        # === IFFT individuales ===
        lowpass = torch.fft.ifft2(torch.fft.ifftshift(fft_lp, dim=(-2, -1)), norm="forward").real
        enhanced = torch.fft.ifft2(torch.fft.ifftshift(fft_enhanced, dim=(-2, -1)), norm="forward").real
    
        # === Normalización (si se desea) ===
        if normalize:
            mean = averages.mean(dim=(-2, -1), keepdim=True)
            std = averages.std(dim=(-2, -1), keepdim=True)
    
            def norm(x):
                return (x - x.mean(dim=(-2, -1), keepdim=True)) / (x.std(dim=(-2, -1), keepdim=True) + eps) * std + mean
    
            lowpass = norm(lowpass)
            enhanced = norm(enhanced)
    
        # === Mezcla final en espacio real ===
        return blend_factor * lowpass + (1 - blend_factor) * enhanced
    
    
    @torch.no_grad()
    def enhance_averages_attenuate_lowfrequencies(
        self,
        averages: torch.Tensor,            # [B, H, W]
        resolutions: torch.Tensor,         # [B] resoluciones FRC por clase
        pixel_size: float,                 # Å/pixel
        low_res_angstrom: float = 15.0,    # corte para bajas frecuencias a atenuar
        order: int = 2,                    # orden del filtro
        blend_factor: float = 0.5,         # mezcla con original
        normalize: bool = True             # conservar contraste
    ) -> torch.Tensor:
        """
        1. Aplica paso bajo según FRC (limitado a 25 Å).
        2. Atenúa bajas frecuencias (<20 Å) restando componente filtrado.
        """
        device = averages.device
        B, H, W = averages.shape
        eps = 1e-8
    
        # Malla radial normalizada
        yy, xx = torch.meshgrid(
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing='ij'
        )
        r = torch.sqrt((xx - W // 2) ** 2 + (yy - H // 2) ** 2)
        r_norm = r / r.max()                          # [H, W]
        r_norm_exp = r_norm.unsqueeze(0)              # [1, H, W]
    
        # --- Frecuencia de Nyquist ---
        nyquist = 1.0 / (2.0 * pixel_size)
        MAX_CUTOFF = 0.475
    
        # === Paso 1: Filtro pasa-bajo por clase (FRC) ===
        res_clamped = torch.clamp(resolutions, max=25.0)  # [B]
        frc_cutoffs = (1.0 / res_clamped.clamp(min=1e-3)) / nyquist / 2  # [B]
        frc_cutoffs = torch.clamp(frc_cutoffs, 0.0, MAX_CUTOFF).view(B, 1, 1)
        lp_filter = 1.0 / (1.0 + (r_norm_exp / (frc_cutoffs + eps)) ** (2 * order))  # [B, H, W]
    
        # FFT y aplicar paso bajo (por clase)
        fft_avg = torch.fft.fft2(averages)
        fft_shift = torch.fft.fftshift(fft_avg, dim=(-2, -1))
        fft_lp = fft_shift * lp_filter
        fft_lp_unshift = torch.fft.ifftshift(fft_lp, dim=(-2, -1))
        filtered_lp = torch.fft.ifft2(fft_lp_unshift).real  # [B, H, W]
    
        # === Paso 2: Atenuar bajas frecuencias (por debajo de 20 Å) ===
        low_cutoff = (1.0 / low_res_angstrom) / nyquist / 2
        low_cutoff = min(low_cutoff, MAX_CUTOFF)
        low_filter = 1.0 / (1.0 + (r_norm / (low_cutoff + eps)) ** (2 * order))  # [H, W]
        low_filter = low_filter.unsqueeze(0)  # [1, H, W]
    
        fft_lp2 = torch.fft.fft2(filtered_lp)
        fft_lp2_shift = torch.fft.fftshift(fft_lp2, dim=(-2, -1))
        fft_lowfreq = fft_lp2_shift * low_filter
        fft_lowfreq_unshift = torch.fft.ifftshift(fft_lowfreq, dim=(-2, -1))
        low_component = torch.fft.ifft2(fft_lowfreq_unshift).real  # [B, H, W]
    
        # === Sustraer bajas frecuencias suavemente ===
        filtered = filtered_lp - blend_factor * low_component
    
        # === Normalización (opcional) ===
        if normalize:
            mean_orig = averages.mean(dim=(-2, -1), keepdim=True)
            std_orig  = averages.std(dim=(-2, -1), keepdim=True)
            mean_filt = filtered.mean(dim=(-2, -1), keepdim=True)
            std_filt  = filtered.std(dim=(-2, -1), keepdim=True)
            filtered  = (filtered - mean_filt) / (std_filt + eps) * std_orig + mean_orig
    
        return filtered
    
    @torch.no_grad()
    def enhance_averages_butterworth(self, 
        averages,
        pixel_size,
        # high_res_angstrom=4,
        low_res_angstrom=20,
        order=2,
        blend_factor=0.5,
        normalize=True
    ):
        """
        Realza altas frecuencias en imágenes promedio usando filtro Butterworth pasa-banda.
        
        Args:
            averages (Tensor): [B, H, W] tensor de clases promedio.
            sampling (float): tamaño del píxel (Å/píxel).
            low_res_angstrom (float): resolución baja (filtro corta debajo de esta).
            high_res_angstrom (float): resolución alta (filtro corta arriba de esta).
            order (int): orden del filtro Butterworth (mayor = transición más dura).
            blend_factor (float): entre 0.0 (solo filtro) y 1.0 (solo original).
            normalize (bool): si True, conserva media y desviación estándar original.
    
        Returns:
            Tensor: clases promedio mejoradas [B, H, W]
        """
        high_res_angstrom = (2 * pixel_size) / 0.95
        # low_res_angstrom = 10 * pixel_size
        # high_res_angstrom = 4
        
        device = averages.device
        B, H, W = averages.shape
        eps = 1e-8
    
        # 1. Construcción del filtro pasa-banda
        nyquist = 1.0 / (2.0 * pixel_size)
        low_cutoff = (1.0 / low_res_angstrom) / nyquist / 2
        high_cutoff = (1.0 / high_res_angstrom) / nyquist / 2
        MAX_CUTOFF = 0.475
        low_cutoff = max(0.0, min(low_cutoff, MAX_CUTOFF))
        high_cutoff = max(0.0, min(high_cutoff, MAX_CUTOFF))
    
        y, x = torch.meshgrid(
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing='ij'
        )
        r = torch.sqrt((x - W//2)**2 + (y - H//2)**2)
        r_norm = r / r.max()
    
        low = 1.0 / (1.0 + (r_norm / (low_cutoff + eps))**(2 * order))
        high = 1.0 / (1.0 + (high_cutoff / (r_norm + eps))**(2 * order))
 
        bp_filter = low * high  # pasa-banda
    
        # 2. Aplicar en espacio de Fourier
        fft_avg = torch.fft.fft2(averages)
        fft_shift = torch.fft.fftshift(fft_avg, dim=(-2, -1))
        fft_filtered = fft_shift * bp_filter  # aplica el filtro
        fft_unshift = torch.fft.ifftshift(fft_filtered, dim=(-2, -1))
        filtered = torch.fft.ifft2(fft_unshift).real
    
        # 3. Normalización para mantener contraste
        if normalize:
            mean_orig = averages.mean(dim=(-2, -1), keepdim=True)
            std_orig = averages.std(dim=(-2, -1), keepdim=True)
            mean_filt = filtered.mean(dim=(-2, -1), keepdim=True)
            std_filt = filtered.std(dim=(-2, -1), keepdim=True)
            filtered = (filtered - mean_filt) / (std_filt + eps) * std_orig + mean_orig
            
        # 4. Fusión con original (mezcla controlada)
        filtered = blend_factor * averages + (1.0 - blend_factor) * filtered
    
        return filtered
    
    @torch.no_grad()
    def highpass_butterworth_soft_batch(self,
        averages: torch.Tensor,         # [B, H, W]
        resolutions: torch.Tensor,      # [B] en Å
        pixel_size: float,              # tamaño del píxel en Å/pix
        order: int = 2,                 # orden del filtro
        a: float = 0.2,                 # mínimo valor en bajas frecuencias
        eps: float = 1e-8,
        normalize: bool = True
    ) -> torch.Tensor: 
        """
        Aplica un filtro paso alto tipo Butterworth suavizado a un batch de imágenes.
        F(f) = a + (1 - a) / (1 + (fc / (f + eps))^(2n))
    
        Cada imagen se filtra con su propia resolución como frecuencia de corte.
        """
        B, H, W = averages.shape
        device = averages.device
    
        # === Malla de frecuencias radiales ===
        fy = torch.fft.fftfreq(H, d=pixel_size).to(device)
        fx = torch.fft.fftfreq(W, d=pixel_size).to(device)
        gy, gx = torch.meshgrid(fy, fx, indexing='ij')
        freq_r = torch.sqrt(gx**2 + gy**2)  # [H, W]
        freq_r = freq_r.unsqueeze(0).expand(B, -1, -1)  # [B, H, W]
    
        # === Frecuencia de corte desde resolución ===
        f_cutoff = (1.0 / resolutions.clamp(min=1e-3)).view(B, 1, 1)  # [B, 1, 1]
    
        # === Filtro paso alto suavizado ===
        filt = a + (1 - a) / (1 + (f_cutoff / (freq_r + eps)) ** (2 * order))  # [B, H, W]
    
        # === FFT e inversión ===
        fft = torch.fft.fft2(averages, norm='forward')  # [B, H, W]
        fft_filt = fft * filt
        filtered = torch.fft.ifft2(fft_filt, norm='forward').real  # [B, H, W]
    
        # === Normalización opcional ===
        if normalize:
            mean_orig = averages.mean(dim=(-2, -1), keepdim=True)
            std_orig = averages.std(dim=(-2, -1), keepdim=True)
            mean_filt = filtered.mean(dim=(-2, -1), keepdim=True)
            std_filt = filtered.std(dim=(-2, -1), keepdim=True)
            filtered = (filtered - mean_filt) / (std_filt + eps) * std_orig + mean_orig
    
        return filtered
    
    
    @torch.no_grad()
    def enhance_averages_butterworth_adaptive(self, 
        averages,       
        frc_res,        
        pixel_size,       # Å/pix
        low_res_floor = 20.0,
        order = 2,
        blend_factor = 0.5,
        normalize = True
    ):
        """
        Aplica filtro Butterworth adaptativo por clase según resolución FRC.
        Si la resolución FRC es ≥ low_res_floor, no se aplica filtrado.
        """
        device = averages.device
        B, H, W = averages.shape
        eps = 1e-8
        nyquist = 1.0 / (2.0 * pixel_size)
    
        # === 1. Coordenadas radiales normalizadas en [0, 1]
        y, x = torch.meshgrid(
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing='ij'
        )
        r = torch.sqrt((x - W // 2) ** 2 + (y - H // 2) ** 2)
        r_norm = r / r.max()  # [H, W] en [0, 1]
        # r_norm = r / (min(H, W) / 2.0)
        r_norm = r_norm.unsqueeze(0).expand(B, -1, -1)  # [B, H, W]
    
        # === 2. Máscara de clases a procesar: sólo si FRC < low_res_floor
        apply_mask = (frc_res < low_res_floor) & torch.isfinite(frc_res) & (frc_res > 0)
        if not apply_mask.any():
            return averages.clone()
    
        # === 3. Configura las resoluciones de corte
        high_res = frc_res.clone()
        low_res = torch.full_like(high_res, low_res_floor)  # fijo en 20 Å
    
        # Para clases que no se procesan, valores dummy
        high_res[~apply_mask] = 1.0
        low_res[~apply_mask] = 1.0
    
        # === 4. Frecuencias normalizadas respecto al Nyquist (∈ [0, 0.5])
        f_low = (1.0 / low_res) / nyquist / 2.0
        f_high = (1.0 / high_res) / nyquist / 2.0
        MAX_CUTOFF = 0.475
        f_low  = torch.clamp(f_low,  min=0.0, max=MAX_CUTOFF)
        f_high = torch.clamp(f_high, min=0.0, max=MAX_CUTOFF)
        f_low = f_low.view(B, 1, 1)
        f_high = f_high.view(B, 1, 1)
    
        # === 5. Filtro Butterworth pasa banda
        low = 1.0 / (1.0 + (r_norm / (f_low + eps)) ** (2 * order))
        high = 1.0 / (1.0 + (f_high / (r_norm + eps)) ** (2 * order))
        bp = low * high  # [B, H, W]
    
        # === 6. FFT y aplicación de filtro
        fft = torch.fft.fft2(averages)
        fft_shifted = torch.fft.fftshift(fft, dim=(-2, -1))
        filtered_fft = fft_shifted * bp
        fft_unshifted = torch.fft.ifftshift(filtered_fft, dim=(-2, -1))
        filtered = torch.fft.ifft2(fft_unshifted).real
    
        # === 7. Normalización de contraste
        if normalize:
            mean_orig = averages.mean(dim=(-2, -1), keepdim=True)
            std_orig = averages.std(dim=(-2, -1), keepdim=True)
            mean_filt = filtered.mean(dim=(-2, -1), keepdim=True)
            std_filt = filtered.std(dim=(-2, -1), keepdim=True)
            filtered = (filtered - mean_filt) / (std_filt + eps) * std_orig + mean_orig
    
        # === 8. Fusión: sólo en clases seleccionadas
        output = averages.clone()
        output[apply_mask] = (
            blend_factor * averages[apply_mask] +
            (1.0 - blend_factor) * filtered[apply_mask]
        )
    
        return output



    def determine_batches(self, free_memory, dim):
        
        if free_memory <= 14: #test with 6Gb GPU
            if dim <= 64:
                expBatchSize = 30000 
                expBatchSize2 = 30000
                numFirstBatch = 1
            elif dim <= 128:
                expBatchSize = 6000 
                expBatchSize2 = 9000
                numFirstBatch = 5
            elif dim <= 256:
                expBatchSize = 1000 
                expBatchSize2 = 2000
                numFirstBatch = 20
                
        elif free_memory > 14 and free_memory < 22: #test with 15Gb GPU
            if dim <= 64:
                expBatchSize = 30000 
                expBatchSize2 = 50000
                numFirstBatch = 1
            elif dim <= 128:
                expBatchSize = 15000 
                # expBatchSize = 10000
                expBatchSize2 = 20000
                # numFirstBatch = 2
                numFirstBatch = 5
            elif dim <= 256:
                expBatchSize = 4000 
                expBatchSize2 = 5000
                numFirstBatch = 6  
                
        else: #test with 23Gb GPU
            if dim <= 64:
                expBatchSize = 30000 
                expBatchSize2 = 60000
                numFirstBatch = 1
            elif dim <= 128:
                expBatchSize = 30000 
                expBatchSize2 = 30000
                numFirstBatch = 1
            elif dim <= 256:
                expBatchSize = 6000 
                expBatchSize2 = 9000
                numFirstBatch = 5 
                
        return(expBatchSize, expBatchSize2, numFirstBatch)
    
       
    
    def determine_ROTandSHIFT(self, iter, mode, dim):
        
        maxShift_20 = round( (dim * 20)/100 )
        maxShift_20 = (maxShift_20//5)*5
        
        maxShift_15 = round( (dim * 15)/100 )
        maxShift_15 = (maxShift_15//4)*4
        
        if mode == "create_classes":
            #print("---Iter %s for creating classes---"%(iter+1))
            # if iter < 5:
            #     ang, shiftMove = (-180, 180, 6), (-maxShift, maxShift+4, 4)
            # elif iter < 8:
            #     ang, shiftMove = (-180, 180, 4), (-8, 10, 2)
            # elif iter < 11:
            #     ang, shiftMove = (-90, 92, 2), (-6, 8, 2)
            # elif iter < 14:
            #     ang, shiftMove = (-30, 31, 1), (-3, 4, 1)
            
            #print("---Iter %s for creating classes---"%(iter+1))
            # if iter < 5:
            #     ang, shiftMove = (-180, 180, 10), (-maxShift_20, maxShift_20+5, 5)
            #     # ang, shiftMove = (-180, 180, 10), (-maxShift_15, maxShift_15+4, 4)
            # elif iter < 10:
            #     ang, shiftMove = (-180, 180, 8), (-maxShift_15, maxShift_15+4, 4)
            # elif iter < 13:
            #     ang, shiftMove = (-180, 180, 6), (-12, 16, 4)
            # elif iter < 16:
            #     ang, shiftMove = (-180, 180, 4), (-8, 10, 2)
            # elif iter < 19:
            #     ang, shiftMove = (-90, 92, 2), (-6, 8, 2)
            # elif iter < 22:
            #     ang, shiftMove = (-30, 31, 1), (-3, 4, 1)   
            
            if iter < 4:
                ang, shiftMove = (-180, 180, 10), (-maxShift_20, maxShift_20+5, 5)
            elif iter < 7:
                ang, shiftMove = (-180, 180, 8), (-maxShift_15, maxShift_15+4, 4)
            elif iter < 10:
                ang, shiftMove = (-180, 180, 6), (-12, 16, 4)
            elif iter < 13:
                ang, shiftMove = (-180, 180, 4), (-8, 10, 2)
            elif iter < 18:
                ang, shiftMove = (-90, 92, 2), (-6, 8, 2)
            # elif iter < 19:
            #     ang, shiftMove = (-30, 31, 1), (-3, 4, 1)           
                
        else:
            #print("---Iter %s for align to classes---"%(iter+1))
            if iter < 1:
                ang, shiftMove = (-180, 180, 6), (-maxShift_15, maxShift_15+4, 4)
            elif iter < 2:
                ang, shiftMove = (-180, 180, 4), (-8, 10, 2)
            elif iter < 3:
                ang, shiftMove = (-90, 92, 2), (-6, 8, 2)
            # elif iter < 4:
            #     ang, shiftMove = (-30, 31, 1), (-3, 4, 1)
           
        vectorRot, vectorshift = self.setRotAndShift(ang, shiftMove)
        return (vectorRot, vectorshift)
    
    

   
    

    

  
    
    
