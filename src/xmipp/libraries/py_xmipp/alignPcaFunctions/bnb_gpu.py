#!/usr/bin/env python3
"""/***************************************************************************
 *
 * Authors:    Erney Ramirez-Aportela
 *
  ***************************************************************************/
"""
import numpy as np
import torch
import time
import torchvision.transforms.functional as T
import torch.nn.functional as F
import kornia


class BnBgpu:
    
    def __init__(self, nBand):

        self.nBand = nBand 
        
        torch.cuda.is_available()
        torch.cuda.current_device()
        self.cuda = torch.device('cuda')
        # self.cuda = torch.device('cpu')
    
    
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
     
    @torch.no_grad()
    def precShiftBand(self, ft, freq_band, grid_flat, coef, shift):
        fourier_band = self.selectFourierBands(ft, freq_band, coef)
        nRef = fourier_band[0].size(0)
        nShift = shift.size(0)
    
        band_shifted = [torch.zeros((nRef*nShift, coef[n]), device=self.cuda) for n in range(self.nBand)]
        ONE = torch.tensor(1, dtype=torch.float32, device=self.cuda)
    
        for n in range(self.nBand):
            # --- Filtro de fase ---
            angles = shift @ grid_flat[n]           # (nShift, coef[n])
            filter = torch.polar(ONE, angles)      # (nShift, coef[n])
    
            band_shifted_complex = fourier_band[n][None, :, :] * filter[:, None, :]  # (nShift, nRef, coef[n])
    
            # --- Concatenar real e imaginario ---
            band_shifted_flat = torch.cat(
                (band_shifted_complex.real, band_shifted_complex.imag), dim=2
            )  
            band_shifted_flat = band_shifted_flat.transpose(0, 1)
    
            band_shifted[n] = band_shifted_flat.reshape(nShift*nRef, coef[n])  # (nShift*nRef, 2*coef[n])
    
        return band_shifted

    
  
    def selectFourierBands(self, ft, freq_band, coef):

        dimFreq = freq_band.shape[1]

        fourier_band = [torch.zeros(int(coef[n]/2), dtype = ft.dtype, device = self.cuda) for n in range(self.nBand)]
        
        freq_band = freq_band.expand(ft.size(dim=0) ,freq_band.size(dim=0), freq_band.size(dim=1))
           
        for n in range(self.nBand):
            fourier_band[n] = ft[:,:,:dimFreq][freq_band == n].view(ft.size(dim=0),int(coef[n]/2))
           
        return fourier_band        
            
    
    @torch.no_grad()
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
    @torch.no_grad()
    def precalculate_projection(self, prjTensorCpu, freqBn, grid_flat, coef, cvecs, rot_tensor, shift):
        device = self.cuda
        prj = prjTensorCpu.to(device, dtype=torch.float32, non_blocking=True)
        N, H, W = prj.shape
    
        rot_tensor = torch.as_tensor(rot_tensor, device=device, dtype=torch.float32).flatten()
        num_angles = rot_tensor.numel()
    
        prj_exp = prj.unsqueeze(0).repeat(num_angles, 1, 1, 1)  # (num_angles, N, H, W)
        prj_exp = prj_exp.view(-1, 1, H, W)                     # (N*num_angles, 1, H, W)
    
        # Rotation matrix
        theta = rot_tensor * math.pi / 180.0  # convertir a radianes
        c = torch.cos(theta)
        s = torch.sin(theta)
        A = torch.zeros((num_angles, 2, 3), device=device)
        A[:,0,0] = c
        A[:,0,1] = -s
        A[:,1,0] = s
        A[:,1,1] = c
    
        A_exp = A.unsqueeze(1).repeat(1, N, 1, 1).view(-1, 2, 3)  # (N*num_angles, 2, 3)
    
        # Grid sampling
        grid = F.affine_grid(A_exp, prj_exp.size(), align_corners=False)
        prj_rot = F.grid_sample(prj_exp, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        prj_rot = prj_rot.squeeze(1)
    
        del prj_exp, A_exp, grid
    
        # FFT y shift
        rotFFT = torch.fft.rfft2(prj_rot, norm="forward")
        shift_tensor = torch.as_tensor(shift, device=device, dtype=torch.float32)
        band_shifted = self.precShiftBand(rotFFT, freqBn, grid_flat, coef, shift_tensor)
        projBatch = self.phiProjRefs(band_shifted, cvecs)
    
        del prj_rot, rotFFT, band_shifted
        return projBatch
    
    
    @torch.no_grad()
    def create_batchExp(self, Texp, freqBn, coef, vecs):
             
        self.batch_projExp = [torch.zeros((Texp.size(dim=0), vecs[n].size(dim=1)), device = self.cuda) for n in range(self.nBand)]
        expFFT = torch.fft.rfft2(Texp, norm="forward")
        del(Texp)
        bandExp = self.selectBandsRefs(expFFT, freqBn, coef)
        self.batch_projExp = self.phiProjRefs(bandExp, vecs)
        del(expFFT , bandExp)
        
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
        
        iter_matches = torch.cat((exp, sel.view(nExp,1), min_score.view(nExp,1), 
                                  rotation, shift_location.view(nExp,1)), dim=1)  
        
        cond = iter_matches[:, 2] < matches[initBatch:initBatch + nExp, 2]
        matches[initBatch:initBatch + nExp] = torch.where(cond.view(nExp, 1), iter_matches, matches[initBatch:initBatch + nExp])       
        # torch.cuda.empty_cache()              
        return(matches)
    
    
    @torch.no_grad()
    def match_batch_initVol(self, batchExp, batchRef, initBatch, matches, rot, nShift):
        
        nExp = int(batchExp[0].size(dim=0) / nShift)
        nShift = torch.tensor(nShift, device=self.cuda)       
                             
        for n in range(self.nBand):
            score = torch.cdist(batchRef[n], batchExp[n])
            
        num_rows, num_cols = score.shape    
        
        num_blocks = score.shape[1] // nShift
        blocks = score.view(num_rows, num_blocks, nShift)
        
        block_min_values, block_min_indices_rows = blocks.min(dim=0)
        block_min_value, block_min_indices_cols = block_min_values.min(dim=1)
        block_min_indices_rows = block_min_indices_rows.gather(1, block_min_indices_cols.unsqueeze(1)).squeeze()  
        del(score)
           
        exp = torch.arange(initBatch, initBatch+nExp, 1, device = self.cuda).view(nExp,1) 
        sel = block_min_indices_rows.type(torch.int64) 
        rotation = torch.full((nExp,1), rot, device = self.cuda) 
        shift_location = (block_min_indices_cols).type(torch.int64)
        
        iter_matches = torch.cat((exp, sel.view(nExp,1), block_min_value.view(nExp,1), 
                          rotation, shift_location.view(nExp,1)), dim=1)      


        cond = iter_matches[:, 2] < matches[initBatch:initBatch + nExp, 2]
        matches[initBatch:initBatch + nExp] = torch.where(cond.view(nExp, 1), iter_matches, matches[initBatch:initBatch + nExp])       
        # torch.cuda.empty_cache()              
        return(matches)
    
    
    @torch.no_grad()
    def match_batch_with_class(self, match_list):
        
        # N = match_list[0].shape[0]
        stacked_tensores = torch.stack(match_list) 
        N = stacked_tensores.shape[1] 

        min_values, min_indices = torch.min(stacked_tensores[:, :, 2], dim=0)        
        resultado = stacked_tensores[min_indices, torch.arange(N)]  
        match_minScore = torch.cat((resultado, min_indices.unsqueeze(1)), dim=1)
        
        return(match_minScore)
    
    
    @torch.no_grad()
    def match_batch_label_minScore(self, tensor):
        col2 = tensor[:, 1]  
        col3 = tensor[:, 2]  
        
        _, indices_col3 = torch.sort(col3, stable=True)
        tensor_sorted_col3 = tensor[indices_col3]
        
        _, indices_col2 = torch.sort(tensor_sorted_col3[:, 1], stable=True)
        sorted_tensor = tensor_sorted_col3[indices_col2]
    
        col2_sorted = sorted_tensor[:, 1]
        is_first_occurrence = torch.cat(
            [torch.tensor([True], device=tensor.device), col2_sorted[1:] != col2_sorted[:-1]]
        )
        
        indicator_column = torch.zeros(tensor.size(0), dtype=torch.int, device=tensor.device)
        indicator_column[indices_col3[indices_col2[is_first_occurrence.nonzero(as_tuple=True)[0]]]] = 1
    
        tensor_with_indicator = torch.cat((tensor, indicator_column.unsqueeze(1)), dim=1)
        
        return tensor_with_indicator
        
       
    
    def center_shifts(self, Texp, initBatch, expBatchSize, prev_shifts):
        batchsize = Texp.shape[0]
        translations = prev_shifts[initBatch:initBatch+expBatchSize]
        TexpView = Texp.view(batchsize, 1, Texp.shape[1], Texp.data.shape[2])
        transforTexp = kornia.geometry.translate(TexpView, translations)
        transforTexp = transforTexp.view(batchsize, Texp.shape[1], Texp.data.shape[2])
        del(Texp)
        del(TexpView)
        del(translations)     
        return(transforTexp)
    
    
    def center_rot(self, Texp, initBatch, expBatchSize, prevPosition):
        batchsize = Texp.shape[0]
        rotation = -prevPosition[initBatch:initBatch+expBatchSize, 0]
        TexpView = Texp.view(batchsize, 1, Texp.shape[1], Texp.data.shape[2])
        transforTexp = kornia.geometry.rotate(TexpView, rotation)
        transforTexp = transforTexp.view(batchsize, Texp.shape[1], Texp.data.shape[2])
        del(Texp)
        del(TexpView)
        del(rotation)
        return(transforTexp)
    
    
    @torch.no_grad()
    def center_particles_inverse(self, Texp, initBatch, expBatchSize, prevPosition):
        
        dim = Texp.size(dim=1)
        rotBatch = -prevPosition[initBatch:initBatch+expBatchSize, 0]   
        batchsize = rotBatch.size(dim=0)
        rotBatch = rotBatch.view(batchsize)  
        translations = prevPosition[initBatch:initBatch+expBatchSize,1:].view(batchsize,2,1)
        
        centerxy = torch.tensor([dim/2,dim/2], device = self.cuda) 
        center = torch.ones(batchsize, 2, device=self.cuda) * centerxy       
        scale = torch.ones(batchsize, 2, device = self.cuda)
        
        #translation matrix
        translation_matrix = torch.eye(3, device=self.cuda).unsqueeze(0).repeat(batchsize, 1, 1)
        translation_matrix[:, :2, 2] = translations.squeeze(-1)
        
        #rotation matrix
        rotation_matrix = kornia.geometry.get_rotation_matrix2d(center, rotBatch, scale)        
        M = torch.bmm(rotation_matrix, translation_matrix)
          
        transforTexp = kornia.geometry.warp_affine(Texp.view(batchsize, 1, dim, dim), M, dsize=(dim, dim), mode='bilinear')
        transforTexp = transforTexp.view(batchsize, dim, dim)
        del(Texp)
        
        return(transforTexp)
        
       
    def create_mask(self, images, rad):

        dim = images.size(dim=1)
        center = dim // 2
        y, x = torch.meshgrid(torch.arange(dim) - center, torch.arange(dim) - center, indexing='ij')
        dist = torch.sqrt(x**2 + y**2).to(images.device)

        self.mask = dist <= rad
        self.mask = self.mask.float() 
        
        return self.mask
    
    
    def create_gaussian_mask(self, images, sigma):
        dim = images.size(dim=1)
        center = dim // 2
        y, x = torch.meshgrid(torch.arange(dim) - center, torch.arange(dim) - center, indexing='ij')
        dist = torch.sqrt(x**2 + y**2).float().to(images.device)  
    
        sigma2 = sigma**2
        K = 1. / (torch.sqrt(2 * torch.tensor(np.pi)) * sigma)**2
    
        self.mask = K * torch.exp(-0.5 * (dist**2 / sigma2))
        self.mask = self.mask / self.mask[center, center].clone()
    
        return self.mask 
            

    
    
