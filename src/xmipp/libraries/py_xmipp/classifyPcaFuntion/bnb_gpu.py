#!/usr/bin/env python3
"""/***************************************************************************
 *
 * Authors:    Erney Ramirez-Aportela
 *
  ***************************************************************************/
"""
import numpy as np
import torch
import torch.nn.functional as F
import kornia
import random
import math



class BnBgpu:
    
    def __init__(self, nBand):

        self.nBand = nBand 
        
        torch.cuda.is_available()
        torch.cuda.current_device()
        self.cuda = torch.device('cuda:0')  
    
    
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

    
    @torch.no_grad()
    def selectFourierBands(self, ft, freq_band, coef):

        dimFreq = freq_band.shape[1]

        fourier_band = [torch.zeros(int(coef[n]/2), dtype = ft.dtype, device = self.cuda) for n in range(self.nBand)]
        
        freq_band = freq_band.expand(ft.size(dim=0) ,freq_band.size(dim=0), freq_band.size(dim=1))
           
        for n in range(self.nBand):
            fourier_band[n] = ft[:,:,:dimFreq][freq_band == n]
            fourier_band[n] = fourier_band[n].reshape(ft.size(dim=0),int(coef[n]/2)) 
                      
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
    
    
    @torch.no_grad()
    def match_batch(self, batchExp, batchRef, initBatch, matches, rot, nShift):
    
        nExp = batchExp[0].size(0)
        nShift = int(nShift)
    
        for n in range(self.nBand):
            score = torch.cdist(batchRef[n], batchExp[n])
    
        min_score, ref = score.min(dim=0)
        del score
    
        target = matches[initBatch:initBatch + nExp]
        improved = min_score < target[:, 2]
    
        if improved.any():
            br = ref[improved]
    
            target[improved, 2] = min_score[improved]
            target[improved, 1] = br.div(nShift, rounding_mode='floor').to(target.dtype)
            target[improved, 4] = br.remainder(nShift).to(target.dtype)
            target[improved, 3] = rot
    
        exp_idx = torch.arange(
            initBatch, initBatch + nExp,
            device=target.device,
            dtype=target.dtype
        )
        target[improved, 0] = exp_idx[improved]
    
        return matches
    
    
    @torch.no_grad()
    def batchExpToCpu(self, Timage, freqBn, coef, cvecs):        

        self.create_batchExp(Timage, freqBn, coef, cvecs)        
        self.batch_projExp = torch.stack(self.batch_projExp)
        batch_projExp_cpu = self.batch_projExp.to("cpu")
        
        return(batch_projExp_cpu)
    
    
    @torch.no_grad()
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
        
    
    @torch.no_grad()
    def get_robust_zscore_thresholds(self, classes, matches, threshold=2.0, bins=200):
        
        thr_low = torch.full((classes,), float('-inf'), device=matches.device)
        thr_high = torch.full((classes,), float('inf'), device=matches.device)
    
        for n in range(classes):
            class_scores = matches[matches[:, 1] == n, 2]
            if class_scores.numel() > 2:
                vals = class_scores.cpu().numpy()
                hist, edges = np.histogram(vals, bins=bins)
                mode_idx = hist.argmax()
    
                # valores dentro del bin de la moda
                bin_mask = (vals >= edges[mode_idx]) & (vals < edges[mode_idx + 1])
                bin_values = class_scores[torch.tensor(bin_mask, device=class_scores.device)]
    
                # centro = mediana dentro del bin de la moda
                if len(bin_values) > 0:
                    center = bin_values.median()
                else:
                    center = torch.tensor((edges[mode_idx] + edges[mode_idx+1]) / 2.0, device=class_scores.device)
    
                # MAD simétrica
                mad = torch.median(torch.abs(class_scores - center)) + 1e-8
    
                thr_low[n] = center - threshold * mad
                thr_high[n] = center + threshold * mad
    
        return thr_low, thr_high
    
    
    @torch.no_grad()
    def create_classes(self, mmap, tMatrix, iter, nExp, expBatchSize, matches, vectorshift, classes, final_classes, freqBn, coef, cvecs, mask, sigma, sampling, cycles):
        
        # print("----------create-classes-------------") 
        iterSplit = 7       
            
        if iter == 2: 
            split = (final_classes - classes) * 60 // 100
        elif 3 <= iter < iterSplit and final_classes > classes:
            split = final_classes - classes
        else:
            split = 0
        
        # --- Z-score thresholds por clase ---    
        if iter < 10: # Filters the best-scoring particles
            thr_low, thr_high = self.get_robust_zscore_thresholds(classes, matches)
            
        total_slots = classes + split
        newCL = [[] for i in range(total_slots)]
        newProj = [[] for i in range(total_slots)]

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
                        
            transforIm, tMatrix[initBatch:endBatch] = self.center_particles_inverse_save_matrix(mmap.data[initBatch:endBatch], tMatrix[initBatch:endBatch], 
                                                                             rotBatch[initBatch:endBatch], translations[initBatch:endBatch], centerxy)
            
   
            if mask:
                sigma_gauss = (0.75*sigma) if (iter < 10 and iter % 2 == 1) else (sigma)# if iter < 10 else sigma

                transforIm = transforIm * self.create_gaussian_mask(transforIm, sigma_gauss)
            else:
                transforIm = transforIm * self.create_circular_mask(transforIm)
                

            proj_batch = self.batchExpToCpu(transforIm, freqBn, coef, cvecs)
            batch_projExp_cpu[count] = proj_batch
            count+=1

            #Create classes for batches
            
            batch_class_indices = matches[initBatch:endBatch, 1].to(self.cuda, non_blocking=True).long()
            batch_scores = matches[initBatch:endBatch, 2].to(self.cuda, non_blocking=True)
            projs_gpu = proj_batch.to(self.cuda, non_blocking=True)[0]
            
            # ---------- ZSCORE FILTER ----------
            if iter < 10:
                low  = thr_low[batch_class_indices]
                high = thr_high[batch_class_indices]
                valid_mask = (batch_scores >= low) & (batch_scores <= high)
            else:
                valid_mask = torch.ones_like(batch_scores, dtype=torch.bool)
    
            if valid_mask.sum() == 0:
                del transforIm
                continue
    
            transforIm = transforIm[valid_mask]
            batch_class_indices = batch_class_indices[valid_mask]
            projs_gpu = projs_gpu[valid_mask]
                
            labels = batch_class_indices
            order = torch.argsort(labels)
            
            imgs_sorted = transforIm[order]
            projs_sorted = projs_gpu[order] 
            lbls_sorted = batch_class_indices[order]
            
            counts = torch.bincount(lbls_sorted, minlength=classes)
            
            start = 0
            for n, c in enumerate(counts):
                if c > 0:
                    newCL[n].append(imgs_sorted[start:start+c])
                    newProj[n].append(projs_sorted[start:start+c])
                start += c
            
            del(transforIm)

        # Concatenación Global
        _, H, W = mmap.data.shape
        newCL = [torch.cat(l, dim=0) if len(l) > 0 else torch.empty((0,H,W), device=self.cuda) for l in newCL]
        newProj = [torch.cat(l, dim=0) if len(l) > 0 else None for l in newProj]

        #  Structural Split  
        if 2 <= iter < iterSplit and split > 0:
            for n in range(classes):
                if newCL[n].shape[0] > 20: # Minimum particles to consider
                    
                    #K-means over PCA proj
                    _, labels = self.kmeans_pytorch_for_averages(newCL[n], newProj[n], cvecs, num_clusters=2, num_iters=15)
                    
                    part_A = newCL[n][labels == 0]
                    part_B = newCL[n][labels == 1]
                    
                    if n < split:
                        newCL[n] = part_A
                        newCL[n + classes] = part_B
                    else:
                        newCL[n] = torch.cat([part_A, part_B], dim=0)
        
        
        clk = self.averages_createClasses(mmap, iter, newCL)       

        if iter > 1:
            # cut = (25 if iter < 5 else 20) if sampling < 3 else (35 if iter < 5 else 30)
            # cut_res = 100 if iter < (iterSplit-1) else 50
            cut=50
            cut_res = 50           
            res_classes = self.frc_resolution_tensor(newCL, sampling, fallback_res=cut_res, rcut=cut)
            clk = self.gaussian_lowpass_filter_2D_adaptive(clk, res_classes, sampling)
            
            boost = None
            clk = self.highpass_cosine_sharpen(clk, res_classes, sampling, factorR = boost)
            
                
        if iter < (iterSplit + 1): #order by size
            
            lengths = torch.tensor([len(cls) for cls in newCL], device=clk.device)
            valid_mask = lengths > 0
            # res_classes = res_classes[valid_mask]
            sizes = lengths[valid_mask]
            clk = clk[valid_mask]
            # clk = clk[torch.argsort(res_classes)]
            clk = clk[torch.argsort(sizes, descending=True)]
            
            

        if iter in [10, 13]:
            clk = clk * self.contrast_dominant_mask(clk, window=3, contrast_percentile=80,
                                intensity_percentile=50, smooth_sigma=1.0)
        if 1 < iter < 7 and iter % 2 == 0:
            clk = clk * self.contrast_dominant_mask(clk, window=3, contrast_percentile=80,
                                intensity_percentile=50, smooth_sigma=1.0)

        
        if iter > 2 and iter < 12:
            for _ in range(2):
                clk = self.center_by_com(clk)  
        
        clk = clk * self.create_circular_mask(clk)                
        
        return(clk, tMatrix, batch_projExp_cpu)
    
    
    def align_particles_to_classes(self, data, cl, tMatrix, iter, expBatchSize, matches, vectorshift, classes, freqBn, coef, cvecs, mask, sigma, sampling):
        
        # print("----------align-to-classes-------------")
                
        #rotate and translations
        rotBatch = -matches[:,3].view(expBatchSize,1)
        translations = list(map(lambda i: vectorshift[i], matches[:, 4].int()))
        translations = torch.tensor(translations, device = self.cuda).view(expBatchSize,2)
        
        centerIm = data.shape[1]/2 
        centerxy = torch.tensor([centerIm,centerIm], device = self.cuda)
                            
        transforIm, tMatrix = self.center_particles_inverse_save_matrix(data, tMatrix, 
                                                                         rotBatch, translations, centerxy)
                
        del rotBatch,translations, centerxy 
        
        if mask:
            transforIm = transforIm * self.create_gaussian_mask(transforIm, sigma)
        else: 
            transforIm = transforIm * self.create_circular_mask(transforIm)
                               
    
        
        batch_projExp_cpu = self.create_batchExp(transforIm, freqBn, coef, cvecs)
        
        if iter == 2:
            newCL = [[] for i in range(classes)]              
            
            for n in range(classes):
                class_images = transforIm[matches[:, 1] == n]
                newCL[n].append(class_images)
                 
            del(transforIm)
            # torch.cuda.empty_cache()
            
            newCL = [torch.cat(class_images_list, dim=0) for class_images_list in newCL] 
            clk = self.averages(data, newCL, classes)          

            
            res_classes = self.frc_resolution_tensor(newCL, sampling)
            
            clk = self.gaussian_lowpass_filter_2D_adaptive(clk, res_classes, sampling)
            
            clk = self.highpass_cosine_sharpen(clk, res_classes, sampling)                       
        
            if not hasattr(self, 'grad_squared'):
                self.grad_squared = torch.zeros_like(cl)
            clk, self.grad_squared = self.update_classes_rmsprop(cl, clk, 0.001, 0.9, 1e-8, self.grad_squared)        
                      
            clk = clk * self.create_circular_mask(clk)
      
        else: 
            del(transforIm)
            # torch.cuda.empty_cache()
            clk = cl  
            
        return (clk, tMatrix, batch_projExp_cpu) 
    
           
    @torch.no_grad()
    def center_particles_inverse_save_matrix(self, data, tMatrix, update_rot, update_shifts, centerxy):
          
        batchsize, H, W = data.shape
            
        rotBatch = update_rot.view(-1)

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
        
        # -------- load data ----------
        Texp = torch.from_numpy(data.astype(np.float32)).to(self.cuda).unsqueeze(1)

        transforIm = kornia.geometry.warp_affine(Texp, M, dsize=(H, W), mode='bilinear', padding_mode='zeros')
        transforIm = transforIm.view(batchsize, H, W)
        del(Texp)
        
        return(transforIm, M)
    
    
    @torch.no_grad()
    def averages_createClasses(self, mmap, iter, newCL): 
        
        classes = len(newCL)       
  
        clk = []
        for n in range(classes):
            if len(newCL[n]) > 0:
                clk.append(torch.mean(newCL[n], dim=0))
            else:
                clk.append(torch.zeros((mmap.data.shape[1], mmap.data.shape[2]), device=newCL[0].device))
        clk = torch.stack(clk)
        return clk
    
    
    @torch.no_grad()
    def averages(self, data, newCL, classes): 
        
        clk = []
        for n in range(classes):
            if len(newCL[n]) > 0:
                clk.append(torch.mean(newCL[n], dim=0))
            else:
                clk.append(torch.zeros((data.shape[1], data.shape[2]), device=newCL[0].device))
        clk = torch.stack(clk)
        return clk
    
    
    @torch.no_grad()
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
    
    
    @torch.no_grad()
    def create_circular_mask(self, images):
        dim = images.size(dim=1)
        center = dim // 2
        y, x = torch.meshgrid(torch.arange(dim) - center, torch.arange(dim) - center, indexing='ij')
        dist = torch.sqrt(x**2 + y**2).float().to(images.device)
        
        # Creamos una máscara circular
        circular_mask = torch.zeros_like(dist)
        circular_mask[dist <= center] = 1.0
        
        return circular_mask
    
    
    @torch.no_grad()
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
         
    
    
    @torch.no_grad()
    def gaussian_lowpass_filter_2D_adaptive(self, imgs, res_angstrom, pixel_size,
                                            floor_res=100.0, clamp_exp=80.0,
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
        
        del fy, fx, gy, gx
    
        # === Filtro Gaussiano adaptativo
        ln2    = torch.log(torch.tensor(2.0, device=device))
        D0     = (1.0 / res_eff).view(B,1,1)                            
        sigma2 = (D0 / torch.sqrt(2*ln2))**2 
        
        # scale_factor = torch.tensor(H/128, device=device)  # referencia 128px
        # sigma2 = ((D0 / torch.sqrt(2*ln2)) * scale_factor)**2            
    
        exponent = (-freq2) / (2*sigma2 + eps)
        
        filt = torch.exp(exponent.clamp(max=clamp_exp))
    
        if hard_cut:
            filt = torch.where(freq2 > D0**2, 0.0, filt)
        
        del sigma2, exponent, freq2, D0
    
        # === Aplicar filtro y transformar inversa
        img_filt = torch.fft.ifft2(torch.fft.fft2(imgs, norm="forward") * filt, norm="forward").real
        img_filt = torch.nan_to_num(img_filt)
        del filt
    
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
            del mean0, std0, mean_f, std_f, valid
    
        return img_filt

    

    def update_classes_rmsprop(self, cl, clk, learning_rate, decay_rate, epsilon, grad_squared):
        
        grad = clk - cl
        
        grad_squared = decay_rate * grad_squared + (1 - decay_rate) * grad**2        
        update = learning_rate * grad / (torch.sqrt(grad_squared) + epsilon)
        cl = torch.add(cl, update)

        return cl, grad_squared
    
    
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
    
    
    @torch.no_grad()
    def frc_resolution_tensor(
            self,
            newCL,                       # [N_i,H,W]
            pixel_size: float,           # Å/px
            frc_threshold: float = 0.143,
            fallback_res: float = 100.0, #40.0,
            rcut: float = 100,
            apply_window: bool = False, #True,
            smooth: bool = True          
    ) -> torch.Tensor:
        """
        Devuelve tensor [n_classes] con la resolución FRC por clase (Å)
        y las curvas FRC por clase.
        """
        
        n_classes = len(newCL)
        h, w = next((imgs.shape[-2], imgs.shape[-1]) for imgs in newCL if imgs.numel() > 0)
        device    = newCL[0].device
        Rmax  = min(h, w) // 2
    
        res_out   = torch.full((n_classes,), float('nan'), device=device)
    
        # --- malla de frecuencias físicas (Å⁻¹) ---
        fy = torch.fft.fftfreq(h, d=pixel_size, device=device)
        fx = torch.fft.rfftfreq(w, d=pixel_size, device=device)
        gy, gx = torch.meshgrid(fy, fx, indexing="ij")
        r = torch.sqrt(gx**2 + gy**2)   
    
        # discretizar radios en bins
        freq_bins = torch.linspace(0, 0.5/pixel_size, Rmax, device=device)
        r_bin = torch.bucketize(r.flatten(), freq_bins) - 1
        r_bin = r_bin.clamp(0, Rmax-1)
        
        del fy, fx, gy, gx, r  
    
        # --- Ventana Hann (opcional) ---
        if apply_window:
            wy = torch.hann_window(h, periodic=False, device=device)
            wx = torch.hann_window(w, periodic=False, device=device)
            window = wy[:, None] * wx[None, :]
            window = window / window.norm() * (h*w)**0.5  # normalización
    
        for c, imgs in enumerate(newCL):
            n = imgs.shape[0]
            if n < 8:
                res_out[c] = 40.0
                continue  
    
            # ---- half maps ----
            perm          = torch.randperm(n, device=device)
            half1, half2  = torch.chunk(imgs[perm], 2, dim=0)
            avg1, avg2    = half1.mean(0), half2.mean(0)
    
            if apply_window:
                avg1 = avg1 * window
                avg2 = avg2 * window
                
    
            # ---- FFT ----
            fft1 = torch.fft.rfft2(avg1, norm="forward")
            fft2 = torch.fft.rfft2(avg2, norm="forward")
            
    
            p1   = (fft1.real**2 + fft1.imag**2)
            p2   = (fft2.real**2 + fft2.imag**2)
            prod = (fft1 * fft2.conj()).real
    
            # ---- FRC anular ----
            frc_num = torch.zeros(Rmax, device=device).scatter_add_(0, r_bin, prod.flatten())
            frc_d1  = torch.zeros(Rmax, device=device).scatter_add_(0, r_bin, p1.flatten())
            frc_d2  = torch.zeros(Rmax, device=device).scatter_add_(0, r_bin, p2.flatten())
            frc     = frc_num / (torch.sqrt(frc_d1*frc_d2) + 1e-12)
    
            if smooth:
                # suavizado con media móvil (kernel triangular 3 pts)
                kernel = torch.tensor([0.25, 0.5, 0.25], device=device).view(1, 1, -1)
                frc = torch.nn.functional.conv1d(frc.view(1,1,-1), kernel, padding=1).view(-1)
    
            # frc_curves[c] = frc
    
            # ---- resolución = cruce con threshold ----
            idx = torch.where(frc < frc_threshold)[0]
            if len(idx) and idx[0] > 0:
                res_out[c] = 1.0 / freq_bins[idx[0]]
    
        # ---- reemplazo de NaN/Inf por fallback ----
        nq_save = (2 * pixel_size) / 0.8
        res_out = torch.nan_to_num(res_out, nan=nq_save,
                                   posinf=nq_save, neginf=nq_save)
        
        res_out = torch.where(res_out > rcut, torch.tensor(fallback_res, device=res_out.device), res_out)
        
        del r_bin, freq_bins
        
        return res_out#, frc_curves, freq_bins

    
    @torch.no_grad()
    def highpass_cosine_sharpen(
        self,
        averages: torch.Tensor,         # [B, H, W]
        resolutions: torch.Tensor,      # [B] Å
        pixel_size: float,              # píxel(Å/pix)
        f_energy: float = 2.0,
        # R_high: float = 25.0,
        boost_max: float = None,        # si None, se ajusta para energía
        sharpen_power: float = None,    # si None, se ajusta automáticamente según resolución
        factorR: float = None,
        eps: float = 1e-8,
        normalize: bool = True,
        max_iter: int = 20
    ) -> torch.Tensor:
        B, H, W = averages.shape
        device = averages.device
    
        # === FFT + original energy ===
        fft = torch.fft.fft2(averages, norm='forward')
        fft_mag2 = torch.abs(fft) ** 2  # [B, H, W]
        energy_orig = torch.sum(fft_mag2, dim=(-2, -1))  # [B]
    
        # === Frecuencias radiales ===
        fy = torch.fft.fftfreq(H, d=pixel_size, device=device)
        fx = torch.fft.fftfreq(W, d=pixel_size, device=device)
        gy, gx = torch.meshgrid(fy, fx, indexing='ij')
        freq_r = torch.sqrt(gx**2 + gy**2).unsqueeze(0).expand(B, -1, -1)  # [B, H, W]
        del fy, fx, gy, gx
    
        # === Frecuencia de corte por imagen ===
        f_cutoff = (1.0 / resolutions.clamp(min=1e-3)).view(B, 1, 1)  # [B, 1, 1]
    
        # === Ajuste dinámico de sharpen_power por resolución ===
            # sharpen_power = (0.1 * resolutions).clamp(min=0.3, max=2.5)
            #sharpen_power = (0.08 * resolutions).clamp(min=0.3, max=2.5)
        if sharpen_power is None:
            # factorR = torch.where(resolutions > 8, 0.1, 0.08)
            if factorR is None:
                factorR = torch.where(
                    resolutions < 10,  torch.tensor(0.1, device=resolutions.device),
                    torch.where(resolutions < 14, torch.tensor(0.08, device=resolutions.device),
                                               torch.tensor(0.06, device=resolutions.device))
                )
            else:
                factorR = torch.as_tensor(factorR, device=resolutions.device, dtype=resolutions.dtype)
                factorR = factorR.expand_as(resolutions)

            sharpen_power = (factorR * resolutions).clamp(min=0.3, max=2.5)
  
            sharpen_power = sharpen_power.view(B, 1, 1)  # broadcasting por imagen
        else:
            # Modo fijo: mismo valor para todas las imágenes
            if not torch.is_tensor(sharpen_power):
                sharpen_power = torch.tensor(float(sharpen_power), device=device)
            sharpen_power = sharpen_power.view(1, 1, 1).expand(B, -1, -1)
    
        # === Filtro en forma de coseno ===
        cos_term = torch.pi * freq_r / (f_cutoff + eps)
        cosine_shape = ((1 - torch.cos(cos_term)) / 2).clamp(min=0.0, max=1.0)
        del cos_term

        cosine_shape = torch.where(freq_r <= f_cutoff, cosine_shape, torch.ones_like(freq_r))
        cosine_shape = cosine_shape ** sharpen_power  # [B, H, W]
    
        # === Ajuste automático de boost_max para duplicar energía ===
        if boost_max is None:
            
            target_energy = f_energy * energy_orig  # [B]
            g_low = torch.ones(B, device=device)
            g_high = torch.full((B,), 1000.0, device=device)  # límite arbitrario
            
            for _ in range(max_iter):
                g_mid = (g_low + g_high) / 2
                g_val = g_mid.view(B, 1, 1)
                energy_mid = torch.sum(fft_mag2 * (1.0 + (g_val - 1.0) * cosine_shape)**2, dim=(-2, -1))
                
                delta = target_energy - energy_mid
                mask_too_low = delta > 0
                g_low = torch.where(mask_too_low, g_mid, g_low)
                g_high = torch.where(~mask_too_low, g_mid, g_high)
            
            boost_max = g_mid.view(B, 1, 1)
            del fft_mag2, energy_orig, g_low, g_high, target_energy
    
        else:
            if not torch.is_tensor(boost_max):
                boost_max = torch.tensor(boost_max, device=device)
            if boost_max.dim() == 0:
                boost_max = boost_max.view(1)
            if boost_max.shape[0] != B:
                boost_max = boost_max.expand(B)
            boost_max = boost_max.view(B, 1, 1)
    
        # === Filtro coseno final con limitación hasta f_cutoff ===
        boost = 1.0 + (boost_max - 1.0) * cosine_shape
        boost = torch.where(freq_r <= f_cutoff, boost, torch.ones_like(freq_r))
        del cosine_shape, freq_r, f_cutoff, boost_max, sharpen_power
    
        # === Aplicar filtro ===
        fft *= boost  
        del boost
        filtered = torch.fft.ifft2(fft, norm='forward').real
        del fft
    
        # === (Opcional) Normalizar contraste en espacio real ===
        if normalize:
            mean_orig = averages.mean(dim=(-2, -1), keepdim=True)
            std_orig = averages.std(dim=(-2, -1), keepdim=True)
            mean_filt = filtered.mean(dim=(-2, -1), keepdim=True)
            std_filt = filtered.std(dim=(-2, -1), keepdim=True)
            filtered = (filtered - mean_filt) / (std_filt + eps) * std_orig + mean_orig
            del mean_orig, std_orig, mean_filt, std_filt
    
        return filtered#, boost_max, sharpen_power
  

    @torch.no_grad()
    def kmeans_pytorch_for_averages(self, Im_tensor, X, eigvect, num_clusters, num_iters=25, verbose=False):

        # X = torch.stack(X)
        X = X.view(Im_tensor.shape[0], eigvect[0].shape[1]).float()
        N, D = X.shape
    
        # Normalization
        X = (X - X.mean(dim=0)) / (X.std(dim=0) + 1e-6)
    
        # --- K-means++ inicialización ---
        centroids = torch.empty((num_clusters, D), device=X.device)
        idx = torch.randint(0, N, (1,), device=X.device)
        centroids[0] = X[idx]
    
        for k in range(1, num_clusters):
            dist = torch.cdist(X, centroids[:k]).min(dim=1)[0]
            probs = dist / (dist.sum() + 1e-8)
            next_idx = torch.multinomial(probs, 1)
            centroids[k] = X[next_idx]
    
        # --- Iteracions ---
        for it in range(num_iters):
    
            distances = torch.cdist(X, centroids, p=2)
            labels = distances.argmin(dim=1)
    
            counts = torch.bincount(labels, minlength=num_clusters).float()
    
            # --- rescate suave de clusters pequeños ---
            min_size = max(4, int(0.01 * N))
            small = counts < min_size
            if small.any():
                worst = distances[:, small].max(dim=0).indices
                centroids[small] = X[worst]
    
            # --- actualización estándar ---
            counts = counts.clamp(min=1)
            centroids.zero_()
            centroids.scatter_add_(0, labels.unsqueeze(1).expand(-1, D), X)
            centroids /= counts.unsqueeze(1)
    
            # --- repulsión débil si colapsan ---
            Cdist = torch.cdist(centroids, centroids)
            mask = (Cdist < 0.15) & (Cdist > 0)
            if mask.any():
                centroids += torch.randn_like(centroids) * 0.01
    
            if verbose:
                inertia = (distances[torch.arange(N), labels] ** 2).sum().item()
                print(f"Iter {it+1:02d}  inertia = {inertia:.2e}")
    
        # --- Promedios robustos ---
        averages = []
        for i in range(num_clusters):
            mask = labels == i
            class_imgs = Im_tensor[mask]
    
            if class_imgs.shape[0] > 0:
                d = distances[mask, i]
                med = d.median()
                mad = torch.median(torch.abs(d - med)) + 1e-6
                good = d < med + 2.5 * mad
    
                if good.sum() > 0:
                    avg = class_imgs[good].mean(dim=0)
                else:
                    avg = class_imgs.mean(dim=0)
            else:
                avg = torch.zeros_like(Im_tensor[0])
    
            averages.append(avg)
    
        # del X, labels, centroids, distances
        del X, centroids, distances
    
        return torch.stack(averages),labels
    

    def determine_batches(self, free_memory, dim):
        
        if free_memory < 13: #test with 6Gb GPU
            if dim <= 64:
                expBatchSize = 30000 
                expBatchSize2 = 30000
                numFirstBatch = 1
                initClBatch = 30000
            elif dim <= 128:
                expBatchSize = 6000 
                expBatchSize2 = 9000
                numFirstBatch = 5
                initClBatch = 15000
            elif dim <= 256:
                expBatchSize = 1000 
                expBatchSize2 = 2000
                numFirstBatch = 20
                initClBatch = 15000
                
        elif free_memory >= 13 and free_memory < 21: #test with 15Gb GPU
            if dim <= 64:
                expBatchSize = 40000 
                expBatchSize2 = 50000
                numFirstBatch = 2
                initClBatch = 80000
            elif dim <= 128:
                expBatchSize = 15000 
                expBatchSize2 = 20000
                numFirstBatch = 5
                initClBatch = 50000
            elif dim <= 256:
                expBatchSize = 5000 
                expBatchSize2 = 6000
                numFirstBatch = 4 
                initClBatch = 15000 
                
        elif free_memory >= 21 and free_memory < 45: #test with 23Gb GPU
            if dim <= 64:
                expBatchSize = 50000 
                expBatchSize2 = 60000
                numFirstBatch = 2
                initClBatch = 100000
            elif dim <= 128:
                expBatchSize = 25000 
                expBatchSize2 = 30000
                numFirstBatch = 3
                initClBatch = 100000
            elif dim <= 256:
                expBatchSize = 6000 
                expBatchSize2 = 9000
                numFirstBatch = 6 
                initClBatch = 20000
        else:  #test with 49Gb GPU
            if dim <= 64:
                expBatchSize = 50000 
                expBatchSize2 = 60000
                numFirstBatch = 2
                initClBatch = 100000
            elif dim <= 128:
                expBatchSize = 50000 
                expBatchSize2 = 60000
                numFirstBatch = 2
                initClBatch = 100000
            elif dim <= 256:
                expBatchSize = 15000 
                expBatchSize2 = 20000
                numFirstBatch = 5 
                initClBatch = 30000
                
                
        return(expBatchSize, expBatchSize2, numFirstBatch, initClBatch)
    
    
    def apply_jitter_annealing(self, min_val, max_val, step, iter, max_iter, 
                           max_range_jitter=0.1, max_step_jitter=0.2):

        # Factor de annealing: empieza en 1 y decae linealmente a 0
        factor = max(0, 1 - iter / max_iter)
    
        # Jitter proporcional al factor
        range_jitter = factor * max_range_jitter
        step_jitter = factor * max_step_jitter
    
        min_j = min_val + random.uniform(-abs(min_val) * range_jitter, abs(min_val) * range_jitter)
        max_j = max_val + random.uniform(-abs(max_val) * range_jitter, abs(max_val) * range_jitter)
        step_j = step + random.uniform(-abs(step) * step_jitter, abs(step) * step_jitter)
        
        return min_j, max_j, max(step_j, 1)  # Step mínimo 1
    
       
    
    def determine_ROTandSHIFT(self, iter, mode, dim):
        
        if dim >= 200:   s, final = [6, 4, 4, 2, 2], 6
        elif dim >= 100: s, final = [5, 4, 4, 2, 2], 6
        else:            s, final = [3, 3, 2, 2, 1], 4
    
        if mode == "create_classes":
            max_s10 = math.ceil((dim * 0.10) / s[0]) * s[0]
            max_s10_f2 = math.ceil((dim * 0.10) / s[1]) * s[1] 
            limit_f3 = 8 if dim < 100 else 12
            max_iter = 18
            
            schedule = [
                (4,  (-180, 180, 10), (-max_s10, max_s10 + s[0], s[0])),
                (7,  (-180, 180, 8),  (-max_s10_f2, max_s10_f2 + s[1], s[1])), 
                (10, (-180, 180, 6),  (-limit_f3, limit_f3 + s[2], s[2])),
                (13, (-90, 94, 4),    (-8, 8 + s[3], s[3])),
                (18, (-90, 92, 2),    (-final, final + s[4], s[4]))
            ]
            
            res_ang, res_shift = schedule[-1][1], schedule[-1][2]
            for it_lim, a_p, s_p in schedule:
                if iter < it_lim:
                    res_ang, res_shift = a_p, s_p
                    break
            ang = self.apply_jitter_annealing(*res_ang, iter, max_iter)
            shiftMove = self.apply_jitter_annealing(*res_shift, iter, max_iter)
    
        else:
            max_s10 = math.ceil((dim * 0.10) / s[2]) * s[2]
            schedule = [
                (1, (-180, 180, 6), (-max_s10, max_s10 + s[2], s[2])),
                (2, (-180, 180, 4), (-8, 8 + s[3], s[3])),
                (3, (-90, 92, 2),   (-final, final + s[4], s[4]))
            ]
            
            ang, shiftMove = schedule[-1][1], schedule[-1][2]
            for it_lim, a_v, s_v in schedule:
                if iter < it_lim:
                    ang, shiftMove = a_v, s_v
                    break
           
        vectorRot, vectorshift = self.setRotAndShift(ang, shiftMove)
        return (vectorRot, vectorshift)
    