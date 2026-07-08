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
import math
import random


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
    def precalculate_projection(self, prjTensorCpu, whitening, freqBn, grid_flat, coef, cvecs, rot_tensor, shift):
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
        #Apply whitening
        # rotFFT = rotFFT * whitening
        shift_tensor = torch.as_tensor(shift, device=device, dtype=torch.float32)
        band_shifted = self.precShiftBand(rotFFT, freqBn, grid_flat, coef, shift_tensor)
        projBatch = self.phiProjRefs(band_shifted, cvecs)
    
        del prj_rot, rotFFT, band_shifted
        return projBatch
    
    
    @torch.no_grad()
    def create_batchExp(self, Texp, whitening, freqBn, coef, vecs):
             
        self.batch_projExp = [torch.zeros((Texp.size(dim=0), vecs[n].size(dim=1)), device = self.cuda) for n in range(self.nBand)]
        expFFT = torch.fft.rfft2(Texp, norm="forward")
        del(Texp)
        #Apply whitening
        # expFFT = expFFT * whitening
        bandExp = self.selectBandsRefs(expFFT, freqBn, coef)
        self.batch_projExp = self.phiProjRefs(bandExp, vecs)
        del(expFFT , bandExp)
        
        torch.cuda.empty_cache()
        return(self.batch_projExp)
    
    
    def compute_radial_whitening_filter(self, images, eps=1e-8):
        """
        Calcula filtro de whitening radial a partir de un stack de imágenes.
        """
    
        device = images.device
        N, H, W = images.shape
    
        fft = torch.fft.rfft2(images, norm="forward")
        psd2d = torch.mean(torch.abs(fft)**2, dim=0)
    
        fy = torch.fft.fftfreq(H, d=1.0, device=device)
        fx = torch.fft.rfftfreq(W, d=1.0, device=device)
        yy, xx = torch.meshgrid(fy, fx, indexing="ij")
    
        r = torch.sqrt(yy**2 + xx**2)
        r = r * min(H, W)
        r = r.round().long()
    
        max_r = r.max().item()
    
        # Radial average
        radial_psd = torch.zeros(max_r + 1, device=device)
        counts = torch.zeros(max_r + 1, device=device)
    
        radial_psd.scatter_add_(0, r.flatten(), psd2d.flatten())
        counts.scatter_add_(0, r.flatten(), torch.ones_like(psd2d.flatten()))
    
        radial_psd /= counts + eps
    
        whitening_filter = 1.0 / torch.sqrt(radial_psd[r] + eps)
    
        whitening_filter[0, 0] = 0.0
    
        return whitening_filter
        
      
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
    
    
    
    def robust_normalize_and_mask2(self, images, radius_ratio=0.85):
        """
        Calcula la máscara y normaliza las imágenes en un solo paso.
        
        Args:
            images: Tensor [N, H, W]
            radius_ratio: Proporción del radio de la máscara (0.85 suele ser ideal).
        """
        N, H, W = images.shape
        device = images.device
        
        # 1. Calcular la máscara circular internamente
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=device),
            torch.linspace(-1, 1, W, device=device),
            indexing='ij'
        )
        dist = torch.sqrt(x**2 + y**2)
        mask = (dist <= radius_ratio).float()
        inv_mask = (dist > radius_ratio) # Área de ruido/fondo
        
        normalized_images = torch.zeros_like(images)
        
        for i in range(N):
            img = images[i]
            
            # 2. Estadísticas del fondo (píxeles fuera del círculo)
            noise_pixels = img[inv_mask]
            if noise_pixels.numel() > 0:
                bg_mean = noise_pixels.mean()
                bg_std = noise_pixels.std()
            else:
                bg_mean, bg_std = img.mean(), img.std()
                
            # 3. Normalización Z-score (Fondo -> 0, Ruido -> varianza 1)
            # Esto nivela el contraste entre diferentes partículas
            img_norm = (img - bg_mean) / (bg_std + 1e-8)
            
            # 4. Aplicar máscara y ReLU
            # El ReLU es vital: elimina el ruido negativo que confunde al PCA
            # normalized_images[i] = torch.relu(img_norm * mask)
            normalized_images[i] = img_norm
            
        return normalized_images
    
    
    def robust_normalize_and_mask(self, images, radius_ratio=0.85):
        N, H, W = images.shape
        device = images.device
        
        # Crear máscara de coseno suave (evita cortes bruscos que odia el PCA)
        y, x = torch.meshgrid(torch.linspace(-1, 1, H, device=device),
                              torch.linspace(-1, 1, W, device=device), indexing='ij')
        dist = torch.sqrt(x**2 + y**2)
        
        # Máscara con caída suave (soft edge)
        soft_mask = torch.clamp((radius_ratio - dist) / 0.1, 0, 1)
        
        normalized_images = torch.zeros_like(images)
        
        for i in range(N):
            img = images[i]
            
            # En lugar de Z-score total, solo restamos el percentil 10 (estimación del fondo)
            # Esto no cambia la escala de la proteína, solo asegura que el fondo sea ~0
            background_level = torch.quantile(img, 0.1)
            img_centered = img - background_level
            
            # Aplicamos la máscara suave sin ReLU agresivo
            # Dejamos que los valores negativos existan pero solo cerca de cero
            normalized_images[i] = img_centered * soft_mask
            
        return normalized_images
    
    
    def zscore_normalization(self, images):
        mean = images.mean(dim=(-2, -1), keepdim=True)
        std = images.std(dim=(-2, -1), keepdim=True)
        images = (images - mean) / (std + 1e-8)
        return images
    
    def gaussian_weighted_zscore_normalization(self, imgs, sigma=42):
        """
        Normaliza promedios de clase que tienen aplicada una máscara gaussiana.
        Usa estadística ponderada para no destruir la atenuación de los bordes.
        
        imgs: Tensor de PyTorch (B, H, W)
        sigma: Desviación estándar de la gaussiana (controla qué tan ancha es)
        """
        H, W = imgs.shape[-2], imgs.shape[-1]
        device = imgs.device
        
        # 1. Recrear la máscara gaussiana (valores de 0 a 1)
        y, x = torch.meshgrid(torch.linspace(-1, 1, H, device=device), 
                              torch.linspace(-1, 1, W, device=device), indexing='ij')
        r2 = x**2 + y**2
        # El peso w será máximo (1.0) en el centro y caerá hacia 0 en las esquinas
        weight_mask = torch.exp(-r2 / (2 * sigma**2))
        
        # Suma total de los pesos (el equivalente al 'número de píxeles')
        sum_w = weight_mask.sum()
        
        # 2. Calcular la Media Ponderada por cada imagen en el batch
        # Multiplicamos la imagen por los pesos para ponderar el centro
        weighted_mean = (imgs * weight_mask).sum(dim=(-2, -1), keepdim=True) / sum_w
        
        # 3. Calcular la Varianza y STD Ponderadas
        weighted_variance = ((imgs - weighted_mean).pow(2) * weight_mask).sum(dim=(-2, -1), keepdim=True) / sum_w
        weighted_std = torch.sqrt(weighted_variance + 1e-8)
        
        # 4. Aplicar el Z-score ponderado
        imgs_norm = (imgs - weighted_mean) / weighted_std
        
        # 5. Crucial: Volver a aplicar la máscara gaussiana
        # Esto asegura que todo lo que se movió por la resta de la media vuelva a decaer a 0 puro
        return imgs_norm * weight_mask
    
    def background_contrast_normalization(self, imgs, bg_radius=0.85):
        """
        Iguala el contraste tomando como referencia el ruido de fondo (solvente).
        bg_radius: Radio (0 a 1) a partir del cual todo se considera fondo/esquinas.
        """
        H, W = imgs.shape[-2], imgs.shape[-1]
        device = imgs.device
        
        # Crear máscara circular (1 en las esquinas/fondo, 0 en el centro donde está la proteína)
        y, x = torch.meshgrid(torch.linspace(-1, 1, H, device=device), 
                              torch.linspace(-1, 1, W, device=device), indexing='ij')
        r = torch.sqrt(x**2 + y**2)
        bg_mask = (r > bg_radius).float()
        
        # Calcular media y std solo en la zona del fondo
        num_bg_pixels = bg_mask.sum()
        bg_mean = (imgs * bg_mask).sum(dim=(-2, -1), keepdim=True) / num_bg_pixels
        
        bg_variance = (((imgs - bg_mean) * bg_mask).pow(2)).sum(dim=(-2, -1), keepdim=True) / num_bg_pixels
        bg_std = torch.sqrt(bg_variance + 1e-8)
        
        # Normalizar TODA la imagen usando la referencia del fondo
        return (imgs - bg_mean) / bg_std
    
    def percentile_contrast_normalization(self, imgs, low_p=1.0, high_p=99.0):
        """
        Iguala el contraste estirando el rango dinámico entre dos percentiles robustos.
        Imgs: Tensor de PyTorch (B, H, W) o (B, C, H, W)
        """
        # Aplanar las dimensiones espaciales para calcular percentiles por imagen
        flat = imgs.flatten(start_dim=-2)
        
        # Calcular los percentiles robustos (convertimos p a fracción de 0 a 1)
        q_low = torch.quantile(flat, low_p / 100.0, dim=-1, keepdim=True).unsqueeze(-1)
        q_high = torch.quantile(flat, high_p / 100.0, dim=-1, keepdim=True).unsqueeze(-1)
        
        # Escalado lineal al rango [0, 1] basado en esos percentiles
        imgs_norm = (imgs - q_low) / (q_high - q_low + 1e-8)
        
        # Recortamos (clip/clamp) para que los pocos píxeles fuera de los percentiles no distorsionen
        return torch.clamp(imgs_norm, 0.0, 1.0)
    
    def contrast_cv(self, images):
        """
        Calcula el coeficiente de variación (CV) de la desviación estándar
        entre partículas.
        """
    
        # std de cada partícula
        particle_stds = images.std(dim=(-2, -1))
    
        # Si hay canales, promediamos sobre ellos
        if particle_stds.ndim > 1:
            particle_stds = particle_stds.mean(dim=-1)
    
        mean_std = particle_stds.mean()
        std_std = particle_stds.std()
    
        cv = std_std / (mean_std + 1e-8)
    
        return cv, mean_std, std_std
    
    def contrast_outliers(self, images):
        particle_std = images.std(dim=(-2, -1))
    
        mean_std = particle_std.mean()
        std_std = particle_std.std()
    
        zscore = (particle_std - mean_std) / (std_std + 1e-8)
    
        return particle_std, zscore
    
    
            
    def search_space_old(self, iter, rot, sh, msh):
        
        if iter == 0:
            angle, shift, maxShift = rot, sh, msh
        elif iter == 8:
            angle, shift, maxShift = 6, 3, 12
        elif iter == 13:
            angle, shift, maxShift = 5, 3, 12
        elif iter == 16:
            angle, shift, maxShift = 5, 3, 12
            
        return(angle, shift, maxShift)
    
    def search_space_old(self, iter, sampling):
        
        if iter == 0:
            angle, shift, maxShift = 8, 6/sampling, 24/sampling
        elif iter == 8:
            angle, shift, maxShift = 6, 6/sampling, 24/sampling
        elif iter == 13:
            angle, shift, maxShift = 5, 6/sampling, 24/sampling
        elif iter == 16:
            angle, shift, maxShift = 5, 6/sampling, 24/sampling
            
        return(angle, shift, maxShift)
    
    #Probar el jitter
    def search_space(self, iter, sampling):
        # 1. Valores base según la iteración
        if iter < 4:
            base_angle, base_shift, maxShift = 8.0, 6.0/sampling, 24.0/sampling
        elif iter < 8:
            base_angle, base_shift, maxShift = 8.0, 6.0/sampling, 24.0/sampling
        elif iter < 13:
            base_angle, base_shift, maxShift = 6.0, 6.0/sampling, 24.0/sampling
        else:
            base_angle, base_shift, maxShift = 5.0, 6.0/sampling, 24.0/sampling
            
        # 2. Aplicamos Jitter decreciente (enfriamiento)
        if iter < 8:
            # Exploración fuerte
            angle_jitter = random.uniform(-0.5, 0.5)
            # El shift varía ligeramente (p.ej., entre -8% y +8% de su valor base)
            shift_jitter = random.uniform(-0.08, 0.08) * base_shift
        elif iter < 13:
            # Ajuste medio
            angle_jitter = random.uniform(-0.2, 0.2)
            shift_jitter = random.uniform(-0.03, 0.03) * base_shift
        else:
            # Refinamiento puro y congelado
            angle_jitter = 0.0
            shift_jitter = 0.0
            
        angle = base_angle + angle_jitter
        shift = base_shift + shift_jitter
        
        return (angle, shift, maxShift)
    
    
    def reconstruct_parameters_old(self, iter, pcaRes, filter):
        
        if iter == 0:
            pcaRes, volRes, angleGallery = 20, 20, 12
            # pcaRes, volRes, angleGallery = 8, 20, 12
        elif iter == 8:
            pcaRes, volRes, angleGallery = 16, 16, 8
            # pcaRes, volRes, angleGallery = 8, 16, 8
        elif iter == 13:
            pcaRes, volRes, angleGallery = pcaRes, filter, 6
        elif iter == 16:
            pcaRes, volRes, angleGallery = pcaRes, filter, 5
            
        return(pcaRes, volRes, angleGallery)
    
    
    def reconstruct_parameters(self, iter, pcaRes, filter):
        
        if iter < 3:
            pcaRes, volRes, angleGallery = 40, 40, 12
        elif iter < 5:
            pcaRes, volRes, angleGallery = 30, 30, 6
        elif iter < 8:
            pcaRes, volRes, angleGallery = 20, 20, 12
            # pcaRes, volRes, angleGallery = 8, 20, 12
        elif iter < 13:
            pcaRes, volRes, angleGallery = 16, 16, 8
            # pcaRes, volRes, angleGallery = 8, 16, 8
        elif iter == 16:
            pcaRes, volRes, angleGallery = pcaRes, filter, 6
        else:
            pcaRes, volRes, angleGallery = pcaRes, filter, 5
            
        return(pcaRes, volRes, angleGallery)


            
    def parameters2(self, iter, maxRes, filter):
        
        if iter == 9:
            volRes, filtRes, angleGallery, angle, shift, maxShift = 16, 16, 8, 6, 3, 12
        elif iter == 14:
            volRes, filtRes, angleGallery, angle, shift, maxShift = maxRes, filter, 6, 5, 3, 12
        elif iter == 17:
            volRes, filtRes, angleGallery, angle, shift, maxShift = maxRes, filter, 5, 5, 3, 12
            
        return(volRes, filtRes, angleGallery, angle, shift, maxShift)
            

    
    
