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
        rotFFT = rotFFT * whitening
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
        expFFT = expFFT * whitening
        bandExp = self.selectBandsRefs(expFFT, freqBn, coef)
        self.batch_projExp = self.phiProjRefs(bandExp, vecs)
        del(expFFT , bandExp)
        
        torch.cuda.empty_cache()
        return(self.batch_projExp)
    
    
    def compute_radial_whitening_filter2(self, images, eps=1e-8):
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
    
    
    def compute_radial_whitening_filter(
            self,
            images,
            pixel_size,
            resolution_limit,
            eps=1e-6):
    
        device = images.device
        N, H, W = images.shape
    
        # 1. PSD en Fourier
        fft = torch.fft.rfft2(images, norm="forward")
        psd2d = torch.mean(torch.abs(fft)**2, dim=0)
    
        # 2. Frecuencias físicas (1/Å)
        fy = torch.fft.fftfreq(H, d=pixel_size, device=device)
        fx = torch.fft.rfftfreq(W, d=pixel_size, device=device)
    
        yy, xx = torch.meshgrid(fy, fx, indexing="ij")
    
        freq = torch.sqrt(yy**2 + xx**2)
    
        # índice radial
        r_idx = (freq * min(H, W)).round().long()
    
        # 3. Promedio radial
        max_r = r_idx.max().item()
    
        radial_psd = torch.zeros(max_r + 1, device=device)
        counts = torch.zeros(max_r + 1, device=device)
    
        radial_psd.scatter_add_(0, r_idx.flatten(), psd2d.flatten())
        counts.scatter_add_(0, r_idx.flatten(), torch.ones_like(psd2d.flatten()))
    
        radial_psd /= (counts + 1e-12)
    
        # 4. Whitening
        whitening_filter = 1.0 / torch.sqrt(radial_psd[r_idx] + eps)
    
        # eliminar DC
        whitening_filter[0, 0] = 0.0
    
    
        # --------------------------------------------------
        # 5. Limitar a resolución máxima
        # --------------------------------------------------
        
        fmax = 1.0 / resolution_limit
        temperature = 0.02 # Controla la suavidad de la caída (ajustar si es necesario)
        
        # Filtro que es 1 hasta fmax y luego cae
        lowpass = 1.0 / (1.0 + torch.exp((freq - fmax) / temperature))
        
        whitening_filter *= lowpass
    
    
        # 6. Normalizar
        whitening_filter /= whitening_filter.mean()
    
    
        return whitening_filter
        
      
# ==================================================================
    # Pegar dentro de la clase BnBgpu (misma indentacion que el resto de metodos)
    # ==================================================================


    @torch.no_grad()
    def normalize_batch(self, batch):
        """Normaliza (norma L2) cada vector de la lista de bandas. Modifica la lista in-place.
        Devuelve (batch, norms): norms[n] son las normas ORIGINALES (nRows,), por si se quiere
        usar la penalizacion de amplitud. Llamar UNA vez sobre batchRef antes del bucle de rotaciones."""
        norms = []
        for n in range(self.nBand):
            nrm = torch.linalg.norm(batch[n], dim=1)
            batch[n] = batch[n] / (nrm[:, None] + 1e-8)
            norms.append(nrm)
        return batch, norms


    @torch.no_grad()
    def match_batch_initVol(self, batchExp, batchRef, initBatch, matches, rot, nShift):
        """Version original (un unico mejor match por average). Unico cambio: squeeze(1)
        en vez de squeeze() para que no falle con nExp == 1."""

        nExp = int(batchExp[0].size(dim=0) / nShift)
        nShift = torch.tensor(nShift, device=self.cuda)

        for n in range(self.nBand):
            batchRef[n] = batchRef[n] / (torch.linalg.norm(batchRef[n], dim=1, keepdim=True) + 1e-8)
            batchExp[n] = batchExp[n] / (torch.linalg.norm(batchExp[n], dim=1, keepdim=True) + 1e-8)
            score = torch.cdist(batchRef[n], batchExp[n])

        num_rows, num_cols = score.shape

        num_blocks = score.shape[1] // nShift
        blocks = score.view(num_rows, num_blocks, nShift)

        block_min_values, block_min_indices_rows = blocks.min(dim=0)
        block_min_value, block_min_indices_cols = block_min_values.min(dim=1)
        block_min_indices_rows = block_min_indices_rows.gather(1, block_min_indices_cols.unsqueeze(1)).squeeze(1)
        del(score)

        exp = torch.arange(initBatch, initBatch + nExp, 1, device=self.cuda).view(nExp, 1)
        sel = block_min_indices_rows.type(torch.int64)
        rotation = torch.full((nExp, 1), rot, device=self.cuda)
        shift_location = (block_min_indices_cols).type(torch.int64)

        iter_matches = torch.cat((exp, sel.view(nExp, 1), block_min_value.view(nExp, 1),
                                  rotation, shift_location.view(nExp, 1)), dim=1)

        cond = iter_matches[:, 2] < matches[initBatch:initBatch + nExp, 2]
        matches[initBatch:initBatch + nExp] = torch.where(cond.view(nExp, 1), iter_matches,
                                                          matches[initBatch:initBatch + nExp])
        return(matches)


    @torch.no_grad()
    def match_batch_best_per_ref(self, batchExp, batchRef, best, best_rot, best_shift, rot, nShift,
                                 ref_norms=None, amp_lambda=0.0, amp_scale=1.0):
        """
        Mantiene, para cada par (average, referencia), el mejor score sobre shifts y rotaciones.

        batchRef debe estar ya normalizado (normalize_batch). batchExp se normaliza aqui.
        best, best_rot, best_shift: (nExp, nRef). `rot` se guarda tal como se recibe
        (mismo convenio que match_batch_initVol: se le pasa -rot).

        Penalizacion de amplitud (opcional, amp_lambda > 0):
            score = d_coseno + amp_lambda * [log(||x|| / (amp_scale * ||r||))]^2
        con ref_norms = normas originales de las referencias (salida de normalize_batch).
        amp_lambda = 0 -> solo coseno (comportamiento anterior).
        """
        nShift = int(nShift)
        nExp = batchExp[0].size(0) // nShift

        for n in range(self.nBand):
            nrm_e = torch.linalg.norm(batchExp[n], dim=1)
            expN = batchExp[n] / (nrm_e[:, None] + 1e-8)
            score = torch.cdist(batchRef[n], expN)               # (nRef, nExp*nShift)

            if amp_lambda > 0 and ref_norms is not None:
                d_amp = torch.log((nrm_e[None, :] + 1e-8) / (amp_scale * ref_norms[n][:, None] + 1e-8))
                d_amp.pow_(2).mul_(amp_lambda)
                score.add_(d_amp)
                del d_amp

        nRef = score.shape[0]
        min_sh, arg_sh = score.view(nRef, nExp, nShift).min(dim=2)   # (nRef, nExp)
        del score
        min_sh, arg_sh = min_sh.t(), arg_sh.t()                      # (nExp, nRef)

        upd = min_sh < best
        best = torch.where(upd, min_sh, best)
        best_rot = torch.where(upd, torch.full_like(best_rot, float(rot)), best_rot)
        best_shift = torch.where(upd, arg_sh.to(best_shift.dtype), best_shift)
        return best, best_rot, best_shift


    @torch.no_grad()
    def topk_matches(self, best, best_rot, best_shift, k=3):
        """
        Devuelve una lista de k tensores (nExp, 5) con el formato de `matches`:
        [exp, ref, score, rot, shift_idx]; el elemento r de la lista es el candidato de rango r
        (0 = mejor). Cada candidato es una referencia distinta.
        """
        k = min(k, best.shape[1])
        nExp = best.shape[0]
        vals, refs = torch.topk(best, k, dim=1, largest=False)       # (nExp, k), ascendente
        rots = torch.gather(best_rot, 1, refs)
        shifts = torch.gather(best_shift, 1, refs)
        exp = torch.arange(nExp, device=best.device, dtype=torch.float32)

        out = []
        for r in range(k):
            out.append(torch.stack((exp, refs[:, r].float(), vals[:, r].float(),
                                    rots[:, r].float(), shifts[:, r].float()), dim=1))
        return out


    @torch.no_grad()
    def soft_weights(self, top, T):
        """
        Pesos blandos (nExp, K) con temperatura T dada. Suman 1 por average.
        La escala es la mediana (entre averages) de la diferencia peor-mejor candidato,
        asi que T es adimensional. OJO: esa escala depende de K.
        """
        s = torch.stack([t[:, 2] for t in top], dim=1)               # (nExp, K)
        if s.shape[1] == 1:
            return torch.ones_like(s)
        gap = (s[:, -1] - s[:, 0]).median() + 1e-8
        return torch.softmax(-(s - s[:, :1]) / (T * gap), dim=1)


    @torch.no_grad()
    def soft_weights_neff(self, top, n_eff, iters=30):
        """
        Pesos blandos (nExp, K) tales que el numero efectivo medio de candidatos
        (exp de la entropia) sea ~ n_eff. Busca la temperatura por biseccion.
        Devuelve (w, T). n_eff = 1 -> asignacion dura; n_eff = K -> pesos uniformes.
        """
        s = torch.stack([t[:, 2] for t in top], dim=1)               # (nExp, K)
        if s.shape[1] == 1:
            return torch.ones_like(s), 0.0

        d = s - s[:, :1]
        gap = (s[:, -1] - s[:, 0]).median() + 1e-8
        lo, hi = 1e-3, 1e3
        w, mid = None, hi
        for _ in range(iters):
            mid = (lo * hi) ** 0.5
            w = torch.softmax(-d / (mid * gap), dim=1)
            ne = torch.exp(-(w * torch.log(w + 1e-12)).sum(dim=1)).mean()
            if ne > n_eff:
                hi = mid
            else:
                lo = mid
        return w, mid
    
    
    @torch.no_grad()
    def match_batch_initVol_original(self, batchExp, batchRef, initBatch, matches, rot, nShift):
        
        nExp = int(batchExp[0].size(dim=0) / nShift)
        nShift = torch.tensor(nShift, device=self.cuda)       
                             
        for n in range(self.nBand):
            ################################
            batchRef[n] = batchRef[n] / (torch.linalg.norm(batchRef[n], dim=1, keepdim=True) + 1e-8)
            batchExp[n] = batchExp[n] / (torch.linalg.norm(batchExp[n], dim=1, keepdim=True) + 1e-8)
            ################################
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
    
    
    @torch.no_grad()
    def filter_scores_mad(self, matches, score_col=2, flag_col=6, threshold=1.5):
        matches = matches.clone()
        
        new_col = torch.ones((matches.shape[0], 1), dtype=matches.dtype, device=matches.device)
        matches = torch.cat((matches, new_col), dim=1)
    
        scores = matches[:, score_col]
    
        if scores.numel() < 5:
            return matches
    
        median = scores.median()
        mad = (scores - median).abs().median()
    
        if mad > 1e-8:
            robust_sigma = 1.4826 * mad
            outliers = scores > median + threshold * robust_sigma
        else:
            # Si MAD es cero, descartar valores estrictamente
            # superiores al percentil 95.
            cutoff = torch.quantile(scores, 0.95)
            outliers = scores > cutoff
    
        # Obtener los índices de los scores atípicos
        outlier_indices = torch.where(outliers)[0]
    
        # Marcar como inválidos los matches con scores demasiado altos
        matches[outlier_indices, flag_col] = 0
    
        return matches
        
       
    
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
    
    
    @torch.no_grad()
    def create_gaussian_mask(self, images, sigma=128/3):
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
    def zscore_normalization_mask(self, images, mask, eps=1e-8):
        """
        Normaliza las imágenes llevando la proteína a media 0 y std 1.
        Usa resta modulada para no aplicar la máscara dos veces.
        """
        mask = mask.to(device=images.device, dtype=images.dtype)
        sum_w = mask.sum()
        
        # 1. Estadísticas ponderadas basadas en la zona de la proteína
        mean_w = (images * mask).sum(dim=(-2, -1), keepdim=True) / sum_w
        var_w = ((images - mean_w).pow(2) * mask).sum(dim=(-2, -1), keepdim=True) / sum_w
        std_w = torch.sqrt(var_w + eps)
        
        # 2. Resta modulada: NO multiplicamos por la máscara al final
        # images_normalized = (images - mean_w * mask) / std_w
        images_normalized = (images - mean_w ) / std_w
        
        return images_normalized
    
    
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
    
    
    def correct_background_and_normalize(self, 
        imgs,
        signal_radius_px,
        target_rms=1.0,
        eps=1e-8,
    ):
        """
        imgs: tensor (N, H, W) o (H, W).
        signal_radius_px: radio de la región central de señal.
        target_rms: RMS deseado para cada imagen dentro de la máscara.
        """
        single_image = imgs.ndim == 2
        if single_image:
            imgs = imgs.unsqueeze(0)
    
        if imgs.ndim != 3:
            raise ValueError("imgs debe tener forma (N,H,W) o (H,W)")
    
        n, h, w = imgs.shape
        device = imgs.device
    
        y = torch.arange(h, device=device, dtype=imgs.dtype) - (h - 1) / 2
        x = torch.arange(w, device=device, dtype=imgs.dtype) - (w - 1) / 2
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        rr = torch.sqrt(xx.square() + yy.square())
    
        signal_mask = rr <= signal_radius_px
        background_mask = rr >= signal_radius_px
    
        if signal_mask.sum() < 10 or background_mask.sum() < 10:
            raise ValueError("Las máscaras contienen muy pocos píxeles.")
    
        # Estimar y restar el fondo individual de cada average
        background = imgs[:, background_mask].median(dim=1).values
        corrected = imgs - background[:, None, None]
    
        # Normalizar la amplitud dentro de la región central
        signal = corrected[:, signal_mask]
        rms = torch.sqrt(signal.square().mean(dim=1))
    
        normalized = corrected * (
            target_rms / rms.clamp_min(eps)
        )[:, None, None]
    
        if single_image:
            return normalized[0]
    
        return normalized
    
    
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
            # base_angle, base_shift, maxShift = 8, 6/sampling, 24/sampling
            base_angle, base_shift, maxShift = 8, 6/sampling, 24/sampling
        elif iter < 8:
            base_angle, base_shift, maxShift = 6, 6/sampling, 24/sampling
            # base_angle, base_shift, maxShift = 6, 4/sampling, 20/sampling
        elif iter < 13:
            base_angle, base_shift, maxShift = 5.0, 6/sampling, 24/sampling
            # base_angle, base_shift, maxShift = 5.0, 4/sampling, 20/sampling
        else:
            base_angle, base_shift, maxShift = 5.0, 6/sampling, 24/sampling
            # base_angle, base_shift, maxShift = 5.0, 4/sampling, 20/sampling
            
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
    
    
    # def reconstruct_parameters(self, iter, pcaRes, filter):
    #
    #     if iter < 2:
    #         pcaRes, volRes, angleGallery = 60, 20, 15
    #     elif iter < 4:
    #         pcaRes, volRes, angleGallery = 60, 20, 12
    #     elif iter < 6:
    #         pcaRes, volRes, angleGallery = 60, 20, 10
    #     elif iter < 8:
    #         pcaRes, volRes, angleGallery = 30, 20, 12
    #     elif iter < 10:
    #         pcaRes, volRes, angleGallery = 20, 20, 10
    #     elif iter < 12:
    #         pcaRes, volRes, angleGallery = 16, 16, 8
    #     elif iter < 14:
    #         pcaRes, volRes, angleGallery = 12, 12, 7
    #     else:
    #         pcaRes, volRes, angleGallery = pcaRes, filter, 5
    #
    #     return(pcaRes, volRes, angleGallery)
    
    
    def reconstruct_parameters(self, iter, pcaRes, filter):
        
        if iter < 3:
            # pcaRes, volRes, angleGallery = 40, 40, 12
            pcaRes, volRes, angleGallery = 60, 20, 15
        elif iter < 5:
            # pcaRes, volRes, angleGallery = 30, 30, 6
            pcaRes, volRes, angleGallery = 30, 20, 12
        elif iter < 8:
            # pcaRes, volRes, angleGallery = 20, 20, 12
            pcaRes, volRes, angleGallery = 16, 16, 12
        elif iter < 13:
            # pcaRes, volRes, angleGallery = 16, 16, 8
            pcaRes, volRes, angleGallery = 12, 12, 8
        elif iter == 16:
            # pcaRes, volRes, angleGallery = pcaRes, filter, 6
            pcaRes, volRes, angleGallery = pcaRes, filter, 5
        else:
            pcaRes, volRes, angleGallery = pcaRes, filter, 5
            # pcaRes, volRes, angleGallery = 16, 16, 5
            
           # Jitter multiplicativo: ±10 %
        # jitter = 0.10
        # factor = 1.0 + (2.0 * torch.rand(1).item() - 1.0) * jitter
        # angleGallery = angleGallery * factor
            
        return(pcaRes, volRes, angleGallery)
            

    
    
