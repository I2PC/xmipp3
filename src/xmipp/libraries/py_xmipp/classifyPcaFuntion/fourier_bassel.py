#!/usr/bin/env python3
"""/***************************************************************************
 *
 * Authors:    Erney Ramirez-Aportela
 *
  ***************************************************************************/
"""

import torch
import torch.nn as nn
from scipy.special import jn_zeros, jv

class FourierBesselExtractor(nn.Module):
    def __init__(self, img_size, k_max, n_max, r_max=None):
        super().__init__()
        self.img_size = img_size
        self.k_max = k_max
        self.n_max = n_max
        self.r_max = r_max if r_max is not None else img_size / 2.0
        
        fb_basis = self._build_fb_basis()
        self.register_buffer('fb_basis', fb_basis)

    def _build_fb_basis(self):
        Y, X = np.ogrid[-self.img_size//2 : self.img_size//2, 
                        -self.img_size//2 : self.img_size//2]
        R = np.sqrt(X**2 + Y**2)
        Phi = np.arctan2(Y, X)
        
        mask = (R <= self.r_max).astype(np.float32)
        R_norm = np.clip(R / self.r_max, 0, 1)
        
        basis = np.zeros((self.k_max, self.n_max, self.img_size, self.img_size), dtype=np.complex64)
        
        for k in range(self.k_max):
            zeros_k = jn_zeros(k, self.n_max)
            for n in range(self.n_max):
                alpha_kn = zeros_k[n]
                j_val = jv(k, alpha_kn * R_norm)
                norm_factor = np.sqrt(2.0) / (self.r_max * np.abs(jv(k + 1, alpha_kn)))
                basis[k, n] = norm_factor * j_val * mask * np.exp(1j * k * Phi)

        return torch.from_numpy(basis)

    def forward(self, fft_images_2d):
        """Proyecta FFTs 2D a coeficientes complejos a_nk [Batch, k_max, n_max]."""
        return torch.einsum('bxy,knxy->bkn', fft_images_2d, torch.conj(self.fb_basis))
    
    
    