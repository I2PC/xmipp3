#!/usr/bin/env python3
"""/***************************************************************************
 *
 * Authors:    Erney Ramirez-Aportela
 *
  ***************************************************************************/
"""

import torch
import torch.nn as nn

class CWF_FourierBesselPCA(nn.Module):
    def __init__(self, num_components, k_max, n_max):
        super().__init__()
        self.num_components = num_components
        self.k_max = k_max
        self.n_max = n_max
        
        # Almacenamiento final
        self.register_buffer('global_eigenvalues', torch.zeros(num_components))
        self.register_buffer('component_k_indices', torch.zeros(num_components, dtype=torch.long))
        self.register_buffer('eigenvectors', torch.zeros(num_components, n_max, dtype=torch.complex64))
        self.register_buffer('clean_covariances', torch.zeros(k_max, n_max, n_max, dtype=torch.complex64))
        
        # Acumuladores en float64 para máxima precisión numérica
        self.sum_XXT = torch.zeros(k_max, n_max, n_max, dtype=torch.complex128)
        self.sum_HHT = torch.zeros(k_max, n_max, n_max, dtype=torch.float64)
        self.total_particles = 0
        self.accumulated_noise_var = 0.0

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.sum_XXT = self.sum_XXT.to(args[0] if args else 'cpu')
        self.sum_HHT = self.sum_HHT.to(args[0] if args else 'cpu')
        return self

    def add_batch(self, a_nk_batch, ctf_batch, noise_var=None):
        """Acumula lotes de coeficientes y CTFs."""
        batch_size = a_nk_batch.shape[0]
        self.total_particles += batch_size
        
        if noise_var is not None:
            self.accumulated_noise_var += noise_var * batch_size
            
        device = a_nk_batch.device
        self.sum_XXT = self.sum_XXT.to(device)
        self.sum_HHT = self.sum_HHT.to(device)
        
        for k in range(self.k_max):
            X_k = a_nk_batch[:, k, :].to(torch.complex128) # [Batch, n_max]
            H_k = ctf_batch[:, k, :].to(torch.float64)     # [Batch, n_max]
            
            self.sum_XXT[k] += torch.matmul(X_k.conj().T, X_k)
            self.sum_HHT[k] += torch.sum(H_k.unsqueeze(2) * H_k.unsqueeze(1), dim=0)

    def finalize_pca(self, fallback_noise_var=0.01):
        """Resuelve Ec. 17 (Desacoplamiento CTF + Shrinkage adaptativo) y extrae base PCA."""
        if self.total_particles < 2:
            raise ValueError("Se necesitan más partículas para calcular la covarianza.")
            
        sigma2 = (self.accumulated_noise_var / self.total_particles) if self.accumulated_noise_var > 0 else fallback_noise_var
        
        eigen_pool = []
        device = self.sum_XXT.device
        
        for k in range(self.k_max):
            C_obs_k = self.sum_XXT[k] / self.total_particles
            M_k = self.sum_HHT[k] / self.total_particles
            
            # Restar ruido en la diagonal
            I_deg = torch.eye(self.n_max, device=device, dtype=torch.complex128) * sigma2
            C_signal_k = C_obs_k - I_deg
            
            # Shrinkage adaptativo para evitar división por cero en ceros de CTF
            max_val = torch.max(torch.abs(M_k))
            adaptive_alpha = max(1e-5, 1e-3 * max_val.item())
            
            # Ecuación 17: Desacoplamiento de CTF
            C_clean_k = C_signal_k / (M_k + adaptive_alpha)
            
            # Descomposición hermítica en doble precisión
            evals_k, evecs_k = torch.linalg.eigh(C_clean_k)
            evals_k_clipped = torch.clamp(evals_k.real, min=0.0)
            
            # Reconstrucción de la covarianza limpia semidefinida positiva
            C_clean_k_pos = torch.matmul(evecs_k, torch.matmul(torch.diag(evals_k_clipped.to(torch.complex128)), evecs_k.conj().T))
            self.clean_covariances[k] = C_clean_k_pos.to(torch.complex64)
            
            for i in range(self.n_max):
                eigen_pool.append({
                    'eval': evals_k_clipped[i].item(),
                    'k_idx': k,
                    'evec': evecs_k[:, i].to(torch.complex64)
                })
                
        # Ordenamiento global de componentes principales
        eigen_pool.sort(key=lambda x: x['eval'], reverse=True)
        
        for i in range(self.num_components):
            comp = eigen_pool[i]
            self.global_eigenvalues[i] = comp['eval']
            self.component_k_indices[i] = comp['k_idx']
            self.eigenvectors[i] = comp['evec']

    def project_wiener(self, a_nk_batch, ctf_batch, fallback_noise_var=0.01):
        """Aplica Filtro de Wiener (Ec. 18) y proyecta sobre los autovectores limpios."""
        B = a_nk_batch.shape[0]
        device = a_nk_batch.device
        pca_scores = torch.zeros(B, self.num_components, device=device)
        
        for b in range(B):
            for k in range(self.k_max):
                a_i = a_nk_batch[b, k, :].unsqueeze(1).to(torch.complex128)
                h_i = ctf_batch[b, k, :].to(torch.float64)
                H_i = torch.diag(h_i.to(torch.complex128))
                C_clean = self.clean_covariances[k].to(torch.complex128)
                
                # Ecuación 18: Operador de Wiener
                term1 = torch.matmul(C_clean, H_i.conj().T)
                inv_target = torch.matmul(H_i, torch.matmul(C_clean, H_i.conj().T)) + (fallback_noise_var * torch.eye(self.n_max, device=device, dtype=torch.complex128))
                
                wiener_gain = torch.matmul(term1, torch.linalg.inv(inv_target))
                f_hat_i = torch.matmul(wiener_gain, a_i).squeeze(1)
                
                # Proyección escalar sobre los componentes principales asignados
                for i in range(self.num_components):
                    if self.component_k_indices[i] == k:
                        evec = self.eigenvectors[i].to(torch.complex128)
                        proj = torch.vdot(evec, f_hat_i)
                        pca_scores[b, i] = torch.abs(proj).real
                        
        return pca_scores
    
    
    
    