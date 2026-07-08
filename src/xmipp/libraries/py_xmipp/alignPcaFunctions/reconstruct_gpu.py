#!/usr/bin/env python3
"""/***************************************************************************
 *
 * Authors:    Erney Ramirez-Aportela
 *
  ***************************************************************************/
"""
from xmippPyModules.alignPcaFunctions.assessment import *
import numpy as np
import torch
import time
import torchvision.transforms.functional as T
import torch.nn.functional as F
import kornia
import math
import re

class reconstruct:
    
    # def __init__(self, nBand):
    def __init__(self):

        # self.nBand = nBand 
        
        torch.cuda.is_available()
        torch.cuda.current_device()
        self.cuda = torch.device('cuda')
        # self.cuda = torch.device('cpu')
    
    
    @torch.no_grad()
    def enforce_hermitian_symmetry(self,vol_f):
        G = vol_f.shape[0]
        device = vol_f.device
    
        center = G // 2
    
        z = torch.arange(G, device=device)
        y = torch.arange(G, device=device)
        x = torch.arange(G, device=device)
    
        zz, yy, xx = torch.meshgrid(z, y, x, indexing='ij')
    
        zz_m = (2*center - zz) % G
        yy_m = (2*center - yy) % G
        xx_m = (2*center - xx) % G
    
        vol_sym = 0.5 * (vol_f + torch.conj(vol_f[zz_m, yy_m, xx_m]))
    
        return vol_sym
 
 
    def _rot(self, axis, theta, device='cuda'):
        # Creamos el tensor directamente en el dispositivo objetivo (evita transferencias lentas)
        axis = torch.tensor(axis, dtype=torch.float32, device=device)
        axis = axis / torch.linalg.norm(axis)
        x, y, z = axis
        
        # Creamos los escalares trigonométricos como tensores en el mismo dispositivo
        theta_tensor = torch.tensor(theta, device=device)
        c = torch.cos(theta_tensor)
        s = torch.sin(theta_tensor)
        C = 1. - c
        
        # Construimos la matriz final directamente en la GPU/CPU
        return torch.tensor([
            [c + x * x * C,     x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s,   c + y * y * C,   y * z * C - x * s],
            [z * x * C - y * s,   z * y * C + x * s, c + z * z * C  ],
        ], device=device)   
    
    def generate_symmetry(self, sym, device='cuda'):
        s = sym.strip().lower()
    
        if s.startswith('c') and s[1:].isdigit():
            n = int(s[1:])
            # Pasamos el "device" a cada llamada de _rot
            return torch.stack([self._rot([0,0,1], 2. * math.pi * k / n, device=device) for k in range(n)])
    
        if s.startswith('d') and s[1:].isdigit():
            n = int(s[1:])
            mats = [self._rot([0,0,1], 2. * math.pi * k / n, device=device) for k in range(n)]
            for k in range(n):
                phi = math.pi * k / n
                axis = [math.cos(phi), math.sin(phi), 0.0]
                mats.append(self._rot(axis, math.pi, device=device))
            return torch.stack(mats)
    
        raise ValueError(f"Unsupported or unknown symmetry label: {sym}")
    
    
    def generate_symmetry_old(self, sym, device='cuda', dtype=torch.float32):
        """
        sym: string tipo 'C4', 'D7', 'C1', etc.
        """
    
        sym = sym.upper()
    
        if sym.startswith('C'):
            n = int(sym[1:])
            return self._generate_Cn(n, device=device, dtype=dtype)
    
        elif sym.startswith('D'):
            n = int(sym[1:])
            return self._generate_Dn(n, device, dtype)
    
        else:
            raise ValueError(f"Simetría no soportada: {sym}")
        
    def _generate_Cn(self, n, device, dtype):
        
        ops = []
        
        if n==1:
            n=2
    
        for k in range(n-1):
            theta = 2 * math.pi * k / n
    
            R = torch.tensor([
                [ math.cos(theta), -math.sin(theta), 0.0],
                [ math.sin(theta),  math.cos(theta), 0.0],
                [ 0.0,              0.0,             1.0]
            ], dtype=dtype, device=device)
    
            ops.append(R)
    
        return torch.stack(ops)  # (n, 3, 3)
    
        
    def _generate_Dn(self, n, device, dtype):

        ops = []
    
        # =========================
        # 1. Rotaciones Cn (eje Z)
        # =========================
        for k in range(n-1):
            theta = 2 * math.pi * k / n
            c, s = math.cos(theta), math.sin(theta)
    
            Rz = torch.tensor([
                [c, -s, 0.0],
                [s,  c, 0.0],
                [0.0, 0.0, 1.0]
            ], dtype=dtype, device=device)
    
            ops.append(Rz)
    
        # =========================
        # 2. C2 base (eje X)
        # =========================
        C2x = torch.tensor([
            [1.0,  0.0,  0.0],
            [0.0, -1.0,  0.0],
            [0.0,  0.0, -1.0]
        ], dtype=dtype, device=device)
    
        # =========================
        # 3. Generar C2 por conjugación
        # =========================
        for k in range(n):
            theta = 2 * math.pi * k / n
            c, s = math.cos(theta), math.sin(theta)
    
            Rz = torch.tensor([
                [c, -s, 0.0],
                [s,  c, 0.0],
                [0.0, 0.0, 1.0]
            ], dtype=dtype, device=device)
    
            C2k = Rz @ C2x @ Rz.T
            ops.append(C2k)
    
        sym_ops = torch.stack(ops)  # (2n, 3, 3)
    
        return sym_ops
    
    
    @torch.no_grad()
    def precompute_blob_table(self, radius = 1.9, table_size = 10000, beta = 15.0, device = "cuda"):
        
        device = torch.device(device)
    
        # ✔ dominio XMIPP correcto: cubo 3D → diagonal máxima
        max_r2 = 3.0 * radius * radius
    
        r2 = torch.linspace(0.0, max_r2, table_size, device=device)
    
        r = torch.sqrt(r2)
    
        inside = torch.clamp(1.0 - (r / radius) ** 2, 0.0, 1.0)
    
        table = torch.i0(beta * torch.sqrt(inside))
        table = table / torch.i0(torch.tensor(beta, device=device))
    
        return table
    
    @torch.no_grad()
    def precompute_blob_table_corregir(self, Xdim, oversamp, radius=1.9, beta=15.0, table_size=10000, device="cuda"):
    
    
        device = torch.device(device)
    
        k_max = torch.sqrt(torch.tensor(3.0, device=device)) * (Xdim / 2.0)
        delta = k_max / (table_size - 1)
        k = torch.arange(table_size, device=device, dtype=torch.float32) * delta
    
        radius_fourier = radius / (oversamp * Xdim)
    
        w = 2.0 * torch.pi * k * radius_fourier
    
        z = torch.sqrt(beta**2 - w**2 + 0j)
        blob_ft = torch.sinh(z) / z
    
        # ✔ normalización (más estable que sinh(beta)/beta)
        iw0 = 1.0 / (torch.sinh(torch.tensor(beta, device=device)) / beta)
        blob_ft = blob_ft.real * iw0
    
        # ✔ volumen discreto
        # padXdim3 = (oversamp * Xdim) ** 3
        # blob_ft = blob_ft * padXdim3
    
        return blob_ft
    
    
    @torch.no_grad()

    def reconstruct_volume_sym(self, mmap, sym, resol, sampling, volume_size, rotations, shifts = None, radius=1.9,
                            oversamp=2.0, batch_size=32):
        
        device = torch.device(self.cuda)
            
        N, H, W = mmap.data.shape
        D = volume_size
        
        f_px = (1.0 / resol) * sampling
        f_px = min(f_px, 0.5)
    
        G = int(oversamp * D)
        if G % 2 == 0:
            G += 1
            
        # Definimos el tamaño de la última dimensión (X) a la mitad
        H_x = G // 2 + 1
    
        rotations = rotations.to(device)
        blob_table = self.precompute_blob_table().to(device)
    
        # =========================
        # SIMETRÍA
        # =========================
        sym_ops = self.generate_symmetry(sym, device=device)
        # sym_ops = generate_symmetry(sym)
        n_sym = sym_ops.shape[0]
        
        # print(f"→ Simetría: {sym} ({n_sym} ops)")
    
        # -------------------------
        # Fourier grid 2D
        # -------------------------
        freq = torch.fft.fftshift(torch.fft.fftfreq(H, device=device))
        uy, ux = torch.meshgrid(freq, freq, indexing="ij")
    
        k2 = ux**2 + uy**2
        valid_mask = (k2 <= f_px**2).reshape(-1)
    
        coords2d = torch.stack(
            [ux.reshape(-1), uy.reshape(-1), torch.zeros(H * H, device=device)],
            dim=0,
        )[:, valid_mask].contiguous().unsqueeze(0)
    
        # -------------------------
        # Volumen MITAD (Optimizado en Memoria)
        # -------------------------
        vol_f = torch.zeros((G, G, H_x), dtype=torch.complex64, device=device)
        weight = torch.zeros((G, G, H_x), dtype=torch.float32, device=device)
    
        # -------------------------
        # Blob (Vecindad corregida a 3x3x3 o 5x5x5 según radio)
        # -------------------------
        max_r2 = 3.0 * radius * radius
        iDelta = (blob_table.numel() - 1) / max_r2
    
        o = torch.tensor([-1, 0, 1], device=device)
        ox, oy, oz = torch.meshgrid(o, o, o, indexing="ij")
        ox = ox.reshape(-1)
        oy = oy.reshape(-1)
        oz = oz.reshape(-1)
    
        ACC = 1e-8
    
        # =====================================================
        # MAIN LOOP
        # =====================================================
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            b = end - start
    
            rot = rotations[start:end]  # (B,3,3)
    
    
            # imgs = torch.as_tensor(mmap.data[start:end], device=device, dtype=torch.float32)
            imgs = torch.as_tensor(np.array(mmap.data[start:end]), device=device, dtype=torch.float32)
            # imgs = mmap.data[start:end]
    
            # imgs = (imgs - imgs.mean(dim=(-2, -1), keepdim=True)) / (
            #     imgs.std(dim=(-2, -1), keepdim=True) + 1e-8
            # )
    
            proj_fft = torch.fft.fftshift(
                torch.fft.fft2(torch.fft.ifftshift(imgs, dim=(-2, -1)), norm="forward"),
                dim=(-2, -1),
            )
            
            if shifts is not None:
                batch_shifts = shifts[start:end].to(device)
                proj_fft = self.apply_fourier_shift(proj_fft, ux, uy, batch_shifts)
    
            vals = proj_fft.reshape(b, -1)[:, valid_mask]
            
            # -------------------------
            # Expansión de Simetría
            # -------------------------
            rot_all = []
            vals_all = []
            for S in sym_ops:
                rot_all.append(S @ rot)
                vals_all.append(vals)
    
            rot_all = torch.cat(rot_all, dim=0)   # (B * n_sym, 3, 3)
            vals_all = torch.cat(vals_all, dim=0) # (B * n_sym, n_valid)
            
            coords3d = torch.bmm(rot_all, coords2d.expand(rot_all.shape[0], -1, -1))
    
            # =====================================================
            # OPTIMIZACIÓN HERMITIANA: FORZAR X >= 0 (IN-PLACE)
            # =====================================================
            flip_mask = (coords3d[:, 0, :] < 0) 
            if flip_mask.any():
                sign = 1.0 - 2.0 * flip_mask.to(coords3d.dtype)
                coords3d *= sign.unsqueeze(1)
                # Conjugamos selectivamente en las posiciones necesarias sin reasignar el tensor completo
                vals_all[flip_mask] = torch.conj(vals_all[flip_mask])
                del sign
            
            vals = vals_all.reshape(-1)
            del flip_mask, vals_all
    
            # -------------------------
            # GRIDDING
            # -------------------------
            grid = (coords3d + 0.5) * (G - 1)
            del coords3d
    
            x = grid[:, 0, :].reshape(-1)
            y = grid[:, 1, :].reshape(-1)
            z = grid[:, 2, :].reshape(-1)
            del grid
    
            x0 = x.floor().long()
            y0 = y.floor().long()
            z0 = z.floor().long()
    
            # --- FASE 1: CALCULAR APORTES DEL BLOB (Liberación temprana de VRAM) ---
            fx = (x - x0).unsqueeze(1)
            fy = (y - y0).unsqueeze(1)
            fz = (z - z0).unsqueeze(1)
            del x, y, z
    
            dx = fx - ox
            dy = fy - oy
            dz = fz - oz
            del fx, fy, fz
    
            r2 = dx * dx + dy * dy + dz * dz
            del dx, dy, dz
    
            idx = (r2 * iDelta).round().long().clamp_(0, blob_table.numel() - 1)
            del r2
    
            w = blob_table[idx]
            del idx
    
            contrib = (vals.unsqueeze(1) * w)
    
            # --- FASE 2: ASIGNAR COORDENADAS DE VECINOS ---
            ix = x0[:, None] + ox
            iy = y0[:, None] + oy
            iz = z0[:, None] + oz
            del x0, y0, z0
    
            # =====================================================
            # GESTIÓN SPARSEA DE VECINOS FRONTERA (IN-PLACE)
            # =====================================================
            cross_mask = (ix < (G // 2))
            if cross_mask.any():
                # Modificación quirúrgica in-place únicamente para los hilos que cruzaron el plano
                ix[cross_mask] = (G - 1) - ix[cross_mask]
                iy[cross_mask] = (G - 1) - iy[cross_mask]
                iz[cross_mask] = (G - 1) - iz[cross_mask]
                contrib[cross_mask] = torch.conj(contrib[cross_mask])
            del cross_mask
        
            iz.clamp_(0, G - 1)
            iy.clamp_(0, G - 1)
            ix.clamp_(G // 2, G - 1) 
    
            # Mapeamos al índice de la grilla mitad
            ix -= (G // 2)
    
            # Aplanado directo
            flat = (iz * (G * H_x) + iy * H_x + ix).reshape(-1)
            del ix, iy, iz
    
            # Inserción indexada limpia
            # vol_f.view(-1).index_add_(0, flat, contrib)
            vol_f.view(-1).index_add_(0, flat, contrib.reshape(-1))
            weight.view(-1).index_add_(0, flat, w.reshape(-1))
            del flat, contrib, w
    
        # -------------------------
        # NORMALIZAR
        # -------------------------    
        pad2D = 1
        corr2D_3D = (pad2D**2) / (H * (oversamp**3))
        mask = weight > ACC
        vol_f[mask] /= weight[mask]
        vol_f[mask] *= corr2D_3D
        vol_f[~mask] = 0.0
        del mask
    
        # -------------------------
        # IFFT REAL OPTIMIZADA (irfftn)
        # -------------------------
        vol_f_unshifted = torch.fft.ifftshift(vol_f, dim=(0, 1))
        del vol_f
        
        vol_real = torch.fft.irfftn(vol_f_unshifted, s=(G, G, G), norm="forward")
        del vol_f_unshifted
        
        vol_real = torch.fft.fftshift(vol_real)
        
        pad = (G - D) // 2
        vol_real = vol_real[pad:pad + D, pad:pad + D, pad:pad + D].contiguous()
    
        # ================= FINAL CORRECTION =================
        ipad_relation = (oversamp / pad2D) ** 3
        coords = torch.arange(-D//2, D//2 + (D % 2), device=device, dtype=torch.float32)
        
        Z, Y, X = torch.meshgrid(coords, coords, coords, indexing='ij')
        r = torch.sqrt(X**2 + Y**2 + Z**2)
        del X, Y, Z
        
        arg = r / (2.0 * D)
        sinc = torch.where(arg.abs() < 1e-8, torch.ones_like(arg), torch.sin(torch.pi * arg) / (torch.pi * arg))
        sinc2 = sinc ** 2
        del sinc
        
        blob_fourier_table = self.precompute_blob_table_corregir(Xdim = volume_size, oversamp=oversamp)        
        num_entries = blob_fourier_table.numel()
        k_max = torch.sqrt(torch.tensor(3.0, device=device)) * (D / 2.0)
        delta = k_max / (num_entries - 1)
        iDelta = 1.0 / delta
        
        idx = (r * iDelta).round().long().clamp(0, num_entries - 1)
        del r
        blob_factor = blob_fourier_table[idx]
        del idx
        
        correction = ipad_relation * sinc2 * blob_factor
        del sinc2, blob_factor
        correction = torch.clamp(correction, min=1e-6)
        
        vol_real /= correction
        del correction
        
        # Centro de masa informativo
        # mass = torch.clamp(vol_real, min=0)
        # z_m, y_m, x_m = torch.meshgrid(torch.arange(D, device=device), torch.arange(D, device=device), torch.arange(D, device=device), indexing="ij")
        # m = mass.sum()
        # print(f"Centro de masa: {(mass * z_m).sum() / m:.2f}, {(mass * y_m).sum() / m:.2f}, {(mass * x_m).sum() / m:.2f}")
        # del mass, z_m, y_m, x_m
    
        return vol_real
    
    
    @torch.no_grad()
    def reconstruct_volume(self, mmap, sym, resol, sampling, volume_size, rotations, shifts = None, radius=1.9,
                            oversamp=2.0, batch_size=32):
        
        device = torch.device(self.cuda)
    
        N, H, W = mmap.data.shape
        D = volume_size
        
        
        f_px = (1.0 / resol) * sampling
        if f_px > 0.5:
            f_px = 0.5
        # scale = f_px / 0.5
    
        G = int(oversamp * D)# * scale)
        G += (G % 2 == 0)
        G2 = G * G
    
        rotations = rotations.to(device)
        blob_table = self.precompute_blob_table().to(device)
        # blob_table = blob_table.to(device)
    
        # =========================
        # SIMETRÍA
        # =========================
        sym_ops = self.generate_symmetry(sym, device=device)
        n_sym = sym_ops.shape[0]
    
        # print(f"→ Simetría: {sym} ({n_sym} ops)")
        # print("Matrices:")
        # print(sym_ops)
        # print(n_sym)
    
        # -------------------------
        # Fourier grid 2D
        # -------------------------
        freq = torch.fft.fftshift(torch.fft.fftfreq(H, device=device))
        uy, ux = torch.meshgrid(freq, freq, indexing="ij")
    
        k2 = ux**2 + uy**2
        valid_mask = (k2.reshape(-1) <= f_px**2)
    
        coords2d = torch.stack(
            [ux.reshape(-1), uy.reshape(-1), torch.zeros(H * H, device=device)],
            dim=0,
        )[:, valid_mask].contiguous().unsqueeze(0)
    
        n_valid = coords2d.shape[-1]
    
        # -------------------------
        # volumen
        # -------------------------
        vol_f = torch.zeros((G, G, G), dtype=torch.complex64, device=device)
        weight = torch.zeros((G, G, G), dtype=torch.float32, device=device)
    
        # -------------------------
        # blob
        # -------------------------
        max_r2 = 3.0 * radius * radius
        iDelta = (blob_table.numel() - 1) / max_r2
    
        # offsets
        o = torch.tensor([-1, 0, 1], device=device)
        ox, oy, oz = torch.meshgrid(o, o, o, indexing="ij")
        ox = ox.reshape(-1)
        oy = oy.reshape(-1)
        oz = oz.reshape(-1)
    
        ACC = 1e-8
    
        # =====================================================
        # MAIN LOOP
        # =====================================================
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            b = end - start
    
            rot = rotations[start:end]  # (B,3,3)
    
    
            # imgs = torch.as_tensor(mmap.data[start:end], device=device, dtype=torch.float32)
            imgs = torch.as_tensor(np.array(mmap.data[start:end]), device=device, dtype=torch.float32)
            # imgs = mmap.data[start:end]
    
            # imgs = (imgs - imgs.mean(dim=(-2, -1), keepdim=True)) / (
            #     imgs.std(dim=(-2, -1), keepdim=True) + 1e-8
            # )
    
            proj_fft = torch.fft.fftshift(
                torch.fft.fft2(torch.fft.ifftshift(imgs, dim=(-2, -1)), norm="forward"),
                dim=(-2, -1),
            )
            
            if shifts is not None:
                batch_shifts = shifts[start:end].to(device)
                proj_fft = self.apply_fourier_shift(proj_fft, ux, uy, batch_shifts)
                
    
            vals = proj_fft.reshape(b, -1)[:, valid_mask]  # (B, n_valid)
    
            # =====================================================
            #  APLICAR SIMETRÍA CORRECTAMENTE
            # =====================================================
            if n_sym > 1:
                # rot_total = rot @ sym
                rot_sym = torch.einsum("bij,sjk->bsik", rot, sym_ops)
                rot_sym = rot_sym.reshape(-1, 3, 3)  # (B*n_sym,3,3)
    
                coords3d = torch.bmm(
                    rot_sym,
                    coords2d.expand(rot_sym.shape[0], -1, -1)
                )
    
                #repetir valores (SIN dividir)
                vals = vals[:, None, :].expand(b, n_sym, n_valid).reshape(-1)
                # vals = vals.unsqueeze(1).repeat(1, n_sym, 1).reshape(-1)
    
            else:
                coords3d = torch.bmm(rot, coords2d.expand(b, -1, -1))
                vals = vals.reshape(-1)
    
            # -------------------------
            # GRID
            # -------------------------
            grid = (coords3d + 0.5) * (G - 1)
    
            x = grid[:, 0].reshape(-1)
            y = grid[:, 1].reshape(-1)
            z = grid[:, 2].reshape(-1)
    
            x0 = x.floor().long()
            y0 = y.floor().long()
            z0 = z.floor().long()
    
            ix = x0[:, None] + ox
            iy = y0[:, None] + oy
            iz = z0[:, None] + oz
    
            ix.clamp_(0, G - 1)
            iy.clamp_(0, G - 1)
            iz.clamp_(0, G - 1)
    
            dx = x.unsqueeze(1) - ix
            dy = y.unsqueeze(1) - iy
            dz = z.unsqueeze(1) - iz
    
            r2 = dx * dx + dy * dy + dz * dz
    
            idx = (r2 * iDelta).round().long().clamp_(0, blob_table.numel() - 1)
            w = blob_table[idx]
    
            flat = (iz * G2 + iy * G + ix).reshape(-1)
            contrib = (vals.unsqueeze(1) * w).reshape(-1)
    
            vol_f.view(-1).index_add_(0, flat, contrib)
            weight.view(-1).index_add_(0, flat, w.reshape(-1))
            # weight.view(-1).index_add_(0, flat, (w / n_sym).reshape(-1))
    
        # -------------------------
        # NORMALIZAR
        # -------------------------    
        pad2D = 1
        corr2D_3D = (pad2D**2) / (H * (oversamp**3))
        mask = weight > ACC
        vol_f[mask] /= weight[mask]
        vol_f[mask] *= corr2D_3D
        vol_f[~mask] = 0.0
    
        
        vol_f = self.enforce_hermitian_symmetry(vol_f)
    
        # -------------------------
        # IFFT
        # -------------------------
        vol_real = torch.fft.ifftn(torch.fft.ifftshift(vol_f), norm="forward").real
        vol_real = torch.fft.fftshift(vol_real)
        
        pad = (G - D) // 2
        vol_real =  vol_real[pad:pad + D, pad:pad + D, pad:pad + D]
    
        # ================= FINAL CORRECTION =================
        
        ipad_relation = (oversamp / pad2D) ** 3   # equivalente XMIPP
        
        # ---------- SINC² ----------
        coords = torch.arange(-D//2, D//2 + (D % 2),
                              device=device, dtype=torch.float32)
        # coords = torch.fft.fftshift(torch.fft.fftfreq(D, device=device)) * D
        
        Z, Y, X = torch.meshgrid(coords, coords, coords, indexing='ij')
        r = torch.sqrt(X**2 + Y**2 + Z**2)
        
        # XMIPP: sinc2(radius / (2*imgSize))
        arg = r / (2.0 * D)
        sinc = torch.where(
            arg.abs() < 1e-8,
            torch.ones_like(arg),
            torch.sin(torch.pi * arg) / (torch.pi * arg)
        )
        
        sinc2 = sinc ** 2
        
        
        # ---------- Fourier blob ----------
        blob_fourier_table = self.precompute_blob_table_corregir(Xdim = volume_size, oversamp=oversamp)
        num_entries = blob_fourier_table.numel()
        
        k_max = torch.sqrt(torch.tensor(3.0, device=device)) * (D / 2.0)
        delta = k_max / (num_entries - 1)
        iDelta = 1.0 / delta
        
        idx = (r * iDelta).round().long().clamp(0, num_entries - 1)
        blob_factor = blob_fourier_table[idx]
        
        
        # ---------- Corrección final ----------
        correction = ipad_relation * sinc2 * blob_factor
        correction = torch.clamp(correction, min=1e-6)
        
        vol_real = vol_real / correction
        
        # mean_factor2 = sinc2.mean()
        # vol_real = vol_real * mean_factor2
        
    
        # # meanFactor2 (muy importante)
        # mean_factor2 = sinc2.mean()
        # vol_real = vol_real * mean_factor2
        #
        # # Normalización final suave (sin std)
        # vol_real = vol_real - vol_real.mean()
        # vol_real = torch.clamp(vol_real, min=0.0)
    
        return vol_real.contiguous()
    
    def apply_fourier_shift(self, proj_fft, ux, uy, shifts):
        
        shifts = shifts.to(torch.float32)
        ux = ux.to(torch.float32)
        uy = uy.to(torch.float32)

        arg = -2j * torch.pi * (ux[None, :, :] * shifts[:, 0, None, None] + 
                                uy[None, :, :] * shifts[:, 1, None, None])
        
        phase_shift = torch.exp(arg)
        return proj_fft * phase_shift
    
    
    # @torch.no_grad()
    # def euler_zyz_to_matrix(self, psi, rot, tilt, degrees=True):
    #     """
    #     Convención:
    #         R = Rz(rot) @ Ry(tilt) @ Rz(psi)
    #
    #     Parámetros:
    #     ----------
    #     psi, rot, tilt : Tensor o array-like
    #
    #     Devuelve
    #     --------
    #     R : (N,3,3)
    #         Matrices de rotación.
    #     """
    #
    #     psi = torch.as_tensor(psi, dtype=torch.float32, device=self.cuda)
    #     rot = torch.as_tensor(rot, dtype=torch.float32, device=self.cuda)
    #     tilt = torch.as_tensor(tilt, dtype=torch.float32, device=self.cuda)
    #
    #     if degrees:
    #         psi  = torch.deg2rad(psi)
    #         rot  = torch.deg2rad(rot)
    #         tilt = torch.deg2rad(tilt)
    #
    #     ca = torch.cos(rot)
    #     sa = torch.sin(rot)
    #
    #     cb = torch.cos(tilt)
    #     sb = torch.sin(tilt)
    #
    #     cc = torch.cos(psi)
    #     sc = torch.sin(psi)
    #
    #     n = psi.shape[0]
    #
    #     R = torch.empty((n, 3, 3), dtype=torch.float32, device=self.cuda)
    #
    #     # Fila 1
    #     R[:, 0, 0] = ca * cb * cc - sa * sc
    #     R[:, 0, 1] = -ca * cb * sc - sa * cc
    #     R[:, 0, 2] = ca * sb
    #
    #     # Fila 2
    #     R[:, 1, 0] = sa * cb * cc + ca * sc
    #     R[:, 1, 1] = -sa * cb * sc + ca * cc
    #     R[:, 1, 2] = sa * sb
    #
    #     # Fila 3
    #     R[:, 2, 0] = -sb * cc
    #     R[:, 2, 1] = sb * sc
    #     R[:, 2, 2] = cb
    #
    #     return R
    #
    
    @torch.no_grad()
    def generate_library_sym(self, angular_step_deg: float = 3.0,
                               n_psi: int = 1,
                               sym: str = "C1",
                               return_angles: bool = True):
        """
          Devuelve:
            R: (N, 3, 3)
            angles: (N, 3) → (psi, rot, tilt) en grados (opcional)
        """
        device = torch.device(self.cuda)
        step = float(angular_step_deg)
        
        # ==================== LÍMITES DE LA UNIDAD ASIMÉTRICA ====================
        # Llamamos a nuestra función auxiliar para obtener los límites de la esfera
        max_tilt, max_rot = self.get_symmetry_limits(sym)
    
        # ==================== TILT ====================
        n_tilt = int(round(max_tilt / step)) + 1
        tilt_deg = torch.linspace(0.0, max_tilt, n_tilt, device=device)
        sin_t = torch.sin(torch.deg2rad(tilt_deg))
    
        # ==================== ROT POR PARALELO ====================
        n_rot_per_tilt = torch.where(
            sin_t < 1e-6,
            torch.ones_like(sin_t, dtype=torch.long),
            torch.clamp(torch.round(max_rot / (step / sin_t)).long(), min=1)
        )
    
        cum_nrot = torch.cumsum(n_rot_per_tilt, dim=0)
        total_rot = cum_nrot[-1].item()
    
        # ==================== INDICES ====================
        tilt_idx = torch.repeat_interleave(torch.arange(n_tilt, device=device), n_rot_per_tilt)
    
        rot_global_idx = torch.arange(total_rot, device=device)
        start_idx = torch.cat([torch.tensor([0], device=device), cum_nrot[:-1]])
        local_idx = rot_global_idx - start_idx[tilt_idx]
    
        # ==================== ÁNGULOS ====================
        rot_deg = (local_idx.float() / n_rot_per_tilt[tilt_idx].float()) * max_rot
        tilt_deg_exp = tilt_deg[tilt_idx]
    
        # ==================== PSI ====================
        if n_psi > 1:
            psi_deg = torch.linspace(0.0, 360.0, n_psi, device=device)[:-1]
    
            rot_deg = rot_deg.repeat_interleave(n_psi)
            tilt_deg_exp = tilt_deg_exp.repeat_interleave(n_psi)
            psi_deg = psi_deg.repeat(total_rot)
        else:
            psi_deg = torch.zeros_like(rot_deg)
    
        # ==================== MATRICES ====================
        assess = evaluation()
        R = assess.euler_zyz_to_matrix(
            psi=psi_deg,
            rot=rot_deg,
            tilt=tilt_deg_exp,
            degrees=True
        )
    
        # print(f"→ Librería generada: {R.shape[0]:,} vistas únicas (ASU para simetría {sym.upper()})")
    
        if return_angles:
            angles = torch.stack([psi_deg, rot_deg, tilt_deg_exp], dim=1)
            return R, angles
        else:
            return R
        
    
    def get_symmetry_limits(self, sym: str):
        """
        Calcula los límites de la Unidad Asimétrica (ASU) para CUALQUIER simetría Cn o Dn.
        Devuelve: (max_tilt, max_rot) en grados.
        """
        sym = sym.upper().strip()
        
        # Caso base sin simetría o simetría trivial
        if sym in ["C1", "C", ""]:
            return 180.0, 360.0
            
        # Extraemos la letra (C o D) y el número (n) usando expresiones regulares
        # Esto nos protege de formatos como "C3", "C 7", "D_12", etc.
        match = re.match(r"([CD])\s*_*(\d+)", sym)
        
        if match:
            letter = match.group(1)
            n = int(match.group(2))
            n = max(1, n) # Previene división por cero si accidentalmente ponen "C0"
            
            # --- Simetría C_n (Cíclica) ---
            if letter == 'C':
                max_tilt = 180.0
                max_rot = 360.0 / n
                
            # --- Simetría D_n (Diédrica) ---
            elif letter == 'D':
                max_tilt = 90.0
                max_rot = 360.0 / n
                
            return max_tilt, max_rot
            
        else:
            print(f"⚠️ Advertencia: Formato de simetría '{sym}' no reconocido. Usando esfera completa (C1).")
            return 180.0, 360.0

    
    @torch.no_grad()
    def generate_library(self, angular_step_deg: float = 3.0,
                                          n_psi: int = 1,
                                          return_angles: bool = True):
        """
          Devuelve:
            R: (N, 3, 3)
            angles: (N, 3) → (psi, rot, tilt) en grados (opcional)
        """
        device = torch.device(self.cuda)
        step = float(angular_step_deg)
    
        # ==================== TILT ====================
        n_tilt = int(round(180.0 / step)) + 1
        tilt_deg = torch.linspace(0.0, 180.0, n_tilt, device=device)

        sin_t = torch.sin(torch.deg2rad(tilt_deg))
    
        # ==================== ROT POR PARALELO ====================
        n_rot_per_tilt = torch.where(
            sin_t < 1e-6,
            torch.ones_like(sin_t, dtype=torch.long),
            torch.clamp(torch.round(360.0 / (step / sin_t)).long(), min=1)
        )
    
        cum_nrot = torch.cumsum(n_rot_per_tilt, dim=0)
        total_rot = cum_nrot[-1].item()
    
        # ==================== INDICES ====================
        tilt_idx = torch.repeat_interleave(torch.arange(n_tilt, device=device), n_rot_per_tilt)
    
        rot_global_idx = torch.arange(total_rot, device=device)
        start_idx = torch.cat([torch.tensor([0], device=device), cum_nrot[:-1]])
        local_idx = rot_global_idx - start_idx[tilt_idx]
    
        # ==================== ÁNGULOS ====================
        rot_deg = (local_idx.float() / n_rot_per_tilt[tilt_idx].float()) * 360.0
        tilt_deg_exp = tilt_deg[tilt_idx]
    
        # ==================== PSI ====================
        if n_psi > 1:
            psi_deg = torch.linspace(0.0, 360.0, n_psi, device=device)[:-1]
    
            rot_deg = rot_deg.repeat_interleave(n_psi)
            tilt_deg_exp = tilt_deg_exp.repeat_interleave(n_psi)
            psi_deg = psi_deg.repeat(total_rot)
        else:
            psi_deg = torch.zeros_like(rot_deg)
    
        # ==================== MATRICES (USANDO FUNCIÓN UNIVERSAL) ====================
        assess = evaluation()
        R = assess.euler_zyz_to_matrix(
            psi=psi_deg,
            rot=rot_deg,
            tilt=tilt_deg_exp,
            degrees=True
        )
    
        # print(f"Generadas {R.shape[0]:,} rotaciones")
        # print(f"   Angular step: {angular_step_deg}° | Psi samples: {n_psi}")
    
        if return_angles:
            angles = torch.stack([psi_deg, rot_deg, tilt_deg_exp], dim=1)
            return R, angles
        else:
            return R
        
    
    @torch.no_grad()
    def generate_library_180(self, angular_step_deg: float = 3.0,
                                          n_psi: int = 1,
                                          return_angles: bool = True):
        device = torch.device(self.cuda)
        step = float(angular_step_deg)
    
        # ==================== TILT (0 a 180) ====================
        n_tilt = int(round(180.0 / step)) + 1
        tilt_deg = torch.linspace(0.0, 180.0, n_tilt, device=device)
        tilt_rad = tilt_deg * torch.pi / 180.0
        sin_t = torch.sin(tilt_rad)
    
        # ==================== ROT POR PARALELO ====================
        n_rot_per_tilt = torch.where(
            sin_t < 1e-6,
            torch.ones_like(sin_t, dtype=torch.long),
            torch.clamp(torch.round(360.0 / (step / sin_t)).long(), min=1)
        )
    
        cum_nrot = torch.cumsum(n_rot_per_tilt, dim=0)
        total_rot = cum_nrot[-1].item()
    
        # ==================== INDICES ====================
        tilt_idx = torch.repeat_interleave(torch.arange(n_tilt, device=device), n_rot_per_tilt)
        rot_global_idx = torch.arange(total_rot, device=device)
        start_idx = torch.cat([torch.tensor([0], device=device), cum_nrot[:-1]])
        local_idx = rot_global_idx - start_idx[tilt_idx]
    
        # ==================== ÁNGULOS (AJUSTE -180 a 180) ====================
        rot_deg = (local_idx.float() / n_rot_per_tilt[tilt_idx].float()) * 360.0
        
        #  Transformar de [0, 360) a (-180, 180]
        rot_deg = torch.where(rot_deg > 180.0, rot_deg - 360.0, rot_deg)
        
        tilt_deg_exp = tilt_deg[tilt_idx]
    
        rot_rad = rot_deg * torch.pi / 180.0
        tilt_rad_exp = tilt_deg_exp * torch.pi / 180.0
    
        # ==================== MATRICES (ZYZ, psi=0) ====================
        c1 = torch.cos(rot_rad)
        s1 = torch.sin(rot_rad)
        c2 = torch.cos(tilt_rad_exp)
        s2 = torch.sin(tilt_rad_exp)
    
        R = torch.zeros((total_rot, 3, 3), device=device)
        R[:, 0, 0] = c1 * c2
        R[:, 0, 1] = -s1
        R[:, 0, 2] = c1 * s2
        R[:, 1, 0] = s1 * c2
        R[:, 1, 1] = c1
        R[:, 1, 2] = s1 * s2
        R[:, 2, 0] = -s2
        R[:, 2, 1] = 0.0
        R[:, 2, 2] = c2
    
        # ==================== PSI (AJUSTE -180 a 180) ====================
        if n_psi > 1:
            # Generamos de 0 a 360 y luego remapeamos
            psi_deg = torch.linspace(0.0, 360.0, n_psi + 1, device=device)[:-1]
            
            #  Transformar de [0, 360) a (-180, 180]
            psi_deg = torch.where(psi_deg > 180.0, psi_deg - 360.0, psi_deg)
            
            psi_rad = psi_deg * torch.pi / 180.0
            cos_p = torch.cos(psi_rad)
            sin_p = torch.sin(psi_rad)
    
            R_psi = torch.zeros((len(psi_rad), 3, 3), device=device)
            R_psi[:, 0, 0] = cos_p
            R_psi[:, 0, 1] = -sin_p
            R_psi[:, 1, 0] = sin_p
            R_psi[:, 1, 1] = cos_p
            R_psi[:, 2, 2] = 1.0
    
            R = torch.einsum('bij,njk->bnik', R, R_psi).reshape(-1, 3, 3)
    
            # Expandir ángulos para el return
            rot_deg = rot_deg.repeat_interleave(len(psi_rad))
            tilt_deg_exp = tilt_deg_exp.repeat_interleave(len(psi_rad))
            psi_deg = psi_deg.repeat(total_rot)
    
        else:
            psi_deg = torch.zeros_like(rot_deg)
    
        if return_angles:
            # Según tu código anterior: [psi, rot, tilt]
            angles = torch.stack([psi_deg, rot_deg, tilt_deg_exp], dim=1)
            return R, angles
        else:
            return R
        
        
        
    def generate_projections(self, vol: torch.Tensor, 
                                     R: torch.Tensor, 
                                     batch_size: int = 48)-> torch.Tensor:
        """
        Generador de proyecciones (Forward Projection).
        """
        device = torch.device(self.cuda)
        # vol = vol.to(device)
        # R = R.to(device)
    
        D = vol.shape[-1]
        N = R.shape[0]
        D_pad = D * 2
        pad = (D_pad - D) // 2
    
        #print(f"Generando proyecciones  → D={D} | Batch={batch_size} | N={N}")
    
        # 1. Padding + FFT3D (una sola vez)
        vol_padded = F.pad(vol.unsqueeze(0).unsqueeze(0), 
                           (pad, pad, pad, pad, pad, pad), 
                           mode='constant', value=0.0)
        
        vol_f = torch.fft.fftshift(torch.fft.fftn(
            torch.fft.ifftshift(vol_padded, dim=(-3,-2,-1)), 
            # norm='ortho'
            norm='forward'
        )).squeeze(0).squeeze(0)
    
        vol_f_real = vol_f.real.unsqueeze(0).unsqueeze(0)
        vol_f_imag = vol_f.imag.unsqueeze(0).unsqueeze(0)
    
        # Grid base centrado
        freqs = torch.linspace(-0.5, 0.5, D_pad, device=device)
        uy, ux = torch.meshgrid(freqs, freqs, indexing='ij')
        grid_2d = torch.stack([ux.flatten(), uy.flatten(), torch.zeros_like(ux.flatten())], dim=0)
    
        # Pre-alocación en CPU
        projections = torch.zeros((N, D, D), dtype=torch.float32, device=device)
    
        for i in range(0, N, batch_size):
            b = min(batch_size, N - i)
            R_batch = R[i:i+b]
    
            # Rotar coordenadas
            rotated_grid = torch.matmul(R_batch, grid_2d)
            sampling_coords = (rotated_grid * 2.0).transpose(1, 2).reshape(b, D_pad, D_pad, 3)
    
            # Interpolación
            s_real = F.grid_sample(vol_f_real.expand(b, -1, -1, -1, -1),
                                   sampling_coords.unsqueeze(1),
                                   mode='bilinear',
                                   padding_mode='zeros',
                                   align_corners=True).squeeze(1).squeeze(1)
    
            s_imag = F.grid_sample(vol_f_imag.expand(b, -1, -1, -1, -1),
                                   sampling_coords.unsqueeze(1),
                                   mode='bilinear',
                                   padding_mode='zeros',
                                   align_corners=True).squeeze(1).squeeze(1)
    
            slice_f = torch.complex(s_real, s_imag)
    
            # IFFT2 (sin ramp ni Wiener)
            proj = torch.fft.fftshift(
                torch.fft.ifft2(torch.fft.ifftshift(slice_f, dim=(-2,-1)), norm='forward'),
                dim=(-2,-1)
            ).real
    
            # Crop al tamaño original
            crop = (D_pad - D) // 2
            proj = proj[:, crop:crop+D, crop:crop+D]
    
            #Normalization    
            # mean = proj.mean(dim=(-2, -1), keepdim=True)
            # std = proj.std(dim=(-2, -1), keepdim=True)
            # proj = (proj - mean) / (std + 1e-8)

            projections[i:i+b] = proj.cpu()
    
    
        return projections
    
    
    def generate_random_angles_scipy(self, num_images, angle_range=(-180, 180)):
        
        rot = np.random.uniform(angle_range[0], angle_range[1], num_images)
        tilt = np.random.uniform(angle_range[0], angle_range[1], num_images)
        psi = np.zeros(num_images)
    
        # Convertir a tensor
        angles = torch.stack([
            torch.tensor(rot,  dtype=torch.float32),
            torch.tensor(tilt, dtype=torch.float32),
            torch.tensor(psi,  dtype=torch.float32)
        ], dim=1)   # (N, 3)
        # print(angles)
    
        # ====================== MATRICES DE ROTACIÓN ======================
        rot_obj = R.from_euler('ZYZ', angles.numpy(), degrees=True)
        rotations = torch.from_numpy(rot_obj.as_matrix()).float()   # (N, 3, 3)
        print(rotations)
    
        return rotations
    
    
    def generate_random_angles(self, num_images, angle_range=(-180, 180)):

        device = self.cuda
        angles_deg = torch.zeros((num_images, 3), device=device, dtype=torch.float32)
        angles_deg[:, 0:2] = torch.rand((num_images, 2), device=device) * \
                             (angle_range[1] - angle_range[0]) + angle_range[0]
    
        rad = torch.deg2rad(angles_deg)
    
        a, b, c = rad[:, 0], rad[:, 1], rad[:, 2]
    
        ca, sa = torch.cos(a), torch.sin(a)
        cb, sb = torch.cos(b), torch.sin(b)
        cc, sc = torch.cos(c), torch.sin(c)
    
        # Construcción directa de R = Rz(a) Ry(b) Rz(c)
        R = torch.zeros((num_images, 3, 3), device=self.cuda)
    
        R[:, 0, 0] = ca*cb*cc - sa*sc
        R[:, 0, 1] = -ca*cb*sc - sa*cc
        R[:, 0, 2] = ca*sb
    
        R[:, 1, 0] = sa*cb*cc + ca*sc
        R[:, 1, 1] = -sa*cb*sc + ca*cc
        R[:, 1, 2] = sa*sb
    
        R[:, 2, 0] = -sb*cc
        R[:, 2, 1] = sb*sc
        R[:, 2, 2] = cb
    
        return R
    
    
    import torch
    

    def mask_otsu_0(self, volume, num_bins=256):
        """
        Calcula el umbral de Otsu para un volumen (o set de imágenes) 
        y pone a cero todos los valores por debajo de ese umbral.

        """
        device = volume.device
        dtype = volume.dtype
        
        # 1. Aplanamos el volumen para analizar sus intensidades
        flat_vol = volume.flatten()
        
        # 2. Calculamos el histograma de forma nativa en PyTorch
        min_val, max_val = flat_vol.min(), flat_vol.max()
        hist = torch.histc(flat_vol, bins=num_bins, min=min_val, max=max_val)
        
        # Centros de los bins (los valores de gris correspondientes)
        bin_centers = torch.linspace(min_val, max_val, num_bins, device=device, dtype=dtype)
        
        # 3. Algoritmo de Otsu Vectorizado
        # Probabilidades de cada bin
        weight1 = torch.cumsum(hist, dim=0)
        weight2 = flat_vol.numel() - weight1
        
        # Medias acumuladas
        mean1 = torch.cumsum(hist * bin_centers, dim=0) / (weight1 + 1e-10)
        mean2 = (mean1[-1] * flat_vol.numel() - torch.cumsum(hist * bin_centers, dim=0)) / (weight2 + 1e-10)
        
        # Varianza inter-clase (queremos maximizar esto)
        variance_between = weight1 * weight2 * (mean1 - mean2) ** 2
        
        # El umbral óptimo es el que maximiza la varianza
        idx_max = torch.argmax(variance_between)
        otsu_threshold = bin_centers[idx_max]
        
        # 4. Aplicamos la máscara: todo lo menor al umbral se vuelve 0
        # filtered_volume = torch.where_(volume >= otsu_threshold, volume, torch.zeros_like(volume))
        filtered_volume = volume.where(volume >= otsu_threshold, torch.zeros_like(volume))
        
        # print(f"-> Umbral de Otsu calculado: {otsu_threshold.item():.4f}")
        return filtered_volume
    
    
    def mask_otsu(self, volume, num_bins=256, sigma=2.0, noise_level=0.0):
        """
        Calcula el umbral de Otsu, genera una máscara y le aplica una 
        caída suave (soft-edge) en los bordes usando un filtro Gaussiano 3D.
        
        Argumentos:
            volume: Tensor 3D (Z, Y, X) en la GPU.
            num_bins: Resolución del histograma para Otsu.
            sigma: Controla qué tan suave/ancha es la caída en los bordes (por defecto 2.0).
        """
        device = volume.device
        dtype = volume.dtype
        
        # 1. CALCULO DEL UMBRAL DE OTSU (Igual que antes)
        flat_vol = volume.flatten()
        min_val, max_val = flat_vol.min(), flat_vol.max()
        hist = torch.histc(flat_vol, bins=num_bins, min=min_val, max=max_val)
        bin_centers = torch.linspace(min_val, max_val, num_bins, device=device, dtype=dtype)
        
        weight1 = torch.cumsum(hist, dim=0)
        weight2 = flat_vol.numel() - weight1
        
        mean1 = torch.cumsum(hist * bin_centers, dim=0) / (weight1 + 1e-10)
        mean2 = (mean1[-1] * flat_vol.numel() - torch.cumsum(hist * bin_centers, dim=0)) / (weight2 + 1e-10)
        
        variance_between = weight1 * weight2 * (mean1 - mean2) ** 2
        idx_max = torch.argmax(variance_between)
        otsu_threshold = bin_centers[idx_max]
        
        # 2. CREAR MÁSCARA BINARIA (0 o 1)
        # 1 donde supera el umbral, 0 donde no
        mask = (volume >= otsu_threshold).to(dtype)
        
        # 3. CREAR UN KÉRNEL GAUSSIANO 3D NATIVO PARA EL SUAVIZADO
        # Definimos el tamaño del bloque del filtro basado en sigma
        radius = int(3 * sigma)
        kernel_size = 2 * radius + 1
        
        # Coordenadas de la rejilla del filtro 3D
        coords = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
        z, y, x = torch.meshgrid(coords, coords, coords, indexing='ij')
        dist_sq = z**2 + y**2 + x**2
        
        # Ecuación Gaussiana 3D
        kernel = torch.exp(-dist_sq / (2 * sigma**2))
        kernel = kernel / kernel.sum() # Normalizamos para que no altere la escala del volumen
        
        # Adecuamos las dimensiones del kernel para F.conv3d -> (out_channels, in_channels, Z, Y, X)
        kernel = kernel.unsqueeze(0).unsqueeze(0)
        
        # 4. APLICAR CONVOLUCIÓN 3D PARA CREAR LA CAÍDA SUAVE
        # Añadimos dimensiones de Batch y Channel a la máscara -> (1, 1, Z, Y, X)
        mask_padded = mask.unsqueeze(0).unsqueeze(0)
        
        # Usamos padding para mantener el tamaño exacto del volumen original
        soft_mask = F.conv3d(mask_padded, kernel, padding=radius)
        
        # Quitamos las dimensiones extra para regresar al tamaño original (Z, Y, X)
        soft_mask = soft_mask.squeeze(0).squeeze(0)
        
        # 5. MULTIPLICAR EL VOLUMEN POR LA MÁSCARA SUAVE
        # Conserva intacto el interior, pone a cero el fondo lejano y suaviza la frontera
        
        if noise_level > 0.0:
            std_vol = volume.std()
            # Modificamos el volumen original añadiendo la perturbación
            # volume = volume + (torch.randn_like(volume) * std_vol * noise_level)
            volume += (torch.randn_like(volume) * std_vol * noise_level)
        
        filtered_volume = volume * soft_mask
        
        return filtered_volume
    


    def apply_spherical_mask(self, vol, radius):
    
        dim = vol.shape[0] 
        device = vol.device
        
        center = dim / 2.0 - 0.5
        
        coordinates = torch.arange(dim, dtype=torch.float32, device=device)
        
        grid_z, grid_y, grid_x = torch.meshgrid(coordinates, coordinates, coordinates, indexing='ij')
        
        squared_distances = (grid_z - center)**2 + (grid_y - center)**2 + (grid_x - center)**2
        
        mask_3d = squared_distances <= (radius ** 2)
        
        return vol * mask_3d
    
    
    def filter_3d(self, volume, sampling_rate, resolution_cutoff, width=0.05):

        device = volume.device
        dtype = volume.dtype
        nz, ny, nx = volume.shape
    
        vol_fft = torch.fft.fftshift(torch.fft.fftn(volume, norm="forward"))
    

        freq_z = torch.fft.fftshift(torch.fft.fftfreq(nz, d=1.0, device=device, dtype=dtype))
        freq_y = torch.fft.fftshift(torch.fft.fftfreq(ny, d=1.0, device=device, dtype=dtype))
        freq_x = torch.fft.fftshift(torch.fft.fftfreq(nx, d=1.0, device=device, dtype=dtype))
    
        f_z, f_y, f_x = torch.meshgrid(freq_z, freq_y, freq_x, indexing='ij')
        
 
        radius = torch.sqrt(f_z**2 + f_y**2 + f_x**2) / sampling_rate
    
        # 3. Definir los parámetros del filtro Raised Cosine
        # Frecuencia crítica en base a la resolución dada (f = 1 / Resolución)
        f_cutoff = 1.0 / resolution_cutoff
        
        # Definimos dónde empieza a caer el coseno (f_in) y dónde llega a cero (f_out)
        f_in = f_cutoff - (width / (2 * sampling_rate))
        f_out = f_cutoff + (width / (2 * sampling_rate))
    
        # Construir la máscara matemática del Raised Cosine

        mask = torch.ones_like(radius)
    
        # Zona de transición: Caída suave del coseno
        # Fórmula: 0.5 * (1 + cos(pi * (f - f_in) / (f_out - f_in)))
        transition_mask = radius >= f_in
        cos_part = 0.5 * (1.0 + torch.cos(math.pi * (radius - f_in) / (f_out - f_in)))
        mask = torch.where(transition_mask, cos_part, mask)
    
        # Zona exterior: Todo a cero por encima de f_out
        mask = torch.where(radius > f_out, torch.zeros_like(mask), mask)
    
        filtered_fft = vol_fft * mask
        
        filtered_volume = torch.fft.ifftn(torch.fft.ifftshift(filtered_fft), norm="forward")
    
        # Retornamos la parte real (las pequeñas partes imaginarias son ruido numérico flotante)
        return filtered_volume.real
    
    
    def generate_random_ellipsoid(self, dim, radius, device):
        """
        Genera una semilla elipsoidal (patata) gaussiana con deformaciones
        aleatorias en sus ejes para inicializaciones multi-referencia.
        """
        # 1. Creamos la cuadrícula 3D centrada
        z, y, x = torch.meshgrid(
            torch.linspace(-dim/2, dim/2, dim, device=device),
            torch.linspace(-dim/2, dim/2, dim, device=device),
            torch.linspace(-dim/2, dim/2, dim, device=device),
            indexing='ij'
        )
        
        # 2. Generamos factores de deformación al azar (entre 0.7 y 1.3)
        fx = 0.7 + np.random.rand() * 0.6
        fy = 0.7 + np.random.rand() * 0.6
        fz = 0.7 + np.random.rand() * 0.6
        
        # 3. Calculamos la distancia elipsoidal modificada
        distancia_elipsoide = torch.sqrt((x * fx)**2 + (y * fy)**2 + (z * fz)**2)
        
        # 4. Generamos el volumen suave con su deformación única
        zeroVol = torch.exp(-((distancia_elipsoide) / radius)**2 * 5.0)
        
        return zeroVol
    
    
    def generate_scaffolding_seed(self, mmap, R, sym, sampling, dim, device):
        """
        Genera un volumen inicial seleccionando 3 class averages al azar
        y pasándoselas a reconstruct_volume_sym en orientaciones ortogonales puras.
        """
        import numpy as np
        import torch
        
        # 1. Mini-clase wrapper para imitar la estructura que espera tu función (.data)
        class DataWrapper:
            def __init__(self, array):
                self.data = array
    
        # 2. Elegimos 3 imágenes al azar sin repetir
        n_images = mmap.data.shape[0]
        chosen_indices = np.random.choice(n_images, size=3, replace=False)
        
        # Extraemos y envolvemos en nuestro wrapper
        subset_images = mmap.data[chosen_indices].astype('float32')
        wrapped_mmap = DataWrapper(subset_images)
        
        # 3. CONSTRUCCIÓN DE MATRICES ORTOGONALES (3, 3, 3)
        # Tu función espera matrices de rotación reales de 3x3 para cada imagen.
        # Definimos: Identidad (Frente), Rotación 90° en X (Perfil), Rotación 90° en Y (Planta)
        r_frente = torch.tensor([[1.0, 0.0, 0.0],
                                 [0.0, 1.0, 0.0],
                                 [0.0, 0.0, 1.0]], dtype=torch.float32)
        
        r_perfil = torch.tensor([[1.0, 0.0, 0.0],
                                 [0.0, 0.0, -1.0],
                                 [0.0, 1.0, 0.0]], dtype=torch.float32)
        
        r_planta = torch.tensor([[0.0, 0.0, 1.0],
                                 [0.0, 1.0, 0.0],
                                 [-1.0, 0.0, 0.0]], dtype=torch.float32)
        
        # Empaquetamos las 3 matrices en un único tensor de lote (3, 3, 3)
        ortho_rotations = torch.stack([r_frente, r_perfil, r_planta], dim=0).to(device)
        
        # 4. Llamamos a TU función respetando tus nombres de parámetros exactos
        # Usamos resol=30 para que el andamio sea suave y conecte bien las masas
        zeroVol = R.reconstruct_volume_sym(
            mmap=wrapped_mmap,
            sym=sym,
            resol=30.0,
            sampling=sampling,
            volume_size=dim,
            rotations=ortho_rotations,
            shifts=None,
            batch_size=3  # Procesamos las 3 imágenes juntas en un único bound
        )
        
        return zeroVol
        
            

    
    
