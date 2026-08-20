#!/usr/bin/env python3
"""/***************************************************************************
 *
 * Authors:    Erney Ramirez-Aportela
 *
  ***************************************************************************/
"""

import argparse
import torch
import torch.nn.functional as F
import mrcfile
import starfile
import numpy as np
import pandas as pd
from io import StringIO
import math


@torch.no_grad()
def fourier_shift_batch(imgs, shifts_x, shifts_y):

    n, h, w = imgs.shape
    device = imgs.device

    # Coordenadas de frecuencia
    ky = torch.fft.fftfreq(h, d=1.0, device=device).reshape(1, h, 1)
    kx = torch.fft.rfftfreq(w, d=1.0, device=device).reshape(1, 1, w//2 + 1)

    # Expandir shifts
    sx = shifts_x.view(n, 1, 1)
    sy = shifts_y.view(n, 1, 1)

    # Transformada real
    F = torch.fft.rfft2(imgs)  # (n,h,w//2+1), compleja

    # Fase para shift
    phase = torch.exp(-2j * torch.pi * (kx * sx + ky * sy))
    F.mul_(phase)  # inplace, ahorra memoria
    del phase

    # Transformada inversa real
    shifted = torch.fft.irfft2(F, s=(h, w))  # devuelve real
    del F

    return shifted



@torch.no_grad()
def average_aligned_particles_ctf(star_path, pix_size, output_mrc=None, device="cuda", batch_size=256):
    """Calcula el promedio alineado de partículas con shift + rot en batches."""
    
    
    particles_df = starfile.read(star_path)
    
    # 3. Extraer los datos directamente de las columnas del DataFrame
    img_paths = particles_df["image"].tolist()
    psi = particles_df["anglePsi"].values
    shiftX = particles_df["shiftX"].values 
    shiftY = particles_df["shiftY"].values 
    
    
    #ctf parameters
    voltage = float(particles_df["ctfVoltage"].values[0])
    cs = float(particles_df["ctfSphericalAberration"].values[0])
    ampC = float(particles_df["ctfQ0"].values[0])
    
    defU = particles_df["ctfDefocusU"].values.astype(np.float32)
    defV = particles_df["ctfDefocusV"].values.astype(np.float32)
    defA = particles_df["ctfDefocusAngle"].values.astype(np.float32)

    # --- Cargar stack ---
    stack_path = img_paths[0].split("@")[1]
    with mrcfile.open(stack_path, permissive=True) as mrc:
        particles = mrc.data.copy()

    n, h, w = particles.shape
    print(f"Total partículas: {n}, tamaño: {h}x{w}")

    # Convertir tensores
    particles = torch.tensor(particles, dtype=torch.float32, device=device)
    angles_rad = torch.tensor(np.deg2rad(psi), dtype=torch.float32, device=device)
    shiftX = torch.tensor(shiftX, dtype=torch.float32, device=device)
    shiftY = torch.tensor(shiftY, dtype=torch.float32, device=device)

    # --- Base grid (solo una vez) ---
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, h, device=device),
        torch.linspace(-1, 1, w, device=device),
        indexing="ij"
    )
    base_grid = torch.stack([xx, yy], dim=-1)  # (h,w,2)
    base_grid_flat = base_grid.view(-1, 2)

    # --- Promedio acumulativo ---
    avg_sum = torch.zeros((h, w), dtype=torch.float32, device=device)
    count = 0
    numerator = torch.zeros((h, w), dtype=torch.complex64, device=device)
    denominator = torch.zeros((h, w), dtype=torch.float32, device=device)

    # --- Procesar por batches ---
    for i in range(0, n, batch_size):
        j = min(i + batch_size, n)
        batch = particles[i:j]
        batch_shx = shiftX[i:j]
        batch_shy = shiftY[i:j]
        batch_ang = angles_rad[i:j]

        # Shift en Fourier
        # shifted = fourier_shift_batch(batch, batch_shx, batch_shy)

        # #rot
        cos = torch.cos(batch_ang)
        sin = torch.sin(batch_ang)
        rot_mats = torch.stack([
            torch.stack([cos, -sin], dim=1),
            torch.stack([sin,  cos], dim=1)
        ], dim=1)  # (B, 2, 2)
        grids = base_grid_flat.unsqueeze(0).matmul(rot_mats.transpose(1, 2))
        grids = grids.view(-1, h, w, 2)

        # Rotar en batch
        imgs = batch.unsqueeze(1)  # (B, 1, H, W) 
        rotated = F.grid_sample(imgs, grids, align_corners=True, padding_mode="zeros")
        
        # --- 2) Shift después (en Fourier) ---
        rotated = rotated.squeeze(1)

        shifted = fourier_shift_batch(rotated, batch_shx, batch_shy)
        # shifted_cpu = shifted.cpu().numpy().astype(np.float32)
        
        Fpart = torch.fft.fft2(shifted)
        
        # defA_rotated = defA[i:j] + psi[i:j]
        defA_rotated = defA[i:j]

        
        ctf_batch = compute_ctfs_batch(dim=h, pixel_size=pix_size, defocus_u=defU[i:j],
                                       defocus_v=defV[i:j], astig_angle_deg=defA_rotated,
                                       voltage_kv=voltage, cs_mm=cs,
                                       amp_contrast=ampC, device=device)

        # Averages
        avg_sum += shifted.sum(dim=0)
        count += shifted.shape[0]
        # numerator += (ctf_batch * Fpart).sum(dim=0)
        # denominator += (ctf_batch.square()).sum(dim=0)

        torch.cuda.empty_cache()

    avg = (avg_sum / count)
    
    # regularizer = 1e-3
    regularizer = 1e-2 * denominator.max()
    
    avg_fft = numerator / (denominator + regularizer)
    # avg = torch.real(torch.fft.ifft2(avg_fft))
    
    avg_cpu = avg.cpu().numpy().astype(np.float32)

    if output_mrc:
        with mrcfile.new(output_mrc, overwrite=True) as mrc:
            mrc.set_data(avg_cpu)
            mrc.voxel_size = pix_size

    return avg



def electron_wavelength(voltage_kv):
    V = voltage_kv * 1000.0

    return (
        12.2639 /
        math.sqrt(V + 0.97845e-6 * V**2)
    )



@torch.no_grad()
def compute_ctfs_batch(
    dim,
    pixel_size,
    defocus_u,
    defocus_v,
    astig_angle_deg,
    voltage_kv=300.0,
    cs_mm=2.7,
    amp_contrast=0.1,
    device="cuda"
):

    lam = electron_wavelength(voltage_kv)

    cs = cs_mm * 1e7

    defocus_u = torch.as_tensor(defocus_u, dtype=torch.float32, device=device)
    defocus_v = torch.as_tensor(defocus_v, dtype=torch.float32, device=device)
    astig_angle_deg = torch.as_tensor(astig_angle_deg, dtype=torch.float32, device=device)

    freq = torch.fft.fftfreq(dim, d=pixel_size, device=device)

    ky, kx = torch.meshgrid(freq, freq, indexing="ij")

    k2 = kx**2 + ky**2

    theta = torch.atan2(ky, kx)

    k2 = k2[None]
    theta = theta[None]

    astig_angle = torch.deg2rad(astig_angle_deg)[:, None, None]

    defocus_u = defocus_u[:, None, None]
    defocus_v = defocus_v[:, None, None]

    defocus = (
        0.5 * (defocus_u + defocus_v)
        +
        0.5 * (defocus_u - defocus_v)
        *
        torch.cos(
            2 * (theta - astig_angle)
        )
    )

    phase_shift = math.atan(
        amp_contrast /
        math.sqrt(
            1 - amp_contrast**2
        )
    )

    gamma = (
        math.pi * lam * defocus * k2
        -
        0.5 * math.pi
        * cs
        * lam**3
        * k2**2
        +
        phase_shift
    )

    ctfs = torch.sin(gamma)

    return ctfs




def main():
    parser = argparse.ArgumentParser(description="Promediar partículas alineadas de un archivo .star de RELION")
    parser.add_argument("star", type=str, help="Ruta al archivo .star de entrada")
    parser.add_argument("--sampling_rate", type=float, help="Pixel size")
    parser.add_argument("--out", type=str, default="average.mrc", help="Ruta de salida para el .mrc promedio")
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"], help="Dispositivo para PyTorch")
    args = parser.parse_args()

    avg = average_aligned_particles_ctf(args.star, args.sampling_rate, output_mrc=args.out, device=args.device)
    print(f"Promedio calculado y guardado en {args.out}")

if __name__ == "__main__":
    main()
    