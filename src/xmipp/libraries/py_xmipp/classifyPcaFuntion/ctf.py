#!/usr/bin/env python3
"""/***************************************************************************
 *
 * Authors:    Erney Ramirez-Aportela
 *
  ***************************************************************************/
"""
import numpy as np
import starfile
import torch
import math

class ctf:
    
    def __init__(self, star_path):
        self.star_path = star_path
        torch.cuda.is_available()
        torch.cuda.current_device()
        self.cuda = torch.device('cuda:0')
    
        #for experimental images with starfile module
    def readCtfParams(self):
        
        df = starfile.read(self.star_path)
        
        #ctf parameters
        self.voltage = float(df["ctfVoltage"].values[0])
        self.cs = float(df["ctfSphericalAberration"].values[0])
        self.ampC = float(df["ctfQ0"].values[0])
        
        
        self.defocus_u = df["ctfDefocusU"].to_numpy(dtype=np.float32)
        self.defocus_v = df["ctfDefocusV"].to_numpy(dtype=np.float32)
        self.defocus_angle = df["ctfDefocusAngle"].to_numpy(dtype=np.float32)
        
        return self.voltage, self.cs, self.ampC, self.defocus_u, self.defocus_v, self.defocus_angle
        
        
    def electron_wavelength(self, voltage_kv):
        V = voltage_kv * 1000.0
    
        return (
            12.2639 /
            math.sqrt(V + 0.97845e-6 * V**2)
        )
        
        
    def compute_ctfs_batch(
        self,
        dim,
        pixel_size,
        angle=0.0,
        particle_indices=None,
        phase_shift_deg=0.0,
        device="cuda",
    ):
        self.readCtfParams()
        print(self.defocus_u)
        exit()

        # Seleccionar únicamente los desemfoques de las partículas del batch activo
        if particle_indices is not None:
            defU_val = self.defocus_u[particle_indices.cpu().numpy()]
            defV_val = self.defocus_v[particle_indices.cpu().numpy()]
            angle_val = self.defocus_angle[particle_indices.cpu().numpy()]
        else:
            defU_val = self.defocus_u
            defV_val = self.defocus_v
            angle_val = self.defocus_angle

        astig_angle_deg = angle_val - angle
        lam = self.electron_wavelength(self.voltage)
        cs_angstrom = self.cs * 1e7

        defU = torch.as_tensor(defU_val, dtype=torch.float32, device=device)[:, None, None]
        defV = torch.as_tensor(defV_val, dtype=torch.float32, device=device)[:, None, None]
        az_rad = torch.deg2rad(
            torch.as_tensor(astig_angle_deg, dtype=torch.float32, device=device)
        )[:, None, None]

        K1 = math.pi * lam
        K2 = (math.pi / 2.0) * cs_angstrom * (lam**3)
        K3 = math.atan(self.ampC / math.sqrt(1.0 - self.ampC**2))
        K5 = math.radians(phase_shift_deg)

        sin_az = torch.sin(az_rad)
        cos_az = torch.cos(az_rad)

        Axx = -(defU * (cos_az**2) + defV * (sin_az**2))
        Ayy = -(defU * (sin_az**2) + defV * (cos_az**2))
        Axy = -(defU - defV) * sin_az * cos_az

        freq = torch.fft.fftfreq(dim, d=pixel_size, device=device)
        ky, kx = torch.meshgrid(freq, freq, indexing="ij")

        X = kx[None, :, :]
        Y = ky[None, :, :]

        u2 = X**2 + Y**2
        u4 = u2**2

        astig_term = Axx * (X**2) + 2.0 * Axy * (X * Y) + Ayy * (Y**2)
        gamma = K1 * astig_term + K2 * u4 - K5 - K3

        return -torch.sin(gamma)
    
    
    def apply_ctf_to_average(self, particles, dim, pixel_size, angle):
        
        Fpart = torch.fft.fft2(particles)
                               
        ctf_batch = self.compute_ctfs_batch(dim, pixel_size, angle)
                               
        numerator = (ctf_batch * Fpart).sum(dim=0)
        denominator = (ctf_batch.square()).sum(dim=0)
        
        regularizer = 1e-2 * denominator.max()
    
        avg_fft = numerator / (denominator + regularizer)
        avg = torch.real(torch.fft.ifft2(avg_fft))
        
        return avg
                                
        

        
 


 