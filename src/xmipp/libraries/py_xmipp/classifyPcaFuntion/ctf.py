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

class ctfClass:

    def __init__(self, star_path, device="cuda:0"):

        self.star_path = star_path
        self.cuda = torch.device(device)

        self.readCtfParams()

        self.lam = self.electron_wavelength(self.voltage)
        self.cs_angstrom = self.cs * 1e7

        self.K1 = math.pi * self.lam

        self.K2 = ( (math.pi / 2.0) * self.cs_angstrom * (self.lam ** 3) )

        self.K3 = math.atan( self.ampC / math.sqrt(1.0 - self.ampC ** 2) )

        # ---------------------------------------------------------
        # Parámetros de partículas como tensores GPU
        # ---------------------------------------------------------
        self.defocus_u_gpu = torch.as_tensor(
            self.defocus_u,
            dtype=torch.float32,
            device=self.cuda
        )

        self.defocus_v_gpu = torch.as_tensor(
            self.defocus_v,
            dtype=torch.float32,
            device=self.cuda
        )

        self.defocus_angle_gpu = torch.as_tensor(
            self.defocus_angle,
            dtype=torch.float32,
            device=self.cuda
        )

        # Cache para las mallas de frecuencia
        self._freq_cache = {}


    def readCtfParams(self):

        df = starfile.read(self.star_path)

        # CTF parameters
        self.voltage = float(df["ctfVoltage"].values[0])
        self.cs = float(df["ctfSphericalAberration"].values[0])
        self.ampC = float(df["ctfQ0"].values[0])

        self.defocus_u = df["ctfDefocusU"].to_numpy(dtype=np.float32)
        self.defocus_v = df["ctfDefocusV"].to_numpy(dtype=np.float32)
        self.defocus_angle = df["ctfDefocusAngle"].to_numpy(dtype=np.float32)

        return (self.voltage, self.cs, self.ampC, self.defocus_u, self.defocus_v, self.defocus_angle)

    # =============================================================
    # ELECTRON WAVELENGTH
    # =============================================================

    def electron_wavelength(self, voltage_kv):

        V = voltage_kv * 1000.0

        return (
            12.2639 /
            math.sqrt(
                V + 0.97845e-6 * V**2
            )
        )

    # =============================================================
    # FREQUENCY GRID
    # =============================================================

    def _get_frequency_grid(
        self,
        dim,
        pixel_size,
        device
    ):
        """
        Construye X, Y, u2 y u4 una sola vez y los reutiliza.
        No modifica ninguna convención matemática.
        """

        key = (dim, float(pixel_size), str(device))

        if key not in self._freq_cache:

            freq = torch.fft.fftfreq(dim, d=pixel_size, device=device)
            ky, kx = torch.meshgrid(freq, freq, indexing="ij")

            X = kx[None, :, :]
            Y = ky[None, :, :]

            u2 = X**2 + Y**2
            u4 = u2**2

            self._freq_cache[key] = (
                X,
                Y,
                u2,
                u4
            )

        return self._freq_cache[key]

    # =============================================================
    # COMPUTE CTF BATCH
    # =============================================================

    @torch.no_grad()
    def compute_ctfs_batch(
        self,
        dim,
        pixel_size,
        angle=0.0,
        particle_indices=None,
        phase_shift_deg=0.0,
        device=None,
    ):

        if device is None:
            device = self.cuda
        else:
            device = torch.device(device)

        # ---------------------------------------------------------
        # Seleccionar parámetros de las partículas
        # ---------------------------------------------------------

        if particle_indices is not None:

            # Los índices ya deberían estar en GPU.
            # Evitamos CPU -> NumPy -> GPU.
            if not torch.is_tensor(particle_indices):
                particle_indices = torch.as_tensor(
                    particle_indices,
                    dtype=torch.long,
                    device=device
                )
            else:
                particle_indices = particle_indices.to(
                    device=device,
                    dtype=torch.long,
                    non_blocking=True
                )

            defU = self.defocus_u_gpu[particle_indices]

            defV = self.defocus_v_gpu[particle_indices]

            angle_val = self.defocus_angle_gpu[particle_indices]

        else:

            defU = self.defocus_u_gpu
            defV = self.defocus_v_gpu
            angle_val = self.defocus_angle_gpu


        astig_angle_deg = angle_val - angle

        defU = defU[:, None, None]
        defV = defV[:, None, None]

        az_rad = torch.deg2rad(
            angle_val.new_tensor(astig_angle_deg)
        )[:, None, None]


        X, Y, u2, u4 = self._get_frequency_grid(
            dim,
            pixel_size,
            device
        )

        # ---------------------------------------------------------
        # Astigmatism
        # ---------------------------------------------------------

        sin_az = torch.sin(az_rad)
        cos_az = torch.cos(az_rad)

        Axx = -(
            defU * (cos_az**2)
            +
            defV * (sin_az**2)
        )

        Ayy = -(
            defU * (sin_az**2)
            +
            defV * (cos_az**2)
        )

        Axy = -(
            defU - defV
        ) * sin_az * cos_az

        # ---------------------------------------------------------
        # Phase
        # ---------------------------------------------------------

        astig_term = (
            Axx * (X**2)
            +
            2.0 * Axy * (X * Y)
            +
            Ayy * (Y**2)
        )

        K5 = math.radians(
            phase_shift_deg
        )

        gamma = (
            self.K1 * astig_term
            +
            self.K2 * u4
            -
            K5
            -
            self.K3
        )

        # EXACTAMENTE tu convención
        return -torch.sin(gamma)

    # =============================================================
    # APPLY CTF TO AVERAGE
    # =============================================================

    @torch.no_grad()
    def apply_ctf_to_average(
        self,
        particles,
        dim,
        pixel_size,
        angle
    ):

        Fpart = torch.fft.fft2(
            particles, norm="forward"
        )

        ctf_batch = self.compute_ctfs_batch(
            dim=dim,
            pixel_size=pixel_size,
            angle=angle,
            device=particles.device
        )

        numerator = (
            ctf_batch * Fpart
        ).sum(dim=0)

        denominator = (
            ctf_batch.square()
        ).sum(dim=0)

        # Mantener exactamente tu regularización
        regularizer = (
            1e-2 * denominator.max()
        )

        avg_fft = (
            numerator /
            (denominator + regularizer)
        )

        avg = torch.real(
            torch.fft.ifft2(avg_fft, norm="forward")
        )

        return avg
    
    
    
    
    
    
    
    
    
    
    # def __init__(self, star_path):
    #     self.star_path = star_path
    #     torch.cuda.is_available()
    #     torch.cuda.current_device()
    #     self.cuda = torch.device('cuda:0')
    #
    #     #for experimental images with starfile module
    # def readCtfParams(self):
    #
    #     df = starfile.read(self.star_path)
    #
    #     #ctf parameters
    #     self.voltage = float(df["ctfVoltage"].values[0])
    #     self.cs = float(df["ctfSphericalAberration"].values[0])
    #     self.ampC = float(df["ctfQ0"].values[0])
    #
    #
    #     self.defocus_u = df["ctfDefocusU"].to_numpy(dtype=np.float32)
    #     self.defocus_v = df["ctfDefocusV"].to_numpy(dtype=np.float32)
    #     self.defocus_angle = df["ctfDefocusAngle"].to_numpy(dtype=np.float32)
    #
    #     return self.voltage, self.cs, self.ampC, self.defocus_u, self.defocus_v, self.defocus_angle
    #
    #
    # def electron_wavelength(self, voltage_kv):
    #     V = voltage_kv * 1000.0
    #
    #     return (
    #         12.2639 /
    #         math.sqrt(V + 0.97845e-6 * V**2)
    #     )
    #
    #
    # def compute_ctfs_batch(
    #     self,
    #     dim,
    #     pixel_size,
    #     angle=0.0,
    #     particle_indices=None,
    #     phase_shift_deg=0.0,
    #     device="cuda",
    # ):
    #     self.readCtfParams()
    #
    #     # Seleccionar únicamente los desemfoques de las partículas del batch activo
    #     if particle_indices is not None:
    #         defU_val = self.defocus_u[particle_indices.cpu().numpy()]
    #         defV_val = self.defocus_v[particle_indices.cpu().numpy()]
    #         angle_val = self.defocus_angle[particle_indices.cpu().numpy()]
    #     else:
    #         defU_val = self.defocus_u
    #         defV_val = self.defocus_v
    #         angle_val = self.defocus_angle
    #
    #     astig_angle_deg = angle_val - angle
    #     lam = self.electron_wavelength(self.voltage)
    #     cs_angstrom = self.cs * 1e7
    #
    #     defU = torch.as_tensor(defU_val, dtype=torch.float32, device=device)[:, None, None]
    #     defV = torch.as_tensor(defV_val, dtype=torch.float32, device=device)[:, None, None]
    #     az_rad = torch.deg2rad(
    #         torch.as_tensor(astig_angle_deg, dtype=torch.float32, device=device)
    #     )[:, None, None]
    #
    #     K1 = math.pi * lam
    #     K2 = (math.pi / 2.0) * cs_angstrom * (lam**3)
    #     K3 = math.atan(self.ampC / math.sqrt(1.0 - self.ampC**2))
    #     K5 = math.radians(phase_shift_deg)
    #
    #     sin_az = torch.sin(az_rad)
    #     cos_az = torch.cos(az_rad)
    #
    #     Axx = -(defU * (cos_az**2) + defV * (sin_az**2))
    #     Ayy = -(defU * (sin_az**2) + defV * (cos_az**2))
    #     Axy = -(defU - defV) * sin_az * cos_az
    #
    #     freq = torch.fft.fftfreq(dim, d=pixel_size, device=device)
    #     ky, kx = torch.meshgrid(freq, freq, indexing="ij")
    #
    #     X = kx[None, :, :]
    #     Y = ky[None, :, :]
    #
    #     u2 = X**2 + Y**2
    #     u4 = u2**2
    #
    #     astig_term = Axx * (X**2) + 2.0 * Axy * (X * Y) + Ayy * (Y**2)
    #     gamma = K1 * astig_term + K2 * u4 - K5 - K3
    #
    #     return -torch.sin(gamma)
    #
    #
    # def apply_ctf_to_average(self, particles, dim, pixel_size, angle):
    #
    #     Fpart = torch.fft.fft2(particles)
    #
    #     ctf_batch = self.compute_ctfs_batch(dim, pixel_size, angle)
    #
    #     numerator = (ctf_batch * Fpart).sum(dim=0)
    #     denominator = (ctf_batch.square()).sum(dim=0)
    #
    #     regularizer = 1e-2 * denominator.max()
    #
    #     avg_fft = numerator / (denominator + regularizer)
    #     avg = torch.real(torch.fft.ifft2(avg_fft))
    #
    #     return avg
                                
        

        
 


 