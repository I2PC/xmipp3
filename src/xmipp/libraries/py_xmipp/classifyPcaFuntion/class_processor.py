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

class Processor:
    
    def __init__(
        self,
        parent,
        mmap,
        iteration,
        cut=50,
        cut_res=50,
        frc_threshold=0.143,
        apply_window=False,
        smooth=True,
        floor_res=100.0,
        clamp_exp=80.0,
        hard_cut=False,
        nyquist_margin=0.95,
        normalize=True,
        f_energy=2.0,
        boost_max=None,
        sharpen_power=None,
        factorR=None,
        eps=1e-8,
        max_iter=20,
    ):
        self.parent = parent
        self.mmap = mmap
        self.iteration = iteration

        self.cut = cut
        self.cut_res = cut_res

        self.frc_threshold = frc_threshold
        self.apply_window = apply_window
        self.smooth = smooth

        self.floor_res = floor_res
        self.clamp_exp = clamp_exp
        self.hard_cut = hard_cut
        self.nyquist_margin = nyquist_margin
        self.normalize = normalize

        self.f_energy = f_energy
        self.boost_max = boost_max
        self.sharpen_power = sharpen_power
        self.factorR = factorR
        self.eps = eps
        self.max_iter = max_iter

        self.device = torch.device(
            getattr(
                parent,
                "device",
                "cuda" if torch.cuda.is_available() else "cpu",
            )
        )

        self.sampling = parent.sampling

        # Cachés para no reconstruir las mismas mallas.
        self._frc_cache = {}
        self._gaussian_cache = {}
        self._sharpen_cache = {}

    # ------------------------------------------------------------------
    # API PRINCIPAL
    # ------------------------------------------------------------------

    @torch.no_grad()
    def process(self, particles):
        """
        particles:
            Tensor CPU [N,H,W]

        Returns:
            result:
                Tensor GPU [H,W]

            resolution:
                Tensor GPU escalar, o None si iteration <= 1
        """

        # ==============================================================
        # 1. CPU -> GPU
        # ==============================================================

        if particles.device != self.device:
            particles_gpu = particles.to(
                device=self.device,
                non_blocking=True,
            )
        else:
            particles_gpu = particles

        # ==============================================================
        # 2. Crear promedio
        # ==============================================================

        average = self.create_average(particles_gpu)

        # Ya no necesitamos la lista/tensor de salida de average.
        # El promedio queda en GPU.
        if self.iteration <= 1:
            del particles_gpu
            return average, None

        # ==============================================================
        # 3. FRC usando LAS MISMAS partículas GPU
        # ==============================================================

        resolution = self.calculate_frc(particles_gpu)

        # ==============================================================
        # 4. Liberar partículas inmediatamente
        # ==============================================================

        del particles_gpu

        # ==============================================================
        # 5. Gaussian lowpass
        # ==============================================================

        average = self.gaussian_lowpass(
            average,
            resolution,
        )

        # ==============================================================
        # 6. Highpass / cosine sharpen
        # ==============================================================

        average = self.cosine_sharpen(
            average,
            resolution,
        )

        return average, resolution

    # ------------------------------------------------------------------
    # AVERAGE
    # ------------------------------------------------------------------

    @torch.no_grad()
    def create_average(self, particles_gpu):
        """
        Usa exactamente la rutina existente del parent.

        averages_createClasses(...) devuelve una lista/batch.
        Aquí procesamos una sola clase, por lo que extraemos [0].
        """

        result = self.parent.averages_createClasses(
            self.mmap,
            self.iteration,
            [particles_gpu],
        )

        average = result[0]

        # Evita conservar accidentalmente el contenedor.
        del result

        return average

    # ------------------------------------------------------------------
    # FRC
    # ------------------------------------------------------------------

    @torch.no_grad()
    def calculate_frc(self, particles):
        """
        FRC para UNA sola clase.
        """

        n, h, w = particles.shape

        if n < 8:
            return particles.new_tensor(40.0)

        device = particles.device

        # --------------------------------------------------------------
        # Cache de malla FRC
        # --------------------------------------------------------------

        cache_key = (h, w, device)

        cache = self._frc_cache.get(cache_key)

        if cache is None:
            rmax = min(h, w) // 2

            fy = torch.fft.fftfreq(
                h,
                d=self.sampling,
                device=device,
            )

            fx = torch.fft.rfftfreq(
                w,
                d=self.sampling,
                device=device,
            )

            gy, gx = torch.meshgrid(
                fy,
                fx,
                indexing="ij",
            )

            r = torch.sqrt(gx * gx + gy * gy)

            freq_bins = torch.linspace(
                0.0,
                0.5 / self.sampling,
                rmax,
                device=device,
            )

            r_bin = torch.bucketize(
                r.flatten(),
                freq_bins,
            ) - 1

            r_bin.clamp_(
                min=0,
                max=rmax - 1,
            )

            if self.apply_window:
                wy = torch.hann_window(
                    h,
                    periodic=False,
                    device=device,
                )

                wx = torch.hann_window(
                    w,
                    periodic=False,
                    device=device,
                )

                window = wy[:, None] * wx[None, :]

                window = (
                    window
                    / window.norm()
                    * (h * w) ** 0.5
                )

                del wy, wx

            else:
                window = None

            del fy, fx, gy, gx, r

            cache = (
                rmax,
                freq_bins,
                r_bin,
                window,
            )

            self._frc_cache[cache_key] = cache

        else:
            rmax, freq_bins, r_bin, window = cache

        # --------------------------------------------------------------
        # Random split
        # --------------------------------------------------------------

        perm = torch.randperm(
            n,
            device=device,
        )

        half1, half2 = torch.chunk(
            particles[perm],
            2,
            dim=0,
        )

        del perm

        avg1 = half1.mean(dim=0)
        avg2 = half2.mean(dim=0)

        del half1, half2

        if window is not None:
            avg1.mul_(window)
            avg2.mul_(window)

        # --------------------------------------------------------------
        # FFT
        # --------------------------------------------------------------

        fft1 = torch.fft.rfft2(
            avg1,
            norm="forward",
        )

        fft2 = torch.fft.rfft2(
            avg2,
            norm="forward",
        )

        del avg1, avg2

        # --------------------------------------------------------------
        # Powers / cross product
        # --------------------------------------------------------------

        p1 = fft1.real.square() + fft1.imag.square()
        p2 = fft2.real.square() + fft2.imag.square()

        prod = (
            fft1 * fft2.conj()
        ).real

        del fft1, fft2

        # --------------------------------------------------------------
        # FRC radial
        # --------------------------------------------------------------

        frc_num = torch.zeros(
            rmax,
            device=device,
            dtype=prod.dtype,
        )

        frc_d1 = torch.zeros_like(frc_num)
        frc_d2 = torch.zeros_like(frc_num)

        frc_num.scatter_add_(
            0,
            r_bin,
            prod.flatten(),
        )

        frc_d1.scatter_add_(
            0,
            r_bin,
            p1.flatten(),
        )

        frc_d2.scatter_add_(
            0,
            r_bin,
            p2.flatten(),
        )

        del prod, p1, p2

        frc = frc_num / (
            torch.sqrt(frc_d1 * frc_d2)
            + 1e-12
        )

        del frc_num, frc_d1, frc_d2

        # --------------------------------------------------------------
        # Smooth
        # --------------------------------------------------------------

        if self.smooth:
            kernel = frc.new_tensor(
                [0.25, 0.5, 0.25]
            ).view(1, 1, -1)

            frc = F.conv1d(
                frc.view(1, 1, -1),
                kernel,
                padding=1,
            ).view(-1)

            del kernel

        # --------------------------------------------------------------
        # Resolución
        # --------------------------------------------------------------

        idx = torch.where(
            frc < self.frc_threshold
        )[0]

        if idx.numel() and idx[0] > 0:
            resolution = 1.0 / freq_bins[idx[0]]
        else:
            # Igual que el fallback original.
            resolution = particles.new_tensor(
                (2.0 * self.sampling) / 0.8
            )

        del frc, idx

        # --------------------------------------------------------------
        # Fallback rcut
        #
        # Sin sincronización GPU -> CPU.
        # --------------------------------------------------------------

        resolution = torch.nan_to_num(
            resolution,
            nan=(2.0 * self.sampling) / 0.8,
            posinf=(2.0 * self.sampling) / 0.8,
            neginf=(2.0 * self.sampling) / 0.8,
        )

        resolution = torch.where(
            resolution > self.cut,
            resolution.new_tensor(self.cut_res),
            resolution,
        )

        return resolution

    # ------------------------------------------------------------------
    # GAUSSIAN LOWPASS
    # ------------------------------------------------------------------

    @torch.no_grad()
    def gaussian_lowpass(
        self,
        image,
        resolution,
    ):
        """
        Gaussian lowpass para UNA imagen [H,W].
        """

        h, w = image.shape
        device = image.device
        eps = self.eps

        # --------------------------------------------------------------
        # Nyquist
        # --------------------------------------------------------------

        nyquist_res = 2.0 * self.sampling
        safe_res = nyquist_res / self.nyquist_margin

        res_eff = torch.nan_to_num(
            resolution,
            nan=self.floor_res,
            posinf=self.floor_res,
            neginf=self.floor_res,
        )

        res_eff = torch.minimum(
            res_eff,
            res_eff.new_tensor(self.floor_res),
        )

        res_eff = torch.clamp(
            res_eff,
            min=safe_res,
        )

        # --------------------------------------------------------------
        # Frequency grid
        # --------------------------------------------------------------

        cache_key = (h, w, device)

        freq2 = self._gaussian_cache.get(
            cache_key
        )

        if freq2 is None:
            fy = torch.fft.fftfreq(
                h,
                d=self.sampling,
                device=device,
            )

            fx = torch.fft.fftfreq(
                w,
                d=self.sampling,
                device=device,
            )

            gy, gx = torch.meshgrid(
                fy,
                fx,
                indexing="ij",
            )

            freq2 = gx.square() + gy.square()

            del fy, fx, gy, gx

            self._gaussian_cache[cache_key] = freq2

        # --------------------------------------------------------------
        # Gaussian
        # --------------------------------------------------------------

        ln2 = image.new_tensor(2.0).log()

        D0 = 1.0 / res_eff

        sigma2 = (
            D0 / torch.sqrt(2.0 * ln2)
        ).square()

        exponent = (
            -freq2
            / (2.0 * sigma2 + eps)
        )

        filt = torch.exp(
            exponent.clamp(
                max=self.clamp_exp
            )
        )

        if self.hard_cut:
            filt = torch.where(
                freq2 > D0.square(),
                torch.zeros_like(filt),
                filt,
            )

        del D0, sigma2, exponent, ln2

        # --------------------------------------------------------------
        # FFT
        # --------------------------------------------------------------

        fft = torch.fft.fft2(
            image,
            norm="forward",
        )

        fft.mul_(filt)

        del filt

        img_filt = torch.fft.ifft2(
            fft,
            norm="forward",
        ).real

        del fft

        img_filt = torch.nan_to_num(
            img_filt
        )

        # --------------------------------------------------------------
        # Restaurar contraste
        # --------------------------------------------------------------

        if self.normalize:
            mean0 = image.mean()
            std0 = image.std()

            mean_f = img_filt.mean()
            std_f = img_filt.std()

            normalized = (
                (img_filt - mean_f)
                / (std_f + eps)
                * std0
                + mean0
            )

            img_filt = torch.where(
                std_f > 1e-6,
                normalized,
                image,
            )

            del (
                mean0,
                std0,
                mean_f,
                std_f,
                normalized,
            )

        return img_filt

    # ------------------------------------------------------------------
    # COSINE SHARPEN
    # ------------------------------------------------------------------

    @torch.no_grad()
    def cosine_sharpen(
        self,
        average,
        resolution,
    ):
        """
        Highpass / cosine sharpen para UNA imagen [H,W].
        """

        h, w = average.shape
        device = average.device
        eps = self.eps

        # --------------------------------------------------------------
        # FFT + energía original
        # --------------------------------------------------------------

        fft = torch.fft.fft2(
            average,
            norm="forward",
        )

        fft_mag2 = (
            fft.real.square()
            + fft.imag.square()
        )

        energy_orig = fft_mag2.sum()

        # --------------------------------------------------------------
        # Frecuencia radial
        # --------------------------------------------------------------

        cache_key = (h, w, device)

        freq_r = self._sharpen_cache.get(
            cache_key
        )

        if freq_r is None:
            fy = torch.fft.fftfreq(
                h,
                d=self.sampling,
                device=device,
            )

            fx = torch.fft.fftfreq(
                w,
                d=self.sampling,
                device=device,
            )

            gy, gx = torch.meshgrid(
                fy,
                fx,
                indexing="ij",
            )

            freq_r = torch.sqrt(
                gx.square() + gy.square()
            )

            del fy, fx, gy, gx

            self._sharpen_cache[cache_key] = freq_r

        # --------------------------------------------------------------
        # Cutoff
        # --------------------------------------------------------------

        f_cutoff = 1.0 / torch.clamp(
            resolution,
            min=1e-3,
        )

        # --------------------------------------------------------------
        # Sharpen power
        # --------------------------------------------------------------

        if self.sharpen_power is None:

            if self.factorR is None:

                factor = torch.where(
                    resolution < 10.0,
                    resolution.new_tensor(0.1),
                    torch.where(
                        resolution < 14.0,
                        resolution.new_tensor(0.08),
                        resolution.new_tensor(0.06),
                    ),
                )

            else:
                factor = resolution.new_tensor(
                    self.factorR
                )

            sharpen_power = (
                factor * resolution
            ).clamp(
                min=0.3,
                max=2.5,
            )

            del factor

        else:
            sharpen_power = resolution.new_tensor(
                float(self.sharpen_power)
            )

        # --------------------------------------------------------------
        # Cosine shape
        # --------------------------------------------------------------

        cos_term = (
            torch.pi
            * freq_r
            / (f_cutoff + eps)
        )

        cosine_shape = (
            (1.0 - torch.cos(cos_term))
            / 2.0
        ).clamp(
            min=0.0,
            max=1.0,
        )

        del cos_term

        cosine_shape = torch.where(
            freq_r <= f_cutoff,
            cosine_shape,
            torch.ones_like(freq_r),
        )

        cosine_shape.pow_(
            sharpen_power
        )

        del sharpen_power

        # --------------------------------------------------------------
        # Boost
        #
        # Para boost_max=None resolvemos directamente la ecuación
        # de energía en vez de hacer 20 FFT/reducciones iterativas.
        # --------------------------------------------------------------

        if self.boost_max is None:

            target_energy = (
                self.f_energy
                * energy_orig
            )

            # E(g) =
            # Σ M * [1 + (g-1)C]^2
            #
            # Sea x = g-1:
            #
            # a*x² + 2*b*x + E0 - target = 0

            a = (
                fft_mag2
                * cosine_shape.square()
            ).sum()

            b = (
                fft_mag2
                * cosine_shape
            ).sum()

            discriminant = (
                b.square()
                - a * (
                    energy_orig
                    - target_energy
                )
            )

            discriminant = torch.clamp(
                discriminant,
                min=0.0,
            )

            sqrt_disc = torch.sqrt(
                discriminant
            )

            # Solución positiva.
            x = (
                -b + sqrt_disc
            ) / (
                a + eps
            )

            boost_max = (
                1.0 + x
            )

            # Si la componente coseno no tiene energía,
            # no existe boost real útil.
            boost_max = torch.where(
                a > eps,
                boost_max,
                boost_max.new_tensor(1.0),
            )

            del (
                target_energy,
                a,
                b,
                discriminant,
                sqrt_disc,
                x,
            )

        else:

            boost_max = average.new_tensor(
                float(self.boost_max)
            )

        # --------------------------------------------------------------
        # Filtro final
        # --------------------------------------------------------------

        boost = (
            1.0
            + (boost_max - 1.0)
            * cosine_shape
        )

        boost = torch.where(
            freq_r <= f_cutoff,
            boost,
            torch.ones_like(boost),
        )

        del (
            cosine_shape,
            freq_r,
            f_cutoff,
            boost_max,
            fft_mag2,
            energy_orig,
        )

        # --------------------------------------------------------------
        # Aplicar filtro
        # --------------------------------------------------------------

        fft.mul_(boost)

        del boost

        filtered = torch.fft.ifft2(
            fft,
            norm="forward",
        ).real

        del fft

        # --------------------------------------------------------------
        # Normalización de contraste
        # --------------------------------------------------------------

        if self.normalize:
            mean_orig = average.mean()
            std_orig = average.std()

            mean_filt = filtered.mean()
            std_filt = filtered.std()

            filtered = (
                (filtered - mean_filt)
                / (std_filt + eps)
                * std_orig
                + mean_orig
            )

            del (
                mean_orig,
                std_orig,
                mean_filt,
                std_filt,
            )

        return filtered

 