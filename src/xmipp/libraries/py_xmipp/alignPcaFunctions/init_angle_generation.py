#!/usr/bin/env python3
"""/***************************************************************************
 *
 * Authors:    Erney Ramirez-Aportela
 *
  ***************************************************************************/
"""


import math
import torch
import torch.nn.functional as F


class CommonLinesInitializer:
    """
    Inicializador rápido experimental mediante líneas comunes.

    Entrada:
        classes: tensor (N,H,W).

    Salida:
        rotations: tensor (N,3,3).

    Convención:
        k3D = R @ k2D
        R = Rz(rot) @ Ry(tilt) @ Rz(psi)

    La primera orientación se fija a identidad.
    """

    def __init__(
        self,
        image_size=128,
        n_angles=90,
        n_radial=32,
        min_radius=4,
        max_radius=50,
        device="cuda",
        iterations=100,
        lr=0.015,
        starts=1,
        seed=1234,
    ):
        self.image_size = image_size
        self.n_angles = n_angles
        self.n_radial = n_radial
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.device = torch.device(device)
        self.iterations = iterations
        self.lr = lr
        self.starts = starts
        self.seed = seed
        self.last_result = None

    # -----------------------------------------------------
    # PREPARACIÓN
    # -----------------------------------------------------

    def _prepare(self, images):
        images = torch.as_tensor(
            images,
            dtype=torch.float32,
            device=self.device
        )

        if images.ndim != 3:
            raise ValueError("Entrada esperada: (N,H,W)")

        if images.shape[-2:] != (
            self.image_size, self.image_size
        ):
            raise ValueError("Tamaño de imagen incorrecto")

        if images.shape[0] < 4:
            raise ValueError("Se necesitan al menos 4 imágenes")

        if not torch.isfinite(images).all():
            raise ValueError("Hay NaN o Inf")

        images = images - images.mean(
            dim=(-2, -1), keepdim=True
        )

        std = images.std(
            dim=(-2, -1), keepdim=True
        ).clamp_min(1e-8)

        return images / std

    # -----------------------------------------------------
    # FOURIER POLAR
    # -----------------------------------------------------

    def _polar_fourier(self, images):
        N, H, W = images.shape

        ft = torch.fft.fftshift(
            torch.fft.fft2(
                torch.fft.ifftshift(
                    images, dim=(-2, -1)
                ),
                norm="forward"
            ),
            dim=(-2, -1)
        )

        radii = torch.linspace(
            self.min_radius,
            self.max_radius,
            self.n_radial,
            device=self.device
        )

        angles = (
            torch.arange(
                self.n_angles,
                device=self.device
            ) * math.pi / self.n_angles
        )

        rr, aa = torch.meshgrid(
            radii, angles, indexing="ij"
        )

        x = rr * torch.cos(aa)
        y = rr * torch.sin(aa)

        gx = 2 * (x + W // 2) / (W - 1) - 1
        gy = 2 * (y + H // 2) / (H - 1) - 1

        grid = torch.stack(
            [gx, gy], dim=-1
        ).unsqueeze(0).expand(N, -1, -1, -1)

        real = F.grid_sample(
            ft.real[:, None],
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True
        )[:, 0]

        imag = F.grid_sample(
            ft.imag[:, None],
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True
        )[:, 0]

        polar = torch.complex(
            real, imag
        ).transpose(1, 2)

        polar = polar - polar.mean(
            dim=-1, keepdim=True
        )

        norm = torch.linalg.vector_norm(
            polar, dim=-1, keepdim=True
        ).clamp_min(1e-8)

        return polar / norm

    # -----------------------------------------------------
    # LINEAS COMUNES
    # -----------------------------------------------------

    def _find_common_lines(self, polar):
        N, A, K = polar.shape

        angles = torch.zeros(
            N, N, 2,
            dtype=torch.long,
            device=self.device
        )

        confidence = torch.zeros(
            N, N,
            dtype=torch.float32,
            device=self.device
        )

        for i in range(N - 1):
            corr = torch.abs(
                torch.matmul(
                    polar[i].conj().unsqueeze(0),
                    polar[i + 1:].transpose(1, 2)
                )
            )

            values, indices = corr.flatten(
                start_dim=1
            ).max(dim=1)

            ai = indices // A
            aj = indices % A

            js = torch.arange(
                i + 1, N,
                device=self.device
            )

            angles[i, js, 0] = ai
            angles[i, js, 1] = aj

            angles[js, i, 0] = aj
            angles[js, i, 1] = ai

            confidence[i, js] = values
            confidence[js, i] = values

        return angles, confidence

    # -----------------------------------------------------
    # ROTACIONES SO(3)
    # -----------------------------------------------------

    @staticmethod
    def _skew(w):
        x, y, z = w.unbind(dim=-1)
        zero = torch.zeros_like(x)

        return torch.stack([
            zero, -z, y,
            z, zero, -x,
            -y, x, zero
        ], dim=-1).reshape(-1, 3, 3)

    def _rotations_from_vectors(self, w):
        w = torch.cat([
            torch.zeros(
                1, 3,
                device=self.device,
                dtype=w.dtype
            ),
            w
        ], dim=0)

        return torch.linalg.matrix_exp(
            self._skew(w)
        )

    # -----------------------------------------------------
    # PERDIDA DE SINCRONIZACION
    # -----------------------------------------------------

    def _synchronization_loss(
        self, rotations, angles, confidence, pair_idx
    ):
        i, j = pair_idx

        theta = angles.float() * (
            math.pi / self.n_angles
        )

        ai = theta[i, j, 0]
        aj = theta[i, j, 1]

        ui = torch.stack([
            torch.cos(ai),
            torch.sin(ai),
            torch.zeros_like(ai)
        ], dim=-1)

        uj = torch.stack([
            torch.cos(aj),
            torch.sin(aj),
            torch.zeros_like(aj)
        ], dim=-1)

        vi = torch.bmm(
            rotations[i], ui.unsqueeze(-1)
        ).squeeze(-1)

        vj = torch.bmm(
            rotations[j], uj.unsqueeze(-1)
        ).squeeze(-1)

        dot = (vi * vj).sum(dim=-1)

        # Las líneas no tienen sentido de dirección.
        error = 1.0 - dot.square()

        weights = confidence[i, j]

        return (
            (error * weights).sum()
            / weights.sum().clamp_min(1e-8)
        )

    # -----------------------------------------------------
    # EULER ZYZ
    # -----------------------------------------------------

    @staticmethod
    def _matrix_to_zyz(R):
        tilt = torch.acos(
            R[:, 2, 2].clamp(-1.0, 1.0)
        )

        sb = torch.sin(tilt)

        rot = torch.atan2(
            R[:, 1, 2],
            R[:, 0, 2]
        )

        psi = torch.atan2(
            R[:, 2, 1],
            -R[:, 2, 0]
        )

        singular = sb.abs() < 1e-6

        rot[singular] = torch.atan2(
            R[singular, 1, 0],
            R[singular, 0, 0]
        )

        psi[singular] = 0.0

        return (
            torch.rad2deg(psi),
            torch.rad2deg(rot),
            torch.rad2deg(tilt)
        )

    # -----------------------------------------------------
    # RESIDUOS
    # -----------------------------------------------------

    @torch.no_grad()
    def _residuals(self, rotations, angles):
        N = rotations.shape[0]

        idx = torch.triu_indices(
            N, N, offset=1,
            device=self.device
        )

        i, j = idx

        theta = angles.float() * (
            math.pi / self.n_angles
        )

        ai = theta[i, j, 0]
        aj = theta[i, j, 1]

        ui = torch.stack([
            torch.cos(ai),
            torch.sin(ai),
            torch.zeros_like(ai)
        ], dim=-1)

        uj = torch.stack([
            torch.cos(aj),
            torch.sin(aj),
            torch.zeros_like(aj)
        ], dim=-1)

        vi = torch.bmm(
            rotations[i], ui.unsqueeze(-1)
        ).squeeze(-1)

        vj = torch.bmm(
            rotations[j], uj.unsqueeze(-1)
        ).squeeze(-1)

        values = 1.0 - (
            (vi * vj).sum(dim=-1)
        ).square()

        residual = torch.zeros(
            N, N, device=self.device
        )

        residual[i, j] = values
        residual[j, i] = values

        return residual

    # -----------------------------------------------------
    # FIT
    # -----------------------------------------------------

    def fit(self, classes):
        images = self._prepare(classes)
        N = images.shape[0]

        polar = self._polar_fourier(images)

        angles, pair_confidence = (
            self._find_common_lines(polar)
        )

        pair_idx = torch.triu_indices(
            N, N, offset=1,
            device=self.device
        )

        # Evitar pesos nulos.
        pair_confidence = pair_confidence.clamp_min(1e-6)

        best_loss = None
        best_R = None

        for start in range(self.starts):
            generator = torch.Generator(
                device=self.device
            )
            generator.manual_seed(
                self.seed + start
            )

            axis = torch.randn(
                N - 1, 3,
                device=self.device,
                generator=generator
            )

            axis = axis / torch.linalg.vector_norm(
                axis, dim=-1, keepdim=True
            ).clamp_min(1e-8)

            angle = math.pi * torch.rand(
                N - 1, 1,
                device=self.device,
                generator=generator
            )

            w = torch.nn.Parameter(axis * angle)

            optimizer = torch.optim.Adam(
                [w], lr=self.lr
            )

            for _ in range(self.iterations):
                optimizer.zero_grad()

                R = self._rotations_from_vectors(w)

                loss = self._synchronization_loss(
                    R, angles,
                    pair_confidence,
                    pair_idx
                )

                loss.backward()
                optimizer.step()

            with torch.no_grad():
                R = self._rotations_from_vectors(w)

                final_loss = self._synchronization_loss(
                    R, angles,
                    pair_confidence,
                    pair_idx
                )

                if (
                    best_loss is None
                    or final_loss < best_loss
                ):
                    best_loss = final_loss.detach().clone()
                    best_R = R.detach().clone()

        with torch.no_grad():
            psi, rot, tilt = self._matrix_to_zyz(best_R)

            residual = self._residuals(
                best_R, angles
            )

            self.last_result = {
                "rotations": best_R,
                "psi": psi,
                "rot": rot,
                "tilt": tilt,
                "shifts": torch.zeros(
                    N, 2, device=self.device
                ),
                "common_lines": angles,
                "pair_confidence": pair_confidence,
                "residuals": residual,
                "loss": best_loss
            }

        return best_R

