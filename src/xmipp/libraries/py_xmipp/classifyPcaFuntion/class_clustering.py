#!/usr/bin/env python3
"""/***************************************************************************
 *
 * Authors:    Erney Ramirez-Aportela
 *
  ***************************************************************************/
"""

"""Clustering de class averages 2D mediante similitud y alineamiento."""

import math
import numpy as np
import torch
import torch.nn.functional as F


class TwoDClassAverageClusterer:
    
    """Agrupa averages 2D similares y devuelve el medoide de cada grupo.

    Entrada: tensor/array [N, H, W] con los class averages.
    Salida de fit_predict: tensor PyTorch [K, H, W], con un average original
    por cluster. K es el número de clusters. No se alinean ni promedian las
    imágenes de salida.

    El medoide es el average del cluster con mayor similitud media respecto
    a los demás miembros. Los atributos labels_, clusters_, representatives_,
    similarity_ y alignment_ contienen información adicional del clustering.

    Se debe especificar exactamente un threshold o n_clusters.
    """

    def preprocess(self, images, size=64, apix=None, lowpass=None, highpass=None):

        """Recorta en Fourier a `size` px (si es menor) y aplica filtros gaussianos.

        Devuelve (imagenes, apix_de_trabajo).

        """

        n, h, w = images.shape

        x = images


        if size is not None and size < h:

            size -= size % 2

            spec = torch.fft.fftshift(torch.fft.fft2(x), dim=(-2, -1))

            lo = h // 2 - size // 2

            spec = spec[:, lo:lo + size, lo:lo + size]

            x = torch.fft.ifft2(torch.fft.ifftshift(spec, dim=(-2, -1))).real

            x = x * (size / h) ** 2

            apix = apix * h / size if apix is not None else None

            h = w = size



        if lowpass is not None or highpass is not None:

            if apix is None:

                raise ValueError("--lowpass/--highpass requieren --apix")

            f = torch.fft.fftfreq(h, device=x.device, dtype=x.dtype)

            fr = torch.sqrt(f[:, None] ** 2 + f[None, :] ** 2)  # ciclos/px

            filt = torch.ones_like(fr)

            ln2 = math.log(2.0)

            if lowpass is not None:

                fc = apix / lowpass

                filt = filt * torch.exp(-ln2 * (fr / fc) ** 2)

            if highpass is not None:

                fh = apix / highpass

                filt = filt * (1.0 - torch.exp(-ln2 * (fr / fh) ** 2))

            x = torch.fft.ifft2(torch.fft.fft2(x) * filt).real



        return x.contiguous(), apix


    def make_polar_grid(self, size, radii, n_angles, device, dtype):

        ang = torch.arange(n_angles, device=device, dtype=dtype) * (2 * math.pi / n_angles)

        px = radii[:, None] * torch.cos(ang)[None, :]   # [R,A]

        py = radii[:, None] * torch.sin(ang)[None, :]

        return px, py
    

    def polar_sample(self, images, px, py, dx, dy):

        """Muestrea las imagenes en circulos centrados en (c+dx, c+dy), c = N//2

        (el centro de rotacin habitual en cryo-EM)."""

        n, h, w = images.shape

        c = h // 2

        gx = 2.0 * (px + c + dx) / (w - 1) - 1.0

        gy = 2.0 * (py + c + dy) / (h - 1) - 1.0

        grid = torch.stack((gx, gy), dim=-1)[None].expand(n, -1, -1, -1)

        out = F.grid_sample(images[:, None], grid, mode="bilinear",

                            padding_mode="border", align_corners=True)

        return out[:, 0]  # [N,R,A]

    def center_polar(self, P, w):

        """Resta la media ponderada sobre el disco (-> correlacion tipo Pearson)."""

        mean = (P.mean(-1) * w).sum(-1) / w.sum()

        return P - mean[:, None, None]

    def angular_spectrum(self, P, w, n_angles, max_harmonic):

        """FFT angular, truncada opcionalmente; devuelve (coeficientes, norma)."""

        C = torch.fft.rfft(P, dim=-1)

        if max_harmonic is not None and max_harmonic < n_angles // 2:

            C = C[..., :max_harmonic + 1]

            P = torch.fft.irfft(C, n=n_angles, dim=-1)  # P filtrada, para la norma

        norm = torch.sqrt((w[None, :] * (P ** 2).sum(-1)).sum(-1))

        return C, norm

    @torch.no_grad()

    def pairwise_alignment(self, images, radius_frac=0.85, n_angles=180, max_shift=2.0,

                           shift_step=1.0, mirror=True, max_harmonic=None,

                           block_mb=512, eps=1e-8):

        n, size, _ = images.shape

        dev, dt = images.device, images.dtype



        rmax = radius_frac * (size // 2) - max_shift

        if rmax < 4:

            raise ValueError("Radio til demasiado pequeo: reduce --max-shift "

                             "o aumenta --radius-frac / --size")

        n_rad = max(8, int(round(rmax)))

        radii = torch.linspace(1.0, rmax, n_rad, device=dev, dtype=dt)

        w = radii.clone()  # elemento de rea r dr dtheta

        # rejilla de shifts dentro de un disco de radio max_shift

        vals = np.arange(-max_shift, max_shift + 1e-9, shift_step) if max_shift > 0 else [0.0]

        shifts = [(float(a), float(b)) for a in vals for b in vals

                  if a * a + b * b <= max_shift ** 2 + 1e-9]

        if (0.0, 0.0) not in shifts:

            shifts.append((0.0, 0.0))

        px, py = self.make_polar_grid(size, radii, n_angles, dev, dt)

        # referencia (sin shift)

        P0 = self.center_polar(self.polar_sample(images, px, py, 0.0, 0.0), w)

        C0, n0 = self.angular_spectrum(P0, w, n_angles, max_harmonic)

        A = (C0 * w[None, :, None]).permute(2, 0, 1).contiguous()  # [F,N,R]



        best = torch.full((n, n), -2.0, device=dev, dtype=dt)

        rot = torch.zeros((n, n), device=dev, dtype=torch.long)

        sidx = torch.zeros((n, n), device=dev, dtype=torch.long)

        mir = torch.zeros((n, n), device=dev, dtype=torch.bool)



        rows = max(1, int(block_mb * 2 ** 20 // (4 * n * n_angles)))



        for k, (dx, dy) in enumerate(shifts):

            Pk = self.center_polar(self.polar_sample(images, px, py, dx, dy), w)

            Ck, nk = self.angular_spectrum(Pk, w, n_angles, max_harmonic)

            Bd = torch.conj(Ck).permute(2, 1, 0)  # [F,R,N]

            Bm = Ck.permute(2, 1, 0) if mirror else None



            for i0 in range(0, n, rows):

                i1 = min(n, i0 + rows)

                Ablk = A[:, i0:i1]                                   # [F,r,R]

                denom = (n0[i0:i1, None] * nk[None, :]).clamp_min(eps)



                cc = torch.fft.irfft(torch.bmm(Ablk, Bd).permute(1, 2, 0),

                                     n=n_angles, dim=-1)             # [r,N,A]

                v, a = cc.max(-1)

                v = v / denom

                use_m = torch.zeros_like(v, dtype=torch.bool)



                if mirror:

                    ccm = torch.fft.irfft(torch.bmm(Ablk, Bm).permute(1, 2, 0),

                                          n=n_angles, dim=-1)

                    vm, am = ccm.max(-1)

                    vm = vm / denom

                    use_m = vm > v

                    v = torch.where(use_m, vm, v)

                    a = torch.where(use_m, am, a)



                sl = slice(i0, i1)

                imp = v > best[sl]

                best[sl] = torch.where(imp, v, best[sl])

                rot[sl] = torch.where(imp, a, rot[sl])

                sidx[sl] = torch.where(imp, torch.full_like(sidx[sl], k), sidx[sl])

                mir[sl] = torch.where(imp, use_m, mir[sl])



        sh = torch.tensor(shifts, device=dev, dtype=dt)[sidx]  # [N,N,2]

        sim = torch.maximum(best, best.T)

        sim.fill_diagonal_(1.0)



        return {

            "similarity": sim,

            "similarity_dir": best,

            "rotation_deg": rot.to(dt) * (360.0 / n_angles),

            "shift_x": sh[..., 0],

            "shift_y": sh[..., 1],

            "mirror": mir,

        }

    def get_top_pairs(self, sim, top_k=20):

        n = sim.shape[0]

        if n < 2:

            return []

        iu = torch.triu_indices(n, n, offset=1, device=sim.device)

        vals = sim[iu[0], iu[1]]

        k = min(top_k, vals.numel())

        tv, ti = torch.topk(vals, k)

        return [(iu[0, t].item(), iu[1, t].item(), v.item()) for v, t in zip(tv, ti)]

    def cluster_classes(self, sim, n_clusters=None, threshold=None):
        """Average-linkage jerarquico implementado con PyTorch, sin SciPy."""
        if (n_clusters is None) == (threshold is None):
            raise ValueError("Indica exactamente uno de n_clusters o threshold.")

        # CPU para no reservar memoria adicional en GPU.
        S = torch.as_tensor(sim, dtype=torch.float64, device="cpu")
        if S.ndim != 2 or S.shape[0] != S.shape[1]:
            raise ValueError("La matriz de similitud debe ser cuadrada.")
        n = S.shape[0]
        if n == 0:
            return np.empty(0, dtype=np.int64)
        if n == 1:
            return np.zeros(1, dtype=np.int64)
        if not torch.isfinite(S).all():
            raise ValueError("La matriz de similitud contiene NaN o infinitos.")

        D = torch.clamp(1.0 - 0.5 * (S + S.T), min=0.0)
        D.fill_diagonal_(0.0)

        if n_clusters is not None:
            target = int(n_clusters)
            if not 1 <= target <= n:
                raise ValueError(f"n_clusters debe estar entre 1 y {n}.")
            cutoff = None
        else:
            threshold = float(threshold)
            if not -1.0 <= threshold <= 1.0:
                raise ValueError("threshold debe estar entre -1 y 1.")
            target = 1
            cutoff = 1.0 - threshold

        groups = {i: [i] for i in range(n)}
        sizes = torch.ones(n, dtype=torch.float64)
        active = torch.ones(n, dtype=torch.bool)
        distances = D.clone()
        n_active = n

        while n_active > target:
            candidate = distances.clone()
            candidate[~active, :] = float("inf")
            candidate[:, ~active] = float("inf")
            candidate.fill_diagonal_(float("inf"))
            flat_idx = torch.argmin(candidate)
            min_dist = candidate.reshape(-1)[flat_idx].item()
            if cutoff is not None and min_dist > cutoff:
                break

            a = int((flat_idx // n).item())
            b = int((flat_idx % n).item())
            if a > b:
                a, b = b, a

            # Average linkage: media ponderada por el numero de elementos.
            new_size = sizes[a] + sizes[b]
            keep = active.clone()
            keep[a] = False
            keep[b] = False
            d_new = (sizes[a] * distances[a, keep] +
                     sizes[b] * distances[b, keep]) / new_size
            distances[a, keep] = d_new
            distances[keep, a] = d_new

            groups[a].extend(groups.pop(b))
            sizes[a] = new_size
            active[b] = False
            n_active -= 1

        final_groups = sorted(groups.values(), key=lambda x: min(x))
        labels = np.empty(n, dtype=np.int64)
        for label, members in enumerate(final_groups):
            labels[members] = label
        return labels

    def __init__(self, threshold=0.90, n_clusters=None, device=None,
                 size=64, radius_frac=0.85, n_angles=180,
                 max_shift=3.0, shift_step=1.0, mirror=False,
                 max_harmonic=None, block_mb=512, apix=None,
                 lowpass=None, highpass=None):
        if (threshold is None) == (n_clusters is None):
            raise ValueError("Indica exactamente uno de threshold o n_clusters.")
        self.threshold = threshold
        self.n_clusters = n_clusters
        self.device = torch.device(
            device if device is not None else
            ("cuda" if torch.cuda.is_available() else "cpu"))
        self.size = size
        self.radius_frac = radius_frac
        self.n_angles = n_angles
        self.max_shift = max_shift
        self.shift_step = shift_step
        self.mirror = mirror
        self.max_harmonic = max_harmonic
        self.block_mb = block_mb
        self.apix = apix
        self.lowpass = lowpass
        self.highpass = highpass

        self.labels_ = None
        self.clusters_ = None
        self.representatives_ = None
        self.representative_averages_ = None
        self.similarity_ = None
        self.alignment_ = None

    def fit(self, averages):
        """Calcula similitudes y clustering. Devuelve `self`."""
        if isinstance(averages, np.ndarray):
            data = torch.from_numpy(np.asarray(averages, dtype=np.float32))
        elif torch.is_tensor(averages):
            data = averages.detach().to(dtype=torch.float32)
        else:
            data = torch.as_tensor(averages, dtype=torch.float32)

        if data.ndim == 2:
            data = data.unsqueeze(0)
        if data.ndim != 3:
            raise ValueError(
                f"averages debe tener forma [N,H,W]; recibido {tuple(data.shape)}")
        if data.shape[-1] != data.shape[-2]:
            raise ValueError("Los class averages deben ser imágenes cuadradas.")
        if data.shape[0] == 0:
            raise ValueError("El stack de class averages está vacío.")
        if not torch.isfinite(data).all():
            raise ValueError("averages contiene NaN o infinitos.")

        # Conservar los averages originales y su dispositivo para devolver
        # los representantes sin transformaciones ni cambios de resolución.
        original_data = data.contiguous()
        data = data.to(self.device).contiguous()
        work, _ = self.preprocess(
            data, size=self.size or None, apix=self.apix,
            lowpass=self.lowpass, highpass=self.highpass)
        alignment = self.pairwise_alignment(
            work, radius_frac=self.radius_frac, n_angles=self.n_angles,
            max_shift=self.max_shift, shift_step=self.shift_step,
            mirror=self.mirror, max_harmonic=self.max_harmonic,
            block_mb=self.block_mb)

        similarity = alignment["similarity"]
        labels = self.cluster_classes(
            similarity, n_clusters=self.n_clusters, threshold=self.threshold)

        groups = [np.flatnonzero(labels == k).tolist()
                  for k in range(int(labels.max()) + 1)]

        # Medoide: miembro con mayor similitud media al resto del cluster.
        representatives = []
        sim_np = similarity.detach().cpu().numpy()
        for members in groups:
            if len(members) == 1:
                representatives.append(members[0])
            else:
                within = sim_np[np.ix_(members, members)].copy()
                np.fill_diagonal(within, np.nan)
                scores = np.nanmean(within, axis=1)
                representatives.append(members[int(np.nanargmax(scores))])

        representative_indices = torch.as_tensor(
            representatives, dtype=torch.long, device=original_data.device)
        representative_averages = original_data.index_select(
            0, representative_indices).contiguous()

        self.labels_ = labels
        self.clusters_ = groups
        self.representatives_ = representatives
        self.representative_averages_ = representative_averages
        self.similarity_ = sim_np
        self.alignment_ = {
            key: value.detach().cpu().numpy()
            for key, value in alignment.items()
        }
        return self

    def fit_predict(self, averages):
        """Devuelve un tensor [K,H,W] con el medoide original de cada cluster.

        K es el número de clusters. Cada imagen de salida es una de las
        imágenes de entrada, seleccionada por su similitud media al resto
        de miembros de su cluster; no se calcula ningún promedio.
        """
        return self.fit(averages).representative_averages_
