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
from xmippPyModules.alignPcaFunctions.reconstruct_gpu import *
 
 
class CommonLinesInitializer:
    """
    Orientaciones iniciales para promedios de clase mediante lineas comunes.
 
    Entrada : classes, tensor (N, H, W), imagenes cuadradas y CENTRADAS.
    Salida  : rotations, tensor (N, 3, 3), con la convencion  k3D = R @ k2D
              (la misma que usa reconstruct_volume_*).  R_0 = identidad (fija el gauge).
 
    Algoritmo
    ---------
    1. Cada imagen se transforma a Fourier y se muestrea en coordenadas polares
       (angulos en [0, pi), radios positivos, banda de frecuencias acotada).
    2. Para cada par de imagenes se busca la linea comun: maximo de la correlacion entre lineas,
       comparando tambien con la linea conjugada (F(-k) = conj F(k) para objetos reales).
       Cada linea queda ORIENTADA: el signo (misma direccion / conjugada) se conserva.
    3. Inicializacion espectral (metodo de autovectores, Singer y Shkolnisky 2011) con las lineas orientadas.
    4. Refinamiento por descenso de gradiente (Adam) en el espacio tangente de SO(3), seguido de
       rondas de reponderacion robusta (los pares incoherentes pierden peso).
 
    El descenso de gradiente desde orientaciones aleatorias NO converge de forma fiable (la cuenca de
    atraccion es de unos 60 grados por imagen), por eso se parte de la solucion espectral.
 
    Limitaciones
    ------------
    - Solo para C1 (con simetria hay varias lineas comunes por par).
    - Las imagenes deben estar centradas (un shift introduce una rampa de fase que rompe la correlacion).
    - La mano es arbitraria: fit(..., mirror=True) devuelve la solucion especular.
    - Revisa last_result["inlier_fraction"]: por encima de ~0.8 la solucion es fiable; por debajo de ~0.5
      probablemente ha fallado (umbrales calibrados con datos sinteticos, pueden variar con datos reales).
    """
 
    def __init__(
        self,
        image_size=None,
        n_angles=90,
        n_radial=None,
        min_radius=3.0,
        max_radius=None,
        pixel_size=None,
        min_resolution=60.0,
        max_resolution=20.0,
        device="cuda",
        refine_iters=300,
        lr=0.01,
        robust_rounds=3,
        robust_sigma=0.1,
        robust_iters=200,
        confidence="max",
        inlier_tol=0.05,
        verbose=True,
        iterations=None,   # alias antiguo de refine_iters
        starts=None,       # ignorado (ya no hay arranques aleatorios)
        seed=None,         # ignorado (el metodo es determinista)
    ):
        """
        Banda de frecuencias de las lineas:
          - si se da pixel_size (A/px): de min_resolution a max_resolution (A),
          - si no: de min_radius a max_radius (px); por defecto max_radius = 0.25 * tamano de imagen.
        n_radial=None -> se elige a partir del ancho de la banda.
        """
        device = torch.device(device)
        if device.type == "cuda" and not torch.cuda.is_available():
            print("AVISO: CUDA no disponible, se usa la CPU", flush=True)
            device = torch.device("cpu")
        if confidence not in ("max", "significance"):
            raise ValueError("confidence debe ser 'max' o 'significance'")
 
        self.device = device
        self.image_size = image_size
        self.n_angles = int(n_angles)
        self.n_radial = n_radial
        self.min_radius = float(min_radius)
        self.max_radius = max_radius
        self.pixel_size = pixel_size
        self.min_resolution = float(min_resolution)
        self.max_resolution = float(max_resolution)
        self.refine_iters = int(iterations) if iterations is not None else int(refine_iters)
        self.lr = float(lr)
        self.robust_rounds = int(robust_rounds)
        self.robust_sigma = float(robust_sigma)
        self.robust_iters = int(robust_iters)
        self.confidence = confidence
        self.inlier_tol = float(inlier_tol)
        self.verbose = verbose
        self.last_result = None
        self.last_diagnosis = None
 
    def _log(self, msg):
        if self.verbose:
            print(msg, flush=True)
 
    # -----------------------------------------------------
    # PREPARACION
    # -----------------------------------------------------
 
    def _prepare(self, images):
        images = torch.as_tensor(images, dtype=torch.float32, device=self.device)
 
        if images.ndim != 3:
            raise ValueError("Entrada esperada: (N,H,W)")
 
        N, H, W = images.shape
        if H != W:
            raise ValueError("Se esperan imagenes cuadradas")
        if self.image_size is not None and H != self.image_size:
            raise ValueError(f"Tamano de imagen {H} distinto de image_size={self.image_size}")
        if N < 4:
            raise ValueError("Se necesitan al menos 4 imagenes")
        if not torch.isfinite(images).all():
            raise ValueError("Hay NaN o Inf")
 
        images = images - images.mean(dim=(-2, -1), keepdim=True)
        std = images.std(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
        return images / std
 
    def _band(self, size):
        if self.pixel_size is not None:
            rmin = size * self.pixel_size / self.min_resolution
            rmax = size * self.pixel_size / self.max_resolution
        else:
            rmin = self.min_radius
            rmax = self.max_radius if self.max_radius is not None else 0.25 * size
        rmin = max(float(rmin), 2.0)
        rmax = min(float(rmax), 0.45 * size)
        if rmax <= rmin + 1.0:
            raise ValueError(f"Banda de frecuencias vacia: {rmin:.1f}..{rmax:.1f} px")
        n_radial = self.n_radial
        if n_radial is None:
            n_radial = int(min(max(2.0 * (rmax - rmin), 16), 64))
        return rmin, rmax, int(n_radial)
 
    # -----------------------------------------------------
    # FOURIER POLAR
    # -----------------------------------------------------
 
    def _polar_fourier(self, images, rmin, rmax, n_radial):
        N, H, W = images.shape
 
        ft = torch.fft.fftshift(
            torch.fft.fft2(torch.fft.ifftshift(images, dim=(-2, -1)), norm="forward"),
            dim=(-2, -1),
        )
 
        radii = torch.linspace(rmin, rmax, n_radial, device=self.device)
        angles = (
            torch.arange(self.n_angles, device=self.device, dtype=torch.float32)
            * math.pi / self.n_angles
        )
        rr, aa = torch.meshgrid(radii, angles, indexing="ij")
 
        x = rr * torch.cos(aa)
        y = rr * torch.sin(aa)
 
        # align_corners=True: -1 <-> pixel 0, +1 <-> pixel (W-1); el centro de ft esta en W//2
        gx = 2 * (x + W // 2) / (W - 1) - 1
        gy = 2 * (y + H // 2) / (H - 1) - 1
 
        grid = torch.stack([gx, gy], dim=-1).unsqueeze(0).expand(N, -1, -1, -1).contiguous()
 
        real = F.grid_sample(
            ft.real[:, None].contiguous(), grid,
            mode="bilinear", padding_mode="zeros", align_corners=True,
        )[:, 0]
        imag = F.grid_sample(
            ft.imag[:, None].contiguous(), grid,
            mode="bilinear", padding_mode="zeros", align_corners=True,
        )[:, 0]
 
        polar = torch.complex(real, imag).transpose(1, 2)          # (N, n_angles, n_radial)
        polar = polar - polar.mean(dim=-1, keepdim=True)
        norm = torch.linalg.vector_norm(polar, dim=-1, keepdim=True).clamp_min(1e-8)
        return polar / norm
 
    # -----------------------------------------------------
    # LINEAS COMUNES (ORIENTADAS)
    # -----------------------------------------------------
 
    def _find_common_lines(self, polar):
        """
        Devuelve:
          ang  (N,N,2) long : indices de angulo de la linea comun en la imagen i y en la j,
          sgn  (N,N)        : +1 si la linea tiene la misma direccion en las dos imagenes,
                              -1 si en una aparece conjugada (direccion opuesta),
          conf (N,N)        : confianza del par.
        """
        N, A, K = polar.shape
 
        ang = torch.zeros(N, N, 2, dtype=torch.long, device=self.device)
        sgn = torch.ones(N, N, dtype=torch.float32, device=self.device)
        conf = torch.zeros(N, N, dtype=torch.float32, device=self.device)
 
        for i in range(N - 1):
            pi_ = polar[i]                                          # (A, K)
            pj = polar[i + 1:]                                      # (M, A, K)
 
            c_same = torch.abs(torch.matmul(pi_.conj().unsqueeze(0), pj.transpose(1, 2)))   # (M, A, A)
            c_conj = torch.abs(torch.matmul(pi_.unsqueeze(0), pj.transpose(1, 2)))          # (M, A, A)
 
            use_conj = c_conj > c_same
            corr = torch.where(use_conj, c_conj, c_same).flatten(start_dim=1)               # (M, A*A)
 
            values, idx = corr.max(dim=1)
            chosen_conj = use_conj.flatten(start_dim=1).gather(1, idx.unsqueeze(1)).squeeze(1)
            s = 1.0 - 2.0 * chosen_conj.float()
 
            if self.confidence == "significance":
                values = (values - corr.median(dim=1).values) / (corr.std(dim=1) + 1e-8)
 
            ai = torch.div(idx, A, rounding_mode="floor")
            aj = idx % A
            js = torch.arange(i + 1, N, device=self.device)
 
            ang[i, js, 0] = ai
            ang[i, js, 1] = aj
            ang[js, i, 0] = aj
            ang[js, i, 1] = ai
 
            sgn[i, js] = s
            sgn[js, i] = s
 
            conf[i, js] = values
            conf[js, i] = values
 
        return ang, sgn, conf
 
    def _oriented_lines(self, ang, sgn, iu, ju):
        """Vectores 3D (en el plano de cada imagen) de la linea comun: v = R_i c_ij = R_j c_ji."""
        theta = ang[iu, ju].float() * (math.pi / self.n_angles)     # (P, 2)
        zeros = torch.zeros(iu.shape[0], device=self.device)
 
        ci = torch.stack([torch.cos(theta[:, 0]), torch.sin(theta[:, 0]), zeros], dim=-1)
        cj = sgn[iu, ju].unsqueeze(-1) * torch.stack(
            [torch.cos(theta[:, 1]), torch.sin(theta[:, 1]), zeros], dim=-1
        )
        return ci, cj
 
    # -----------------------------------------------------
    # ROTACIONES SO(3)
    # -----------------------------------------------------
 
    @staticmethod
    def _skew(w):
        x, y, z = w.unbind(dim=-1)
        zero = torch.zeros_like(x)
        return torch.stack(
            [zero, -z, y, z, zero, -x, -y, x, zero], dim=-1
        ).reshape(-1, 3, 3)
 
    def _expm_so3(self, w):
        """Exponencial de so(3) (formula de Rodrigues), w: (N,3) -> (N,3,3)."""
        th = torch.linalg.vector_norm(w, dim=-1)
        th2 = th * th
        small = th < 1e-4
        th_safe = th.clamp_min(1e-8)
 
        a = torch.where(small, 1.0 - th2 / 6.0, torch.sin(th_safe) / th_safe)
        b = torch.where(small, 0.5 - th2 / 24.0, (1.0 - torch.cos(th_safe)) / (th_safe * th_safe))
 
        K = self._skew(w)
        eye = torch.eye(3, device=w.device, dtype=w.dtype)
        return eye + a[:, None, None] * K + b[:, None, None] * torch.matmul(K, K)
 
    # -----------------------------------------------------
    # INICIALIZACION ESPECTRAL
    # -----------------------------------------------------
 
    def _spectral_init(self, ci, cj, w, iu, ju, N):
        """
        Metodo de autovectores. S (2N x 2N) tiene bloques S_ij = w_ij c_ij c_ji^T; sus tres mayores
        autovectores generan las dos primeras columnas de cada rotacion. Una matriz 3x3 se obtiene
        con minimos cuadrados a partir de la ortonormalidad de cada bloque.
        """
        dt = torch.float64
        P = iu.shape[0]
 
        c1 = ci[:, :2].to(dt)
        c2 = cj[:, :2].to(dt)
        ws = (w / w.mean()).to(dt)
 
        B = ws.view(P, 1, 1) * c1.unsqueeze(2) * c2.unsqueeze(1)              # (P, 2, 2)
        S = torch.zeros(N, 2, N, 2, dtype=dt, device=self.device)
        S[iu, :, ju, :] = B
        S[ju, :, iu, :] = B.transpose(1, 2)
        S = S.reshape(2 * N, 2 * N)
 
        _, evecs = torch.linalg.eigh(S)
        V = evecs[:, -3:].reshape(N, 2, 3)                                    # (N, 2, 3)
 
        v00, v01, v02 = V[:, 0, 0], V[:, 0, 1], V[:, 0, 2]
        v10, v11, v12 = V[:, 1, 0], V[:, 1, 1], V[:, 1, 2]
 
        # V_i Q V_i^T = I_2, con Q simetrica = (Q00, Q01, Q02, Q11, Q12, Q22)
        r00 = torch.stack([v00 * v00, 2 * v00 * v01, 2 * v00 * v02, v01 * v01, 2 * v01 * v02, v02 * v02], dim=1)
        r11 = torch.stack([v10 * v10, 2 * v10 * v11, 2 * v10 * v12, v11 * v11, 2 * v11 * v12, v12 * v12], dim=1)
        r01 = torch.stack(
            [v00 * v10, v00 * v11 + v01 * v10, v00 * v12 + v02 * v10,
             v01 * v11, v01 * v12 + v02 * v11, v02 * v12], dim=1
        )
        Mls = torch.cat([r00, r01, r11], dim=0)                               # (3N, 6)
        ones = torch.ones(N, dtype=dt, device=self.device)
        rhs = torch.cat([ones, torch.zeros(N, dtype=dt, device=self.device), ones], dim=0)
 
        eye6 = torch.eye(6, dtype=dt, device=self.device)
        q = torch.linalg.solve(Mls.T @ Mls + 1e-12 * eye6, Mls.T @ rhs)
 
        Q = torch.stack([
            torch.stack([q[0], q[1], q[2]]),
            torch.stack([q[1], q[3], q[4]]),
            torch.stack([q[2], q[4], q[5]]),
        ])
        ev, U = torch.linalg.eigh(Q)
        ev = ev.clamp_min(1e-8)
        Amat = U @ torch.diag(ev.sqrt()) @ U.T                                # A A^T = Q
 
        Mi = torch.matmul(Amat.T, V.transpose(1, 2))                          # (N, 3, 2) = R_i^(12)
        Us, _, Vh = torch.linalg.svd(Mi, full_matrices=False)
        Qi = torch.matmul(Us, Vh)                                             # columnas ortonormales
        r3 = torch.linalg.cross(Qi[:, :, 0], Qi[:, :, 1], dim=-1)
        R = torch.stack([Qi[:, :, 0], Qi[:, :, 1], r3], dim=2)                # (N, 3, 3)
 
        R = torch.matmul(R[0].T, R)                                           # gauge: R_0 = I
        return R.to(torch.float32)
 
    # -----------------------------------------------------
    # REFINAMIENTO
    # -----------------------------------------------------
 
    def _pair_residuals(self, R, ci, cj, iu, ju):
        vi = torch.einsum("pab,pb->pa", R[iu], ci)
        vj = torch.einsum("pab,pb->pa", R[ju], cj)
        return 1.0 - (vi * vj).sum(dim=-1)
 
    def _refine(self, R, ci, cj, w, iu, ju, iters, lr):
        """
        Minimiza  sum_p w_p (1 - v_i . v_j),  v_i = R_i c_ij,  v_j = R_j c_ji,
        con Adam sobre incrementos en el espacio tangente (R <- exp([d]x) R). El gradiente respecto
        a d_i es -w (v_i x v_j) y respecto a d_j su opuesto. La primera rotacion queda fija.
        """
        N = R.shape[0]
        m = torch.zeros(N, 3, device=R.device, dtype=R.dtype)
        v = torch.zeros_like(m)
        b1, b2, eps = 0.9, 0.999, 1e-8
 
        for t in range(1, iters + 1):
            vi = torch.einsum("pab,pb->pa", R[iu], ci)
            vj = torch.einsum("pab,pb->pa", R[ju], cj)
            coef = -w.unsqueeze(-1) * torch.linalg.cross(vi, vj, dim=-1)
 
            g = torch.zeros_like(m)
            g.index_add_(0, iu, coef)
            g.index_add_(0, ju, -coef)
            g[0] = 0.0
 
            m = b1 * m + (1 - b1) * g
            v = b2 * v + (1 - b2) * g * g
            step = (m / (1 - b1 ** t)) / ((v / (1 - b2 ** t)).sqrt() + eps)
 
            cur = lr * 0.5 * (1.0 + math.cos(math.pi * t / iters))          # decaimiento coseno
            R = torch.matmul(self._expm_so3(-cur * step), R)
 
        return R
 
    # -----------------------------------------------------
    # EULER ZYZ  (R = Rz(rot) Ry(tilt) Rz(psi))
    # -----------------------------------------------------
 
    @staticmethod
    def _matrix_to_zyz(R):
        tilt = torch.acos(R[:, 2, 2].clamp(-1.0, 1.0))
        sb = torch.sin(tilt)
 
        rot = torch.atan2(R[:, 1, 2], R[:, 0, 2])
        psi = torch.atan2(R[:, 2, 1], -R[:, 2, 0])
 
        singular = sb.abs() < 1e-6
        rot[singular] = torch.atan2(R[singular, 1, 0], R[singular, 0, 0])
        psi[singular] = 0.0
 
        return torch.rad2deg(psi), torch.rad2deg(rot), torch.rad2deg(tilt)
 
    # -----------------------------------------------------
    # VALIDACION CONTRA LA VERDAD (datos simulados)
    # -----------------------------------------------------
 
    @staticmethod
    def pose_error(R_est, R_true):
        """
        Error angular (grados) por imagen frente a rotaciones verdaderas con la convencion k3D = R k2D.
        Descuenta la rotacion global (R_est[0] = I) y la mano (solucion especular J R J, J = diag(1,1,-1)).
        """
        R_est = R_est.to(torch.float32)
        R_true = R_true.to(device=R_est.device, dtype=torch.float32)
        ref = torch.matmul(R_true[0].T, R_true)
        J = torch.diag(torch.tensor([1.0, 1.0, -1.0], device=R_est.device))
 
        def ang(A, B):
            tr = torch.matmul(A.transpose(1, 2), B).diagonal(dim1=1, dim2=2).sum(-1)
            return torch.rad2deg(torch.acos(((tr - 1.0) / 2.0).clamp(-1.0, 1.0)))
 
        e_same = ang(R_est, ref)
        e_mirr = ang(R_est, torch.matmul(torch.matmul(J, ref), J))
        return e_same if e_same.mean() <= e_mirr.mean() else e_mirr
    
    
    def _select_pair_subset(self, N, pair_fraction=1.0, generator=None):
        """
        Selecciona una fracción aleatoria de parejas, garantizando
        que el grafo de imágenes permanezca conectado.

        Devuelve:
            iu, ju: índices de las parejas seleccionadas.
        """

        if not 0.0 < pair_fraction <= 1.0:
            raise ValueError("pair_fraction debe estar en el intervalo (0, 1].")

        # Número total de parejas posibles
        n_total = N * (N - 1) // 2

        # Número de parejas que queremos utilizar
        n_selected = int(math.ceil(pair_fraction * n_total))

        # Para mantener el grafo conectado necesitamos al menos N-1 parejas
        if n_selected < N - 1:
            raise ValueError(
                f"pair_fraction={pair_fraction} selecciona {n_selected} parejas, "
                f"pero se necesitan al menos {N-1} para conectar {N} imágenes. "
                f"Usa pair_fraction >= {(N-1)/n_total:.4f}."
            )

        # Generar una permutación aleatoria de las imágenes
        perm = torch.randperm(
            N,
            device=self.device,
            generator=generator
        ).tolist()

        # Crear un árbol aleatorio conectado
        selected = set()

        for k in range(N - 1):
            a = perm[k]
            b = perm[k + 1]
            selected.add((min(a, b), max(a, b)))

        # Añadir parejas aleatorias hasta alcanzar n_selected
        remaining = [
            (i, j)
            for i in range(N)
            for j in range(i + 1, N)
            if (i, j) not in selected
        ]

        n_extra = n_selected - len(selected)

        if n_extra > 0:
            indices = torch.randperm(
                len(remaining),
                device=self.device,
                generator=generator
            )[:n_extra].tolist()

            for idx in indices:
                selected.add(remaining[idx])

        # Convertir a tensores
        pairs = sorted(selected)

        iu = torch.tensor(
            [p[0] for p in pairs],
            dtype=torch.long,
            device=self.device
        )

        ju = torch.tensor(
            [p[1] for p in pairs],
            dtype=torch.long,
            device=self.device
        )

        return iu, ju
 
    # -----------------------------------------------------
    # FIT
    # -----------------------------------------------------
 
    def fit(self, classes, mirror=False, pair_fraction=1.0, image_fraction=1.0,
            band_jitter=0.0, generator=None, seed=None):

        images_all = self._prepare(classes)
        N_all, size = images_all.shape[0], images_all.shape[1]

        # Generador reproducible
        if generator is None:
            generator = torch.Generator(device=self.device)

            if seed is not None:
                generator.manual_seed(int(seed))
            else:
                generator.seed()
                
          # --- subconjunto de averages ---
        if image_fraction < 1.0:
            n_sub = max(8, int(math.ceil(image_fraction * N_all)))
            image_idx = torch.randperm(N_all, device=self.device, generator=generator)[:n_sub].sort().values
        else:
            image_idx = torch.arange(N_all, device=self.device)
        images = images_all[image_idx]
        N = images.shape[0]

        # -------------------------------------------------
        # FOURIER Y LINEAS COMUNES
        # -------------------------------------------------

        rmin, rmax, n_radial = self._band(size)
        
        if band_jitter > 0:
            u = torch.rand(2, generator=generator, device=self.device) * 2 - 1
            rmax = min(rmax * (1.0 + band_jitter * float(u[0])), 0.45 * size)
            rmin = max(rmin * (1.0 + 0.5 * band_jitter * float(u[1])), 2.0)
            rmax = max(rmax, rmin + 2.0)
            if self.n_radial is None:
                n_radial = int(min(max(2.0 * (rmax - rmin), 16), 64))

        band_txt = f"{rmin:.1f}..{rmax:.1f} px"

        if self.pixel_size is not None:
            band_txt += (
                f" ({size * self.pixel_size / rmin:.0f}"
                f"..{size * self.pixel_size / rmax:.0f} A)"
            )

        self._log(
            f"CommonLines: N={N}, banda {band_txt}, "
            f"{n_radial} radios, {self.n_angles} angulos"
        )

        polar = self._polar_fourier(
            images, rmin, rmax, n_radial
        )

        # Se calculan las líneas comunes para TODAS las parejas
        ang, sgn, conf = self._find_common_lines(polar)

        # -------------------------------------------------
        # SELECCION DE PAREJAS
        # -------------------------------------------------

        iu, ju = self._select_pair_subset(
            N,
            pair_fraction=pair_fraction,
            generator=generator
        )

        n_total = N * (N - 1) // 2
        n_selected = len(iu)

        self._log(
            f"CommonLines: usando {n_selected}/{n_total} parejas "
            f"({100*n_selected/n_total:.1f} %)"
        )

        # Líneas orientadas de las parejas seleccionadas
        ci, cj = self._oriented_lines(ang, sgn, iu, ju)

        # -------------------------------------------------
        # INICIALIZACION ESPECTRAL
        # -------------------------------------------------

        w0 = conf[iu, ju].clamp_min(1e-9)
        w0 = w0 / w0.sum()

        R = self._spectral_init(
            ci, cj, w0, iu, ju, N
        )

        # -------------------------------------------------
        # REFINAMIENTO
        # -------------------------------------------------

        R = self._refine(
            R, ci, cj, w0, iu, ju,
            self.refine_iters, self.lr
        )

        for _ in range(self.robust_rounds):

            res = self._pair_residuals(
                R, ci, cj, iu, ju
            )

            w = (
                w0 * torch.exp(
                    -0.5 * (res / self.robust_sigma) ** 2
                )
            ).clamp_min(1e-12)

            w = w / w.sum()

            R = self._refine(
                R, ci, cj, w, iu, ju,
                self.robust_iters, self.lr
            )

        # -------------------------------------------------
        # EVALUACION CON TODAS LAS PAREJAS
        # -------------------------------------------------

        with torch.no_grad():

            iu_all, ju_all = torch.triu_indices(
                N, N, offset=1, device=self.device
            )

            ci_all, cj_all = self._oriented_lines(
                ang, sgn, iu_all, ju_all
            )

            res = self._pair_residuals(
                R, ci_all, cj_all, iu_all, ju_all
            )

            # Confianza normalizada sobre todas las parejas
            w_eval = conf[iu_all, ju_all].clamp_min(1e-9)
            w_eval = w_eval / w_eval.sum()

            inlier_fraction = (
                res < self.inlier_tol
            ).float().mean().item()

            loss = (w_eval * res).sum()

            residual = torch.zeros(
                N, N, device=self.device
            )

            residual[iu_all, ju_all] = res
            residual[ju_all, iu_all] = res

            # -------------------------------------------------
            # MIRROR
            # -------------------------------------------------

            if mirror:
                J = torch.diag(
                    torch.tensor(
                        [1.0, 1.0, -1.0],
                        device=self.device
                    )
                )

                R = torch.matmul(
                    torch.matmul(J, R), J
                )

            psi, rot, tilt = self._matrix_to_zyz(R)

            self.last_result = {
                "rotations": R,
                "psi": psi,
                "rot": rot,
                "tilt": tilt,
                "shifts": torch.zeros(
                    N, 2, device=self.device
                ),
                "common_lines": ang,
                "common_line_sign": sgn,
                "pair_confidence": conf,
                "residuals": residual,
                "loss": loss.detach().clone(),
                "inlier_fraction": inlier_fraction,
                "mirrored": bool(mirror),

                # Información adicional de pair_fraction
                "used_pairs": torch.stack([iu, ju], dim=1),
                "n_used_pairs": n_selected,
                "n_total_pairs": n_total,
                "pair_fraction": n_selected / n_total,
                "image_idx": image_idx,
            }

        self._log(
            f"CommonLines: perdida = {loss.item():.4f}, "
            f"pares coherentes (residuo < {self.inlier_tol}) = "
            f"{100 * inlier_fraction:.1f} %"
        )

        if inlier_fraction < 0.5:
            self._log(
                "AVISO: menos del 50 % de los pares es coherente; "
                "la solucion probablemente no es fiable "
                "(ruido alto, banda inadecuada, averages sin centrar "
                "o simetria distinta de C1)."
            )

        return R
    
    
    @torch.no_grad()
    def compute_common_lines(self, classes):
        """
        Lineas comunes de todas las parejas. Solo dependen de las imagenes: calculalas una vez
        y pasalas como `lines=` a diagnose_orientations / reorient_averages en cada iteracion.
        Devuelve (ang, sgn, conf).
        """
        images = self._prepare(classes)
        rmin, rmax, n_radial = self._band(images.shape[1])
        polar = self._polar_fourier(images, rmin, rmax, n_radial)
        return self._find_common_lines(polar)
 
    @torch.no_grad()
    def diagnose_orientations(
        self,
        classes,
        rotations,
        shifts=None,
        inlier_tol=None,
        outlier_threshold=3.5,
        symmetry=None,
        symmetry_ops=None,
        lines=None,
        scale_floor=0.05,
    ):
        """
        Detecta averages cuyas orientaciones son inconsistentes con las lineas comunes
        estimadas entre todos los averages.
 
        Parametros
        ----------
        classes : tensor (N,H,W), averages centrados. Puede ser None si se pasa `lines`.
        rotations : tensor (N,3,3), con la convencion k3D = R @ k2D.
        shifts : se acepta por compatibilidad, no se usa.
        inlier_tol : umbral del residuo 1 - v_i.v_j. Debe reflejar la precision de las poses:
            con un error de d grados por pose, un par correcto puede dar hasta 1 - cos(2d)
            (0.03 para 7 grados, 0.06 para 10, 0.13 para 15). Por defecto, self.inlier_tol.
        outlier_threshold : umbral del z robusto de la puntuacion (3.5 es el estandar).
        symmetry : None, "c1" o un nombre soportado por reconstruct().generate_symmetry (p. ej. "d2").
        symmetry_ops : alternativa a `symmetry`: tensor (G,3,3) de rotaciones propias que actuan sobre
            vectores 3D. Evita instanciar la clase de reconstruccion.
        lines : (ang, sgn, conf) devuelto por compute_common_lines (o result["lines"]) para no recalcular.
        scale_floor : suelo de la escala robusta (en unidades de fraccion de parejas incoherentes);
            evita marcar diferencias insignificantes cuando casi todos los averages son coherentes.
 
        Puntuacion
        ----------
        Por average, fraccion ponderada de parejas incoherentes (residuo > tol), normalizada con
        mediana y MAD entre averages. Es mas robusta al ruido que la mediana del residuo: una pose
        mala queda cerca del azar (tol/2 por pareja en C1) y una buena muy por encima.
 
        Convencion: si la fraccion coherente global es cercana al azar, prueba rotations.transpose(1, 2).
        """
 
        # -------------------------------------------------
        # PREPARACION
        # -------------------------------------------------
 
        R = torch.as_tensor(rotations, dtype=torch.float32, device=self.device)
        if R.ndim != 3 or R.shape[1:] != (3, 3):
            raise ValueError(f"rotations debe tener forma (N,3,3), pero tiene {tuple(R.shape)}")
        if not torch.isfinite(R).all():
            raise ValueError("Hay NaN o Inf en rotations")
        N = R.shape[0]
        if N < 4:
            raise ValueError("Se necesitan al menos 4 averages")
 
        if shifts is not None:
            shifts = torch.as_tensor(shifts)
            if shifts.shape != (N, 2):
                raise ValueError(f"shifts debe tener forma ({N},2)")
 
        tol = self.inlier_tol if inlier_tol is None else float(inlier_tol)
 
        # -------------------------------------------------
        # GRUPO DE SIMETRIA
        # -------------------------------------------------
 
        S = None
        if symmetry_ops is not None:
            S = torch.as_tensor(symmetry_ops, dtype=R.dtype, device=self.device)
            symmetry = symmetry if symmetry is not None else "custom"
        elif symmetry is not None and str(symmetry).lower() not in ("c1", "none"):
            S = reconstruct().generate_symmetry(symmetry, device=self.device)
            S = torch.as_tensor(S, dtype=R.dtype, device=self.device)
 
        if S is not None:
            if S.ndim != 3 or S.shape[1:] != (3, 3):
                raise ValueError("El grupo de simetria debe tener forma (G,3,3)")
            eye = torch.eye(3, device=self.device, dtype=S.dtype)
            if not torch.allclose(torch.matmul(S, S.transpose(1, 2)), eye.expand_as(S), atol=1e-3):
                raise ValueError("Las operaciones de simetria no son ortogonales")
            if (torch.linalg.det(S) < 0).any():
                self._log("AVISO: el grupo de simetria contiene operaciones con determinante negativo")
 
        # -------------------------------------------------
        # LINEAS COMUNES (calculadas o reutilizadas)
        # -------------------------------------------------
 
        if lines is None:
            if classes is None:
                raise ValueError("Hay que pasar classes o lines")
            ang, sgn, conf = self.compute_common_lines(classes)
        else:
            ang, sgn, conf = (t.to(self.device) for t in lines)
            if ang.shape[0] != N:
                raise ValueError(f"lines corresponde a {ang.shape[0]} averages y rotations a {N}")
 
        self._log(f"Diagnose orientations: N={N}, tol={tol}, symmetry={symmetry}")
 
        # -------------------------------------------------
        # RESIDUOS DE TODAS LAS PAREJAS
        # -------------------------------------------------
 
        iu, ju = torch.triu_indices(N, N, offset=1, device=self.device)
        ci, cj = self._oriented_lines(ang, sgn, iu, ju)
 
        vi = torch.einsum("pij,pj->pi", R[iu], ci)        # R_i c_ij
        vj = torch.einsum("pij,pj->pi", R[ju], cj)        # R_j c_ji
 
        if S is None:
            res = 1.0 - torch.sum(vi * vj, dim=1)
            best_symmetry = torch.zeros(len(iu), dtype=torch.long, device=self.device)
        else:
            # el grupo es cerrado respecto a la inversa: basta aplicarlo a un vector
            candidates = torch.stack(
                [1.0 - torch.sum(torch.einsum("ij,pj->pi", S[k], vi) * vj, dim=1) for k in range(len(S))],
                dim=0,
            )
            res, best_symmetry = torch.min(candidates, dim=0)
 
        # -------------------------------------------------
        # MATRICES (N,N)
        # -------------------------------------------------
 
        residual_matrix = torch.zeros(N, N, device=self.device)
        confidence_matrix = torch.zeros(N, N, device=self.device)
        symmetry_matrix = torch.full((N, N), -1, dtype=torch.long, device=self.device)
 
        pair_conf = conf[iu, ju].clamp_min(0.0)
 
        residual_matrix[iu, ju] = res
        residual_matrix[ju, iu] = res
        confidence_matrix[iu, ju] = pair_conf
        confidence_matrix[ju, iu] = pair_conf
        symmetry_matrix[iu, ju] = best_symmetry
        symmetry_matrix[ju, iu] = best_symmetry
 
        # -------------------------------------------------
        # ESTADISTICAS POR AVERAGE (vectorizadas, diagonal excluida)
        # -------------------------------------------------
 
        W = confidence_matrix.clone()
        W.fill_diagonal_(0.0)
        if W.sum() <= 1e-12:
            W = torch.ones(N, N, device=self.device)
            W.fill_diagonal_(0.0)
 
        row_w = W.sum(dim=1).clamp_min(1e-12)
        average_mean = (W * residual_matrix).sum(dim=1) / row_w
        bad_pair_fraction = (W * (residual_matrix > tol).float()).sum(dim=1) / row_w
 
        Rn = residual_matrix.clone()
        Rn.fill_diagonal_(float("nan"))
        average_median = torch.nanmedian(Rn, dim=1).values
        mean_confidence = confidence_matrix.sum(dim=1) / (N - 1)
 
        total_w = 0.5 * W.sum()
        total_wr = 0.5 * (W * residual_matrix).sum()
        leave_one_out_mean = (total_wr - (W * residual_matrix).sum(dim=1)) / (total_w - row_w).clamp_min(1e-12)
 
        # -------------------------------------------------
        # PUNTUACION ROBUSTA (fraccion de parejas incoherentes)
        # -------------------------------------------------
 
        center = bad_pair_fraction.median()
        mad = (bad_pair_fraction - center).abs().median()
        robust_scale = (1.4826 * mad).clamp_min(float(scale_floor))
 
        outlier_scores = (bad_pair_fraction - center) / robust_scale
        suspect_mask = outlier_scores > outlier_threshold
        suspect_indices = torch.where(suspect_mask)[0]
 
        # -------------------------------------------------
        # CONSISTENCIA GLOBAL
        # -------------------------------------------------
 
        pair_weights = pair_conf if pair_conf.sum() > 1e-12 else torch.ones_like(pair_conf)
        global_mean = (res * pair_weights).sum() / pair_weights.sum().clamp_min(1e-12)
        global_inlier_fraction = (res < tol).float().mean()
 
        n_ops = 1 if S is None else len(S)
        chance_inlier_fraction = min(1.0, n_ops * tol / 2.0)       # direcciones aleatorias: P(1 - cos < tol) = tol/2
 
        # -------------------------------------------------
        # RESULTADO
        # -------------------------------------------------
 
        result = {
            "pair_residuals": residual_matrix,
            "pair_confidence": confidence_matrix,
 
            "average_mean_residual": average_mean,
            "average_median_residual": average_median,
            "bad_pair_fraction": bad_pair_fraction,
            "average_mean_confidence": mean_confidence,
 
            "outlier_scores": outlier_scores,
            "suspect_mask": suspect_mask,
            "suspect_indices": suspect_indices,
 
            "global_mean_residual": global_mean,
            "global_inlier_fraction": global_inlier_fraction,
            "chance_inlier_fraction": chance_inlier_fraction,
            "leave_one_out_mean": leave_one_out_mean,
 
            "common_lines": ang,
            "common_line_sign": sgn,
            "lines": (ang, sgn, conf),                 # para reutilizar en la siguiente llamada
 
            "rotations": R,
            "inlier_tol": tol,
            "outlier_threshold": outlier_threshold,
            "scale_floor": float(scale_floor),
 
            "symmetry": symmetry,
            "symmetry_group": S,
            "best_symmetry_index": best_symmetry,
            "symmetry_matrix": symmetry_matrix,
        }
 
        self.last_diagnosis = result
 
        # -------------------------------------------------
        # INFORMACION
        # -------------------------------------------------
 
        gi = global_inlier_fraction.item()
        self._log(f"Residuo global: {global_mean.item():.4f}")
        self._log(f"Parejas coherentes: {100 * gi:.1f} % (azar aprox. {100 * chance_inlier_fraction:.1f} %)")
        if gi < 2.0 * chance_inlier_fraction:
            self._log("AVISO: la fraccion coherente es cercana al azar; revisa la convencion de R "
                      "(prueba la transpuesta), la simetria, el tol o el centrado de los averages.")
        self._log(f"Averages sospechosos: {len(suspect_indices)}")
        if len(suspect_indices) > 0:
            self._log(f"Indices sospechosos: {suspect_indices.tolist()}")
 
        # if S is not None:
        #     for k in range(len(S)):
        #         self._log(f"Simetria {k}: {(best_symmetry == k).sum().item()} parejas")
 
        # self.print_average_scores(outlier_scores, suspect_mask, bad_pair_fraction)
 
        return result
    
    
 
    def print_average_scores(self, scores, suspect, fractions=None, only_suspects=True):
        """Imprime la puntuacion de los averages (por defecto, solo los sospechosos). Respeta self.verbose."""
        if not self.verbose:
            return
 
        scores = scores.detach().cpu().numpy()
        suspect = suspect.detach().cpu().numpy()
        fractions = None if fractions is None else fractions.detach().cpu().numpy()
 
        rows = [i for i in range(len(scores)) if (suspect[i] or not only_suspects)]
        if not rows:
            print("Ningun average sospechoso.", flush=True)
            return
 
        print("\nAverage | Puntuacion | Parejas incoherentes | Estado")
        print("-----------------------------------------------------")
        for i in rows:
            frac = "      -" if fractions is None else f"{100 * fractions[i]:6.1f} %"
            print(f"{i:7d} | {scores[i]:10.2f} | {frac:>20s} | {'MAL' if suspect[i] else 'BIEN'}")
            
            
 
    @torch.no_grad()
    def reorient_averages(self, classes, rotations, indices, lines=None, exclude=None, sigma=0.1, iters=8):
        """
        Reorienta los averages `indices` a partir de las poses de los demas (Procrustes con
        reponderacion robusta). Solo C1. Los averages en `indices` y en `exclude` no se usan como
        referencia. Devuelve las rotaciones (N,3,3) con esos averages corregidos.
        """
        R = torch.as_tensor(rotations, dtype=torch.float32, device=self.device).clone()
        N = R.shape[0]
 
        if lines is None:
            ang, sgn, conf = self.compute_common_lines(classes)
        else:
            ang, sgn, conf = (t.to(self.device) for t in lines)
 
        indices = [int(i) for i in (indices.tolist() if torch.is_tensor(indices) else indices)]
        skip = set(indices) | set(int(i) for i in (exclude or []))
        keep = torch.tensor([j for j in range(N) if j not in skip], device=self.device)
        if len(keep) < 3:
            raise ValueError("Hay demasiados averages excluidos como referencia")
 
        for i in indices:
            th_i = ang[i, keep, 0].float() * (math.pi / self.n_angles)
            th_j = ang[i, keep, 1].float() * (math.pi / self.n_angles)
            z = torch.zeros_like(th_i)
 
            c = torch.stack([torch.cos(th_i), torch.sin(th_i), z], dim=-1)                       # c_ij
            cj = sgn[i, keep].unsqueeze(-1) * torch.stack(
                [torch.cos(th_j), torch.sin(th_j), z], dim=-1)                                   # c_ji orientada
            t = torch.einsum("jab,jb->ja", R[keep], cj)                                           # R_j c_ji
 
            w0 = conf[i, keep].clamp_min(1e-9)
            w = w0.clone()
            Ri = R[i]
            for _ in range(iters):
                M = torch.einsum("j,ja,jb->ab", w, t, c)
                U, _, Vh = torch.linalg.svd(M)
                d = torch.sign(torch.linalg.det(U @ Vh))
                D = torch.diag(torch.stack([torch.ones_like(d), torch.ones_like(d), d]))
                Ri = U @ D @ Vh
                res = 1.0 - ((c @ Ri.T) * t).sum(-1)
                w = w0 * torch.exp(-0.5 * (res / sigma) ** 2)
            R[i] = Ri
 
        return R
    
    
    def select_non_suspect(self, img_idx, rotM, shiftM=None, w_ent=None, diagnosis=None):
        """
        Conserva únicamente las entradas correspondientes a averages
        que NO han sido identificados como sospechosos.
        """
    
        if diagnosis is None:
            raise ValueError("Es necesario proporcionar el diagnóstico.")
    
        suspect_mask = diagnosis["suspect_mask"].to(
            device=img_idx.device,
            dtype=torch.bool
        )
    
        if len(img_idx) != len(suspect_mask):
            raise ValueError(
                "El número de imágenes no coincide con suspect_mask."
            )
    
        keep_mask = ~suspect_mask
        sel = torch.nonzero(keep_mask).squeeze(1)
    
        img_idx = img_idx[sel]
    
        rotM = rotM[sel.to(rotM.device)]
    
        if shiftM is not None:
            shiftM = shiftM[sel.to(shiftM.device)]
    
        if w_ent is not None:
            w_ent = w_ent[sel.to(w_ent.device)]
    
        return img_idx, rotM, shiftM, w_ent

 

