#!/usr/bin/env python3
"""/***************************************************************************
 *
 * Authors:    Erney Ramirez-Aportela
 *
  ***************************************************************************/
"""

"""
Estrategia para las primeras iteraciones de BnBgpu.create_classes:
 
  1. Marcar partículas malas (no encajan con ningún average, o energía anómala)
     ANTES de dividir, para que los scores de split midan estructura y no basura.
  2. Dividir las clases buenas con mayor exceso de varianza.
  3. Con los huecos que queden (más una pequeña reserva) crear clases basura
     sembradas con las partículas marcadas.
  4. Al final de la iteración, cluster de averages casi idénticos
     (clases basura y recién divididas protegidas).
 
"""

import numpy as np
import torch
 
 
class Split:
    

    # ------------------------------------------------------------------
    # Estadística robusta
    # ------------------------------------------------------------------
    @torch.no_grad()
    def robust_outliers(self, x, k=3.5, side="high"):
        """True donde x es atípico respecto a la mediana/MAD de x."""
        x = x.float()
        fin = torch.isfinite(x)
        if int(fin.sum()) < 20:
            return ~fin
        xf = x[fin]
        med = xf.median()
        mad = 1.4826 * (xf - med).abs().median() + 1e-8
        z = (x - med) / mad
        out = (z > k) if side == "high" else (z.abs() > k)
        return out | ~fin
 
    # ------------------------------------------------------------------
    # Partículas malas
    # ------------------------------------------------------------------
    @torch.no_grad()
    def flag_junk_particles(self, dist, proj, ref_mask=None,
                            k_dist=3.5, max_frac=0.15, k_norm=4.0):
        """
        dist: (B,) distancia (matching normalizado) al mejor average.
        proj: (B,D) proyección PCA SIN normalizar de la partícula.
        ref_mask: partículas usadas para estimar mediana/MAD de dist
                  (se excluyen las que ya están en clases basura).
 
        Marca como mala una partícula si:
          - su distancia es un atípico por la derecha (no encaja con nadie),
            con un tope de max_frac del lote, o
          - la energía de su proyección es anómala por arriba o por abajo
            (hielo, carbono, agregados, partículas vacías). El matching
            normalizado ignora la energía, por eso se mira aparte.
        """
        dist = torch.nan_to_num(dist.float(), nan=1e9, posinf=1e9)
        norm = proj.float().norm(dim=1).clamp_min(1e-8)
 
        if ref_mask is None or int(ref_mask.sum()) < 20:
            ref_mask = torch.ones_like(dist, dtype=torch.bool)
        d_ref = dist[ref_mask]
 
        med = d_ref.median()
        mad = 1.4826 * (d_ref - med).abs().median() + 1e-8
        thr = med + k_dist * mad
        if max_frac is not None:
            thr = torch.maximum(thr, torch.quantile(d_ref, 1.0 - max_frac))
 
        junk = dist >= thr
        junk |= self.robust_outliers(norm.log(), k=k_norm, side="both")
        return junk
 
    @torch.no_grad()
    def seed_junk_classes(self, newCL, pool_imgs, pool_proj, cvecs, slots,
                          min_per_class=50):
        """Reparte el pool de partículas malas en las clases basura (k-means en PCA)."""
        if not slots or not pool_imgs:
            return []
 
        Pi = torch.cat(pool_imgs)
        Pp = torch.cat(pool_proj).float()
        Pp = Pp / Pp.norm(dim=1, keepdim=True).clamp_min(1e-8)   # misma métrica que el matching
        N = Pi.shape[0]
 
        m = min(len(slots), N // min_per_class)
        if m < 1:
            return []
 
        if m == 1:
            lab = torch.zeros(N, dtype=torch.long, device=Pi.device)
        else:
            _, lab = self.kmeans_pytorch_for_averages(Pi, Pp, cvecs, num_clusters=m)
 
        used = []
        for k in range(m):
            sel = lab == k
            if sel.any():
                newCL[slots[k]] = Pi[sel]
                used.append(slots[k])
        return used
 
    @torch.no_grad()
    def route_junk(self, matches, proj_raw, cl, freqBn, coef, cvecs,
                   k_dist=3.5, max_frac=0.15, k_norm=4.0):
        """
        Reasigna a clases basura las partículas malas que el matching mandó a una
        clase buena. Se llama justo después del matching y antes de actualizar
        clases / alinear partículas. Modifica `matches` in situ:
          - columna 1 (clase): clase basura con centroide PCA más cercano (coseno)
          - columnas 3 y 4 (rotación, índice de shift): 0, para no alinear ruido
        proj_raw: (N,D) proyecciones PCA SIN normalizar, mismas filas que matches.
        Devuelve el número de partículas movidas.
        """
        junk_cls = getattr(self, 'junk_cls', None)
        if junk_cls is None or junk_cls.numel() != cl.shape[0] or not bool(junk_cls.any()):
            return 0
        if proj_raw.shape[0] != matches.shape[0]:
            return 0
 
        jidx = torch.where(junk_cls)[0]
        labels = matches[:, 1].long()
        in_junk = junk_cls[labels]
 
        flag = self.flag_junk_particles(matches[:, 2], proj_raw, ref_mask=~in_junk,
                                        k_dist=k_dist, max_frac=max_frac, k_norm=k_norm)
        move = flag & ~in_junk
        n_move = int(move.sum())
        if n_move == 0:
            return 0
 
        cent = self.create_batchExp(cl[jidx], freqBn, coef, cvecs)[0].float()
        cent = cent / cent.norm(dim=1, keepdim=True).clamp_min(1e-8)
        x = proj_raw[move].float()
        x = x / x.norm(dim=1, keepdim=True).clamp_min(1e-8)
        best = (x @ cent.T).argmax(dim=1)
 
        matches[move, 1] = jidx[best].to(matches.dtype)
        matches[move, 3] = 0
        matches[move, 4] = 0
        return n_move
 
    # ------------------------------------------------------------------
    # Split por exceso de varianza
    # ------------------------------------------------------------------
    @torch.no_grad()
    def split_score(self, newProj, min_ratio=3.0, min_n=20, edge_factor=1.1):
        """
        newProj: lista de K tensores (n_k, D) o None (proyecciones PCA por clase).
        Las proyecciones se normalizan a norma 1, igual que en el matching;
        si no, el eje de mayor varianza sería la variación de contraste.
        Devuelve scores (K,) y ejes (K,D).
        """
        K = len(newProj)
        ref = next((P for P in newProj if P is not None), None)
        if ref is None:
            return None, None
        dev, D = ref.device, ref.shape[1]
 
        scores = torch.zeros(K, device=dev)
        axes = torch.zeros(K, D, device=dev)
 
        idx = [k for k, P in enumerate(newProj)
               if P is not None and P.shape[0] >= max(min_n, min_ratio * D)]
        if not idx:
            return scores, axes
 
        covs, ns = [], []
        for k in idx:
            P = newProj[k].float()
            P = P / P.norm(dim=1, keepdim=True).clamp_min(1e-8)
            Pc = P - P.mean(0, keepdim=True)
            covs.append(Pc.T @ Pc)
            ns.append(P.shape[0])
 
        n = torch.tensor(ns, device=dev, dtype=torch.float32)
        C = torch.stack(covs) / (n - 1).view(-1, 1, 1)
 
        ev, evec = torch.linalg.eigh(C)                 # una sola llamada batched
        top = ev[:, -1]
        ax = evec[:, :, -1]
 
        gamma = D / n
        sigma2 = ev.median(dim=1).values / (1.0 - gamma / 3.0)
        lam_plus = edge_factor * sigma2 * (1.0 + gamma.sqrt()) ** 2
 
        sc = n * (top - lam_plus).clamp(min=0)
 
        idx_t = torch.tensor(idx, device=dev)
        scores[idx_t] = sc
        axes[idx_t] = ax
        return scores, axes
 
    @torch.no_grad()
    def _otsu_cut_1d(self, t, min_frac=0.1):
        """Corte 1D que maximiza la varianza entre grupos. Devuelve labels 0/1 o None."""
        n = t.numel()
        ts, order = torch.sort(t)
        csum = torch.cumsum(ts, 0)
        total = csum[-1]
        k = torch.arange(1, n, device=t.device, dtype=ts.dtype)
        m1 = csum[:-1] / k
        m2 = (total - csum[:-1]) / (n - k)
        between = k * (n - k) * (m1 - m2) ** 2
 
        lo = max(int(min_frac * n), 2)
        valid = (k >= lo) & (k <= n - lo)
        if not valid.any():
            return None
        between = torch.where(valid, between, torch.full_like(between, -1.0))
 
        kbest = int(between.argmax()) + 1
        labels = torch.zeros(n, dtype=torch.long, device=t.device)
        labels[order[kbest:]] = 1
        return labels
 
    @torch.no_grad()
    def _split_labels(self, P, edge_factor=1.1, max_dims=5, min_frac=0.1, iters=10):
        """Divide una clase en 2: k-means en el subespacio con señal, inicializado con Otsu."""
        P = P.float()
        P = P / P.norm(dim=1, keepdim=True).clamp_min(1e-8)
        n, D = P.shape
        Pc = P - P.mean(0, keepdim=True)
        ev, evec = torch.linalg.eigh(Pc.T @ Pc / (n - 1))
        ev, evec = ev.flip(0), evec.flip(1)
 
        gamma = D / n
        sigma2 = ev.median() / (1.0 - gamma / 3.0)
        lam_plus = edge_factor * sigma2 * (1.0 + gamma ** 0.5) ** 2
 
        m = max(1, min(int((ev > lam_plus).sum()), max_dims))
        Z = Pc @ evec[:, :m]
 
        labels = self._otsu_cut_1d(Z[:, 0], min_frac)
        if labels is None:
            return None
 
        lo = max(int(min_frac * n), 2)
        for _ in range(iters):
            cent = torch.stack([Z[labels == 0].mean(0), Z[labels == 1].mean(0)])
            new = torch.cdist(Z, cent).argmin(dim=1)
            c1 = int(new.sum())
            if c1 < lo or n - c1 < lo:
                break
            if (new == labels).all():
                break
            labels = new
        return labels
