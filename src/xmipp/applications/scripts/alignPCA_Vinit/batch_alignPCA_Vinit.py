#!/usr/bin/env python3
"""/***************************************************************************
 *
 * Authors:    Erney Ramirez-Aportela
 *
  ***************************************************************************/
"""

import argparse
import os
import sys
import random

import mrcfile
import numpy as np
import torch
from xmippPyModules.alignPcaFunctions.pca_gpu import *
from xmippPyModules.alignPcaFunctions.bnb_gpu import *
from xmippPyModules.alignPcaFunctions.reconstruct_gpu import *
from xmippPyModules.alignPcaFunctions.assessment import *
from xmippPyModules.alignPcaFunctions.init_angle_generation import *


# Iteraciones en las que se reconstruye con la mitad de los averages (grupos A/B)
AB_ITERS = 7


# ----------------------------------------------------------------------
# I/O
# ----------------------------------------------------------------------
def read_images(mrcfilename):
    with mrcfile.open(mrcfilename, permissive=True) as f:
        return f.data.astype(np.float32)


def load_vol(file_path, device='cuda'):
    with mrcfile.open(file_path, mode='r') as mrc:
        data = mrc.data.astype(np.float32)
        vol = torch.from_numpy(data).to(device)
        vol = (vol - vol.mean()) / vol.std()
    return vol


def save_vol(volume, filename, psize):
    vol_np = volume.detach().cpu().numpy().astype('float32')
    with mrcfile.new(filename, overwrite=True) as mrc:
        mrc.set_data(vol_np)
        mrc.voxel_size = psize


def save_proj(projections, filename, psize):
    proj_np = projections.detach().cpu().numpy().astype("float32")
    with mrcfile.new(filename, overwrite=True) as mrc:
        mrc.set_data(proj_np)
        mrc.voxel_size = psize


# ----------------------------------------------------------------------
# Rejilla de frecuencias (para los shifts en Fourier)
# ----------------------------------------------------------------------
def flatGrid(freq_band, nBand):
    dim, _ = freq_band.shape
    device = freq_band.device

    fx = torch.fft.rfftfreq(dim, d=0.5 / np.pi, device=device)
    fy = torch.fft.fftfreq(dim, d=0.5 / np.pi, device=device)

    grid_x, grid_y = torch.meshgrid(fx, fy, indexing='xy')
    del fx, fy

    grid_flat = []
    for n in range(nBand):
        mask = (freq_band == n)
        fx_n = grid_x[mask]
        fy_n = grid_y[mask]
        grid_flat.append(torch.stack([fx_n, fy_n], dim=0))
        del mask, fx_n, fy_n

    del grid_x, grid_y
    return grid_flat


# ----------------------------------------------------------------------
# Deterministic annealing: utilidades
# ----------------------------------------------------------------------
def da_neff(it, neff0, neff1, da_iters):
    """Numero efectivo objetivo de candidatos: lineal neff0 -> neff1 a lo largo del DA."""
    if da_iters <= 1:
        return neff1
    return neff0 + (neff1 - neff0) * it / (da_iters - 1)


def build_da_entries(bnb, assess, top, w, angle_triplet, expStar, vectorshift, nExp, apply_shifts):
    """
    top: lista de K tensores (nExp, 5) [exp, ref, score, rot, shift_idx]
    w:   (nExp, K) pesos blandos
    Devuelve entradas (imagen, orientacion): img_idx, rotM, shiftM, w_ent,
    ademas de los indices validos y los matches del rango 0.
    """
    vi, rm, sm, we = [], [], [], []
    v0, m0 = None, None

    for r, t in enumerate(top):
        m = bnb.filter_scores_mad(bnb.match_batch_label_minScore(t))
        v, rM, sM = assess.estimatePose(
            angle_triplet, expStar, m, vectorshift, nExp, apply_shifts,
            filter_matches=False, filter_score=False
        )
        if r == 0:
            v0, m0 = v, m
        else:
            # rangos > 0 solo para averages cuyo rango 0 es valido
            keep = torch.isin(v, v0.to(v.device))
            v = v[keep]
            rM = rM[keep.to(rM.device)]
            if sM is not None:
                sM = sM[keep.to(sM.device)]

        vi.append(v)
        rm.append(rM)
        sm.append(sM)
        we.append(w[v.to(w.device).long(), r])

    img_idx = torch.cat([x.to(v0.device) for x in vi])
    rotM = torch.cat([x.to(rm[0].device) for x in rm])
    shiftM = None
    if sm[0] is not None:
        shiftM = torch.cat([x.to(sm[0].device) for x in sm])
    w_ent = torch.cat(we)

    return img_idx, rotM, shiftM, w_ent, v0, m0


def select_subset(img_idx, rotM, shiftM, w_ent, grupo):
    """Se queda solo con las entradas cuyas imagenes pertenecen a `grupo`."""
    sel = np.flatnonzero(np.isin(img_idx.detach().cpu().numpy(), grupo))
    sel = torch.as_tensor(sel, dtype=torch.long)

    img_idx = img_idx[sel.to(img_idx.device)]
    rotM = rotM[sel.to(rotM.device)]
    if shiftM is not None:
        shiftM = shiftM[sel.to(shiftM.device)]
    if w_ent is not None:
        w_ent = w_ent[sel.to(w_ent.device)]
    return img_idx, rotM, shiftM, w_ent


# ----------------------------------------------------------------------
# Ranking de clases (successive halving)
# ----------------------------------------------------------------------
def class_ranking(heldout_hist, active, it, window=2):
    """
    Score held-out medio de las ultimas `window` iteraciones (una con grupo A y otra con B).
    Menor = mejor. Devuelve (clases ordenadas de mejor a peor, dict clase -> score).
    """
    vals = {}
    for i in active:
        h = [heldout_hist[i][t] for t in range(it - window + 1, it + 1) if t in heldout_hist[i]]
        vals[i] = float(np.mean(h)) if h else float("inf")
    order = sorted(active, key=lambda c: vals[c])
    return order, vals


def write_scores_csv(path, rows):
    with open(path, "w") as f:
        f.write("iter,class,mean_score,heldout_score\n")
        for it, c, s, h in rows:
            f.write("%d,%d,%.6f,%s\n" % (it, c, s, "nan" if h is None else "%.6f" % h))


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="align images")
    parser.add_argument("-i", "--exp", help="input mrc file for experimental images", required=True)
    parser.add_argument("-r", "--refVol", help="input reference volume")
    parser.add_argument("-s", "--sampling", type=float, help="pixel size of the images", required=True)
    parser.add_argument("-sym", "--symmetry", type=str, default="C1", help="Symmetry group")
    parser.add_argument("-o", "--output", help="Root directory for the output files", required=True)
    parser.add_argument("-stExp", "--sartExp", help="star file for experimental images", required=True)
    parser.add_argument("-radius", type=int, help="radius for circular mask (in pixels)")
    parser.add_argument("-vr", "--volRes", type=float, default=8.0, help="Final volume resolution (A)")
    parser.add_argument("--apply_shifts", action="store_true", help="Apply starfile shifts to experimental images")
    parser.add_argument("--posit", action="store_true", help="Apply relu function")
    parser.add_argument("-nCl", "--numCl", type=int, default=1, help="number of classes for initial model")
    parser.add_argument("--save_class", action="store_true", help="(sin implementar) Save the corresponding class in output xmd")
    parser.add_argument("--contrast_mask", action="store_true", help="Apply a mask based on local contrast to the images and to the reconstructed volume")
    # For training
    parser.add_argument("-t", "--training", help="number of image for training (no usado: se usan todos)", required=True)
    parser.add_argument("-hr", "--highres", help="highest resolution to consider", required=True)
    parser.add_argument("-p", "--perc", help="PCA percentage (between 0-1)", required=True)
    # Reproducibilidad
    parser.add_argument("--seed", type=int, default=None, help="semilla (angulos aleatorios, reparto A/B, jitter)")
    # Deterministic annealing
    parser.add_argument("--da", action="store_true", help="activa el deterministic annealing en las primeras iteraciones (por defecto: comportamiento original)")
    parser.add_argument("--da_iters", type=int, default=7, help="iteraciones con DA (<= 7); solo con --da")
    parser.add_argument("--K", type=int, default=5, help="candidatos por average (fijo durante el DA)")
    parser.add_argument("--neff0", type=float, default=3.0, help="n. efectivo de candidatos al inicio del DA")
    parser.add_argument("--neff1", type=float, default=1.2, help="n. efectivo de candidatos al final del DA")
    # Penalizacion de amplitud (0 = solo coseno)
    parser.add_argument("--amp_lambda", type=float, default=0.0, help="peso de la penalizacion de amplitud en el DA")
    # Successive halving de clases
    parser.add_argument("--halve_iters", type=str, default="none",
                        help="iteraciones (separadas por coma) tras las que se descarta la peor parte de las clases; 'none' = nunca")
    parser.add_argument("--keep_frac", type=float, default=0.5, help="fraccion de clases que se conservan en cada descarte")

    args = parser.parse_args()

    expFile = args.exp
    initVol = args.refVol
    sampling = args.sampling
    sym = args.symmetry
    amax = 180.0
    output = args.output
    expStar = args.sartExp
    radius = args.radius
    volRes = args.volRes
    posit = args.posit
    apply_shifts = args.apply_shifts
    contrast_mask = args.contrast_mask
    numCl = args.numCl
    # PCA
    highRes = float(args.highres)
    per_eig_value = float(args.perc)
    # DA
    da_iters = min(args.da_iters, AB_ITERS) if args.da else 0
    if not args.da and args.amp_lambda > 0:
        print("AVISO: --amp_lambda solo tiene efecto con --da; se ignora.", flush=True)

    K = args.K
    neff0, neff1 = args.neff0, args.neff1
    amp_lambda = args.amp_lambda
    # Halving
    if args.halve_iters.strip().lower() in ("", "none"):
        halve_iters = []
    else:
        halve_iters = [int(x) for x in args.halve_iters.split(",") if x.strip()]
    keep_frac = args.keep_frac

    # Reproducibilidad
    seed = args.seed if args.seed is not None else int.from_bytes(os.urandom(4), "little")
    print(f"seed = {seed}", flush=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    cuda = torch.device('cuda')

    if volRes <= 1 / (2 * sampling):
        volRes = 1 / (2 * sampling) + 0.5

    # Read Experimental Images
    mmap = mrcfile.mmap(expFile, permissive=True)
    nExp = mmap.data.shape[0]
    dim = mmap.data.shape[1]
    nIter = 20

    positivity = bool(posit)

    # Create initial references
    R = reconstruct()
    angular_step = 15
    transf, angle_triplet = R.generate_library_sym(angular_step, sym=sym)

    all_refs_cpu = [None] * numCl

    # Reading experimental images (solo una vez)
    expImages = read_images(expFile)
    texp = torch.from_numpy(expImages).to(cuda)
    del expImages

    if initVol:
        for i in range(numCl):
            zeroVol = load_vol(initVol)
            zeroVol = R.mask_otsu(zeroVol, sigma=4.0, noise_level=0.0)
            if posit:
                zeroVol = torch.relu(zeroVol)
            ref = R.generate_projections(zeroVol, transf)
            all_refs_cpu[i] = ref.detach().cpu()
            del zeroVol, ref
    else:
        for i in range(numCl):
            random_angles = R.generate_random_angles(nExp)
            zeroVol = R.reconstruct_volume_weight(mmap, sym, 60, sampling, dim, random_angles, posit=positivity)
            file = output + "-iter%s_zerovol.mrc" % (i)
            save_vol(zeroVol.cpu(), file, sampling)
            zeroVol = R.filter_3d(zeroVol, sampling, 60.0)
            zeroVol = zeroVol * R.contrast_dominant_mask_3d(zeroVol, window=3, contrast_percentile=90,
                                                            intensity_percentile=70, smooth_sigma=1.0)
            zeroVol = R.apply_spherical_mask(zeroVol, radius)

            if posit:
                zeroVol = torch.relu(zeroVol)
            ref = R.generate_projections(zeroVol, transf)
            all_refs_cpu[i] = ref.detach().cpu()

            del zeroVol, ref

    nBand = 1
    bnb = BnBgpu(nBand)
    assess = evaluation()

    texp = bnb.zscore_normalization(texp)
    texp *= bnb.create_gaussian_mask(texp, 42)

    if radius:
        texp *= bnb.create_mask(texp, radius)
    # posit
    if posit:
        texp = torch.relu(texp)

    whitening = 1

    Ntrain = texp.shape[0]

    # pca
    pca = PCAgpu(nBand)
    maxRes = 60
    freqBn, cvecs, coef = pca.calculatePCAbasis(texp, Ntrain, nBand, dim, sampling, maxRes,
                                                minRes=530, per_eig=per_eig_value, batchPCA=True)

    grid_flat = flatGrid(freqBn, nBand)

    next_angle_triplet = None

    # Reparto A/B COMPARTIDO por todas las clases (el held-out es comparable entre clases)
    ab_split = None

    # Clases activas (successive halving) y registros
    active = list(range(numCl))
    heldout_hist = [dict() for _ in range(numCl)]
    log_rows = []

    for current_iter in range(nIter):
        print("----------Iter %s------------" % current_iter, flush=True)

        pcaRes, filtRes, angular_step = bnb.reconstruct_parameters(current_iter, highRes, volRes)
        ang, shiftMove, maxshift = bnb.search_space(current_iter, sampling)
        angSet = (-amax, amax, ang)
        shiftSet = (-maxshift, maxshift + shiftMove, shiftMove)
        vectorRot, vectorshift = bnb.setRotAndShift(angSet, shiftSet)
        vectorRot.sort()
        nShift = len(vectorshift)

        use_da = current_iter < da_iters
        use_ab = current_iter < AB_ITERS

        if current_iter < 7 and contrast_mask:
            texp_align = texp * bnb.contrast_dominant_mask(texp, window=3, contrast_percentile=80,
                                                           intensity_percentile=50, smooth_sigma=1.0)
        else:
            texp_align = texp

        matches = [None] * numCl

        for i in active:
            # Reading references particles
            tref = all_refs_cpu[i].to(cuda)

            tref = bnb.zscore_normalization(tref)
            tref *= bnb.create_gaussian_mask(tref, 42)

            if radius:
                tref = tref * bnb.create_mask(tref, radius)

            batch_projRef = bnb.create_batchExp(tref, whitening, freqBn, coef, cvecs)

            w_ent = None

            if use_da:
                # ------------------------------------------------------
                # DETERMINISTIC ANNEALING: top-K candidatos con pesos
                # ------------------------------------------------------
                _, ref_norms = bnb.normalize_batch(batch_projRef)      # una sola vez

                amp_scale = 1.0
                if amp_lambda > 0:
                    # escala global entre amplitudes de averages y de referencias
                    proj0 = bnb.precalculate_projection(texp_align, whitening, freqBn, grid_flat,
                                                        coef, cvecs, 0, vectorshift[:1])
                    amp_scale = (torch.linalg.norm(proj0[0], dim=1).median() / ref_norms[0].median()).item()
                    del proj0

                nRef = batch_projRef[0].shape[0]
                best = torch.full((nExp, nRef), float("inf"), device=cuda)
                best_rot = torch.zeros((nExp, nRef), device=cuda)
                best_shift = torch.zeros((nExp, nRef), device=cuda)

                for rot in vectorRot:
                    batch_projExp = bnb.precalculate_projection(texp_align, whitening, freqBn, grid_flat,
                                                                coef, cvecs, -rot, vectorshift)
                    best, best_rot, best_shift = bnb.match_batch_best_per_ref(
                        batch_projExp, batch_projRef, best, best_rot, best_shift, -rot, nShift,
                        ref_norms=ref_norms, amp_lambda=amp_lambda, amp_scale=amp_scale)
                    del batch_projExp

                n_eff = da_neff(current_iter, neff0, neff1, da_iters)
                top = bnb.topk_matches(best, best_rot, best_shift, k=K)
                w, T = bnb.soft_weights_neff(top, n_eff)                # (nExp, K)
                # print("Pesos de los 5 candidatos:")
                # print(w[:10].detach().cpu().numpy())
                # print("Temperatura:", T)
                del best, best_rot, best_shift

                ent = -(w * torch.log(w + 1e-12)).sum(dim=1)
                print(f"DA: n_eff objetivo = {n_eff:.2f}, T = {T:.3f}, K = {len(top)}, "
                      f"candidatos efectivos = {ent.exp().mean().item():.2f}", flush=True)

                img_idx, rotM, shiftM, w_ent, valid_indices, matches[i] = build_da_entries(
                    bnb, assess, top, w, angle_triplet, expStar, vectorshift, nExp, apply_shifts)
                del top, w

            else:
                # ------------------------------------------------------
                # Camino original: un unico mejor match por average
                # ------------------------------------------------------
                matches[i] = torch.full((nExp, 5), float("Inf"), device=cuda)

                for rot in vectorRot:
                    batch_projExp = bnb.precalculate_projection(texp_align, whitening, freqBn, grid_flat,
                                                                coef, cvecs, -rot, vectorshift)
                    matches[i] = bnb.match_batch_initVol(batch_projExp, batch_projRef, 0, matches[i], -rot, nShift)
                    del batch_projExp

                matches[i] = bnb.match_batch_label_minScore(matches[i])
                matches[i] = bnb.filter_scores_mad(matches[i])

                valid_indices, rotM, shiftM = assess.estimatePose(
                    angle_triplet, expStar, matches[i], vectorshift,
                    nExp, apply_shifts,
                    filter_matches=(7 < current_iter < 16), filter_score=(7 < current_iter < 16)
                )
                img_idx = valid_indices

            score = matches[i][valid_indices, 2].mean()
            print(f"class {i}: num averages = {len(valid_indices)}, mean score = {score.item()}", flush=True)

            held_val = None

            # ----------------------------------------------------------
            # Grupos A/B (leave-out): reconstruyo con la mitad
            # ----------------------------------------------------------
            if use_ab:
                if ab_split is None:
                    perm = np.random.default_rng(seed).permutation(nExp)
                    ab_split = (perm[:nExp // 2], perm[nExp // 2:])

                grupo = ab_split[0] if current_iter % 2 == 0 else ab_split[1]

                # El grupo de esta iteracion no estaba en la referencia (se construyo con el otro grupo):
                # su score es un score held-out (mismo conjunto de averages para todas las clases).
                if current_iter > 0:
                    in_group = torch.as_tensor(np.isin(valid_indices.detach().cpu().numpy(), grupo),
                                               device=valid_indices.device)
                    if in_group.any():
                        held_val = matches[i][valid_indices[in_group], 2].mean().item()
                        heldout_hist[i][current_iter] = held_val
                        print(f"class {i}: held-out score = {held_val:.5f}", flush=True)

                label = "A" if current_iter % 2 == 0 else "B"
                img_idx, rotM, shiftM, w_ent = select_subset(img_idx, rotM, shiftM, w_ent, grupo)
                print(f"-> [Iter {current_iter}] class {i}, grupo {label}: {len(grupo)} averages, "
                      f"{len(img_idx)} entradas de reconstruccion", flush=True)

            log_rows.append((current_iter, i, score.item(), held_val))

            vol = R.reconstruct_volume_weight(mmap, sym, filtRes, sampling, dim, rotM, shifts=shiftM,
                                           posit=positivity, img_idx=img_idx, weights=w_ent)

            vol = R.filter_3d(vol, sampling, filtRes)

            if current_iter < 7 and contrast_mask:
                vol = vol * R.contrast_dominant_mask_3d(vol, window=3, contrast_percentile=90,
                                                        intensity_percentile=70, smooth_sigma=1.0)

            # posit
            if posit:
                vol = torch.relu(vol)

            vol = R.apply_spherical_mask(vol, radius)

            file = output + "_iter%s_class%s.mrc" % (current_iter + 1, i)
            save_vol(vol.cpu(), file, sampling)

            # La galeria es compartida: la regenera la primera clase activa
            # if i == active[0] and current_iter in (2, 4, 6, 8, 10, 12, 14):
            if i == active[0] and current_iter in (3, 5, 8, 13, 16):
                print(f"ANGULAR STEP = {angular_step}")
                transf, next_angle_triplet = R.generate_library_sym(angular_step, sym=sym)

            ref = R.generate_projections(vol, transf)
            all_refs_cpu[i] = ref.detach().cpu()
            del tref, vol, ref

        # --------------------------------------------------------------
        # Ranking de clases y successive halving
        # --------------------------------------------------------------
        end_of_da = (da_iters > 0 and current_iter == da_iters - 1)
        if current_iter >= 1 and (current_iter in halve_iters or end_of_da) and len(active) > 1:
            order, vals = class_ranking(heldout_hist, active, current_iter)
            print("Ranking de clases por held-out (media de las ultimas 2 iteraciones, menor = mejor):")
            for pos, c in enumerate(order):
                print(f"   {pos + 1:2d}. class {c}: {vals[c]:.5f}")

            if current_iter in halve_iters and np.isfinite([vals[c] for c in order]).any():
                n_keep = max(1, int(np.ceil(len(active) * keep_frac)))
                kept = sorted(order[:n_keep])
                dropped = [c for c in active if c not in kept]
                for c in dropped:
                    all_refs_cpu[c] = None
                print(f"   -> se conservan las clases {kept}; se descartan {dropped}", flush=True)
                active = kept

        write_scores_csv(output + "_scores.csv", log_rows)

        if next_angle_triplet is not None:
            angle_triplet = next_angle_triplet
            next_angle_triplet = None
            # Actualizo PCA
            # if current_iter not in (2, 4):
            if current_iter not in (3, 5):
            # if current_iter != 3:
                del freqBn, coef, grid_flat, cvecs
                Ntrain = texp.shape[0]
                freqBn, cvecs, coef = pca.calculatePCAbasis(
                    texp, Ntrain, nBand, dim, sampling, pcaRes,
                    minRes=530, per_eig=per_eig_value, batchPCA=True
                )
                grid_flat = flatGrid(freqBn, nBand)



