#!/usr/bin/env python3

"""/***************************************************************************
 *
 * Authors:    Carlos Oscar Sorzano coss@cnb.csic.es
 *
* CSIC
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
 * 02111-1307  USA
 *
 *  All comments concerning this program package may be sent to the
 *  e-mail address 'xmipp@cnb.csic.es'
  ***************************************************************************/
"""

import jax
import jax.numpy as jnp
from jax import lax
from jax.scipy.ndimage import map_coordinates
from flax import linen as nn
from flax import serialization
import optax
from flax.training import train_state
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import math
import numpy as np
from scipy.ndimage import zoom
from sklearn.cluster import KMeans
import os
import sys
import time
import xmippLib
from xmipp_script import XmippScript

DT_IMAGE = jnp.bfloat16

def precompute_original_grid(H: int, W: int):
    ys = jnp.arange(H, dtype=jnp.float32)
    xs = jnp.arange(W, dtype=jnp.float32)
    grid_y, grid_x = jnp.meshgrid(ys, xs, indexing="ij")  # [H,W]

    cy = (H - 1) * 0.5
    cx = (W - 1) * 0.5

    # Centered output coordinates (x first, then y): [H,W,2]
    coords = jnp.stack([grid_x - cx, grid_y - cy], axis=-1)  # [H,W,2]
    return coords, cx, cy


def make_centered_warper_multichan(coords: jnp.ndarray, cx: float, cy: float):
    """
    Centered inverse-affine warper for multi-channel images.

    Inputs
    ------
    x: [B, H, W, K]
    M: [B, 2, 2]    (rotation / linear transform; assumed orthonormal if you use M.T as inverse)
    t: [B, 2]       (shift in centered coordinates, in pixels)

    Returns
    -------
    warped: [B, H, W, K]
    """
    # sample a single channel image [H,W]
    def _sample_2d(img2d, sx, sy):
        c = jnp.stack([sy, sx], axis=0)          # [2, H, W] (y,x)
        return map_coordinates(img2d, c, order=1, mode="nearest")  # [H,W]

    # warp one image with K channels: img [H,W,K]
    def _warp_single(img, M, t):
        t = t.astype(coords.dtype)
        Minv = M.T  # for pure rotation matrices; if you later allow scaling/shear, use jnp.linalg.inv(M)

        # inverse map in centered coords: src_c = (coords - t) @ Minv^T
        src_c = (coords - t[None, None, :]) @ Minv.T
        sx = src_c[..., 0] + cx
        sy = src_c[..., 1] + cy

        # Apply to each channel independently (K times) keeping sx,sy fixed
        # img: [H,W,K] -> [K,H,W]
        img_khw = jnp.moveaxis(img, -1, 0)

        warped_khw = jax.vmap(_sample_2d, in_axes=(0, None, None))(img_khw, sx, sy)  # [K,H,W]
        warped_hwk = jnp.moveaxis(warped_khw, 0, -1)  # [H,W,K]
        return warped_hwk

    # vmap over batch
    return jax.vmap(_warp_single, in_axes=(0, 0, 0))


def _load_parallel(fnImgs):
    with ThreadPoolExecutor() as executor:
        imgs = list(
            executor.map(lambda f: np.asarray(xmippLib.Image(f).getData()),
                         fnImgs))
    return imgs

def _per_image_standardize_np(im, eps=1e-6):
    m = float(im.mean())
    s = float(im.std())
    return (im - m) / (s + eps)


def _gaussian_mask_2d(H, W, sigma):
    cy = (H - 1) * 0.5
    cx = (W - 1) * 0.5
    yy, xx = np.ogrid[:H, :W]
    r2 = (yy - cy) ** 2 + (xx - cx) ** 2
    return np.exp(-0.5 * r2 / (sigma * sigma)).astype(np.float32)


def _radial_freq_grid(H, W):
    # cycles/pixel in [-0.5, 0.5)
    fy = np.fft.fftfreq(H, d=1.0)
    fx = np.fft.fftfreq(W, d=1.0)
    FY, FX = np.meshgrid(fy, fx, indexing="ij")
    return np.sqrt(FX * FX + FY * FY).astype(np.float32)  # [H,W]


def _make_filter_bank_masks(H, W, K=5, fmin=0.0, fmax=0.5):
    R = _radial_freq_grid(H, W)  # [H,W]
    edges = np.linspace(fmin, fmax, K + 1, dtype=np.float32)
    masks = []
    for k in range(K):
        lo, hi = float(edges[k]), float(edges[k + 1])
        if k == K - 1:
            m = (R >= lo) & (R <= hi)  # include Nyquist edge in last bin
        else:
            m = (R >= lo) & (R < hi)
        masks.append(m.astype(np.float32))
    # [K,H,W]
    return np.stack(masks, axis=0), edges

def preload_resize_normalize(fnImgs,
                             out_hw,
                             dtype=np.float32,
                             K=5):
    """
    Returns:
      Y: [N, Hout, Wout, K]  (K band-filtered images, Gaussian-masked in real space)
    """
    Hout = Wout = int(out_hw)
    imgs_np = _load_parallel(fnImgs)

    N = len(imgs_np)
    if N == 0:
        raise ValueError("No images provided")

    # infer input size from first image
    Hin, Win = imgs_np[0].shape
    zoom_y = Hout / Hin
    zoom_x = Wout / Win

    # Precompute masks (shared across all images)
    sigma = Hout / 6.0
    gmask = _gaussian_mask_2d(Hout, Wout, sigma=sigma)          # [H,W]
    fb_masks, fb_edges = _make_filter_bank_masks(Hout, Wout, K=K, fmin=0.0, fmax=0.5)  # [K,H,W]

    # Output: [N,H,W,K]
    Y = np.empty((N, Hout, Wout, K), dtype=dtype)

    for i, im in enumerate(imgs_np):
        im = _per_image_standardize_np(im)  # float32
        im_resized = zoom(im, (zoom_y, zoom_x), order=1).astype(np.float32)  # [H,W]

        F = np.fft.fft2(im_resized)  # complex64/128 depending on numpy build

        # Apply each band mask, iFFT back, then apply real-space Gaussian mask
        for k in range(K):
            Fb = F * fb_masks[k]                  # band-limited spectrum
            band = np.fft.ifft2(Fb).real          # [H,W] float
            band = band * gmask                   # real-space Gaussian mask
            Y[i, :, :, k] = band.astype(dtype)

    return Y

def get_xmipp_preloaded_array(fnImgs, XdimOut, K=5):
    pre = preload_resize_normalize(fnImgs, XdimOut, K=K, dtype=np.float32)
    pre = jax.device_put(pre)  # explicit device transfer once
    return pre.astype(DT_IMAGE)

@jax.jit
def train_step(state, batch, margin=0.2):
    def _loss_only(p):
        return triplet_loss(p, state.apply_fn, batch, margin=margin)
    (loss, metrics), grads = jax.value_and_grad(_loss_only, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, metrics

@partial(jax.jit, static_argnums=(0,))
def run_epoch(step_fn, state, key, steps_per_epoch, margin):
    def body(i, carry):
        state, key, loss_sum, dap_sum, dan_sum, viol_sum = carry
        state, key, metrics = step_fn(state, key, margin)
        loss_sum = loss_sum + metrics["loss"]
        dap_sum  = dap_sum  + metrics["d_ap"]
        dan_sum  = dan_sum  + metrics["d_an"]
        viol_sum = viol_sum + metrics["viol"]
        return (state, key, loss_sum, dap_sum, dan_sum, viol_sum)

    carry0 = (
        state,
        key,
        jnp.array(0.0, jnp.float32),
        jnp.array(0.0, jnp.float32),
        jnp.array(0.0, jnp.float32),
        jnp.array(0.0, jnp.float32),
    )
    return lax.fori_loop(0, steps_per_epoch, body, carry0)

def make_batcher(pre_dev, warper, batch_size, sigma_shift,
                 a_range=(0.7,1.4), b_range=(-0.2,0.2)):

    N = pre_dev.shape[0]

    def sample_T(k, B):
        k, k_th, k_sh, k_a, k_b = jax.random.split(k, 5)
        theta = jax.random.uniform(k_th, (B,), dtype=jnp.float32,
                                   minval=-jnp.pi, maxval=jnp.pi)
        cos, sin = jnp.cos(theta), jnp.sin(theta)
        M = jnp.stack([jnp.stack([cos, -sin], axis=-1),
                       jnp.stack([sin,  cos], axis=-1)], axis=-2)  # [B,2,2]
        t = jax.random.normal(k_sh, (B, 2), dtype=jnp.float32) * sigma_shift
        a = jax.random.uniform(k_a, (B,), dtype=DT_IMAGE,
                               minval=a_range[0], maxval=a_range[1])
        b = jax.random.uniform(k_b, (B,), dtype=DT_IMAGE,
                               minval=b_range[0], maxval=b_range[1])
        return k, M, t, a, b

    @jax.jit
    def next_triplet(key):
        key, k_idx, kA, kP, kN = jax.random.split(key, 5)
        idx = jax.random.randint(k_idx, (2 * batch_size,), 0, N)

        base = pre_dev[idx]              # [2B,H,W] bf16 on device
        base_ap = base[:batch_size]
        base_n  = base[batch_size:]

        kA, MA, tA, aA, bA = sample_T(kA, batch_size)
        kP, MP, tP, aP, bP = sample_T(kP, batch_size)
        kN, MN, tN, aN, bN = sample_T(kN, batch_size)

        xa = aA[:, None, None, None] * warper(base_ap, MA, tA) + \
             bA[:, None, None, None]
        xp = aP[:, None, None, None] * warper(base_ap, MP, tP) + \
             bP[:, None, None, None]
        xn = aN[:, None, None, None] * warper(base_n, MN, tN) + \
             bN[:, None, None, None]

        xa = xa.astype(DT_IMAGE); xp = xp.astype(DT_IMAGE); xn = xn.astype(DT_IMAGE)
        return key, (xa, xp, xn)

    @jax.jit
    def step(state, key, margin):
        key, batch = next_triplet(key)
        state, metrics = train_step(state, batch, margin=margin)
        return state, key, metrics

    return next_triplet, step

class TripletNet(nn.Module):
    d: int = 128  # embedding dimension
    compute_dtype: any = jnp.bfloat16
    param_dtype: any = jnp.float32

    @nn.compact
    def __call__(self, x):  # x: [B,H,W,K]
        x = x.astype(self.compute_dtype)

        x = nn.Conv(32, (5,5), (2,2), dtype=self.compute_dtype,
                    param_dtype=self.param_dtype)(x); x = nn.relu(x)
        x = nn.Conv(64, (5,5), (2,2), dtype=self.compute_dtype,
                    param_dtype=self.param_dtype)(x); x = nn.relu(x)
        x = nn.Conv(64, (5,5), (2,2), dtype=self.compute_dtype,
                    param_dtype=self.param_dtype)(x); x = nn.relu(x)
        x = nn.Conv(128, (5,5), (1,1), dtype=self.compute_dtype,
                    param_dtype=self.param_dtype)(x); x = nn.relu(x)

        x = jnp.mean(x, axis=(1, 2))  # [B,C]
        x = nn.Dense(256, dtype=self.compute_dtype,
                     param_dtype=self.param_dtype)(x);
        x = nn.relu(x)
        z = nn.Dense(self.d, dtype=self.compute_dtype,
                     param_dtype=self.param_dtype)(x)

        # L2-normalize for cosine distance (unit sphere)
        z = z.astype(self.compute_dtype)  # normalize in fp32
        z = z / (jnp.linalg.norm(z, axis=-1, keepdims=True) + 1e-9)
        return z

def triplet_loss(params, apply_fn, batch, margin=0.2):
    xa, xp, xn = batch  # [B,d,d] each
    ea = apply_fn({'params': params}, xa)  # [B,d], unit norm
    ep = apply_fn({'params': params}, xp)
    en = apply_fn({'params': params}, xn)

    # cosine distances: d = 1 - dot(u,v)
    dap = 1.0 - jnp.sum(ea * ep, axis=1)   # [B]
    dan = 1.0 - jnp.sum(ea * en, axis=1)   # [B]

    # classic margin triplet: max(0, d_ap - d_an + m)
    losses = jnp.maximum(0.0, dap - dan + margin)
    loss = jnp.mean(losses)

    # helpful metrics
    frac_viol = jnp.mean((dap + margin > dan).astype(jnp.float32))
    metrics = {
        "loss": loss,
        "d_ap": jnp.mean(dap),
        "d_an": jnp.mean(dan),
        "viol": frac_viol,  # fraction violating the margin
    }
    return loss, metrics

def create_train_state(rng, model, XdimOut, K=5, lr=3e-4):
    dummy = jnp.zeros((2,XdimOut,XdimOut, K), dtype=jnp.float32)
    variables = model.init(rng, dummy)
    params = variables["params"]
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(lr),
    )
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

def embed_batch(state, x):
    return state.apply_fn({'params': state.params}, x)

def sample_embeddings(state, next_triplet, key, embedding_points, batch_size):
    """
    Generate synthetic examples with the trained generator, embed them,
    and return a NumPy array [embedding_points, embedding_dim].
    """
    out = []
    n_done = 0

    while n_done < embedding_points:
        key, batch = next_triplet(key)
        xa, xp, xn = batch

        # use anchors only
        emb = embed_batch(state, xa)
        emb = np.asarray(jax.device_get(emb), dtype=np.float32)

        remaining = embedding_points - n_done
        if emb.shape[0] > remaining:
            emb = emb[:remaining]

        out.append(emb)
        n_done += emb.shape[0]

        if n_done % max(batch_size, 1000) == 0 or n_done == embedding_points:
            print(f"[embed] collected {n_done}/{embedding_points}")

    return np.concatenate(out, axis=0)

class ScriptDeepEmbedTrain(XmippScript):
    def __init__(self):
        XmippScript.__init__(self)

    def defineParams(self):
        self.addUsageLine('Train a deep center model')
        ## params
        self.addParamsLine(' -i <metadata>                : xmd file with the list of images')
        self.addParamsLine(' --omodel <fnModel>           : Model filename')
        self.addParamsLine('[--batchSize <N=8>]           : Batch size')
        self.addParamsLine('[--imgSize <Xdim=64>]         : Training image size')
        self.addParamsLine('[--gpu <id=0>]                : GPU Id')
        self.addParamsLine('[--learningRate <lr=0.0001>]  : Learning rate')
        self.addParamsLine('[--maxEpochs <N=100>]         : Max. Epochs')
        self.addParamsLine('[--sigmaShift <s=10>]         : Std.Dev. of the simulated shifts')
        self.addParamsLine('[--embeddingDim <s=128>]      : Embedding dimension')
        self.addParamsLine('[--embeddingPoints <N=100000>]: Embedding points')
        self.addParamsLine('[--embeddingK <K=100>]        : Embedding clusters')
        self.addParamsLine('--ocentroids <fn>             : Output file for centroids (.npy or .npz)')

    def run(self):
        fnXmd = self.getParam("-i")
        fnModel = self.getParam("--omodel")
        maxEpochs = int(self.getParam("--maxEpochs"))
        batch_size = int(self.getParam("--batchSize"))
        XdimOut = int(self.getParam("--imgSize"))
        gpuId = self.getParam("--gpu")
        learning_rate = float(self.getParam("--learningRate"))
        sigma_shift = float(self.getParam("--sigmaShift"))
        embeddingDim = int(self.getParam("--embeddingDim"))
        embeddingPoints = int(self.getParam("--embeddingPoints"))
        embeddingK = int(self.getParam("--embeddingK"))
        fnCentroids = self.getParam("--ocentroids")

        if not gpuId.startswith('-1'):
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = gpuId
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
        os.environ['TF_GPU_ALLOCATOR']='cuda_malloc_async'

        Xdim, _, _, _, _ = xmippLib.MetaDataInfo(fnXmd)
        mdExp = xmippLib.MetaData(fnXmd)
        fnImgs = mdExp.getColumnValues(xmippLib.MDL_IMAGE)

        model = TripletNet(d=embeddingDim)
        state = create_train_state(jax.random.PRNGKey(0), model, XdimOut,
                                   5, learning_rate)

        pre_dev = get_xmipp_preloaded_array(fnImgs, XdimOut)  # GPU once
        coords, cx, cy = precompute_original_grid(XdimOut, XdimOut)
        warper = make_centered_warper_multichan(coords, cx, cy)

        key = jax.random.PRNGKey(42)
        next_triplet, step = make_batcher(pre_dev, warper, batch_size,
                                          sigma_shift)
        margin = 0.2

        # Warmup (optional): trigger compilation before timing
        key, batch = next_triplet(key)
        state, _ = train_step(state, batch, margin=margin)
        jax.block_until_ready(state.params)

        N = int(pre_dev.shape[0])  # number of images
        steps_per_epoch = math.ceil(
            N / batch_size)  # define one epoch as ~one pass worth of batches

        t0 = time.time()
        for epoch in range(1, maxEpochs + 1):
            state, key, loss_sum, dap_sum, dan_sum, viol_sum = run_epoch(
                step, state, key, steps_per_epoch, margin)

            loss_avg = float(loss_sum / steps_per_epoch)
            dap_avg = float(dap_sum / steps_per_epoch)
            dan_avg = float(dan_sum / steps_per_epoch)
            viol_avg = float(viol_sum / steps_per_epoch)

            dt = time.time() - t0
            print(f"[epoch {epoch:4d}/{maxEpochs}] loss={loss_avg:.8f}  "
                  f"d_ap={dap_avg:.8f}  d_an={dan_avg:.8f}  "
                  f"viol%={100 * viol_avg:.1f}  ({dt:.1f}s/epoch)")
            t0 = time.time()

        with open(fnModel, "wb") as f:
            f.write(serialization.to_bytes(state))

        print("Generating synthetic embeddings for centroid estimation...")
        emb_key = jax.random.PRNGKey(12345)
        Xemb = sample_embeddings(state, next_triplet, emb_key,
                                 embeddingPoints, batch_size)

        print(f"Running K-means with K={embeddingK} on {Xemb.shape[0]} "
              f"points...")
        kmeans = KMeans(
            n_clusters=embeddingK,
            init="k-means++",
            n_init=10,
            max_iter=100,
            random_state=0,
        )
        kmeans.fit(Xemb)
        centroids = kmeans.cluster_centers_

        # Save only centroids as .npy
        np.save(fnCentroids, centroids)

if __name__ == '__main__':
    exitCode = ScriptDeepEmbedTrain().tryRun()
    sys.exit(exitCode)
