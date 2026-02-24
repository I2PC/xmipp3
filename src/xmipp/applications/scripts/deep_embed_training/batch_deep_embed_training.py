#!/usr/bin/env python3

import jax
import jax.numpy as jnp
import numpy as np
import os
import sys
import xmippLib
from xmipp_script import XmippScript

def bilinear_sample_2d(img: jnp.ndarray, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """
    Bilinear sample a single 2D image.

    Parameters
    ----------
    img : [H,W]
    x,y : [Hout,Wout] float coordinates in source image space

    Returns
    -------
    out : [Hout,Wout]
    """
    H, W = img.shape

    # Keep inside valid range; -1.0001 trick avoids x1==W at the right border
    x = jnp.clip(x, 0.0, W - 1.0001)
    y = jnp.clip(y, 0.0, H - 1.0001)

    x0 = jnp.floor(x).astype(jnp.int32)
    y0 = jnp.floor(y).astype(jnp.int32)
    x1 = jnp.clip(x0 + 1, 0, W - 1)
    y1 = jnp.clip(y0 + 1, 0, H - 1)

    wx = x - x0.astype(jnp.float32)
    wy = y - y0.astype(jnp.float32)

    # gather with linear indices (fast + vmap/jit friendly)
    flat = img.reshape(-1)

    def gather(ix, iy):
        idx = (iy * W + ix).reshape(-1)
        return jnp.take(flat, idx).reshape(x.shape)

    I00 = gather(x0, y0)
    I10 = gather(x1, y0)
    I01 = gather(x0, y1)
    I11 = gather(x1, y1)

    return ((1 - wx) * (1 - wy) * I00 +
            wx * (1 - wy) * I10 +
            (1 - wx) * wy * I01 +
            wx * wy * I11)

def precompute_original_grid(H: int, W: int):
    ys = jnp.arange(H, dtype=jnp.float32)
    xs = jnp.arange(W, dtype=jnp.float32)
    grid_y, grid_x = jnp.meshgrid(ys, xs, indexing="ij")  # [H,W]

    cy = (H - 1) * 0.5
    cx = (W - 1) * 0.5

    # Centered output coordinates (x first, then y): [H,W,2]
    coords = jnp.stack([grid_x - cx, grid_y - cy], axis=-1)  # [H,W,2]
    return coords, cx, cy

def precompute_resize_grid(Hout: int, Wout: int,
                           Hin: int, Win: int,
                           dtype=jnp.float32):
    ys = jnp.linspace(0.0, Hin - 1.0, Hout, dtype=dtype)
    xs = jnp.linspace(0.0, Win - 1.0, Wout, dtype=dtype)
    grid_y, grid_x = jnp.meshgrid(ys, xs, indexing="ij")
    coords = jnp.stack([grid_x, grid_y], axis=-1)  # [H,W,2]
    return coords

def make_centered_warper(coords: jnp.ndarray, cx: float, cy: float):
    """
    Build warp(x, M, t) applying an inverse affine warp around the image center.

    x: [B,H,W], M: [B,2,2], t: [B,2]  ->  [B,H,W]
    coords: [H,W,2] in centered coordinates
    """
    def _warp_single(img, M, t):
        t = t.astype(coords.dtype)
        Minv = M.T

        # inverse map in centered coords: src_c = (coords - t) @ Minv^T
        src_c = (coords - t[None, None, :]) @ Minv.T

        sx = src_c[..., 0] + cx
        sy = src_c[..., 1] + cy
        return bilinear_sample_2d(img, sx, sy)

    def warp(x, M, t):
        return jax.vmap(_warp_single, in_axes=(0, 0, 0))(x, M, t)

    return warp

def _iter_xmipp_images(fnImgs):
    for fn in fnImgs:
        img = xmippLib.Image(fn).getData()
        img = jnp.asarray(img)
        yield img.astype(jnp.float32)

def get_xmipp_ds(fnImgs,
                 x: jnp.ndarray,
                 y: jnp.ndarray,
                 seed: int = 0):
    imgs = list(_iter_xmipp_images(fnImgs))

    rng = np.random.default_rng(seed)
    idx = np.arange(len(imgs))

    while True:
        rng.shuffle(idx)

        for k in idx:
            im = bilinear_sample_2d(imgs[k], x, y)

            yield {"image": im}

def norm_per_image(x: jnp.ndarray, eps: float = 1e-6) -> jnp.ndarray:
    """
    Per-image standardization (zero mean, unit std) for gray-level invariance.
    x: [B,H,W] in [0,1] -> standardized [B,H,W]
    """
    m = x.mean(axis=(1, 2), keepdims=True)
    v = jnp.var(x, axis=(1, 2), keepdims=True)
    return (x - m) / jnp.sqrt(v + eps)

def make_triplet_gen(
    rng,
    ds,
    batch_size,
    warper,
    a_range=(0.7, 1.4),
    b_range=(-0.2, 0.2),
    sigma_shift=1.0,
):
    """
    Yields (xa, xp, xn) where xa & xp are two aug views of the SAME base image,
    and xn is an aug view of a DIFFERENT base image. Shapes: [B,64, 64], normalized values
    """
    ds_iter = iter(ds)
    key = rng

    def sample_T(k, B):
        """
        Returns:
          k : updated PRNGKey
          M : [B,2,2] rotation matrices (no mirror)
          t : [B,2] pixel shifts ~ N(0, sigma_shift^2) per axis
          a : [B] gray gain
          b : [B] gray bias
        """
        k, k_th, k_sh, k_a, k_b = jax.random.split(k, 5)
        theta = jax.random.uniform(k_th, (B,), dtype=jnp.float32,
                                   minval=-jnp.pi, maxval=jnp.pi)
        cos, sin = jnp.cos(theta), jnp.sin(theta)
        M = jnp.stack([jnp.stack([cos, -sin], axis=-1),
                       jnp.stack([sin,  cos], axis=-1)], axis=-2)  # [B,2,2]
        t = jax.random.normal(k_sh, (B, 2), dtype=jnp.float32) * float(sigma_shift)
        a = jax.random.uniform(k_a, (B,), dtype=jnp.float32, minval=a_range[0], maxval=a_range[1])
        b = jax.random.uniform(k_b, (B,), dtype=jnp.float32, minval=b_range[0], maxval=b_range[1])
        return k, M, t, a, b

    while True:
        # 2B bases: first B for (anchor,positive), second B for negatives
        imgs = [next(ds_iter)["image"] for _ in range(2 * batch_size)]
        base = jnp.stack(imgs, 0).astype(jnp.float32)
        base_ap = base[:batch_size]
        base_n  = base[batch_size:]

        # keys
        key, kA, kP, kN = jax.random.split(key, 4)

        # sample transforms
        kA, MA, tA, aA, bA = sample_T(kA, batch_size)
        kP, MP, tP, aP, bP = sample_T(kP, batch_size)
        kN, MN, tN, aN, bN = sample_T(kN, batch_size)

        # apply transforms + gray
        xa = aA[:, None, None] * warper(base_ap, MA, tA) + bA[:, None, None]
        xp = aP[:, None, None] * warper(base_ap, MP, tP) + bP[:, None, None]
        xn = aN[:, None, None] * warper(base_n,  MN, tN) + bN[:, None, None]

        # Normalize values
        xa = norm_per_image(xa)
        xp = norm_per_image(xp)
        xn = norm_per_image(xn)

        yield xa, xp, xn

def show_triplet_batch(train_gen, fnDir, n_examples=5):
    """Display n_examples triplets (anchor, positive, negative) from generator."""
    xa, xp, xn = next(train_gen)           # each [B,28,28]
    xa, xp, xn = np.array(xa), np.array(xp), np.array(xn)

    I=xmippLib.Image()
    print("Estoy aqui")
    for i in range(n_examples):
        I.setData(xa[i])
        I.write(os.path.join(fnDir,"anchor%02d.mrc"%i))
        I.setData(xp[i])
        I.write(os.path.join(fnDir,"positive%02d.mrc"%i))
        I.setData(xn[i])
        I.write(os.path.join(fnDir,"negative%02d.mrc"%i))
        print("Saved ",i)

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

    def run(self):
        fnXmd = self.getParam("-i")
        fnModel = self.getParam("--omodel")
        maxEpochs = int(self.getParam("--maxEpochs"))
        batch_size = int(self.getParam("--batchSize"))
        XdimOut = int(self.getParam("--imgSize"))
        gpuId = self.getParam("--gpu")
        learning_rate = float(self.getParam("--learningRate"))
        sigma_shift = float(self.getParam("--sigmaShift"))

        if not gpuId.startswith('-1'):
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = gpuId
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
        os.environ['TF_GPU_ALLOCATOR']='cuda_malloc_async'

        Xdim, _, _, _, _ = xmippLib.MetaDataInfo(fnXmd)
        mdExp = xmippLib.MetaData(fnXmd)
        fnImgs = mdExp.getColumnValues(xmippLib.MDL_IMAGE)

        coordsResize = precompute_resize_grid(XdimOut, XdimOut, Xdim, Xdim)
        train_ds = get_xmipp_ds(fnImgs, coordsResize[...,0], coordsResize[...,1])
        coords, cx, cy = precompute_original_grid(XdimOut, XdimOut)

        warper = make_centered_warper(coords, cx, cy)
        rng = jax.random.PRNGKey(42)
        train_gen = make_triplet_gen(rng, train_ds, batch_size, warper, sigma_shift=sigma_shift)
        # show_triplet_batch(train_gen, ".") # Debugging code

        return 0

if __name__ == '__main__':
    exitCode = ScriptDeepEmbedTrain().tryRun()
    sys.exit(exitCode)
