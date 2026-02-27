#!/usr/bin/env python3

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax import serialization
import optax
from flax.training import train_state

import math
import numpy as np
from scipy.ndimage import zoom
import os
import sys
import time
import xmippLib
from xmipp_script import XmippScript

DT_IMAGE = jnp.bfloat16

def bilinear_sample_2d(img: jnp.ndarray, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    H, W = img.shape

    x = jnp.clip(x, 0.0, W - 1.0001)
    y = jnp.clip(y, 0.0, H - 1.0001)

    x0 = jnp.floor(x).astype(jnp.int32)
    y0 = jnp.floor(y).astype(jnp.int32)
    x1 = jnp.clip(x0 + 1, 0, W - 1)
    y1 = jnp.clip(y0 + 1, 0, H - 1)

    # weights MUST be float32 for stable geometry
    wx = x - x0.astype(jnp.float32)
    wy = y - y0.astype(jnp.float32)

    flat = img.reshape(-1)

    def gather(ix, iy):
        idx = (iy * W + ix).reshape(-1)
        return jnp.take(flat, idx).reshape(x.shape)

    I00 = gather(x0, y0)
    I10 = gather(x1, y0)
    I01 = gather(x0, y1)
    I11 = gather(x1, y1)

    # result dtype follows img dtype; that’s fine (bf16) but you can cast if needed
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

def _load_xmipp_images_numpy(fnImgs):
    """Load all images as float32 numpy arrays (2D)."""
    imgs = []
    for fn in fnImgs:
        im = xmippLib.Image(fn).getData()
        im = np.asarray(im, dtype=np.float32)
        # squeeze common singleton dims
        if im.ndim == 3:
            if im.shape[-1] == 1:
                im = im[..., 0]
            elif im.shape[0] == 1:
                im = im[0]
            else:
                raise ValueError(f"Expected 2D or singleton 3D image, got {im.shape} for {fn}")
        imgs.append(im)
    return imgs

def _per_image_standardize_np(im, eps=1e-6):
    m = float(im.mean())
    s = float(im.std())
    return (im - m) / (s + eps)

def preload_resize_normalize(fnImgs,
                             out_hw,
                             dtype=np.float32):

    Hout = Wout = out_hw
    imgs_np = _load_xmipp_images_numpy(fnImgs)

    N = len(imgs_np)
    if N == 0:
        raise ValueError("No images provided")

    # infer input size from first image
    Hin, Win = imgs_np[0].shape
    zoom_y = Hout / Hin
    zoom_x = Wout / Win

    Y = np.empty((N, Hout, Wout), dtype=dtype)
    for i, im in enumerate(imgs_np):
        im = _per_image_standardize_np(im)
        im_resized = zoom(im, (zoom_y, zoom_x), order=1)
        Y[i] = im_resized.astype(dtype)
    return Y

def get_xmipp_preloaded_array(fnImgs, XdimOut):
    pre = preload_resize_normalize(fnImgs, XdimOut, dtype=np.float32)  # numpy [N,H,W]
    pre = jax.device_put(pre)  # explicit device transfer once
    return pre.astype(DT_IMAGE)

def get_xmipp_ds(fnImgs, XdimOut, seed: int = 0):
    """
    Preload + resize once, then yield dicts {"image": ...} quickly.
    """
    pre = preload_resize_normalize(fnImgs, XdimOut)

    rng = np.random.default_rng(seed)
    idx = np.arange(pre.shape[0])

    while True:
        rng.shuffle(idx)
        for k in idx:
            yield {"image": pre[k]}

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
        a = jax.random.uniform(k_a, (B,), dtype=jnp.float32, minval=a_range[0], maxval=a_range[1])
        b = jax.random.uniform(k_b, (B,), dtype=jnp.float32, minval=b_range[0], maxval=b_range[1])
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

        # warper uses bf16 images but float32 coords; fine
        xa = aA[:, None, None] * warper(base_ap, MA, tA) + bA[:, None, None]
        xp = aP[:, None, None] * warper(base_ap, MP, tP) + bP[:, None, None]
        xn = aN[:, None, None] * warper(base_n,  MN, tN) + bN[:, None, None]

        # Optional: cast to bf16 after augmentation to keep the model input bf16
        xa = xa.astype(DT_IMAGE); xp = xp.astype(DT_IMAGE); xn = xn.astype(DT_IMAGE)
        return key, (xa, xp, xn)

    return next_triplet

class Encoder(nn.Module):
    d: int = 128 # Size of the output layer
    @nn.compact
    def __call__(self, x):
        x = x[..., None]
        x = nn.Conv(32, (5,5), (2,2))(x); x = nn.relu(x)
        x = nn.Conv(64, (5,5), (2,2))(x); x = nn.relu(x)
        x = nn.Conv(64, (5,5), (2,2))(x); x = nn.relu(x)
        x = nn.Conv(128, (5,5), (1,1))(x); x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(256)(x); x = nn.relu(x)
        return nn.Dense(self.d)(x)

class TripletNet(nn.Module):
    d: int = 128 # Size of the encoding embedding
    @nn.compact
    def __call__(self, x):  # x: [B,d,d]
        z = Encoder(self.d)(x)
        z = z / (jnp.linalg.norm(z, axis=1, keepdims=True) + 1e-9)
        return z  # [B,d], unit-norm

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

def create_train_state(rng, model, XdimOut, lr=3e-4):
    dummy = jnp.zeros((2,XdimOut,XdimOut), dtype=jnp.float32)
    variables = model.init(rng, dummy)
    params = variables["params"]
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(lr),
    )
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

@jax.jit
def train_step(state, batch, margin=0.2):
    def _loss_only(p):
        return triplet_loss(p, state.apply_fn, batch, margin=margin)
    (loss, metrics), grads = jax.value_and_grad(_loss_only, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, metrics

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

        model = TripletNet(d=128)
        state = create_train_state(jax.random.PRNGKey(0), model, XdimOut,
                                   learning_rate)

        pre_dev = get_xmipp_preloaded_array(fnImgs, XdimOut)  # GPU once
        coords, cx, cy = precompute_original_grid(XdimOut, XdimOut)
        warper = make_centered_warper(coords, cx, cy)

        key = jax.random.PRNGKey(42)
        next_triplet = make_batcher(pre_dev, warper, batch_size, sigma_shift)

        margin = 0.2

        # Warmup (optional): trigger compilation before timing
        key, batch = next_triplet(key)
        state, _ = train_step(state, batch, margin=margin)
        jax.block_until_ready(state.params)

        N = int(pre_dev.shape[0])  # number of images
        steps_per_epoch = math.ceil(
            N / batch_size)  # define one epoch as ~one pass worth of batches
        global_step = 0

        t0 = time.time()
        for epoch in range(1, maxEpochs + 1):
            # running sums for epoch averages
            loss_sum = 0.0
            dap_sum = 0.0
            dan_sum = 0.0
            viol_sum = 0.0

            for _ in range(steps_per_epoch):
                key, batch = next_triplet(key)
                state, metrics = train_step(state, batch, margin=margin)
                global_step += 1

                loss_sum += float(metrics["loss"])
                dap_sum += float(metrics["d_ap"])
                dan_sum += float(metrics["d_an"])
                viol_sum += float(metrics["viol"])

            # epoch summary
            loss_avg = loss_sum / steps_per_epoch
            dap_avg = dap_sum / steps_per_epoch
            dan_avg = dan_sum / steps_per_epoch
            viol_avg = viol_sum / steps_per_epoch

            dt = time.time() - t0
            print(f"[epoch {epoch:4d}/{maxEpochs}] loss={loss_avg:.8f}  "
                  f"d_ap={dap_avg:.8f}  d_an={dan_avg:.8f}  "
                  f"viol%={100 * viol_avg:.1f}  ({dt:.1f}s/epoch)")
            t0 = time.time()

        with open(fnModel, "wb") as f:
            f.write(serialization.to_bytes(state))

if __name__ == '__main__':
    exitCode = ScriptDeepEmbedTrain().tryRun()
    sys.exit(exitCode)
