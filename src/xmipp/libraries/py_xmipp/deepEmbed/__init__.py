
import os
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import jax
import jax.numpy as jnp
from jax import lax
from jax.scipy.ndimage import map_coordinates

from flax import linen as nn
from flax import serialization
from flax.training import train_state

import optax
import numpy as np
from scipy.ndimage import zoom

import xmippLib

DT_IMAGE = jnp.bfloat16

# -----------------------------------------------------------------------------
# Environment
# -----------------------------------------------------------------------------

def setup_gpu(gpu_id):
    if gpu_id is not None and not str(gpu_id).startswith("-1"):
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
    os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"


# -----------------------------------------------------------------------------
# Geometry / warping
# -----------------------------------------------------------------------------

def precompute_original_grid(H: int, W: int):
    ys = jnp.arange(H, dtype=jnp.float32)
    xs = jnp.arange(W, dtype=jnp.float32)
    grid_y, grid_x = jnp.meshgrid(ys, xs, indexing="ij")

    cy = (H - 1) * 0.5
    cx = (W - 1) * 0.5

    coords = jnp.stack([grid_x - cx, grid_y - cy], axis=-1)  # [H,W,2]
    return coords, cx, cy


def make_centered_warper_multichan(coords: jnp.ndarray, cx: float, cy: float):
    def _sample_2d(img2d, sx, sy):
        c = jnp.stack([sy, sx], axis=0)  # [2,H,W] = (y,x)
        return map_coordinates(img2d, c, order=1, mode="nearest")

    def _warp_single(img, M, t):
        t = t.astype(coords.dtype)
        Minv = M.T  # valid for rotations

        src_c = (coords - t[None, None, :]) @ Minv.T
        sx = src_c[..., 0] + cx
        sy = src_c[..., 1] + cy

        img_khw = jnp.moveaxis(img, -1, 0)
        warped_khw = jax.vmap(_sample_2d, in_axes=(0, None, None))(img_khw, sx, sy)
        warped_hwk = jnp.moveaxis(warped_khw, 0, -1)
        return warped_hwk

    return jax.vmap(_warp_single, in_axes=(0, 0, 0))


# -----------------------------------------------------------------------------
# Preprocessing
# -----------------------------------------------------------------------------

def _load_parallel(fn_imgs):
    with ThreadPoolExecutor() as executor:
        imgs = list(executor.map(lambda f: np.asarray(xmippLib.Image(f).getData()), fn_imgs))
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
    fy = np.fft.fftfreq(H, d=1.0)
    fx = np.fft.fftfreq(W, d=1.0)
    FY, FX = np.meshgrid(fy, fx, indexing="ij")
    return np.sqrt(FX * FX + FY * FY).astype(np.float32)


def _make_filter_bank_masks(H, W, K=5, fmin=0.0, fmax=0.5):
    R = _radial_freq_grid(H, W)
    edges = np.linspace(fmin, fmax, K + 1, dtype=np.float32)
    masks = []
    for k in range(K):
        lo, hi = float(edges[k]), float(edges[k + 1])
        if k == K - 1:
            m = (R >= lo) & (R <= hi)
        else:
            m = (R >= lo) & (R < hi)
        masks.append(m.astype(np.float32))
    return np.stack(masks, axis=0), edges


def make_preprocess_context(out_hw, K=5, sigma=None, fmin=0.0, fmax=0.5):
    Hout = Wout = int(out_hw)
    if sigma is None:
        sigma = Hout / 6.0
    gmask = _gaussian_mask_2d(Hout, Wout, sigma=sigma)
    fb_masks, fb_edges = _make_filter_bank_masks(Hout, Wout, K=K, fmin=fmin, fmax=fmax)
    return {
        "out_hw": Hout,
        "K": K,
        "gmask": gmask,
        "fb_masks": fb_masks,
        "fb_edges": fb_edges,
    }


def preprocess_images(fn_imgs, preprocess_ctx, dtype=np.float32):
    """
    Deterministic preprocessing:
      standardize -> resize -> fft band split -> ifft -> gaussian mask

    Returns:
      Y: [N,H,W,K]
    """
    imgs_np = _load_parallel(fn_imgs)
    N = len(imgs_np)
    if N == 0:
        raise ValueError("No images provided")

    Hout = Wout = int(preprocess_ctx["out_hw"])
    fb_masks = preprocess_ctx["fb_masks"]
    gmask = preprocess_ctx["gmask"]
    K = int(preprocess_ctx["K"])

    Hin, Win = imgs_np[0].shape
    zoom_y = Hout / Hin
    zoom_x = Wout / Win

    Y = np.empty((N, Hout, Wout, K), dtype=dtype)

    for i, im in enumerate(imgs_np):
        im = _per_image_standardize_np(im)
        im_resized = zoom(im, (zoom_y, zoom_x), order=1).astype(np.float32)
        F = np.fft.fft2(im_resized)

        for k in range(K):
            Fb = F * fb_masks[k]
            band = np.fft.ifft2(Fb).real
            band = band * gmask
            Y[i, :, :, k] = band.astype(dtype)

    return Y


def get_xmipp_preloaded_array(fn_imgs, XdimOut, K=5):
    preprocess_ctx = make_preprocess_context(XdimOut, K=K)
    pre = preprocess_images(fn_imgs, preprocess_ctx, dtype=np.float32)
    pre = jax.device_put(pre)
    return pre.astype(DT_IMAGE)


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------

class TripletNet(nn.Module):
    d: int = 128
    compute_dtype: any = jnp.bfloat16
    param_dtype: any = jnp.float32

    @nn.compact
    def __call__(self, x):
        x = x.astype(self.compute_dtype)

        x = nn.Conv(32, (5, 5), (2, 2), dtype=self.compute_dtype,
                    param_dtype=self.param_dtype)(x)
        x = nn.relu(x)

        x = nn.Conv(64, (5, 5), (2, 2), dtype=self.compute_dtype,
                    param_dtype=self.param_dtype)(x)
        x = nn.relu(x)

        x = nn.Conv(64, (5, 5), (2, 2), dtype=self.compute_dtype,
                    param_dtype=self.param_dtype)(x)
        x = nn.relu(x)

        x = nn.Conv(128, (5, 5), (1, 1), dtype=self.compute_dtype,
                    param_dtype=self.param_dtype)(x)
        x = nn.relu(x)

        x = jnp.mean(x, axis=(1, 2))
        x = nn.Dense(256, dtype=self.compute_dtype,
                     param_dtype=self.param_dtype)(x)
        x = nn.relu(x)

        z = nn.Dense(self.d, dtype=self.compute_dtype,
                     param_dtype=self.param_dtype)(x)

        z = z.astype(jnp.float32)
        z = z / (jnp.linalg.norm(z, axis=-1, keepdims=True) + 1e-9)
        return z


def create_train_state(rng, model, XdimOut, K=5, lr=3e-4):
    dummy = jnp.zeros((2, XdimOut, XdimOut, K), dtype=jnp.float32)
    variables = model.init(rng, dummy)
    params = variables["params"]
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(lr),
    )
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


def save_train_state(state, fn_model):
    with open(fn_model, "wb") as f:
        f.write(serialization.to_bytes(state))


def load_train_state(fn_model, XdimOut, embedding_dim, K=5, lr=3e-4, rng=None):
    if rng is None:
        rng = jax.random.PRNGKey(0)
    model = TripletNet(d=embedding_dim)
    state = create_train_state(rng, model, XdimOut, K=K, lr=lr)
    with open(fn_model, "rb") as f:
        state = serialization.from_bytes(state, f.read())
    return model, state


def make_embedder(apply_fn):
    @jax.jit
    def _embed(params, x):
        return apply_fn({'params': params}, x)
    return _embed


# -----------------------------------------------------------------------------
# Training helpers
# -----------------------------------------------------------------------------

def triplet_loss(params, apply_fn, batch, margin=0.2):
    xa, xp, xn = batch
    ea = apply_fn({'params': params}, xa)
    ep = apply_fn({'params': params}, xp)
    en = apply_fn({'params': params}, xn)

    dap = 1.0 - jnp.sum(ea * ep, axis=1)
    dan = 1.0 - jnp.sum(ea * en, axis=1)

    losses = jnp.maximum(0.0, dap - dan + margin)
    loss = jnp.mean(losses)

    frac_viol = jnp.mean((dap + margin > dan).astype(jnp.float32))
    metrics = {
        "loss": loss,
        "d_ap": jnp.mean(dap),
        "d_an": jnp.mean(dan),
        "viol": frac_viol,
    }
    return loss, metrics


@jax.jit
def train_step(state, batch, margin=0.2):
    def _loss_only(p):
        return triplet_loss(p, state.apply_fn, batch, margin=margin)
    (loss, metrics), grads = jax.value_and_grad(_loss_only, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, metrics


@partial(jax.jit, static_argnums=(0,))
def run_epoch(step_fn, pre_dev, state, key, steps_per_epoch, margin):
    def body(i, carry):
        state, key, loss_sum, dap_sum, dan_sum, viol_sum = carry
        state, key, metrics = step_fn(pre_dev, state, key, margin)
        loss_sum = loss_sum + metrics["loss"]
        dap_sum = dap_sum + metrics["d_ap"]
        dan_sum = dan_sum + metrics["d_an"]
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


def make_batcher(
    warper,
    batch_size,
    sigma_shift,
    ds_len,
    a_range=(0.7, 1.4), b_range=(-0.2, 0.2)
):

    N = ds_len

    def sample_T(k, B):
        k, k_th, k_sh, k_a, k_b = jax.random.split(k, 5)
        theta = jax.random.uniform(k_th, (B,), dtype=jnp.float32,
                                   minval=-jnp.pi, maxval=jnp.pi)
        cos, sin = jnp.cos(theta), jnp.sin(theta)
        M = jnp.stack([jnp.stack([cos, -sin], axis=-1),
                       jnp.stack([sin,  cos], axis=-1)], axis=-2)
        t = jax.random.normal(k_sh, (B, 2), dtype=jnp.float32) * sigma_shift
        a = jax.random.uniform(k_a, (B,), dtype=DT_IMAGE,
                               minval=a_range[0], maxval=a_range[1])
        b = jax.random.uniform(k_b, (B,), dtype=DT_IMAGE,
                               minval=b_range[0], maxval=b_range[1])
        return k, M, t, a, b

    @jax.jit
    def next_triplet(pre_dev, key):
        key, k_idx, kA, kP, kN = jax.random.split(key, 5)
        idx = jax.random.randint(k_idx, (2 * batch_size,), 0, N)

        base = pre_dev[idx]
        base_ap = base[:batch_size]
        base_n = base[batch_size:]

        kA, MA, tA, aA, bA = sample_T(kA, batch_size)
        kP, MP, tP, aP, bP = sample_T(kP, batch_size)
        kN, MN, tN, aN, bN = sample_T(kN, batch_size)

        xa = aA[:, None, None, None] * warper(base_ap, MA, tA) + bA[:, None, None, None]
        xp = aP[:, None, None, None] * warper(base_ap, MP, tP) + bP[:, None, None, None]
        xn = aN[:, None, None, None] * warper(base_n, MN, tN) + bN[:, None, None, None]

        xa = xa.astype(DT_IMAGE)
        xp = xp.astype(DT_IMAGE)
        xn = xn.astype(DT_IMAGE)
        return key, (xa, xp, xn)

    @jax.jit
    def step(pre_dev, state, key, margin):
        key, batch = next_triplet(pre_dev, key)
        state, metrics = train_step(state, batch, margin=margin)
        return state, key, metrics

    return next_triplet, step


def sample_embeddings(pre_dev, state, next_triplet, key, embedding_points, batch_size):
    out = []
    n_done = 0

    while n_done < embedding_points:
        key, batch = next_triplet(pre_dev, key)
        xa, _, _ = batch

        emb = state.apply_fn({'params': state.params}, xa)
        emb = np.asarray(jax.device_get(emb), dtype=np.float32)

        remaining = embedding_points - n_done
        if emb.shape[0] > remaining:
            emb = emb[:remaining]

        out.append(emb)
        n_done += emb.shape[0]

    return np.concatenate(out, axis=0)


# -----------------------------------------------------------------------------
# Centroids / prediction
# -----------------------------------------------------------------------------

def load_centroids(fn_centroids, normalize=True):
    C = np.load(fn_centroids)
    C = np.asarray(C, dtype=np.float32)
    if normalize:
        C /= np.maximum(np.linalg.norm(C, axis=1, keepdims=True), 1e-12)
    return C


@jax.jit
def assign_to_centroids(embeddings, centroids):
    sims = embeddings @ centroids.T
    labels = jnp.argmax(sims, axis=1)
    scores = jnp.max(sims, axis=1)
    return labels, scores