# **************************************************************************
# *
# * Authors:  Mikel Iceta (miceta@cnb.csic.es) 2026
# * Authors:  Carlos Oscar Sorzano (coss@cnb.csic.es)
# *
# * Unidad de  Bioinformatica of Centro Nacional de Biotecnologia , CSIC
# *
# * This program is free software; you can redistribute it and/or modify
# * it under the terms of the GNU General Public License as published by
# * the Free Software Foundation; either version 2 of the License, or
# * (at your option) any later version.
# *
# * This program is distributed in the hope that it will be useful,
# * but WITHOUT ANY WARRANTY; without even the implied warranty of
# * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# * GNU General Public License for more details.
# *
# * You should have received a copy of the GNU General Public License
# * along with this program; if not, write to the Free Software
# * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
# * 02111-1307  USA
# *
# *  All comments concerning this program package may be sent to the
# *  e-mail address 'scipion@cnb.csic.es'
# *
# **************************************************************************

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
CONV_DEPTH = 6
CONV_INITIAL_CHANNELS = 32
FILTER_BANK_BANDS = 5
DROPOUT_PROB = 0.1

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
# Preprocessing
# -----------------------------------------------------------------------------

def _load_parallel(fn_imgs, n_cores=None):
    with ThreadPoolExecutor(max_workers=n_cores) as executor:
        imgs = list(executor.map(lambda f: np.asarray(xmippLib.Image(f).getData()), fn_imgs))
    return np.stack(imgs) # (B, H, W)

def _per_image_standardize_jnp(x, eps=1e-6):
    mean = jnp.mean(x, axis=(1,2), keepdims=True)
    std = jnp.std(x, axis=(1,2), keepdims=True)
    return (x - mean) / (std + eps)

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

def _make_filter_bank_masks(H, W, K=4, fmin=0.0, fmax=0.5):
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

def preprocess_images(fn_imgs, preprocess_ctx, dtype=np.float32, n_cores=None):
    """
    Deterministic preprocessing:
      resize (if required) -> fft band split -> ifft -> gaussian mask

    Returns:
      Y: [N,H,W,K]
    """
    imgs_np = _load_parallel(fn_imgs, n_cores)
    N = len(imgs_np)
    if N == 0:
        raise ValueError("No images provided!")

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
        #im_resized = zoom(im, (zoom_y, zoom_x), order=1).astype(np.float32)
        im_resized = im
        F = np.fft.fft2(im_resized)

        # Generate the K band-limited images in the spatial domain
        for k in range(K):
            Fb = F * fb_masks[k]
            band = np.fft.ifft2(Fb).real
            # Apply the Gaussian mask to kill borders
            band = band * gmask
            Y[i, :, :, k] = band.astype(dtype)

    return Y

def xmipp_preload_all(fn_imgs, XdimOut, K=4):
    preprocess_ctx = make_preprocess_context(XdimOut, K=K)
    pre = preprocess_images(fn_imgs, preprocess_ctx, dtype=np.float32)
    pre = jax.device_put(pre)
    return pre.astype(DT_IMAGE)

def xmipp_train_batch_generator(fn_imgs, batch_size, XdimOut, K=4, n_cores=None, augment=True):
    """
    fn_imgs: list of str, paths to the images
    batch_size: int, number of particles per batch
    XdimOut: int, output dimension of the preprocessed images
    K: filterbank parameter, number of bands
    augment: bool, whether to apply random augmentations
    """
    preprocess_ctx = make_preprocess_context(XdimOut, K=K)
    n_imgs = len(fn_imgs)
    idxs = np.arange(n_imgs)

    # Infinitely generate for training
    while True:
        np.random.shuffle(idxs)

        for start in range(0, n_imgs, batch_size):
            batch_idxs = idxs[start : start + batch_size]
            batch_fns = [fn_imgs[i] for i in batch_idxs]

            # Load in parallel
            imgs = _load_parallel(batch_fns, n_cores) # (B, H, W)

            # Preprocess data (generate filterbank, PSD, resize, etc)
            pre = preprocess_images(batch_fns, preprocess_ctx, dtype=np.float32, n_cores=n_cores) # (B, Hout, Wout, K+2)

            # On-the-fly augmentation
            if augment:
                for i in range(pre.shape[0]):
                    angle = np.random.choice([0, 90, 180, 270])
                    pre[i] = np.rot90(pre[i], k=angle // 90, axes=(0, 1))
                    pre[i] = np.fliplr(pre[i]) if np.random.rand() < 0.5 else pre[i]

            # Send to GPU
            pre = jax.device_put(pre.astype(DT_IMAGE))

            yield pre.astype(np.float32), labels_batch


# -----------------------------------------------------------------------------
# Block definitions
# -----------------------------------------------------------------------------

class MultiScaleDilatedBlock(nn.Module):
    features: int
    dilation_rates: tuple = (1, 2, 3)
    num_groups: int = 8
    dropout_rate: float = 0.1
    use_residual: bool = True

    @nn.compact
    def __call__(self, x, training: bool = True):
        branches = []

        def norm_act(y):
            y = nn.GroupNorm(num_groups=self.num_groups)(y)
            return nn.gelu(y)
        
        # Branch 1: 3x3 for fine details
        b1 = nn.Conv(features=self.features, kernel_size=(3,3), padding="SAME")(x)
        branches.append(norm_act(b1))

        # Branch 2: 5x5 for more global, denoised context
        b2 = nn.Conv(features=self.features, kernel_size=(5,5), padding="SAME")(x)
        branches.append(norm_act(b2))

        # Branches 3+: Dilated convolutions
        # Larger receptive fields without pooling, cool for low SNR lol
        for d in self.dilation_rates:
            bd = nn.Conv(
                features=self.features,
                kernel_size=(3, 3),
                padding="SAME",
                kernel_dilation=(d, d),
            )(x)
            branches.append(norm_act(bd))

        # Fuse everything! 
        x_out = jnp.concatenate(branches, axis=-1)
        x_out = nn.Conv(features=self.features, kernel_size=(1, 1), padding="SAME")(x_out)
        x_out = nn.GroupNorm(num_groups=self.num_groups)(x_out)

        if self.dropout_rate > 0.0:
            x_out = nn.Dropout(rate=self.dropout_rate)(x_out, deterministic=not training)

        # Manage residual connections. Stabilizes training on cryo data a lot
        if self.use_residual:
            if x.shape[-1] != self.features: # Project if needed for residual addition
                x = nn.Conv(self.features, kernel_size=(1, 1), padding="SAME")(x)
            x_out = x_out + x

        return nn.gelu(x_out)
    
class BranchGatingBlock(nn.Module):
    n_branches: int = 3

    @nn.compact
    def __call__(self, features):
        # features: list of tensors [(B,H,W,C), ...]
        pooled = [jnp.mean(f, axis=(1,2)) for f in features] # List of (B,C)'s

        x = jnp.concatenate(pooled, axis=-1) # (B, C*n_branches)
        x = nn.Dense(features=32)(x)
        x = nn.gelu(x)
        x = nn.Dense(self.n_branches)(x)

        weights = nn.softmax(x, axis=-1) # (B, n_branches)
        return weights

class AttentionPoolingBlock(nn.Module):
    hidden_dim: int = 32
    
    @nn.compact
    def __call__(self, x):
        # x shape: (B, H, W, C)

        # Compute attention logits
        attn = nn.Conv(features=self.hidden_dim, kernel_size=(1,1), padding="SAME")(x)
        attn = nn.gelu(attn)
        attn = nn.Conv(features=1, kernel_size=(1,1), padding="SAME")(attn)
        
        # Flatten spatially
        B, H, W = attn.shape
        attn = attn.reshape(B, H*W)

        # Softmax to get attention weights
        attn = nn.softmax(attn, axis=-1)

        x_flat = x.reshape(B, H*W, -1)

        # Weighted sum pooling
        pooled = jnp.sum(
            x_flat * attn[..., None], axis=1
        )   # (B, C)

        return pooled

class MultiHeadAttentionPoolingBlock(nn.Module):
    num_heads: int = 4

    @nn.compact
    def __call__(self, x):
        heads = []

        for _ in range(self.num_heads):
            attn = nn.conv(1, (1,1))(x)
            B, H ,W , _ = attn.shape

            attn = attn.reshape(B, H*W)
            # Add 0.7 temperature to make it softer, 
            # since we are pooling over a lot of elements
            attn = nn.softmax(attn / 0.7, axis=-1)
            x_flat = x.reshape(B, H*W, -1)

            pooled = jnp.sum(x_flat * attn[..., None], axis=1)
            heads.append(pooled)

        return jnp.concatenate(heads, axis=-1)

# -----------------------------------------------------------------------------
# Network definitions
# -----------------------------------------------------------------------------

@jax.jit
def train_step(state, batch):
    dropout_rng, new_dropout_rng = jax.random.split(state.dropout_rng)

    images, labels = batch
    labels = labels.reshape(-1, 1) # Assuming bin cls for simplicity

    def loss_fn(params):
        logits = state.apply_fn(
            {"params": params},
            batch,
            training=True,
            rngs={"dropout": dropout_rng}
        )

        loss = optax.sigmoid_binary_cross_entropy(logits, labels)
        loss = jnp.mean(loss)

        return loss, logits
    
    (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)

    state = state.apply_gradients(grads=grads)
    state = state.replace(dropout_rng=new_dropout_rng)

    return state, loss, logits

@jax.jit
def val_step(params, batch):
    images, labels = batch
    labels = labels.reshape(-1, 1) # Assuming bin cls for simplicity

    logits = model.apply(
        {"params": params},
        images,
        training=False
    )

    loss = optax.sigmoid_binary_cross_entropy(logits, labels)
    loss = jnp.mean(loss)

    probs = jax.nn.sigmoid(logits)

    return loss, probs

def create_lr_schedule(base_lr, warmup_steps=500, total_steps=10000):
    warmup = optax.linear_schedule(
        init_value=0.0, end_value=base_lr, transition_steps=warmup_steps
    )

    cosine = optax.cosine_decay_schedule(
        init_value=base_lr, decay_steps=total_steps - warmup_steps
    )

    return optax.join_schedules(
        schedules=[warmup, cosine],
        boundaries=[warmup_steps]
    )

def compute_auc(probs, labels):
    """
    probs: (N, )
    labels: (N, ) being {0, 1}
    """

    # Sort by predicted probabilities
    order = jnp.argsort(probs)
    labels = labels[order]

    # Count POS / NEG
    pos = jnp.sum(labels)
    neg = len(labels) - pos

    # Rank sum method for AUC calculation
    ranks = jnp.arange(1, len(labels) + 1)
    pos_ranks = jnp.sum(ranks * labels)
    auc = (pos_ranks - pos * (pos + 1) / 2) / (pos * neg + 1e-8)

    return auc

class CryoCNNet(nn.Module):
    features: int = CONV_INITIAL_CHANNELS
    depth: int = CONV_DEPTH

    @nn.compact
    def __call__(self, x, training: bool = True):
        # Split inputs
        # The shape of x is (B,H,W,n+2)
        # raw image + n filter bank outputs + PSD
        raw = x[..., :1] # Original image
        bands = x[..., 1:-1] # Filter bank outputs
        psd = x[..., -1:] # Radial PSD

        # PSD can be heavy tail!
        # You better log it
        psd = jnp.log(psd + 1e-6)

        # Data normalization in a separate manner
        raw = _per_image_standardize_jnp(raw)
        bands = _per_image_standardize_jnp(bands)
        psd = _per_image_standardize_jnp(psd)

        # Encode features
        raw_feat = nn.Conv(features=self.features, kernel_size=(3,3), padding="SAME")(raw)
        band_feat = nn.conv(features=self.features, kernel_size=(3,3), padding="SAME")(bands)
        psd_feat = nn.Conv(features=self.features, kernel_size=(3,3), padding="SAME")(psd)

        # Gating for branch weighting
        weights = BranchGatingBlock()([raw_feat, band_feat, psd_feat])
        w_raw, w_band, w_psd = jnp.split(weights, 3, axis=-1)
        # Reshape for proper broadcasting
        w_raw = w_raw[:, None, None, :]
        w_band = w_band[:, None, None, :]
        w_psd = w_psd[:, None, None, :]
        # Weight the features
        raw_feat *= w_raw
        band_feat *= w_band
        psd_feat *= w_psd

        # Fusion of all features
        x = jnp.concatenate([raw_feat, band_feat, psd_feat], axis=-1)
        x = nn.Conv(features=self.features, kernel_size=(1,1), padding="SAME")(x)

        # Stack of CNNBlocks
        for _ in range(self.depth):
            x = MultiScaleDilatedBlock(features=self.features)(x, training=training)
            self.features *= 2

        # Attention! Pooling
        x = MultiHeadAttentionPoolingBlock()(x) # Attention-based pooling

        # Classifier head
        x = nn.Dropout(rate=DROPOUT_PROB)(x, deterministic=not training)
        x = nn.Dense(features=512)(x)
        x = nn.gelu(x)
        logits = nn.Dense(features=1)(x)

        # Return the probabilities per image
        # Ojo! Apply sigmoid_binary_cross_entropy in the loss function 
        # for better numerical stability with mixed precision
        return logits
