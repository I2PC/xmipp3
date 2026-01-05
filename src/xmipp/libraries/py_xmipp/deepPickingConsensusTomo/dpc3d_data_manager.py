#!/usr/bin/env python3
""""
**************************************************************************
*
* Authors:  Mikel Iceta Tena (miceta@cnb.csic.es)
* 
*
* Unidad de  Bioinformatica of Centro Nacional de Biotecnologia , CSIC
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
*  e-mail address 'scipion@cnb.csic.es'
*
* Initial version: jan 2025
**************************************************************************
"""

import tensorflow as tf
import numpy as np

class DataManDPC3D:
    def __init__(self, patch_size, overlap, batch_size, augment, noise_std)-> None:
        self.patch_size = patch_size
        self.overlap = overlap
        self.batch_size = batch_size

        self.augment = augment
        self.noise_std = noise_std
        self._rng = np.random.default_rng()

    # ---- Training ----

    def get_train_dataset(self, volumes, labels):
        ds = tf.data.Dataset.from_generator(
            lambda: self.train_generator(volumes, labels),
            output_signature=(
                tf.TensorSpec(
                    shape=(*self.patch_size, 1),
                    dtype=tf.float32,
                ),
                tf.TensorSpec(
                    shape=(),
                    dtype=tf.float32,
                ),
            ),
        )

        ds = ds.batch(self.batch_size)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds

    def train_generator(self, volumes, labels):
        while True:
            idx = self._rng.integers(len(volumes))
            vol = self.normalize(volumes[idx])
            patch = self.random_patch(vol)
            label = labels[idx]

            if self.augment:
                patch = self.augment_patch(patch)

            yield patch[..., None], label


    def get_val_dataset(self):
        pass
    

    # ---- Inference ----
    def generate_inference_patches(self, volume):
        ...

    def blend_patches(self, predictions, locations, volume_shape):
        ...
    
    # ---- Augmentation and image operations ----
    def normalize(self, vol):
        mean = np.mean(vol)
        std = np.std(vol) + 1e-6
        return (vol - mean) / std
    
    def augment_patch(self, vol):
        # Random flips
        for axis in (0, 1, 2):
            if self._rng.random() < 0.5:
                vol = np.flip(vol, axis=axis)

        # Random 90deg rots in XY
        k = self._rng.integers(0, 4)
        vol = np.rot90(vol, k, axes=(1, 2))

        # Additive noise
        if self.noise_std > 0:
            vol = vol + self._rng.normal(0, self.noise_std, vol.shape)

        return vol
        
    def random_patch(self, vol):
            z, x, y = vol.shape
            pz, px, py = self.patch_size

            z0 = self._rng.integers(0, z - pz + 1)
            x0 = self._rng.integers(0, x - px + 1)
            y0 = self._rng.integers(0, y - py + 1)

            patch = vol[z0:z0+pz, x0:x0+px, y0:y0+py]

            return patch

def generate_patch_grid(volume_shape, patch_size, overlap):
    steps = []
    for dim, p, o in zip(volume_shape, patch_size, overlap):
        step = p - o
        coords = list(range(0, max(dim - p + 1, 1), step))
        if coords[-1] + p < dim:
            coords.append(dim - p)
        steps.append(coords)

    grid = []
    for z in steps[0]:
        for x in steps[1]:
            for y in steps[2]:
                grid.append((z, x, y))
    return grid

def extract_patch(volume, start, patch_size):
    z, x, y = start
    pz, px, py = patch_size
    return volume[z:z+pz, x:x+px, y:y+py]

def hann_window_3d(shape):
    wz = np.hanning(shape[0])
    wx = np.hanning(shape[1])
    wy = np.hanning(shape[2])
    return wz[:, None, None] * wx[None, :, None] * wy[None, None, :]

def blend_patches(preds, coords, volume_shape):
    output = np.zeros(volume_shape, dtype=np.float32)
    weight = np.zeros(volume_shape, dtype=np.float32)

    window = hann_window_3d(preds[0].shape)

    for pred, (z, x, y) in zip(preds, coords):
        output[z:z+pred.shape[0],
               x:x+pred.shape[1],
               y:y+pred.shape[2]] += pred * window
        weight[z:z+pred.shape[0],
               x:x+pred.shape[1],
               y:y+pred.shape[2]] += window

    return output / np.maximum(weight, 1e-6)

def patch_generator(volumes, masks, patch_size):
    while True:
        idx = np.random.randint(len(volumes))
        vol = volumes[idx]
        mask = masks[idx]
        patch = random_patch(vol, patch_size)
        label = random_patch(mask, patch_size)
        yield patch[..., None], label[..., None]


# For augmentation:
# Random flips (x,y,z)
# random 90º rots in XY
# additive gaussian noise
# contrast jitter (small)