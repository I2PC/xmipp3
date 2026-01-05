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

try:
    import tensorflow_addons as tfa
    HAS_TFA = True
except ImportError:
    HAS_TFA = False

ARCH_REGISTRY = {}

def register_arch(name):
    def wrapper(fn):
        ARCH_REGISTRY[name] = fn
        return fn
    return wrapper


def norm_layer(norm, groups=8):
    if norm == "group":
        if not HAS_TFA:
            raise ImportError("GroupNorm requires tensorflow-addons")
        return tfa.layers.GroupNormalization(groups=groups)
    elif norm == "instance":
        if not HAS_TFA:
            raise ImportError("InstanceNorm requires tensorflow-addons")
        return tfa.layers.InstanceNormalization()
    else:
        raise ValueError(f"Unknown norm type: {norm}")

def conv_block(
    x,
    filters,
    kernel_size,
    norm="group",
    groups=8,
    activation="relu",
):
    x = tf.keras.layers.Conv3D(
        filters,
        kernel_size,
        padding="same",
        use_bias=False,
    )(x)
    x = norm_layer(norm, groups)(x)
    x = tf.keras.layers.Activation(activation)(x)
    return x

def channel_attention(x, reduction=8):
    channels = x.shape[-1]
    y = tf.keras.layers.GlobalAveragePooling3D()(x)
    y = tf.keras.layers.Dense(
        max(channels // reduction, 1),
        activation="relu",
    )(y)
    y = tf.keras.layers.Dense(channels, activation="sigmoid")(y)
    y = tf.keras.layers.Reshape((1, 1, 1, channels))(y)
    return tf.keras.layers.Multiply()([x, y])

def encoder_block(
    x,
    filters,
    anisotropic=True,
    norm="group",
    groups=8,
):
    if anisotropic:
        kernel = (1, 3, 3)
        pool = (1, 2, 2)
    else:
        kernel = (3, 3, 3)
        pool = (2, 2, 2)

    x = conv_block(x, filters, kernel, norm, groups)
    x = conv_block(x, filters, kernel, norm, groups)
    skip = x
    x = tf.keras.layers.MaxPool3D(pool_size=pool)(x)
    return x, skip

def decoder_block(
    x,
    skip,
    filters,
    anisotropic=True,
    norm="group",
    groups=8,
):
    if anisotropic:
        up = (1, 2, 2)
        kernel = (1, 3, 3)
    else:
        up = (2, 2, 2)
        kernel = (3, 3, 3)

    x = tf.keras.layers.UpSampling3D(size=up)(x)
    x = tf.keras.layers.Concatenate()([x, skip])
    x = conv_block(x, filters, kernel, norm, groups)
    x = conv_block(x, filters, kernel, norm, groups)
    return x


@register_arch("unet3d")
def build_unet_3d(
    input_shape,
    base_channels=16,
    depth=4,
    anisotropic_levels=2,
    norm="group",
    groups=8,
):
    """
    Cryo-ET specific UNet3D with anisotropic early layers and channel attention.

    Parameters
    ----------
    input_shape : tuple
        (Z, X, Y, C) or (None, None, None, C)
    base_channels : int
        Number of channels at first level
    depth : int
        Total UNet depth
    anisotropic_levels : int
        Number of encoder/decoder levels using anisotropic ops
    """

    inputs = tf.keras.Input(shape=input_shape)
    x = inputs
    skips = []

    # Encoder
    for d in range(depth):
        filters = base_channels * (2 ** d)
        anisotropic = d < anisotropic_levels
        x, skip = encoder_block(
            x,
            filters,
            anisotropic=anisotropic,
            norm=norm,
            groups=groups,
        )
        skips.append(skip)

    # Bottleneck
    filters = base_channels * (2 ** depth)
    x = conv_block(x, filters, (3, 3, 3), norm, groups)
    x = conv_block(x, filters, (3, 3, 3), norm, groups)
    x = channel_attention(x)

    # Decoder
    for d in reversed(range(depth)):
        filters = base_channels * (2 ** d)
        anisotropic = d < anisotropic_levels
        x = decoder_block(
            x,
            skips[d],
            filters,
            anisotropic=anisotropic,
            norm=norm,
            groups=groups,
        )

    # Output (force FP32 for numerical stability)
    outputs = tf.keras.layers.Conv3D(
        1,
        kernel_size=1,
        activation="sigmoid",
        dtype="float32",
    )(x)

    return tf.keras.Model(inputs, outputs, name="cryoet_unet3d")


