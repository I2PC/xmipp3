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

CONSIDER_ANISOTROPY = True # Change to false to do P,P,P instead of Pz,Px,Py
PATCH_FULLVOLUME = 0
PATCH_PATCHES = 1
PATCH_OVERLAPPEDATCHES = 2
PATCH_DS_FULLVOLUME = 3
PATCH_OPTIONS = [PATCH_FULLVOLUME, PATCH_PATCHES, PATCH_OVERLAPPEDATCHES, PATCH_DS_FULLVOLUME]
BATCH_DECISION_SMALL_BOX = 96
BATCH_DECISION_MIN_BOX = 64
BATCH_DECISION_BSIZES = [8, 4, 2, 1]

class ParaManDPC3D:
    # NN parameters
    batch_size: int # N of images per batch
    base_channels: int # Number of base channels
    depth: int # Depth of the U-Net
    patch_size: tuple[int, int, int] # Patch size (Pz, Px, Py) if anisotropic or (P, P, P) if isotropic
    patch_kind: int # Patch kind (from PATCH_OPTIONS)
    patch_do_overlap: bool # Whether to do overlapping patches or not
    patch_overlap_fraction: float # Fraction of overlap between patches (0.0 to 0.2)
    
    # Data parameters
    ds_factor: int # Downsampling factor if any
    dtype_bytes: int # Bytes per data type (2 for FP16, 4 for FP32)
    in_box_size: tuple[int, int, int] # Input box size
    tightness: bool # Is the extraction box tight on the data?

    # GPU-related information
    gpu_alignment : int # Alignment for the GPU (8 or 16 bytes for Tensor cores)
    gpu_index: int # index of the GPU to use
    gpu_max_safe_vram_gb: float # Max safe VRAM in GB for the selected GPU

    def __init__(self, in_box_size: tuple[int, int, int], gpu_index: int = 0, tightness: bool = False) -> None:
        self.base_channels = 16
        self.depth = 4
        self.patch_size = in_box_size # Initialize to box size
        self.patch_do_overlap = False # By default, no overlap
        self.patch_overlap_fraction = 0.2

        self.ds_factor = 1
        self.dtype_bytes = 2  # FP16
        self.in_box_size = in_box_size
        self.tightness = tightness

        self.gpu_alignment = get_gpu_base_alignment(gpu_index)
        self.gpu_index = gpu_index
        self.gpu_max_safe_vram_gb = getSafeVramGbytes(device=gpu_index)

        self.adjust_model_to_gpu()      

    def adjust_model_to_gpu(self) -> None:
        print(f"Adjusting model to GPU {self.gpu_index} with max safe VRAM {self.gpu_max_safe_vram_gb:.2f} GB")
        print(f"Original input box size: {self.in_box_size}")
        
        # Find a proper batch size
        for B in BATCH_DECISION_BSIZES:
            candidates = generate_candidates(bbox=self.in_box_size, tight=self.tightness)
            # Try largest patches first
            for P in sorted(candidates, key=lambda x: x[0]*x[1]*x[2], reverse=True):
                if fits_vram(patch_size=P, batch_size=B, max_vram_gb=self.gpu_max_safe_vram_gb):
                    self.batch_size = B
                    self.patch_size = P
                    self.patch_kind = PATCH_OVERLAPPEDATCHES
            raise RuntimeError("No valid patch/batch configuration found!")
        
def getSafeVramGbytes(device: int = 0, safety_factor: float = 0.9) -> float:
    info = tf.config.experimental.get_device_details(
        tf.config.list_physical_devices('GPU')[device]
    )
    return info['memory_limit'] / (1024 ** 3) * safety_factor

def get_gpu_base_alignment(index: int) -> int:
    """
    Auto-detect GPU alignment based on its name.
    Will choose 8 if any older card is found, else 16 for Tensor cores.
    """
    gpus = tf.config.list_physical_devices(f'GPU')
    if not gpus:
        raise RuntimeError("No GPU devices found.")
    details = tf.config.experimental.get_device_details(gpus[index])
    name = details.get('device_name', '').lower()

    new_gpu_keywords = [
        'turing', 'ampere', 'hopper', 'ada', 'lovelace', 'blackwell', 't4', 'rtx'
    ]

    if any(k in name for k in new_gpu_keywords):
        print("Detected modern GPU architecture, using alignment 16.")
        return 16  # Safe and better for newer GPUs (Tensor cores)
    else:
        print("Detected older GPU architecture, using alignment 8.")
        return 8  # Older architectures  

def estimate_3d_vram_aniso(patch_size, base_channels, depth, batch_size, dtype_bytes=2, overhead_factor=1.25):
    """Estimate VRAM for a 3D encoder with anisotropic patches."""
    Pz, Px, Py = patch_size
    total_bytes = 0
    channels = base_channels
    d, h, w = Pz, Px, Py

    for level in range(depth):
        # Two convs per level
        for _ in range(2):
            voxels = d * h * w
            total_bytes += batch_size * voxels * channels * dtype_bytes * 2
        # Downsample
        d, h, w = d // 2, h // 2, w // 2
        channels *= 2

    # Bottleneck
    voxels = d * h * w
    total_bytes += batch_size * voxels * channels * dtype_bytes * 2

    total_bytes *= overhead_factor
    return total_bytes / (1024**3)  # GB

def ceil_to_multiple(x, base) -> int:
    return int(((x + base - 1) // base) * base)

def generate_candidates(bbox: tuple[int, int, int],
                        scales = (0.9, 0.8, 0.7),
                        min_patch = (32, 32, 32),
                        max_patch = (256, 256, 256),
                        alignment = (16, 16, 16),
                        tight = False
                        ) -> list[ tuple[int, int, int] ]:
    
    Dz, Dx, Dy = bbox
    candidates = set()
    if tight:
        candidates.add(bbox)
        return list(candidates)

    for s in scales:
        raw = (
            int(s * Dz),
            int(s * Dx),
            int(s * Dy),
        )
        aligned = tuple(
            ceil_to_multiple(
            max(min_patch[i], min(raw[i], max_patch[i])),
            alignment[i],
            ) 
            for i in range(3)
        )

        candidates.add(aligned)
    
    return list(candidates)

def fits_vram(
    patch_size: tuple[int,int,int],
    batch_size: int,
    max_vram_gb: float,
    base_channels: int = 16,
    depth: int = 4,
    dtype_bytes: int = 2,
    overhead_factor: float = 1.25,
    safety_factor: float = 0.85,
) -> bool:
    est = estimate_3d_vram_aniso(
        patch_size=patch_size,
        base_channels=base_channels,
        depth=depth,
        batch_size=batch_size,
        dtype_bytes=dtype_bytes,
        overhead_factor=overhead_factor,
    )
    return est <= max_vram_gb * safety_factor
