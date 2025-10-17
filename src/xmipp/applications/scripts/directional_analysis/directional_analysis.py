#!/usr/bin/env python3

# ***************************************************************************
# * Authors:     Oier Lauzirika Zarrabeitia (oierlauzi@bizkaia.eu)
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
# *  e-mail address 'xmipp@cnb.csic.es'
# ***************************************************************************/

import argparse
import os.path

import numpy as np
import pandas as pd
import torch
import collections
import itertools
import sklearn.decomposition

import xmippPyModules.swiftalign.image as image
import xmippPyModules.swiftalign.metadata as md
import xmippPyModules.swiftalign.operators as operators
import xmippPyModules.swiftalign.transform as transform

class CudaComputeContext:
    def __init__(self, device: torch.device):
        self.device = device
        self.compute_stream = torch.cuda.Stream(device=device)
        self.h2d_transfer_stream = torch.cuda.Stream(device=device)
        self.d2h_transfer_stream = torch.cuda.Stream(device=device)
    
class DirectionalTransformer:
    def __init__(self, 
                 interpolation: str = 'bilinear',
                 padding: str = 'zeros' ):
        self.interpolation = interpolation
        self.padding = padding
        self.flattener = None
    
    def setup(self,
              direction_angles: torch.Tensor,
              compute_context: CudaComputeContext,
              mask: torch.Tensor ):
        
        with torch.cuda.stream(compute_context.h2d_transfer_stream):
            mask = mask.to(compute_context.device, non_blocking=True)
        
        direction_angles = torch.deg2rad(direction_angles)
        direction_matrix = transform.euler_to_matrix(
            direction_angles[...,0], 
            direction_angles[...,1],
            direction_angles[...,2]
        )
        self.inv_direction_matrix = direction_matrix.mT
        
        compute_context.compute_stream.wait_stream(compute_context.h2d_transfer_stream)
        with torch.cuda.stream(compute_context.compute_stream):
            self.flattener = operators.MaskFlattener(mask)
    
    def transform(self,
                  images: torch.Tensor,
                  angles: torch.Tensor,
                  shifts: torch.Tensor,
                  compute_context: CudaComputeContext,
                  out: torch.Tensor = None ):
        if out is None:
            raise RuntimeError('Not implemented when out is None')
        
        with torch.cuda.stream(compute_context.h2d_transfer_stream):
            images = images.to(compute_context.device, non_blocking=True)
        
        centre = torch.tensor(images.shape[-2:]) / 2
        angles = torch.deg2rad(angles)
        pose = transform.euler_to_matrix(
            angles[...,0], 
            angles[...,1],
            angles[...,2]
        )
        pose = self.inv_direction_matrix @ pose
        
        matrices = torch.empty(
            images.shape[:2] + (2, 3),
            pin_memory=True,
            dtype=pose.dtype
        )
        matrices = transform.align_inplane(
            matrices_3d=pose,
            shifts=shifts,
            centre=centre,
            out=matrices
        )

        with torch.cuda.stream(compute_context.h2d_transfer_stream):
            matrices = matrices.to(compute_context.device, non_blocking=True)
            
        compute_context.compute_stream.wait_stream(compute_context.h2d_transfer_stream)
        with torch.cuda.stream(compute_context.compute_stream):
            images = transform.affine_2d(
                images=images, 
                matrices=matrices,
                interpolation=self.interpolation,
                padding=self.padding
            )
            
            result = self.flattener(images)
        
        compute_context.d2h_transfer_stream.wait_stream(compute_context.compute_stream)
        with torch.cuda.stream(compute_context.d2h_transfer_stream):
            out.copy_(result, non_blocking=True)
        
        return out

def preprocess_direction(direction_md: pd.DataFrame,
                         direction_angles: np.ndarray,
                         mask: torch.Tensor,
                         transformer: DirectionalTransformer,
                         compute_context: CudaComputeContext,
                         reader: image.CachingReader,
                         batch_size: int = 256):
    images_paths = list(map(image.parse_path, direction_md[md.IMAGE]))
    angles = direction_md[[md.ANGLE_ROT, md.ANGLE_TILT, md.ANGLE_PSI]].to_numpy()
    shifts = direction_md[[md.SHIFT_X, md.SHIFT_Y]].to_numpy()

    start = 0
    n_elements = len(direction_md)

    data = torch.empty(
        (n_elements, torch.count_nonzero(mask)), 
        device='cpu',
        pin_memory=True
    )
    transformer.setup(
        direction_angles=torch.as_tensor(direction_angles, dtype=torch.float32), 
        mask=mask,
        compute_context=compute_context
    )
    
    while start < n_elements:
        end = min(start + batch_size, n_elements)
        
        images = reader.read_batch(
            images_paths[start:end], 
            pin_memory=True,
            dtype=torch.float32
        )
        transformer.transform(
            images=images,
            angles=torch.as_tensor(angles[start:end], dtype=torch.float32),
            shifts=torch.as_tensor(shifts[start:end], dtype=torch.float32),
            compute_context=compute_context,
            out=data[start:end]
        )
        
        start = end
            
    return data

def save_direction(index: int, pca, data: np.ndarray, mask: np.ndarray, output_root: str):
    mean = pca.mean_.squeeze()
    average_image = np.zeros_like(mask, dtype=mean.dtype)
    average_image[mask] = mean
    
    image.write(average_image, os.path.join(output_root, f'{index:06d}_average.mrc'))
    
    components = pca.components_
    eigen_images = np.zeros((len(components), ) + mask.shape, dtype=components.dtype)
    eigen_images[:,mask] = components
    image.write(eigen_images, os.path.join(output_root, f'{index:06d}_eigen.mrcs'), image_stack=True)
    
    np.save(os.path.join(output_root, f'{index:06d}_data.npy'), data)

def run(directional_md_path: str, 
        output_root: str, 
        batch_size: int,
        device_names: list,
        n_components: int ):

    if device_names:
        devices = list(map(torch.device, device_names))
    else:
        devices = [torch.device('cpu')]
    
    n_in_flight_directions = 3*len(devices)
    compute_contexts = [CudaComputeContext(device) for device in devices]
    transformer = DirectionalTransformer()
    particle_reader = image.CachingReader()
    mask_reader = image.CachingReader()
    
    directional_md = md.read(directional_md_path)
    in_flight_directions = collections.deque()
    pca = sklearn.decomposition.PCA(n_components=n_components)
    for (i, row), compute_context in zip(directional_md.iterrows(), itertools.cycle(compute_contexts)):
        direction_md_path = row[md.SELFILE]
        mask_path = image.parse_path(row[md.MASK])
        direction_angles = row[[md.ANGLE_ROT, md.ANGLE_TILT, md.ANGLE_PSI]]
        direction_md = md.read(direction_md_path)
        mask = mask_reader.read(mask_path, pin_memory=True, dtype=torch.bool)
        
        data = preprocess_direction(
            direction_md=direction_md,
            direction_angles=direction_angles,
            mask=mask,
            transformer=transformer,
            compute_context=compute_context,
            reader=particle_reader,
            batch_size=batch_size
        )
        in_flight_directions.append((i+1, data, mask))
        
        if len(in_flight_directions) > n_in_flight_directions:
            index, data, mask = in_flight_directions.popleft()
            data = pca.fit_transform(data.numpy())
            save_direction(index, pca, data, mask.numpy(), output_root)
            
    while in_flight_directions:
        index, data, mask = in_flight_directions.popleft()
        data = pca.fit_transform(data.numpy())
        save_direction(index, pca, data, mask.numpy(), output_root)
        
if __name__ == '__main__':
    # Define the input
    parser = argparse.ArgumentParser(prog = 'Directionally analyze images')
    parser.add_argument('-i', required=True)
    parser.add_argument('-o', required=True)
    parser.add_argument('-k', default=8, type=int)
    parser.add_argument('--batch', type=int, default=256)
    parser.add_argument('--device', nargs='*')

    # Parse
    args = parser.parse_args()

    # Run the program
    run(
        directional_md_path= args.i,
        output_root = args.o,
        batch_size = args.batch,
        device_names = args.device,
        n_components=args.k
    )
