#!/usr/bin/env python

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

from typing import Optional, Sequence

import torch
import argparse
import math
import pandas as pd

import xmippPyModules.swiftalign.image as image
import xmippPyModules.swiftalign.metadata as md
import xmippPyModules.swiftalign.alignment as alignment
import xmippPyModules.swiftalign.classification as classification
import xmippPyModules.swiftalign.operators as operators
import xmippPyModules.swiftalign.transform as transform

def _dataframe_batch_generator(df: pd.DataFrame, batch_size: int) -> pd.DataFrame:
    for i in range(0, len(df), batch_size):
        start = i
        end = start + batch_size
        yield df[start:end]
        
def run(images_md_path: str, 
        output_root: str, 
        scratch_path: Optional[str],
        mask_path: Optional[str],
        align_to: Optional[Sequence[int]],
        batch_size: int,
        device_names: list ):
    
    # Devices
    if device_names:
        devices = list(map(torch.device, device_names))
    else:
        devices = [torch.device('cpu')]
    transform_device = devices[0]
    pin_memory = transform_device.type=='cuda'
    
    # Read input files
    images_md = md.sort_by_image_filename(md.read(images_md_path))
    images_paths = list(map(image.parse_path, images_md[md.IMAGE]))
    images_dataset = image.torch_utils.Dataset(images_paths)
    images_loader = torch.utils.data.DataLoader(
        images_dataset,
        batch_size=batch_size,
        drop_last=False,
        pin_memory=pin_memory,
        num_workers=1
    )
    image_size = md.get_image2d_size(images_md)
    
    # Read the mask
    if mask_path is not None:
        mask_path = image.parse_path(mask_path)
        mask = image.read(mask_path.filename)
        if mask_path.position_in_stack is not None:
            mask = mask[mask_path.position_in_stack-1]
        mask = torch.as_tensor(mask, dtype=bool)
    else:
        mask = torch.ones(image_size, dtype=bool)
    
    # Compute the alignment matrix
    align_to_quaternion = None
    if align_to is not None:
        align_to_quaternion = transform.euler_to_quaternion(
            torch.tensor(math.radians(float(align_to[0]))),
            torch.tensor(math.radians(float(align_to[1]))),
            torch.tensor(math.radians(float(align_to[2])))
        )
    
    # Create the flattener and transformer
    flattener = operators.MaskFlattener(
        mask=mask,
        device=transform_device
    )
    image_transformer = alignment.InPlaneTransformCorrector(
        flattener=flattener,
        align_to_quaternion=align_to_quaternion,
        device=transform_device
    )

    # Create the storage for the training set.
    # This will be LARGE. Therefore provide a MMAP path
    training_set_shape = (len(images_md), flattener.get_length())
    if scratch_path is not None:
        size = math.prod(training_set_shape)
        storage = torch.FloatStorage.from_file(scratch_path, shared=True, size=size)
        scratch = torch.FloatTensor(storage=storage)
        scratch = scratch.view(training_set_shape)
    else:
        scratch = torch.empty(training_set_shape, device=transform_device)

    average, direction, projections = classification.aligned_2d_classification(
        image_transformer(zip(images_loader, _dataframe_batch_generator(images_md, batch_size))),
        scratch
    )

    # Write average
    output_average_path = output_root + 'average.mrc'
    output_average = torch.zeros(mask.shape, dtype=average.dtype)
    output_average[mask] = average.to(output_average.device)
    image.write(output_average.numpy(), output_average_path)
    
    # Write direction
    output_direction_path = output_root + 'eigen_image.mrc'
    output_direction = torch.zeros(mask.shape, dtype=direction.dtype)
    output_direction[mask] = direction.to(output_direction.device)
    image.write(output_direction.numpy(), output_direction_path)
    
    # Write metadata
    output_particles_md_path = output_root + 'classification.xmd'
    images_md[md.SCORE_BY_PCA_RESIDUAL] = projections.cpu().numpy()
    md.write(images_md, output_particles_md_path)

if __name__ == '__main__':
    # Define the input
    parser = argparse.ArgumentParser(
                        prog = 'Align Nearest Neighbor Training',
                        description = 'Align Cryo-EM images using a fast Nearest Neighbor approach')
    parser.add_argument('-i', required=True)
    parser.add_argument('-o', required=True)
    parser.add_argument('--scratch')
    parser.add_argument('--mask')
    parser.add_argument('--align_to', nargs=3)
    parser.add_argument('--batch', type=int, default=1024)
    parser.add_argument('--device', nargs='*')

    # Parse
    args = parser.parse_args()

    # Run the program
    run(
        images_md_path = args.i,
        output_root = args.o,
        scratch_path = args.scratch,
        mask_path = args.mask,
        align_to = args.align_to,
        batch_size = args.batch,
        device_names = args.device
    )