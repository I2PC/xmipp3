from pathlib import Path

import mrcfile
import numpy as np
import pandas as pd
import torch

MDL_REF_COLUMN = "ref"
MDL_ITEM_ID_COLUMN = "itemId"


def read_images(data: pd.DataFrame, device: torch.device | str = "cpu") -> torch.Tensor:
    """
    Reads a metadata DataFrame and returns the requested images as a tensor.

    Parameters
    ----------
    data: pd.DataFrame
        DataFrame containing the images metadata.
        Must contain the field `"image"` with a path to the location of each
        image in the image stack, in the format ``"index@path/to/stack.mrcs"``,
        with 1-based indices.

    device: torch.device or str
        Device where the output tensors should be placed.

    Returns
    -------
    torch.Tensor
        Tensor of shape ``(n_images, height, width)`` on the specified device,
        containing the particle images.
    """
    image_names = data["image"].to_numpy()

    # Image names have the form "000001@Runs/path-to-images.mrcs", with 1-based indices
    indices = np.array(
        [int(name.split("@", maxsplit=1)[0]) - 1 for name in image_names],
        dtype=np.int64,
    )

    stack_name = str(image_names[0]).split("@", maxsplit=1)[1]
    stack_path = Path(stack_name)

    # Memory-map the stack instead of loading the full .mrcs into RAM.
    with mrcfile.mmap(stack_path, permissive=True, mode="r") as mrc:
        stack = mrc.data
        if stack.ndim == 2:
            stack = stack[np.newaxis, ...]
        particles_np = np.asarray(stack[indices], dtype=np.float32)

    # Convert everything to tensors
    particles = torch.as_tensor(particles_np, dtype=torch.float32, device=device)

    return particles
