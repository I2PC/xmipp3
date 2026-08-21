from pathlib import Path
from typing import Tuple, Union, Iterable, Optional

import numpy as np
import mrcfile
import starfile
import pandas as pd
import torch

MDL_REF_COLUMN = "ref"

def read_images(
    data: pd.DataFrame,
    class_id: Optional[int] = None,
    device: Union[torch.device, str] = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Reads a metadata DataFrame and returns the requested images as a tensor.

    Parameters
    ----------
    data: pd.DataFrame
        DataFrame containing the images metadata. 
        Must contain the field `"image"` with a path to the location of each 
        image in the image stack, in the format ``"index@path/to/stack.mrcs"``, 
        with 1-based indices.

    class_id: int or None, optional
        Class ID identifying the images that should be read. If set to None,
        all the images in ``data`` will be read. Default is None.

    device: torch.device or str
        Device where the output tensors should be placed.

    Returns
    -------
    torch.Tensor
        Images loaded in the specified device.
    """
    if class_id is None:
        used_data = data
    else:
        used_data = data[data[MDL_REF_COLUMN] == class_id]

    image_names = used_data["image"].to_numpy()

    # Image names have the form "000001@Runs/path-to-images.mrcs", with 1-based indices
    indices = np.array(
        [int(name.split("@", maxsplit=1)[0]) - 1 for name in image_names],
        dtype=np.int64,
    )

    stack_name = str(image_names[0]).split("@", maxsplit=1)[1]
    stack_path = Path(stack_name)

    # Memory-map the stack instead of loading the full .mrcs into RAM.
    with mrcfile.mmap(stack_path, permissive=True, mode="r") as mrc:
        particles_np = np.asarray(mrc.data[indices], dtype=np.float32)

    # Convert everything to tensors
    particles = torch.as_tensor(particles_np, dtype=torch.float32, device=device)

    return particles


def write_star_with_weights(
    input_star: Path,
    output_star: Path,
    weights_list: Iterable[np.ndarray],
    column_names: Iterable[str] = ["RobustWeight"],
) -> None:
    """
    Reads the input .star or .xmd file and saves a .star file with an extra column
    containing the weights
    """
    for weights in weights_list:
        if weights.ndim != 1:
            raise ValueError(f"Expected one-dimensional weights, got {weights.ndim = }")

    data = starfile.read(input_star)

    for weights, column_name in zip(weights_list, column_names):
        if len(weights) != len(data):
            raise ValueError(
                f"Number of weights ({len(weights)}) does not match number of particles "
                f"({len(data)})."
            )
        data[column_name] = weights

    starfile.write(data, output_star)
