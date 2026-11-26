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
        with 1-based indices. Images can come from multiple stacks that
        share the same image dimensions.

    device: torch.device or str
        Device where the output tensors should be placed.

    Returns
    -------
    torch.Tensor
        Tensor of shape ``(n_images, height, width)`` on the specified device,
        containing the particle images in metadata row order.
    """
    image_names = data["image"].astype(str)
    if image_names.empty:
        raise ValueError("No particle images were requested.")

    references = image_names.str.partition("@")
    references.columns = ["index", "separator", "path"]
    try:
        image_indices = references["index"].map(int)
    except ValueError as exc:
        raise ValueError(f"Invalid particle image index: {exc}") from exc
    invalid = references["path"].eq("") | image_indices.lt(1)
    if invalid.any():
        raise ValueError(
            f"Invalid particle image reference {image_names[invalid].iloc[0]!r}: "
            "expected a 1-based index and a stack path."
        )

    indices = image_indices.to_numpy()
    positions_by_stack = references.groupby("path", sort=False).indices
    particles_np = None
    for stack_path, positions in positions_by_stack.items():
        stack_indices = indices[positions]
        # Materialize only the requested images while the stack is mapped.
        with mrcfile.mmap(stack_path, permissive=True, mode="r") as mrc:
            stack = mrc.data
            if stack.ndim == 2:
                stack = stack[np.newaxis, ...]
            max_index = stack_indices.max()
            if max_index > len(stack):
                raise ValueError(
                    f"Particle image index {max_index} exceeds the "
                    f"{len(stack)} images in stack {stack_path}."
                )
            selected = np.asarray(
                stack[stack_indices.astype(np.intp, copy=False) - 1], dtype=np.float32
            )

        if len(positions_by_stack) == 1:
            particles_np = selected
            break

        if particles_np is None:
            particles_np = np.empty(
                (len(image_names), *selected.shape[1:]), dtype=np.float32
            )
        elif selected.shape[1:] != particles_np.shape[1:]:
            raise ValueError(
                f"Stack {stack_path} has image shape {selected.shape[1:]}, "
                f"expected {particles_np.shape[1:]}."
            )
        particles_np[positions] = selected

    return torch.as_tensor(particles_np, dtype=torch.float32, device=device)
