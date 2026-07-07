import torch
import torch.nn.functional as F


def fourier_shift_batch(
    imgs: torch.Tensor, shift_x: torch.Tensor, shift_y: torch.Tensor
) -> torch.Tensor:
    """
    Apply subpixel shifts to a batch of 2D images using the Fourier shift theorem.

    Parameters
    ----------
    imgs : torch.Tensor
        Input tensor of shape (N, H, W) containing a batch of 2D images.
    shift_x : torch.Tensor
        Tensor of shape (N,) with translations along the X axis (columns),
        expressed in pixels.
    shift_y : torch.Tensor
        Tensor of shape (N,) with translations along the Y axis (rows),
        expressed in pixels.

    Returns
    -------
    torch.Tensor
        Tensor of shape (N, H, W) containing the shifted images.

    Notes
    -----
    The shifts are applied in the Fourier domain by multiplying the real FFT
    of each image by a complex phase factor. This enables subpixel-accurate
    translations without interpolation artifacts.

    A real-to-complex FFT (`rfft2`) is used for efficiency, exploiting the
    Hermitian symmetry of real-valued inputs. The shifted images are recovered
    via the inverse real FFT (`irfft2`).
    """
    n, h, w = imgs.shape

    # Frequency coordinates
    ky = torch.fft.fftfreq(h, d=1.0, device=imgs.device).reshape(1, h, 1)
    kx = torch.fft.rfftfreq(w, d=1.0, device=imgs.device).reshape(1, 1, w // 2 + 1)

    # Expand shifts
    sx = shift_x.view(n, 1, 1)
    sy = shift_y.view(n, 1, 1)

    # Calculate phase and shift images
    phase = torch.exp(-2j * torch.pi * (kx * sx + ky * sy))
    fourier_images = torch.fft.rfft2(imgs)
    fourier_images.mul_(phase)
    del phase

    # Return real space images
    return torch.fft.irfft2(fourier_images, s=(h, w))


def rotate_batch(imgs: torch.Tensor, angles: torch.Tensor):
    n, h, w = imgs.shape

    # Build rotation matrices
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    zeros = torch.zeros_like(cos)

    rot_mats = torch.stack(
        [
            torch.stack([cos, -sin, zeros], dim=1),
            torch.stack([sin, cos, zeros], dim=1),
        ],
        dim=1,
    )

    # Build affine rotation grid for grid_sample
    grids = F.affine_grid(rot_mats, size=(n, 1, h, w), align_corners=True)

    # Apply rotation through sampling
    return F.grid_sample(
        imgs.unsqueeze(1), grids, align_corners=True, padding_mode="zeros"
    ).squeeze(1)


def align_particles_batch(
    particles: torch.Tensor,
    psi: torch.Tensor,
    shiftX: torch.Tensor,
    shiftY: torch.Tensor,
    batch_size: int = 256,
    inplace: bool = True,
    shift_first: bool = True,
):
    """
    Aligns a set of particles using batched Fourier shifts and spatial rotations.

    The alignment consists of:
    - Subpixel translations applied in Fourier space.
    - In-plane rotations applied via grid sampling.

    Parameters
    ----------
    particles : torch.Tensor
        Tensor of shape (N, H, W) containing the unaligned particle images.
    psi : torch.Tensor
        Tensor of shape (N,) containing in-plane rotation angles (in radians).
    shiftX : torch.Tensor
        Tensor of shape (N,) containing X shifts.
    shiftY : torch.Tensor
        Tensor of shape (N,) containing Y shifts.
    batch_size : int, optional
        Number of particles processed per batch, by default 256.
    inplace : bool, optional
        If True, overwrites the input `particles` tensor to save memory.
        If False, allocates a new tensor for the aligned output.
        Default is True.
    shift_first: bool, optional
        Convention followed for the alignment of the particles: whether
        the shifts are applied before or after the rotations. Default is True.
        - `shift_first = True` corresponds to RELION's alignment conventions.
        - `shift_first = False` corresponds to XMIPP's alignment conventions.

    Returns
    -------
    torch.Tensor
        Tensor of shape (N, H, W) containing the aligned particle images.

    Notes
    -----
    - Translations are applied using the Fourier shift theorem for
      subpixel accuracy.
    - Rotations are applied using `torch.nn.functional.grid_sample`.
    - The coordinate grid is defined in the range [-1, 1] with
      `align_corners=True`.
    """
    n = particles.size(0)

    # Initialize aligned images tensor
    if inplace:
        aligned = particles
    else:
        aligned = torch.empty_like(particles)

    # Process particles in batches
    for i in range(0, n, batch_size):
        j = min(i + batch_size, n)

        # Read batch data
        batch = particles[i:j]
        batch_shx = shiftX[i:j]
        batch_shy = shiftY[i:j]
        batch_ang = psi[i:j]

        if shift_first:
            # Shift first, then rotate (RELION convention)
            shifted = fourier_shift_batch(batch, batch_shx, batch_shy)
            aligned_batch = rotate_batch(shifted, batch_ang)
        else:
            # Rotate first, then shift (XMIPP convention)
            rotated = rotate_batch(batch, batch_ang)
            aligned_batch = fourier_shift_batch(rotated, batch_shx, batch_shy)

        # Save rotated images to aligned tensor
        aligned[i:j] = aligned_batch

    return aligned
