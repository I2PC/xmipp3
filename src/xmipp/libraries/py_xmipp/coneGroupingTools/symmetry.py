"""Generate proper rotations in the Xmipp symmetry coordinate conventions."""

from __future__ import annotations

import re

import numpy as np

SUPPORTED_SYMMETRIES = "Cn and Dn (n >= 1), T, O, I, and I1–I4"
_GOLDEN_RATIO = (1 + np.sqrt(5)) / 2
_SHORT = 1 / np.hypot(1, _GOLDEN_RATIO)
_LONG = _GOLDEN_RATIO * _SHORT
_X = (1, 0, 0)
_Z = (0, 0, 1)


def generate_cn_matrices(n: int) -> np.ndarray:
    """Return the cyclic rotations about Z."""
    angles = np.arange(n) * (2 * np.pi / n)
    cosine, sine = np.cos(angles), np.sin(angles)
    rotations = np.zeros((n, 3, 3))
    rotations[:, 0, 0] = rotations[:, 1, 1] = cosine
    rotations[:, 1, 0] = sine
    rotations[:, 0, 1] = -sine
    rotations[:, 2, 2] = 1
    return rotations


def generate_dn_matrices(n: int) -> np.ndarray:
    """Extend the cyclic group with a half-turn about X."""
    cyclic = generate_cn_matrices(n)
    half_turn = np.diag([1, -1, -1])
    return np.concatenate([cyclic, half_turn @ cyclic])


def _rotation(axis: tuple[float, float, float], fold: int) -> np.ndarray:
    """Build a single rotation by 2π/fold using Rodrigues' formula."""
    axis_vector = np.asarray(axis, dtype=np.float64)
    axis_vector /= np.linalg.norm(axis_vector)
    x, y, z = axis_vector
    cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    angle = 2 * np.pi / fold
    cosine, sine = np.cos(angle), np.sin(angle)
    return cosine * np.eye(3) + sine * cross + (1 - cosine) * np.outer(
        axis_vector, axis_vector
    )


def _generate_group(size: int, first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Expand products of the two generators in discovery order."""
    rotations = np.empty((size, 3, 3))
    rotations[0] = np.eye(3)
    count = 1
    for i in range(size):
        if i == count:
            raise RuntimeError(
                f"Symmetry generators produced {count} rotations, expected {size}."
            )
        for generator in (first, second):
            candidate = rotations[i] @ generator
            # Merge round-off copies of the same rotation.
            differences = np.max(np.abs(rotations[:count] - candidate), axis=(1, 2))
            if np.any(differences <= 1e-10):
                continue
            if count == size:
                raise RuntimeError(
                    f"Symmetry generators produced more than {size} rotations."
                )
            rotations[count] = candidate
            count += 1
    return rotations


# Axes follow XmippCore's fillSymmetryClass definitions. Use their analytic
# coordinates so products close without the error from rounded axis literals.
def generate_t_matrices() -> np.ndarray:
    return _generate_group(12, _rotation(_Z, 3), _rotation((0, np.sqrt(2), 1), 2))


def generate_o_matrices() -> np.ndarray:
    return _generate_group(24, _rotation((1, 1, 1), 3), _rotation(_Z, 4))


def generate_i1_matrices() -> np.ndarray:
    return _generate_group(60, _rotation(_X, 2), _rotation((_LONG, 0, -_SHORT), 5))


def generate_i2_matrices() -> np.ndarray:
    return _generate_group(60, _rotation(_Z, 2), _rotation((_SHORT, 0, _LONG), 5))


def generate_i3_matrices() -> np.ndarray:
    return _generate_group(60, _rotation((-_SHORT, 0, _LONG), 2), _rotation(_Z, 5))


def generate_i4_matrices() -> np.ndarray:
    return _generate_group(60, _rotation((_SHORT, 0, _LONG), 2), _rotation((2, 0, 1), 5))


_POLYHEDRAL_GENERATORS = {
    "T": generate_t_matrices,
    "O": generate_o_matrices,
    "I": generate_i2_matrices,
    "I1": generate_i1_matrices,
    "I2": generate_i2_matrices,
    "I3": generate_i3_matrices,
    "I4": generate_i4_matrices,
}


def parse_symmetry_group_string(sym: str | None) -> tuple[str, int | None]:
    """Return the normalized group and its order for cyclic or dihedral groups."""
    symbol = "C1" if sym is None else sym.strip().upper()
    if symbol in _POLYHEDRAL_GENERATORS:
        return symbol, None

    match = re.fullmatch(r"([CD])([0-9]+)", symbol)
    if match and (order := int(match[2])) > 0:
        return match[1], order

    raise ValueError(
        f"Unsupported symmetry group {sym!r}. Supported groups are {SUPPORTED_SYMMETRIES}."
    )


def generate_symmetries(sym: str | None) -> np.ndarray:
    """Return the right-hand rotations for E @ S, with identity first."""
    group, order = parse_symmetry_group_string(sym)
    if order is None:
        return _POLYHEDRAL_GENERATORS[group]()
    if group == "C":
        return generate_cn_matrices(order)
    return generate_dn_matrices(order)
