import math
from pathlib import Path

import numpy as np
import torch as th
from md_et.nn.dist import assert_dtype, set_dtype
from loguru import logger
from scipy.spatial.transform import Rotation
from torch.nn import functional as F


def read_quat_file(
    filepath: Path, device: str = "cpu", dtype: th.dtype | None = None
) -> th.Tensor:
    quaternions = []
    dtype = set_dtype(dtype)
    with filepath.open("r") as f:
        for line in f:
            l = line.strip()
            if l.startswith("#") or not l:
                continue
            if l.startswith("format"):
                continue
            parts = l.split()
            if len(parts) >= 4:
                try:
                    w, x, y, z = map(float, parts[:4])
                    quaternions.append([w, x, y, z])
                except ValueError:
                    continue

    if not quaternions:
        raise ValueError("No valid quaternions found in file")
    quaternions = th.tensor(quaternions, dtype=dtype)
    scipy_quats = quaternions[:, [1, 2, 3, 0]]
    return quads_to_matrices(scipy_quats).to(device)


def abs_pairwise_cosine_distance(quads: th.Tensor) -> th.Tensor:
    quads_pair_idx = th.combinations(
        th.arange(quads.shape[0]), 2, with_replacement=False
    )
    abs_cos = th.arccos(
        th.abs(
            F.cosine_similarity(
                quads[quads_pair_idx[:, 0]], quads[quads_pair_idx[:, 1]], dim=1
            )
        )
    )
    return abs_cos


def optimize_approx_equidistant_rotations(
    n: int,
    n_steps: int = 4000,
    lr: float = 0.005,
    fp_type: th.dtype | None = None,
    device: str = "cpu",
) -> th.Tensor:  # (n, 3, 3)
    random_quads = th.randn(n, 4, device=device)
    random_quads = random_quads / th.linalg.norm(random_quads, axis=1, keepdims=True)

    def objective(quads: th.Tensor) -> th.Tensor:
        dists = abs_pairwise_cosine_distance(quads)
        return -th.log(dists).sum()

    random_quads = random_quads.requires_grad_()
    optimizer = th.optim.Adam([random_quads], lr=lr)
    for i in range(n_steps):
        optimizer.zero_grad()
        random_quads.data = random_quads.data / th.linalg.norm(
            random_quads.data, axis=1, keepdims=True
        )
        loss = objective(random_quads)
        loss.backward()
        optimizer.step()
        if i % 400 == 0:
            logger.info(f"Step {i} loss: {loss.item()}")

    fp_type = set_dtype(fp_type)
    optimized_quads = (
        (random_quads / th.linalg.norm(random_quads, axis=1, keepdims=True))
        .detach()
        .to(fp_type)
    )
    matrices = quads_to_matrices(optimized_quads)
    return matrices


def get_super_fibonacci_spiral_rotations(
    n: int, device="cpu", dtype: th.dtype | None = None
) -> th.Tensor:
    assert n > 1, "n must be greater than 1"
    if n < 64:
        logger.warning(
            "for small n less than 64, it is recommended to use optimized_approx_equidistant_rotations"
        )
    dtype = set_dtype(dtype)
    phi = 1.533751168755204288118041
    theta = 2**0.5
    quads = th.zeros((n, 4), dtype=dtype)
    for i in range(n):
        s = i + 0.5
        t = s / n
        d = 2 * th.pi * s
        r = t**0.5
        R = (1 - t) ** 0.5
        alpha = d / theta
        beta = d / phi
        quads[i] = th.tensor(
            [
                r * math.sin(alpha),
                r * math.cos(alpha),
                R * math.sin(beta),
                R * math.cos(beta),
            ],
            dtype=dtype,
        )
    matrices = quads_to_matrices(quads)
    return matrices.to(device)


def quads_to_matrices(quads: th.Tensor) -> th.Tensor:
    return th.tensor(
        Rotation.from_quat(quads.cpu().numpy()).as_matrix(), dtype=quads.dtype
    ).to(quads.device)


def matrices_to_quads(matrices: th.Tensor) -> th.Tensor:
    return th.tensor(
        Rotation.from_matrix(matrices.cpu().numpy()).as_quat(), dtype=matrices.dtype
    ).to(matrices.device)


def get_approximate_equidistant_rotation_cover(
    n: int, device="cpu", dtype: th.dtype | None = None
) -> th.Tensor:
    dtype = set_dtype(dtype)
    assert n > 1, "n must be greater than 1"
    if n < 64:
        return optimize_approx_equidistant_rotations(n, device=device, fp_type=dtype)
    else:
        return get_super_fibonacci_spiral_rotations(n, device, dtype)


def get_random_rotations(n_samples, device, dtype: th.dtype | None = None) -> th.Tensor:
    R = Rotation.random(
        n_samples,
    ).as_matrix()
    dtype = set_dtype(dtype)
    R = th.tensor(R, dtype=dtype)
    return R.to(device)


def get_random_reflections(
    n_samples, device, reflection_share=0.5, eps=1e-9, dtype: th.dtype | None = None
) -> th.Tensor:
    dtype = set_dtype(dtype)
    # get random normal vectors
    normals = th.randn(n_samples, 3, dtype=dtype).to(device)  # (n_samples, 3)
    normals = normals / (th.norm(normals, dim=1, keepdim=True) + eps)  # (n_samples, 3)

    # get householder matrix
    normals = normals.unsqueeze(2)
    outer = th.matmul(normals, normals.transpose(1, 2))  # (n_samples, 3, 3)
    identity = th.eye(3, dtype=normals.dtype, device=normals.device).unsqueeze(
        0
    )  # (1, 3, 3)
    householder = identity.repeat(n_samples, 1, 1)  # (n_samples, 3, 3)
    # selectively reflect
    sample_mask = th.rand(n_samples) < reflection_share
    householder[sample_mask] -= 2 * outer[sample_mask]
    assert_dtype(householder, dtype)
    return householder


def get_grid_rotations(
    grid_size: int, device, dtype: th.dtype | None = None
) -> th.Tensor:
    dtype = set_dtype(dtype)
    pitch = th.linspace(0, 2 * th.pi, grid_size)
    roll = th.linspace(0, 2 * th.pi, grid_size)
    yaw = th.linspace(0, 2 * th.pi, grid_size)
    pitch, roll, yaw = th.meshgrid(pitch, roll, yaw)
    pitch = pitch.flatten()
    roll = roll.flatten()
    yaw = yaw.flatten()
    angles = th.stack([yaw, pitch, roll], dim=-1)  # (grid_size^3, 3)
    R = Rotation.from_euler(
        "xyz", angles, degrees=False
    ).as_matrix()  # (grid_size^3, 3, 3)
    return th.tensor(R, dtype=dtype).to(device)
