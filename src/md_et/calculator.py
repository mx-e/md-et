"""ASE calculator interface for MD-ET models."""

from pathlib import Path
from functools import partial
from typing import Literal

import numpy as np
import torch as th
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from hydra_zen import instantiate, load_from_yaml
from loguru import logger

from md_et.nn.checkpoint import load_checkpoint
from md_et.nn.types import Property as Props
from md_et.nn.filters import remove_net_force, remove_net_torque
from md_et.nn.loaders import batch_tall, collate_fn
from md_et.nn.transforms import center_positions_on_centroid
from md_et.constants import (
    ANG_TO_BOHR,
    HARTREE_TO_EV,
    HARTREE_BOHR_TO_EV_ANG,
    HARTREE_BOHR2_TO_EV_ANG2,
)


def _make_collate_fn(device: str):
    """Create a collate function for model input preparation."""
    return partial(
        collate_fn,
        device=device,
        batch_func=batch_tall,
        pre_batch_preprocessors=[center_positions_on_centroid],
        props={
            Props.positions: "positions",
            Props.atomic_numbers: "atomic_numbers",
            Props.charge: "charge",
            Props.multiplicity: "multiplicity",
        },
    )


def _load_model(run_dir_path, device=None, checkpoint_name="best_model"):
    """
    Load a PairEncoder model from a training run checkpoint.

    Args:
        run_dir_path: Path to the training run directory containing
            .hydra/config.yaml and ckpts/
        device: Target device (default: cuda if available, else cpu)
        checkpoint_name: Name of checkpoint file without .pth extension

    Returns:
        Loaded model in eval mode
    """
    if device is None:
        device = "cuda" if th.cuda.is_available() else "cpu"
    th.set_float32_matmul_precision("high")
    run_dir_path = Path(run_dir_path)
    model_run_conf_path = run_dir_path / ".hydra" / "config.yaml"
    model_run_conf = load_from_yaml(model_run_conf_path)
    model_conf = model_run_conf["train"]["model"]
    model_conf["_target_"] = "md_et.nn.pair_encoder.PairEncoder"
    model = instantiate(model_conf)
    checkpoint_path = run_dir_path / "ckpts" / (checkpoint_name + ".pth")
    load_checkpoint(model, checkpoint_path)
    model.eval().to(device)
    return model


def load_calculator(
    run_dir_path: str | Path,
    device: str | None = None,
    checkpoint_name: str = "best_model",
    filter_forces: bool = True,
) -> "MDETCalculator":
    """
    Load an MD-ET calculator from a training run checkpoint.

    This is the main entry point for using MD-ET models. It loads the model
    architecture from the Hydra config and the weights from the checkpoint,
    and wraps everything in an ASE-compatible calculator.

    Args:
        run_dir_path: Path to the training run directory. Must contain:
            - .hydra/config.yaml (model architecture config)
            - ckpts/{checkpoint_name}.pth (model weights)
        device: Torch device string (e.g. "cuda", "cpu").
            Default: "cuda" if available, else "cpu".
        checkpoint_name: Name of checkpoint file without .pth extension.
            Default: "best_model".
        filter_forces: Whether to apply force filtering (remove net force/torque).
            Use True for MD simulations (momentum conservation).
            Use False for geometry optimization and Hessian computation.

    Returns:
        Configured MDETCalculator instance.

    Example::

        from md_et import load_calculator

        calc = load_calculator("path/to/training_run")
        atoms.calc = calc
        energy = atoms.get_potential_energy()  # eV
        forces = atoms.get_forces()            # eV/A
    """
    if device is None:
        device = "cuda" if th.cuda.is_available() else "cpu"
    model = _load_model(Path(run_dir_path), device, checkpoint_name)
    return MDETCalculator(model, device, filter_forces=filter_forces)


class MDETCalculator(Calculator):
    """
    ASE Calculator for the MD-ET (PairEncoder) ML force field.

    Computes energies, forces, and Hessians for molecular systems.

    Units:
        - Input positions: Angstrom (standard ASE)
        - Output energy: eV
        - Output forces: eV/Angstrom
        - Output Hessian: eV/Angstrom^2
    """

    implemented_properties = ["energy", "forces"]

    def __init__(
        self,
        model,
        device: str,
        filter_forces: bool = True,
    ):
        """
        Args:
            model: The PairEncoder model instance.
            device: Torch device string.
            filter_forces: If True, remove net force and net torque from
                predicted forces (for momentum conservation during MD).
                Set to False for geometry optimization or Hessian computation.
        """
        super().__init__()
        self.model = model
        self.device = device
        self.filter_forces = filter_forces
        self.model.eval()
        self.last_hessian_antisym_ratio: float | None = None
        self.collate_fn = _make_collate_fn(device)

    def convert_atoms_to_model_input(self, atoms: Atoms) -> dict:
        """Convert ASE Atoms to model input dict."""
        positions_bohr = atoms.positions * ANG_TO_BOHR
        data = {
            "positions": positions_bohr,
            "atomic_numbers": atoms.numbers,
            "charge": atoms.info.get("charge", 0),
            "multiplicity": 1,
        }
        return self.collate_fn([data])

    def calculate(
        self, atoms=None, properties=None, system_changes=all_changes
    ) -> None:
        if properties is None:
            properties = ["forces", "energy"]
        Calculator.calculate(self, atoms, properties, system_changes)

        data = self.convert_atoms_to_model_input(atoms)
        positions = data[Props.positions]
        positions.requires_grad_(True)

        outputs = self.model(data)

        # Get energy
        if Props.formation_energy in outputs or Props.energy in outputs:
            energy = outputs.get(Props.formation_energy, outputs.get(Props.energy))
            energy_ev = energy.squeeze() * HARTREE_TO_EV
        else:
            energy_ev = th.tensor(0.0, device=self.device)

        # Get forces
        forces = outputs[Props.forces]

        if self.filter_forces:
            n_nodes = (data[Props.atomic_numbers] != 0).sum(dim=1)
            forces = remove_net_force(forces, n_nodes)
            forces = remove_net_torque(positions, forces, n_nodes)

        forces = forces.squeeze() * HARTREE_BOHR_TO_EV_ANG

        forces_np = forces.squeeze().detach().cpu().numpy()
        energy_val = energy_ev.detach().cpu().item()

        # Validate outputs
        if not np.isfinite(forces_np).all():
            pos = atoms.positions
            logger.warning(
                f"MDETCalculator: forces contain NaN/inf! "
                f"n_atoms={len(atoms)}, positions range: [{pos.min():.2f}, {pos.max():.2f}]"
            )
        if not np.isfinite(energy_val):
            logger.warning(f"MDETCalculator: energy is NaN/inf! energy={energy_val}")

        max_force_magnitude = np.linalg.norm(forces_np, axis=-1).max()
        if max_force_magnitude > 100.0:
            logger.warning(
                f"MDETCalculator: excessive force magnitude {max_force_magnitude:.1f} eV/A "
                f"(model may be extrapolating badly)"
            )

        self.results = {
            "forces": forces_np,
            "energy": energy_val,
        }

    def get_hessian(self, atoms: Atoms = None) -> np.ndarray:
        """
        Compute the Hessian matrix via automatic differentiation.

        The Hessian is the matrix of second derivatives of energy with
        respect to atomic positions, computed as the negative Jacobian
        of forces w.r.t. positions.

        Note: Force filtering is bypassed for Hessian computation.

        Args:
            atoms: ASE Atoms object. If None, uses self.atoms.

        Returns:
            Symmetrized Hessian matrix in eV/A^2, shape (3N, 3N).
        """
        if atoms is None:
            atoms = self.atoms
        if atoms is None:
            raise ValueError("atoms not set")

        n_atoms = len(atoms)
        positions_bohr = atoms.positions * ANG_TO_BOHR

        # Prepare input tensors
        positions_t = (
            th.tensor(positions_bohr, dtype=th.float32).unsqueeze(0).to(self.device)
        )
        atomic_numbers_t = (
            th.tensor(atoms.numbers, dtype=th.int64).unsqueeze(0).to(self.device)
        )

        # Build input dict (without collate_fn preprocessing for Hessian)
        data = {
            Props.positions: positions_t,
            Props.atomic_numbers: atomic_numbers_t,
            Props.mask: th.ones_like(atomic_numbers_t, dtype=th.bool),
            Props.charge: th.tensor(
                [[atoms.info.get("charge", 0)]], dtype=th.int64, device=self.device
            ),
            Props.multiplicity: th.ones(
                (1, 1), dtype=th.int64, device=self.device
            ),
        }

        def force_fn(pos):
            """Compute forces given positions tensor."""
            inp = data.copy()
            inp[Props.positions] = pos
            return self.model(inp)[Props.forces].view(-1, 3)

        # Compute Jacobian of forces w.r.t. positions
        J = th.autograd.functional.jacobian(
            force_fn, positions_t, create_graph=False
        )
        J = J.detach().cpu().squeeze().numpy().reshape(n_atoms * 3, n_atoms * 3)

        # Hessian is negative Jacobian of forces, converted to eV/A^2
        H_raw = -J * HARTREE_BOHR2_TO_EV_ANG2

        # Track antisymmetric component as model quality diagnostic
        H_sym = (H_raw + H_raw.T) / 2
        H_antisym = (H_raw - H_raw.T) / 2
        sym_norm = np.linalg.norm(H_sym)
        antisym_norm = np.linalg.norm(H_antisym)
        self.last_hessian_antisym_ratio = antisym_norm / (sym_norm + 1e-10)

        if self.last_hessian_antisym_ratio > 0.1:
            logger.warning(
                f"Hessian has {self.last_hessian_antisym_ratio:.1%} antisymmetric component "
                f"(non-conservative forces detected)"
            )

        return H_sym
