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


AVAILABLE_VARIANTS = ["4l", "5l", "12l"]
DEFAULT_VARIANT = "12l"


def _resolve_model_source(source: str | Path, variant: str | None = None) -> Path:
    """
    Resolve a model source to a local directory path.

    If source is a local path that exists, return it directly.
    Otherwise, treat it as a Hugging Face Hub repo ID and download it.

    Args:
        source: Local path or HF Hub repo ID (e.g. "mx-e/md-et-v2")
        variant: Model variant subfolder (e.g. "4l", "5l", "12l").
            Only used when downloading from HF Hub.

    Returns:
        Path to a local directory containing the model files
    """
    local_path = Path(source)
    if local_path.exists():
        return local_path

    # Treat as Hugging Face Hub repo ID
    from huggingface_hub import snapshot_download

    variant = variant or DEFAULT_VARIANT
    if variant not in AVAILABLE_VARIANTS:
        raise ValueError(
            f"Unknown variant '{variant}'. Available: {AVAILABLE_VARIANTS}"
        )

    logger.info(f"Downloading model from Hugging Face Hub: {source} (variant={variant})")
    cache_dir = snapshot_download(
        repo_id=str(source),
        repo_type="model",
        allow_patterns=[f"{variant}/**"],
    )
    return Path(cache_dir) / variant


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
    source: str | Path = "mx-e/md-et-v2",
    variant: str | None = None,
    device: str | None = None,
    checkpoint_name: str = "best_model",
    filter_forces: bool = True,
) -> "MDETCalculator":
    """
    Load an MD-ET calculator from a local path or Hugging Face Hub.

    This is the main entry point for using MD-ET models. It loads the model
    architecture from the Hydra config and the weights from the checkpoint,
    and wraps everything in an ASE-compatible calculator.

    Args:
        source: Either a local path to a training run directory, or a
            Hugging Face Hub repo ID. Default: "mx-e/md-et-v2".
            The directory must contain:
            - .hydra/config.yaml (model architecture config)
            - ckpts/{checkpoint_name}.pth (model weights)
        variant: Model variant when loading from HF Hub. Available:
            - "4l"  - 4 layers, 256 dim (fastest, 42 MB)
            - "5l"  - 5 layers, 256 dim (balanced, 49 MB)
            - "12l" - 12 layers, 192 dim (most accurate, 53 MB, default)
            Ignored when source is a local path.
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

        # Default (12-layer model from HF Hub)
        calc = load_calculator()

        # Faster 4-layer variant
        calc = load_calculator(variant="4l")

        # From a local path
        calc = load_calculator("path/to/training_run")

        atoms.calc = calc
        energy = atoms.get_potential_energy()  # eV
        forces = atoms.get_forces()            # eV/A
    """
    if device is None:
        device = "cuda" if th.cuda.is_available() else "cpu"
    run_dir = _resolve_model_source(source, variant=variant)
    model = _load_model(run_dir, device, checkpoint_name)
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

    # =========================================================================
    # Batched operations (for PES exploration with PRFO optimizer)
    # =========================================================================

    def get_batched_forces_and_energy(
        self, atoms_list: list[Atoms]
    ) -> list[tuple[np.ndarray, float]]:
        """
        Compute forces and energy for a batch of Atoms in one forward pass.

        All atoms in the batch must have the same number of atoms and species.

        Args:
            atoms_list: List of ASE Atoms objects (same molecule, different positions).

        Returns:
            List of (forces_eV_Ang, energy_eV) tuples.
        """
        if not atoms_list:
            return []

        B = len(atoms_list)
        positions_list = [
            th.tensor(a.positions * ANG_TO_BOHR, dtype=th.float32)
            for a in atoms_list
        ]
        positions_t = th.stack(positions_list).to(self.device)  # (B, N, 3)

        atomic_numbers_t = (
            th.tensor(atoms_list[0].numbers, dtype=th.int64)
            .unsqueeze(0)
            .expand(B, -1)
            .to(self.device)
        )

        charge_val = atoms_list[0].info.get("charge", 0)
        data = {
            Props.positions: positions_t,
            Props.atomic_numbers: atomic_numbers_t,
            Props.mask: th.ones_like(atomic_numbers_t, dtype=th.bool),
            Props.charge: th.full((B, 1), charge_val, dtype=th.int64, device=self.device),
            Props.multiplicity: th.ones((B, 1), dtype=th.int64, device=self.device),
        }

        with th.no_grad():
            outputs = self.model(data)
            energy = outputs.get(Props.formation_energy, outputs.get(Props.energy))
            forces_batched = outputs[Props.forces]

            if self.filter_forces:
                n_nodes = (data[Props.atomic_numbers] != 0).sum(dim=1)
                forces_batched = remove_net_force(forces_batched, n_nodes)
                forces_batched = remove_net_torque(data[Props.positions], forces_batched, n_nodes)

            forces_ev = (forces_batched * HARTREE_BOHR_TO_EV_ANG).cpu().numpy()
            energy_ev = (energy.squeeze(-1) * HARTREE_TO_EV).cpu().numpy()

        return [(forces_ev[i], float(energy_ev[i])) for i in range(B)]

    def get_batched_hessians(
        self, atoms_list: list[Atoms], hessian_batch_size: int = 16
    ) -> list[tuple[np.ndarray, float, np.ndarray]]:
        """
        Compute forces, energy, and Hessian for a batch of Atoms.

        Uses vmap(jacfwd(...)) to compute all Hessians in parallel on GPU.
        Falls back to sequential computation if vmap is not available.
        Processes in chunks of hessian_batch_size to manage memory.

        All atoms in the batch must have the same number of atoms and species.

        Args:
            atoms_list: List of ASE Atoms objects (same molecule, different positions).
            hessian_batch_size: Max number of Hessians to compute in parallel.
                Halved automatically on OOM.

        Returns:
            List of (forces_eV_Ang, energy_eV, hessian_eV_Ang2) tuples.
            Hessian is symmetrized, shape (3N, 3N).
        """
        if not atoms_list:
            return []

        B = len(atoms_list)
        n_atoms = len(atoms_list[0])

        # Build batched input tensors
        positions_list = [
            th.tensor(a.positions * ANG_TO_BOHR, dtype=th.float32)
            for a in atoms_list
        ]
        positions_t = th.stack(positions_list).to(self.device)  # (B, N, 3)

        atomic_numbers_t = (
            th.tensor(atoms_list[0].numbers, dtype=th.int64)
            .unsqueeze(0)
            .expand(B, -1)
            .to(self.device)
        )

        charge_val = atoms_list[0].info.get("charge", 0)
        data = {
            Props.positions: positions_t,
            Props.atomic_numbers: atomic_numbers_t,
            Props.mask: th.ones_like(atomic_numbers_t, dtype=th.bool),
            Props.charge: th.full((B, 1), charge_val, dtype=th.int64, device=self.device),
            Props.multiplicity: th.ones((B, 1), dtype=th.int64, device=self.device),
        }

        # Batched forward pass for energy/forces
        with th.no_grad():
            outputs = self.model(data)
            energy = outputs.get(Props.formation_energy, outputs.get(Props.energy))
            forces_batched = outputs[Props.forces]

            if self.filter_forces:
                n_nodes = (data[Props.atomic_numbers] != 0).sum(dim=1)
                forces_batched = remove_net_force(forces_batched, n_nodes)
                forces_batched = remove_net_torque(data[Props.positions], forces_batched, n_nodes)

            forces_ev = (forces_batched * HARTREE_BOHR_TO_EV_ANG).cpu().numpy()
            energy_ev = (energy.squeeze(-1) * HARTREE_TO_EV).cpu().numpy()

        # Batched Hessian via vmap(jacfwd(...))
        th.cuda.empty_cache() if self.device != "cpu" else None

        try:
            jacobians = self._batched_jacobians_vmap(
                positions_t, atoms_list[0], hessian_batch_size
            )
        except Exception as e:
            logger.warning(f"vmap+jacfwd failed ({e}), falling back to sequential Jacobians")
            jacobians = self._batched_jacobians_sequential(positions_t, data, n_atoms)

        results = []
        for i in range(B):
            J = jacobians[i]  # (3N, 3N)
            H_raw = -J * HARTREE_BOHR2_TO_EV_ANG2
            H_sym = (H_raw + H_raw.T) / 2
            results.append((forces_ev[i], float(energy_ev[i]), H_sym))

        return results

    def _batched_jacobians_vmap(
        self, positions_t: th.Tensor, ref_atoms: Atoms, chunk_size: int = 16
    ) -> np.ndarray:
        """Compute Jacobians for all samples using vmap(jacfwd(...)).

        Forward-mode AD (jacfwd) is used because for this square Jacobian
        (3N -> 3N) it requires the same number of passes as reverse-mode
        but each pass is cheaper (no activation storage / tape replay).

        Processes in chunks to avoid GPU OOM, halving chunk size on failure.

        Returns:
            numpy array of shape (B, 3N, 3N)
        """
        n_atoms = len(ref_atoms)
        B = positions_t.shape[0]

        # Static inputs (same for all samples)
        an = th.tensor(ref_atoms.numbers, dtype=th.int64, device=self.device)
        charge_val = ref_atoms.info.get("charge", 0)
        charge = th.tensor([[charge_val]], dtype=th.int64, device=self.device)
        mult = th.ones((1, 1), dtype=th.int64, device=self.device)
        mask = th.ones_like(an, dtype=th.bool).unsqueeze(0)

        params_and_buffers = {
            **dict(self.model.named_parameters()),
            **dict(self.model.named_buffers()),
        }
        model = self.model

        def single_force_fn(positions):
            """positions: (N, 3) -> forces: (N*3,)"""
            data = {
                Props.positions: positions.unsqueeze(0),
                Props.atomic_numbers: an.unsqueeze(0),
                Props.mask: mask,
                Props.charge: charge,
                Props.multiplicity: mult,
            }
            out = th.func.functional_call(model, params_and_buffers, (data,))
            return out[Props.forces].reshape(-1)

        batched_jac_fn = th.func.vmap(th.func.jacfwd(single_force_fn))

        all_jacobians = []
        start = 0
        while start < B:
            chunk = positions_t[start : start + chunk_size]
            try:
                J_chunk = batched_jac_fn(chunk)
                J_chunk = J_chunk.reshape(-1, n_atoms * 3, n_atoms * 3)
                all_jacobians.append(J_chunk.detach().cpu().numpy())
                del J_chunk
                if self.device != "cpu":
                    th.cuda.empty_cache()
                start += chunk_size
            except RuntimeError as e:
                if "out of memory" in str(e) and chunk_size > 1:
                    chunk_size = max(1, chunk_size // 2)
                    logger.warning(f"vmap OOM, halving chunk size to {chunk_size}")
                    del chunk
                    if self.device != "cpu":
                        th.cuda.empty_cache()
                else:
                    raise

        return np.concatenate(all_jacobians, axis=0)

    def _batched_jacobians_sequential(
        self, positions_t: th.Tensor, data: dict, n_atoms: int
    ) -> np.ndarray:
        """Fallback: compute Jacobians one at a time.

        Returns:
            numpy array of shape (B, 3N, 3N)
        """
        B = positions_t.shape[0]
        jacobians = []
        for i in range(B):
            pos_i = positions_t[i : i + 1].clone().detach()

            single_data = {
                Props.positions: pos_i,
                Props.atomic_numbers: data[Props.atomic_numbers][i : i + 1],
                Props.mask: data[Props.mask][i : i + 1],
                Props.charge: data[Props.charge][i : i + 1],
                Props.multiplicity: data[Props.multiplicity][i : i + 1],
            }

            def force_fn(pos, d=single_data):
                inp = d.copy()
                inp[Props.positions] = pos
                return self.model(inp)[Props.forces].view(-1, 3)

            J = th.autograd.functional.jacobian(force_fn, pos_i, create_graph=False)
            J = J.detach().cpu().squeeze().numpy().reshape(n_atoms * 3, n_atoms * 3)
            jacobians.append(J)

        return np.stack(jacobians)
