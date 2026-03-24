"""Tests for the MD-ET ASE calculator.

Tests are parametrized over all model variants (4l, 5l, 12l).
Requires: `huggingface-cli login` with an account that has access to mx-e/md-et-v2.
"""

import numpy as np
import pytest


def test_load_from_hub(calculator_unfiltered):
    """Test that load_calculator successfully loads a model from HF Hub."""
    assert calculator_unfiltered is not None
    assert calculator_unfiltered.model is not None


def test_energy_and_forces(calculator_unfiltered, water_atoms):
    """Test that energy and forces are computed and finite."""
    water_atoms.calc = calculator_unfiltered

    energy = water_atoms.get_potential_energy()
    forces = water_atoms.get_forces()

    assert np.isfinite(energy)
    assert forces.shape == (3, 3)
    assert np.isfinite(forces).all()


def test_forces_shape_methane(calculator_unfiltered, methane_atoms):
    """Test forces shape for a larger molecule."""
    methane_atoms.calc = calculator_unfiltered
    forces = methane_atoms.get_forces()
    assert forces.shape == (5, 3)


def test_energy_changes_with_displacement(calculator_unfiltered, water_atoms):
    """Test that energy changes when atoms are displaced."""
    water_atoms.calc = calculator_unfiltered
    energy_1 = water_atoms.get_potential_energy()

    displaced = water_atoms.copy()
    displaced.positions[0, 0] += 0.1
    displaced.calc = calculator_unfiltered
    energy_2 = displaced.get_potential_energy()

    assert energy_1 != energy_2


def test_force_filtering(calculator_filtered, water_atoms):
    """Test that force filtering removes net force and torque."""
    water_atoms.calc = calculator_filtered
    forces_filtered = water_atoms.get_forces()

    net_force = forces_filtered.sum(axis=0)
    np.testing.assert_allclose(net_force, 0.0, atol=1e-4)


def test_hessian(calculator_unfiltered, water_atoms):
    """Test Hessian computation: shape, symmetry, finiteness."""
    water_atoms.calc = calculator_unfiltered
    hessian = calculator_unfiltered.get_hessian(water_atoms)

    n_atoms = len(water_atoms)
    assert hessian.shape == (3 * n_atoms, 3 * n_atoms)
    np.testing.assert_allclose(hessian, hessian.T, atol=1e-5)
    assert np.isfinite(hessian).all()


def test_deterministic_output(calculator_unfiltered, water_atoms):
    """Test that the model produces deterministic results."""
    water_atoms.calc = calculator_unfiltered

    energy_1 = water_atoms.get_potential_energy()
    forces_1 = water_atoms.get_forces().copy()

    energy_2 = water_atoms.get_potential_energy()
    forces_2 = water_atoms.get_forces().copy()

    assert energy_1 == energy_2
    np.testing.assert_array_equal(forces_1, forces_2)


def test_charge_handling(calculator_unfiltered):
    """Test that charge info is passed through correctly."""
    from ase import Atoms

    atoms = Atoms(
        "OH",
        positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.97]],
        info={"charge": -1},
    )
    atoms.calc = calculator_unfiltered
    energy = atoms.get_potential_energy()
    assert np.isfinite(energy)


def test_invalid_variant():
    """Test that an invalid variant raises ValueError."""
    from md_et import load_calculator

    with pytest.raises(ValueError, match="Unknown variant"):
        load_calculator(variant="99l")


def test_default_loads_12l():
    """Test that default load_calculator uses the 12l variant."""
    from md_et import load_calculator
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    calc = load_calculator(device=device)
    # 12l has 12 transformer layers
    assert len(calc.model.layers) == 12


def test_batched_forces_and_energy(calculator_unfiltered, water_atoms):
    """Test batched energy/forces: consistent with single-molecule results."""
    water_atoms.calc = calculator_unfiltered

    single_energy = water_atoms.get_potential_energy()
    single_forces = water_atoms.get_forces().copy()

    # Create a batch of 3 copies with slight perturbations
    from ase import Atoms
    batch = [water_atoms.copy()]
    for dx in [0.01, -0.01]:
        a = water_atoms.copy()
        a.positions[0, 0] += dx
        batch.append(a)

    results = calculator_unfiltered.get_batched_forces_and_energy(batch)

    assert len(results) == 3
    # First item should match single-molecule result
    forces_0, energy_0 = results[0]
    np.testing.assert_allclose(energy_0, single_energy, atol=1e-6)
    np.testing.assert_allclose(forces_0, single_forces, atol=1e-5)

    # All results should be finite
    for forces, energy in results:
        assert np.isfinite(energy)
        assert np.isfinite(forces).all()

    # Perturbed molecules should have different energies
    assert results[1][1] != results[0][1]


def test_batched_hessians(calculator_unfiltered, water_atoms):
    """Test batched Hessian: shape, symmetry, consistency with single Hessian."""
    single_hessian = calculator_unfiltered.get_hessian(water_atoms)
    n_atoms = len(water_atoms)

    # Batch of 2 copies
    batch = [water_atoms.copy(), water_atoms.copy()]
    results = calculator_unfiltered.get_batched_hessians(batch, hessian_batch_size=2)

    assert len(results) == 2
    for forces, energy, hessian in results:
        assert hessian.shape == (3 * n_atoms, 3 * n_atoms)
        np.testing.assert_allclose(hessian, hessian.T, atol=1e-5)
        assert np.isfinite(hessian).all()
        assert np.isfinite(forces).all()
        assert np.isfinite(energy)

    # First Hessian should match single-molecule result
    # Tolerance accounts for jacfwd vs jacobian numerical differences in float32
    np.testing.assert_allclose(results[0][2], single_hessian, atol=0.02, rtol=0.03)


def test_batched_hessians_empty(calculator_unfiltered):
    """Test that empty batch returns empty list."""
    assert calculator_unfiltered.get_batched_hessians([]) == []
    assert calculator_unfiltered.get_batched_forces_and_energy([]) == []
