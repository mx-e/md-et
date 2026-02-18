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
