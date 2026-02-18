"""Tests for the MD-ET ASE calculator."""

import numpy as np
import pytest
import torch


def test_load_calculator(model_dir, device):
    """Test that load_calculator successfully loads a model."""
    from md_et import load_calculator

    calc = load_calculator(model_dir, device=device)
    assert calc is not None
    assert calc.model is not None
    assert calc.device == device


def test_energy_and_forces(model_dir, device, water_atoms):
    """Test that energy and forces are computed correctly."""
    from md_et import load_calculator

    calc = load_calculator(model_dir, device=device, filter_forces=False)
    water_atoms.calc = calc

    energy = water_atoms.get_potential_energy()
    forces = water_atoms.get_forces()

    # Energy should be a finite scalar
    assert np.isfinite(energy)

    # Forces should have correct shape (n_atoms, 3)
    assert forces.shape == (3, 3)
    assert np.isfinite(forces).all()


def test_forces_shape_methane(model_dir, device, methane_atoms):
    """Test forces shape for a larger molecule."""
    from md_et import load_calculator

    calc = load_calculator(model_dir, device=device, filter_forces=False)
    methane_atoms.calc = calc
    forces = methane_atoms.get_forces()
    assert forces.shape == (5, 3)


def test_energy_changes_with_displacement(model_dir, device, water_atoms):
    """Test that energy changes when atoms are displaced."""
    from md_et import load_calculator

    calc = load_calculator(model_dir, device=device, filter_forces=False)

    water_atoms.calc = calc
    energy_1 = water_atoms.get_potential_energy()

    # Displace one atom
    displaced = water_atoms.copy()
    displaced.positions[0, 0] += 0.1
    displaced.calc = calc
    energy_2 = displaced.get_potential_energy()

    assert energy_1 != energy_2


def test_force_filtering(model_dir, device, water_atoms):
    """Test that force filtering removes net force and torque."""
    from md_et import load_calculator

    calc_filtered = load_calculator(model_dir, device=device, filter_forces=True)
    water_atoms.calc = calc_filtered
    forces_filtered = water_atoms.get_forces()

    # Net force should be approximately zero with filtering
    net_force = forces_filtered.sum(axis=0)
    np.testing.assert_allclose(net_force, 0.0, atol=1e-4)


def test_hessian(model_dir, device, water_atoms):
    """Test Hessian computation."""
    from md_et import load_calculator

    calc = load_calculator(model_dir, device=device, filter_forces=False)
    water_atoms.calc = calc

    hessian = calc.get_hessian(water_atoms)

    n_atoms = len(water_atoms)
    # Hessian should be (3N x 3N)
    assert hessian.shape == (3 * n_atoms, 3 * n_atoms)

    # Hessian should be symmetric
    np.testing.assert_allclose(hessian, hessian.T, atol=1e-5)

    # Hessian should be finite
    assert np.isfinite(hessian).all()


def test_hessian_antisym_ratio(model_dir, device, water_atoms):
    """Test that the antisymmetric ratio diagnostic is tracked."""
    from md_et import load_calculator

    calc = load_calculator(model_dir, device=device, filter_forces=False)
    water_atoms.calc = calc
    calc.get_hessian(water_atoms)

    assert calc.last_hessian_antisym_ratio is not None
    assert calc.last_hessian_antisym_ratio >= 0


def test_deterministic_output(model_dir, device, water_atoms):
    """Test that the model produces deterministic results."""
    from md_et import load_calculator

    calc = load_calculator(model_dir, device=device, filter_forces=False)
    water_atoms.calc = calc

    energy_1 = water_atoms.get_potential_energy()
    forces_1 = water_atoms.get_forces().copy()

    energy_2 = water_atoms.get_potential_energy()
    forces_2 = water_atoms.get_forces().copy()

    assert energy_1 == energy_2
    np.testing.assert_array_equal(forces_1, forces_2)


def test_charge_handling(model_dir, device):
    """Test that charge info is passed through correctly."""
    from ase import Atoms
    from md_et import load_calculator

    # Charged molecule (e.g., OH-)
    atoms = Atoms(
        "OH",
        positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.97]],
        info={"charge": -1},
    )

    calc = load_calculator(model_dir, device=device, filter_forces=False)
    atoms.calc = calc
    energy = atoms.get_potential_energy()
    assert np.isfinite(energy)
