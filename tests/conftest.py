from pathlib import Path

import pytest


FIXTURES_DIR = Path(__file__).parent / "fixtures"
MODEL_DIR = FIXTURES_DIR / "combined_v3_direct"


@pytest.fixture
def model_dir():
    """Path to the test model directory."""
    if not MODEL_DIR.exists():
        pytest.skip("Test fixtures not found. Copy model weights to tests/fixtures/")
    return MODEL_DIR


@pytest.fixture
def device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def water_atoms():
    """Simple water molecule for testing."""
    from ase import Atoms
    return Atoms(
        "H2O",
        positions=[
            [0.0, 0.0, 0.0],
            [0.0, 0.757, 0.587],
            [0.0, -0.757, 0.587],
        ],
    )


@pytest.fixture
def methane_atoms():
    """Methane molecule for testing."""
    from ase import Atoms
    return Atoms(
        "CH4",
        positions=[
            [0.000, 0.000, 0.000],
            [0.629, 0.629, 0.629],
            [-0.629, -0.629, 0.629],
            [-0.629, 0.629, -0.629],
            [0.629, -0.629, -0.629],
        ],
    )
