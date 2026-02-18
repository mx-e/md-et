import pytest

HF_REPO_ID = "mx-e/md-et-v2"


@pytest.fixture(scope="session", params=["4l", "5l", "12l"])
def calculator_unfiltered(request):
    """Load each model variant once for the test session (no force filtering)."""
    from md_et import load_calculator
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    return load_calculator(HF_REPO_ID, variant=request.param, device=device, filter_forces=False)


@pytest.fixture(scope="session")
def calculator_filtered():
    """Load default model with force filtering for filter-specific tests."""
    from md_et import load_calculator
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    return load_calculator(HF_REPO_ID, device=device, filter_forces=True)


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
