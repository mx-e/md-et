# MD-ET

Machine-learned molecular dynamics with edge transformers. ASE calculator interface for the PairEncoder force field model.

## Installation

```bash
uv pip install -e .
```

## Usage

```python
from md_et import load_calculator

# Load from a training run directory
calc = load_calculator("path/to/training_run")

# Use with ASE
from ase import Atoms
atoms = Atoms("H2O", positions=[[0, 0, 0], [0, 0.757, 0.587], [0, -0.757, 0.587]])
atoms.calc = calc

energy = atoms.get_potential_energy()  # eV
forces = atoms.get_forces()            # eV/Angstrom
hessian = calc.get_hessian(atoms)      # eV/Angstrom^2
```
