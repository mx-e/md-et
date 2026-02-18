# MD-ET

Machine-learned molecular dynamics with edge transformers. ASE calculator interface for the PairEncoder force field model.

## Installation

```bash
pip install git+https://github.com/YOUR_ORG/md-et.git
```

## Getting access to model weights

Model weights are hosted on Hugging Face Hub with gated access.

1. Create a [Hugging Face](https://huggingface.co) account
2. Go to [your-org/md-et-v1](https://huggingface.co/your-org/md-et-v1) and request access
3. Once approved, log in locally:
   ```bash
   pip install huggingface-hub
   huggingface-cli login
   ```

## Usage

```python
from md_et import load_calculator

# Load from Hugging Face Hub (downloads and caches automatically)
calc = load_calculator("your-org/md-et-v1")

# Or load from a local path
calc = load_calculator("/path/to/training_run")

# Use with ASE
from ase import Atoms

atoms = Atoms("H2O", positions=[[0, 0, 0], [0, 0.757, 0.587], [0, -0.757, 0.587]])
atoms.calc = calc

energy = atoms.get_potential_energy()  # eV
forces = atoms.get_forces()            # eV/Angstrom
hessian = calc.get_hessian(atoms)      # eV/Angstrom^2
```

### Options

```python
# For MD simulations (default: filter_forces=True removes net force/torque)
calc = load_calculator("your-org/md-et-v1", filter_forces=True)

# For geometry optimization or Hessian computation
calc = load_calculator("your-org/md-et-v1", filter_forces=False)

# Specify device
calc = load_calculator("your-org/md-et-v1", device="cpu")

# Use a different checkpoint
calc = load_calculator("your-org/md-et-v1", checkpoint_name="model_final")
```
