# MD-ET

Machine-learned molecular dynamics with edge transformers. ASE calculator interface for the PairEncoder force field model.

**Paper:** Max Eissler, Tim Korjakow, Stefan Ganscha, Oliver T. Unke,
Klaus-Robert Müller, Stefan Gugler, *"How simple can you go? An off-the-shelf
transformer approach to molecular dynamics"*,
[arXiv:2503.01431](https://arxiv.org/abs/2503.01431). **If you use MD-ET in your
research, please cite this paper** (see [Citation](#citation)).

## Installation

```bash
pip install git+https://github.com/mx-e/md-et.git
```

## Model weights

Model weights are hosted openly on the Hugging Face Hub at
[mx-e/md-et-v2](https://huggingface.co/mx-e/md-et-v2) and are downloaded
automatically on first use — no account or token required. `load_calculator()`
fetches the requested variant (see [Model variants](#model-variants)) and caches
it under `~/.cache/huggingface`.

## Usage

```python
from md_et import load_calculator

# Load the default model (12-layer, most accurate)
calc = load_calculator()

# Use with ASE
from ase import Atoms

atoms = Atoms("H2O", positions=[[0, 0, 0], [0, 0.757, 0.587], [0, -0.757, 0.587]])
atoms.calc = calc

energy = atoms.get_potential_energy()  # eV
forces = atoms.get_forces()            # eV/Angstrom
hessian = calc.get_hessian(atoms)      # eV/Angstrom^2
```

### Model variants

Three model sizes are available, selectable via the `variant` parameter:

| Variant | Layers | Embedding dim | Checkpoint size | Notes |
|---------|--------|---------------|-----------------|-------|
| `"4l"` | 4 | 256 | 42 MB | Fastest |
| `"5l"` | 5 | 256 | 49 MB | Balanced |
| `"12l"` | 12 | 192 | 53 MB | Most accurate (default) |

```python
# Fast 4-layer model
calc = load_calculator(variant="4l")

# Balanced 5-layer model
calc = load_calculator(variant="5l")

# Full 12-layer model (default)
calc = load_calculator(variant="12l")
```

### Options

```python
# For MD simulations (default: filter_forces=True removes net force/torque)
calc = load_calculator(filter_forces=True)

# For geometry optimization or Hessian computation
calc = load_calculator(filter_forces=False)

# Specify device
calc = load_calculator(device="cpu")

# Load from a local path instead of HF Hub
calc = load_calculator("/path/to/training_run")
```

## License

Apache License 2.0 — see [LICENSE](LICENSE). Please retain attribution to the
MD-ET authors and cite the paper below when you use MD-ET.

## Citation

If you use MD-ET, please cite (see also [CITATION.cff](CITATION.cff)):

```bibtex
@article{mdet2025,
  title   = {How simple can you go? An off-the-shelf transformer approach to
             molecular dynamics},
  author  = {Eissler, Max and Korjakow, Tim and Ganscha, Stefan and
             Unke, Oliver T. and M\"uller, Klaus-Robert and Gugler, Stefan},
  journal = {arXiv preprint arXiv:2503.01431},
  year    = {2025}
}
```

