# Exact exterior coupling for an excavation PINN

Code and final data for **Material-weighted PINN coupling with an exact elastic
exterior for excavation unloading**, by Pengcheng Zhu and Tielin Chen.

The model couples a causal, mixed displacement--stress PINN to a fixed
differentiable elastic exterior. The exterior compliance also defines the
material-weighted interface residual. The interior has non-associated matrix
and weak-plane plasticity.

## Contents

- `code/`: the field model, exterior operator, constitutive equations,
  Gauss--Newton update, field extraction and plotting code.
- `configs/unloading.json`: material, architecture and numerical settings of
  the saved calculation.
- `results/model/model.pt`: consolidated final weights for all twelve spatial
  increment networks, together with their architecture and material parameters.
- `results/unloading/`: small load-summary and fixed-location response tables,
  with the units and grid definitions used by the field exporter.
- `tests/`: checks of the interface residual and strain-output conventions.

The twelve load states describe the reported physical loading history.
Full field grids and wall profiles are generated locally from these weights.
Sampling arrays, optimizer states, optimization trajectories, intermediate
checkpoints, exploratory results and manuscript working files are excluded.

## Installation

Use Python 3.11 or newer. The reference environment uses PyTorch 2.8.0; field
queries and plotting work on the CPU. CUDA is useful for large training grids.

```bash
python -m venv .venv
# Activate .venv using the command for your shell.
python -m pip install -r requirements.txt
```

For GPU training, install the PyTorch build matching your CUDA environment.
The original calculation used PyTorch 2.8.0 with CUDA 12.6 and float64 tensors.

## Plot the supplied response tables

```bash
python code/plot_unloading_results.py --figure stress
```

This writes PDF and editable SVG versions of the fixed-location response figure to
`figures/`. Times New Roman is used when installed; Matplotlib's bundled STIX
font provides a portable alternative.

Generate the full field grid and wall profiles before drawing all three figures:

```bash
python code/extract_unloading_results.py --device cpu
python code/plot_unloading_results.py
```

Use `--device cuda` for GPU evaluation. The generated `fields.npz` and
`wall_profiles.csv` are excluded from Git. Individual figures can be selected
with `--figure wall`, `--figure plastic` or `--figure stress`.

## Query the trained model

Run from the repository root:

```python
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path("code").resolve()))
from publication_model import load_model
from extract_unloading_results import sample_fields

run = Path("results/model")
bundle, saved, model = load_model(run, device="cpu")
history = sample_fields(bundle, model, np.array([[1.25, 0.5]]))
print(history["displacement_mm"].shape)  # (point, load step, component)
print(history["plastic_strain"].shape)   # (point, load step, component)
```

Coordinates are divided by the tunnel radius. Displacements are in mm; strains
are dimensionless. Plastic-strain components are `(xx, yy, zz, 2xy)`. Weak-plane
slip is a continuum shear strain. `results/unloading/metadata.json` defines the
scales, grid, fixed locations and all exported strain measures.

Recompute the exported data in a separate directory:

```bash
python code/extract_unloading_results.py --device cpu --output results/recomputed
```

Use `--device cuda` for GPU evaluation. To recompute only the two stress paths
from the saved location definitions:

```bash
python code/extract_unloading_results.py --device cpu --stress-paths-only
```

## Numerical method

The main implementation files are:

- `code/pilot/causal_mixed.py`: spatial increment networks and causal residuals.
- `code/pilot/plasticity.py`: yield surfaces, non-associated flow and
  complementarity.
- `code/pilot/torch_sf.py` and `code/sf_core/material.py`: anisotropic exterior
  displacement-to-traction map and inverse-stiffness metric.
- `code/causal_response_problem.py`: material-weighted interface residual.
- `code/train_causal_response.py`: the damped Gauss--Newton update used
  for the final material-weighted loading stages.
- `code/case.py`: construct a model and quadrature from the supplied settings.

The training routines are Python APIs. `case.make_case(spec, device)` constructs
the geometry and a model; `model.append_increment()` adds a trainable loading
increment. `CausalResponseProblem` builds the residual and its derivatives.
`propose(problem, diagonal, state, check)` computes an update, where `diagonal`
is the parameter scaling and `state` supplies `mu` and `radius`. Install an
update with `problem.install(trial)` and obtain its complete spatial model
with `problem.assemble(problem.theta)`.

The consolidated weights reproduce the reported loading path. The calculation
used staged sampling and optimizer changes with manually selected endpoints.
The supplied settings describe the final stage; they are not a single training
schedule that regenerates the entire path. Optimizer states are not included.

## Checks

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q tests
```

## Data files

| File in `results/unloading/` | Contents |
| --- | --- |
| `load_summary.csv` | Wall and field summaries for each load step |
| `fixed_location_histories.csv` | Deformation and slip histories at locations A and B |
| `fixed_location_stress_paths.csv` | Stress and slip histories including the initial stress state |
| `metadata.json` | Units, material constants, field definitions and fixed locations |

CSV data and model evaluations use float64. A and B are selected from the
final wall profiles and retained at fixed coordinates throughout the history.
The field exporter generates a 128 by 720 annular grid at all 12 load steps
and wall profiles at 0.5 degree spacing. Grid arrays use float32 for plotting.

Please cite the paper when using the code or data. Contact:
[Tielin Chen](mailto:tlchen1@bjtu.edu.cn).
