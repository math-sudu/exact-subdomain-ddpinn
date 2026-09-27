# Exact exterior coupling for an excavation PINN

Code and final data for **Exact exterior coupling and energy-based interface
weighting for physics-informed modeling of anisotropic solids**.
<!-- BEGIN_IDENTIFIED -->
by Pengcheng Zhu and Tielin Chen.
<!-- END_IDENTIFIED -->

The model couples a causal, mixed displacement--stress PINN to a fixed
differentiable elastic exterior. The exterior compliance also defines the
material-weighted interface residual. The interior has non-associated matrix
and weak-plane plasticity.

## Contents

- `code/`: the field model, exterior operator, constitutive equations,
  Gauss--Newton update, field extraction and plotting code.
- `configs/initialization.json`: final-stage settings of the shared pointwise
  initialization calculation.
- `configs/orientation_extension.json`: the projected nonlinear calculation.
- `results/model/model.pt`: consolidated final weights for all twelve spatial
  increment networks, together with their architecture and material parameters.
- `results/initialization/model.pt`: the pointwise networks used to initialize
  the three matched nonlinear cases.
- `results/unloading/`: small load-summary and fixed-location response tables,
  with the units and grid definitions used by the field exporter.
- `tests/`: checks of the interface residual and strain-output conventions.

The twelve load states describe the reported physical loading history.
Full field grids and wall profiles are generated locally from these weights.
The paired studies also supply their training trajectories.

## Extended studies

`configs/interface_extension.json`
defines twenty-seven elastic trials across four interface metrics, two
anisotropy strengths and an equal-area inclined elliptical opening.
Each metric starts from the same stored initial
field for each seed. `configs/orientation_extension.json` defines matched
unloading histories with oblique, parallel and perpendicular weak-plane traces
relative to the elastic bedding. An inactive elastic prefix is shared; each
later increment starts from the corresponding original increment. The neural
outputs are projected onto Chebyshev--Fourier fields before forming their
physical residuals. Radial order 32 and angular modes 0--64 are retained, with
256 angular projection nodes. Quadrature uses 48 radial Gauss points, 256
angles per ring and 1,024 points per boundary. Ring angles are staggered;
independent evaluation doubles the counts and shifts the angles.

`results/interface_extension/` and `results/orientation_extension/` contain
the final networks, training trajectories and compact response tables.
Reconstruct the field grids from the supplied networks before recomputing the
summaries and figures:

```bash
python code/export_extension_fields.py
python code/summarize_extensions.py
python code/plot_extensions.py
```

The field exporter evaluates all 27 elastic models and three nonlinear
histories without training. Use `--device cuda` for GPU evaluation or
`--study elastic` / `--study orientation` to select a study. Large field arrays
are generated locally and excluded from Git.

To train the paired elastic study in a new directory:

```bash
python code/run_interface_extension.py --output results/recomputed_interface --device cuda
```

The orientation runner can likewise use `--output results/recomputed_orientation`.
It initializes from `results/initialization/model.pt`. Complete final checkpoints are
retained without optimizer history; an interrupted active increment retains
the optimizer state needed for continuation.

## Installation

Use Python 3.11 or newer. The reference environment uses PyTorch 2.8.0; field
queries and plotting work on the CPU. CUDA is useful for large training grids.

```bash
python -m venv .venv
# Activate .venv using the command for your shell.
python -m pip install -r requirements.txt
```

For GPU training, install the PyTorch build matching your CUDA environment.
The reference environment is PyTorch 2.8.0 with CUDA 12.6; field queries use
float64 tensors.

## Plot the supplied response tables

```bash
python code/plot_unloading_results.py --figure stress
```

This writes PDF and editable SVG versions of the fixed-location response figure to
`figures/`. Times New Roman is used when installed; Matplotlib's bundled STIX
font provides a portable alternative.

Generate the field grid, wall profiles and radial histories before drawing
all five result figures:

```bash
python code/extract_unloading_results.py --device cpu
python code/extract_radial_kinematics.py
python code/plot_unloading_results.py
```

Use `--device cuda` for GPU evaluation. The generated field, wall-profile and
radial-history arrays are excluded from Git. Individual figures can be selected
with `--figure displacement`, `--figure wall`, `--figure stress`,
`--figure plastic` or `--figure radial`.
The displacement figure uses vector increments before taking their magnitude.
The plotting entry points accept `--data` and `--figure-directory` for
separate figure variants with the same panels.
The radial figure combines the full displacement strain, interface motion
and signed path integrals along the fixed A/B rays. The compact
`radial_contributions.csv` contains all twelve states at both locations.

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
- `code/pilot/spectral_mixed.py`: differentiable polar projection, resolved
  basis derivatives and frozen projected history.
- `code/run_orientation_extension.py`: the matched nonlinear training protocol.
- `code/pilot/plasticity.py`: yield surfaces, non-associated flow and
  complementarity.
- `code/pilot/torch_sf.py` and `code/sf_core/material.py`: anisotropic exterior
  displacement-to-traction map and inverse-stiffness metric.
- `code/causal_response_problem.py`: material-weighted interface residual.
- `code/train_causal_response.py`: the damped Gauss--Newton update used
  for the pointwise initialization calculation.
- `code/case.py`: construct a model and quadrature from the supplied settings.

The initialization routines are Python APIs. `case.make_case(spec, device)` constructs
the geometry and a model; `model.append_increment()` adds a trainable loading
increment. `CausalResponseProblem` builds the residual and its derivatives.
`propose(problem, diagonal, state, check)` computes an update, where `diagonal`
is the parameter scaling and `state` supplies `mu` and `radius`. Install an
update with `problem.install(trial)` and obtain its complete spatial model
with `problem.assemble(problem.theta)`.

The reported nonlinear loading paths are the projected models in
`results/model/` and `results/orientation_extension/`.
Their common initial fields are supplied in `results/initialization/model.pt`,
with model settings in `configs/initialization.json`.

<!-- BEGIN_IDENTIFIED -->
The pointwise initializer was released at commit
[`7d1fc50`](https://github.com/math-sudu/exact-subdomain-ddpinn/commit/7d1fc50cdf836be9db7f168d7e43a30c1fc0a647).
<!-- END_IDENTIFIED -->

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

Please cite the paper when using the code or data.
<!-- BEGIN_IDENTIFIED -->
Contact:
[Tielin Chen](mailto:tlchen1@bjtu.edu.cn).
<!-- END_IDENTIFIED -->
