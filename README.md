# Exact exterior coupling for an excavation PINN

Code and final data for **Physics-informed modeling of anisotropic excavation
with exact exterior coupling and spatial loading histories**.
<!-- BEGIN_IDENTIFIED -->
by Pengcheng Zhu and Tielin Chen.
<!-- END_IDENTIFIED -->

The model couples spatial displacement increments to a fixed differentiable
anisotropic elastic exterior. A local zero-dilation weak-plane return supplies
stress and plastic-strain histories, and a virtual-work residual enforces
equilibrium. The exterior compliance defines the interface residual metric.
Matrix yielding is monitored as an applicability condition. Material parameters
and loads define computational scenarios; they are not calibrated observations.

## Contents

- `code/`: the field model, exterior operator, constitutive equations,
  Gauss--Newton update, field extraction and plotting code.
- `configs/initialization.json`: final-stage settings of the shared pointwise
  initialization calculation.
- `configs/orientation_extension.json`: the local-return nonlinear calculation.
- `configs/orientation_spectral.json`: the earlier mixed formulation used by
  the initialization and projection tests.
- `results/model/model.pt`: consolidated final weights for all twelve spatial
  increment networks, together with their architecture and material parameters.
- `results/initialization/model.pt`: the pointwise networks used to initialize
  the three matched nonlinear cases.
- `results/orientation_initialization/`: same-orientation mixed onset weights
  used to start the local-return histories.
- `results/history_analysis/`: fixed gauges, slip sectors and signed regional
  source integrals for the three completed histories.
- `results/unloading/`: small load-summary and fixed-location response tables,
  with the units and grid definitions used by the field exporter.
- `tests/`: checks of the interface residual and strain-output conventions.

The twelve load states describe the reported physical loading history.
Full field grids and wall profiles are generated locally from these weights.
The paired studies also supply their training trajectories.

## Extended studies

`configs/interface_extension.json`
defines 24 paired circular elastic trials across four interface metrics and
two anisotropy strengths, plus three calculations for an equal-area inclined
elliptical opening.
Each metric starts from the same stored initial
field for each seed. `configs/orientation_extension.json` defines matched
unloading histories with oblique, parallel and perpendicular weak-plane traces
relative to the elastic bedding. Their exact elastic prefixes contain four,
two and three increments, respectively. The first plastic increment starts
from the supplied same-orientation mixed onset; subsequent increments copy
the preceding trained displacement. The two active displacement outputs are
projected onto Chebyshev--Fourier fields before local constitutive return.
Radial order 32 and angular modes 0--64 are retained, with
256 angular projection nodes. Quadrature uses 48 radial Gauss points, 256
angles per ring and 1,024 points per boundary. Ring angles are staggered;
off-training-grid evaluation doubles the counts and shifts the angles.

`results/interface_extension/` and `results/orientation_extension/` contain
the final networks, training trajectories and compact response tables.
Reconstruct the field grids from the supplied networks before recomputing the
summaries and figures:

```bash
python code/export_extension_fields.py
python code/summarize_extensions.py --study elastic
python code/plot_extensions.py --figure interface
python code/plot_extensions.py --figure geometry
```

The field exporter evaluates all 27 elastic models and three nonlinear
histories without training. Use `--device cuda` for GPU evaluation or
`--study elastic` / `--study orientation` to select a study. Large field arrays
are generated locally and excluded from Git.

To train the paired elastic study in a new directory:

```bash
python code/run_interface_extension.py --output results/recomputed_interface --device cuda
```

To start a returned history from its supplied onset (example: oblique case):

```bash
python code/run_orientation_extension.py --case bedding_oblique --output results/recomputed_orientation --initialize-checkpoint results/orientation_initialization/bedding_oblique/model.pt --through-step 5 --device cuda
```

The runner saves a pending increment after each block of 12 residual evaluations,
using at most 600 Krylov iterations per linear solve. Repeat the command to
continue that increment. After assessing its deformation, source balance and
boundary response, `--accept-step 5 --through-step 6` freezes it and starts the
next increment. This explicit staging preserves the reported training procedure;
the gradient-reduction setting is an optimization control. Final checkpoints and
their trajectories are supplied so field reconstruction requires no training.

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

## Reconstruct the manuscript response figures

```bash
python code/extract_unloading_results.py --device cpu
python code/extract_radial_kinematics.py
python code/plot_unloading_results.py --figure displacement
python code/plot_unloading_results.py --figure wall
python code/analyze_unloading_history.py
python code/analyze_gauge_sources.py --root results/orientation_extension --cases bedding_oblique bedding_aligned bedding_crossed --steps 1 2 3 4 5 6 7 8 9 10 11 12 --output results/history_analysis/sources
python code/plot_history_analysis.py
```

Run `export_extension_fields.py --study orientation` first for the orientation
slip maps. These commands write PDF and editable SVG figures to `figures/`.
Times New Roman is used when installed, with STIX as a portable alternative.
Use `--device cuda` for GPU field evaluation. Generated dense arrays are excluded
from Git.
The displacement figure uses vector increments before taking their magnitude.
The plotting entry points accept `--data` and `--figure-directory` for
separate figure variants with the same panels.
The whole-space figure combines stress paths and signed radial accumulation
along fixed rays. The compact
`radial_contributions.csv` contains all twelve states at both locations.
The source analysis keeps signed contributions of actual stepwise plastic
strains to fixed horizontal and vertical endpoint gauges. Its two regions are
defined by the sign of `(x.n)*(x.t)` in material axes. Source closure remains a
reported numerical discrepancy; the crossed final vertical discrepancy of
about -0.00312 mm limits interpretation of small last-increment contributions.

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

- `code/pilot/causal_mixed.py`: base increment networks and mixed initializer.
- `code/pilot/spectral_mixed.py`: differentiable polar projection, resolved
  basis derivatives and frozen projected history.
- `code/run_orientation_extension.py`: the matched nonlinear training protocol.
- `code/pilot/joint_return.py`: closed-form local weak-plane return.
- `code/pilot/returned_orientation.py`: displacement-only neural residuals.
- `code/pilot/virtual_work.py`: exterior-coupled virtual-work equilibrium.
- `code/pilot/plasticity.py`: yield surfaces and flow directions.
- `code/analyze_gauge_sources.py`: signed fixed-gauge source integration.
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
Their mixed onset weights are supplied in `results/orientation_initialization/`.
The earlier pointwise precursor is in `results/initialization/model.pt`, with
settings in `configs/initialization.json`.

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
