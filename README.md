# Operator-embedded spectral PINN for tunnel unloading

Code and final data for **Anisotropic elastoplastic deformation during tunnel
unloading using an operator-embedded spectral PINN**.
<!-- BEGIN_IDENTIFIED -->
by Pengcheng Zhu and Tielin Chen.
<!-- END_IDENTIFIED -->

The model describes small-strain plane-strain elastoplastic deformation during
tunnel wall traction release. A fixed anisotropic exterior operator enters
differentiable PINN training, and a Chebyshev--Fourier projection retains
spatial displacement increments and their derivatives. A local zero-dilation
weak-plane return supplies stress and plastic-strain histories, while
virtual work enforces equilibrium. Exterior compliance defines the interface
residual metric. The results connect the anisotropic elastic background and
evolving plastic shear to section shortening and finite-span radial deformation.
Weak-plane slip is represented by continuum plastic shear strain in a
continuous displacement field. Material parameters and loads define prescribed
computational scenarios.

## Contents

- `code/`: the field model, exterior operator, constitutive equations,
  Gauss--Newton update, field extraction and plotting code.
- `configs/initialization.json`: final-stage settings of the shared pointwise
  initialization calculation.
- `configs/orientation_extension.json`: the local-return nonlinear calculation.
- `configs/orientation_bedding_crossed.json`: the perpendicular case's final-step settings.
- `configs/orientation_spectral.json`: the earlier mixed formulation used by
  the initialization and projection tests.
- `results/model/model.pt`: consolidated final weights for all twelve spatial
  increment networks, together with their architecture and material parameters.
- `results/initialization/model.pt`: the pointwise networks used to initialize
  the three matched nonlinear cases.
- `results/orientation_initialization/`: same-orientation initial weights
  (mixed onsets for oblique/parallel and a trained 3a history for perpendicular).
- `results/history_analysis/`: fixed gauges, slip sectors and signed regional
  source integrals for the three completed histories.
- `results/history_resolution/`: coupled domain, spatial and load-step
  experiments, their settings and trajectories, and independent response,
  weak-balance and exterior-yield checks against the reference histories.
- `results/controlled_history/`: parallel 4a paths solved from
  the first plastic increment with two predictors and common physical
  acceptance checks; saved records identify accepted and pending increments.
- `results/tensile_history/`: source partitions by end-of-increment normal
  stress, including separate positive and negative gauge work.
- `results/late_step_continuation/`: pending final-increment sensitivity
  states on identical saved prefixes and their comparisons.
- `results/unloading/`: small load-summary and fixed-location response tables,
  with the units and grid definitions used by the field exporter.
- `tests/`: checks of the interface residual and strain-output conventions.

The reference histories use twelve load states; the load-step studies use 24.
Full field grids and wall profiles are generated locally from these weights.
The paired studies also supply their training trajectories.

The current parallel result is
`results/controlled_history/parallel_source_final/adopted/model.pt`, with
matching settings, trajectory and response tables in the same directory.
It uses a 4a interface, projection (48,80) and twelve increments. Horizontal
and vertical total shortenings are 0.10792587 and 0.53167376 mm; plastic
sources are -0.03190980 and +0.07168952 mm. The saved final increment retains
its actual wall/weak residual flags. Other states in the parent directory
are optimization experiments. Reconstruct the selected response with:

```bash
python code/summarize_history_study.py results/controlled_history/parallel_source_final/adopted/model.pt --output results/reconstructed_parallel --include-pending
```

The standalone reference selector is `code/current_histories.py`: all three
reference checkpoints are under `results/orientation_extension/<case>/`.
The project paths `orientation_return/` and `orientation_expanded/` retained
inside historical diagnostic records map to those same reference states.
Each reference's saved initializer is supplied under
`results/orientation_initialization/<case>/model.pt`. The six subsequent
studies keep their names under `results/history_resolution/`: `parallel_r4`,
`parallel_r4_j48_m80`, `oblique_j40_m80`, `oblique_n24`,
`perpendicular_j48_m80`, and `perpendicular_n24`.

The initializer and settings of those six studies are:

| Study directory | Input checkpoint directory | R/a; J, M; volume quadrature; increments |
| --- | --- | --- |
| `parallel_r4` | `orientation_extension/bedding_aligned` | 4; 40, 64; 64 by 256; 12 |
| `parallel_r4_j48_m80` | `history_resolution/parallel_r4` | 4; 48, 80; 96 by 512; 12 |
| `oblique_j40_m80` | `orientation_extension/bedding_oblique` | 3; 40, 80; 64 by 320; 12 |
| `oblique_n24` | `orientation_extension/bedding_oblique` | 3; 32, 64; 48 by 256; 24 |
| `perpendicular_j48_m80` | `orientation_extension/bedding_crossed` | 4; 48, 80; 96 by 512; 12 |
| `perpendicular_n24` | `orientation_extension/bedding_crossed` | 4; 40, 64; 96 by 512; 24 |

All input directories in the table are relative to `results/` and contain
`model.pt`. The spatially refined parallel and perpendicular directories also
supply the pending and constrained-head states for steps 11 and 12. Their
`constrained_head_solves` metadata specifies the actual constraint regions,
margin floor and optimization outcome. The final `model.pt` in each directory
is the complete history used for its reported response table.

Use `code/recompute_orientation_history.py --source <reference-model>
--output <new-directory> --through-step <N>` with the domain, projection,
quadrature and load-step arguments shown by `--help` to resolve a changed
history. Only the exact elastic prefix is accepted from the source; the
remaining increments are solved with their newly accumulated histories.
`code/summarize_history_study.py` produces the response and exterior readouts;
`code/assess_history_resolution.py` produces the weak-test and local-force
checks. The small endpoint/source discrepancy measures the selected functional,
and the weak residual is not an elastoplastic field-error bound.

`code/refine_returned_head.py` applies the documented constrained recovery
of the existing displacement head. Its independent source grid and explicit
local constraint regions address unsupported return queries found during
assessment. The pending inputs, earlier states and branch assessments are
retained in the corresponding history-resolution directory. The output stays
pending until its independent history queries have been assessed; optimizer
success alone is not a field-accuracy criterion. Final summaries retain the
solve counts, source-predictor restarts and constrained recovery results.

## Extended studies

`configs/interface_extension.json`
defines 24 paired circular elastic trials across four interface metrics and
two anisotropy strengths, plus three calculations for an equal-area inclined
elliptical opening.
Each metric starts from the same stored initial
field for each seed. `configs/orientation_extension.json` defines
unloading histories with oblique, parallel and perpendicular weak-plane traces
relative to the elastic bedding. Their exact elastic prefixes contain four,
two and three increments, respectively. The first plastic increment starts
from the supplied same-orientation mixed onset for oblique/parallel;
subsequent increments copy the preceding trained displacement. The perpendicular
history uses same-orientation trained step networks as initial guesses on an
enlarged 4a annulus, solving increments in load order with the accepted prefix
fixed. The two active displacement outputs are
projected onto Chebyshev--Fourier fields before local constitutive return.
Oblique/parallel histories use an interface at 3a and radial order 32;
the perpendicular history uses 4a and order 40. All use angular modes 0--64
and 256 projection nodes. Volume quadrature has 48 radial Gauss points and
256 angles per ring for oblique/parallel, 64 by 256 for the perpendicular
plastic increments, and 96 by 512 for its final increment. Each boundary
uses 1,024 points. Field exports read each checkpoint's own settings.

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

## Full-history optimization experiment

The parallel 4a experiment fixes the twelve increments, material, projection
and quadrature, and changes the displacement predictors. To reconstruct its
endpoint, regional-source and exterior checks from the retained weights:

```bash
python code/summarize_history_study.py results/controlled_history/parallel_source/model.pt --output results/reconstructed_source --include-pending
python code/summarize_history_study.py results/controlled_history/parallel_preceding/model.pt --output results/reconstructed_preceding --include-pending
```

To solve both histories again in new directories:

```bash
python code/resolve_controlled_history.py --source results/history_resolution/parallel_r4_j48_m80/model.pt --output results/recomputed_source --predictor source --weak-load-scale '1.6358928798171821'
python code/resolve_controlled_history.py --source results/history_resolution/parallel_r4_j48_m80/model.pt --output results/recomputed_preceding --predictor preceding --onset-scale 0.5 --weak-load-scale '1.6358928798171821'
```

Each run starts from the exact elastic prefix and re-solves every plastic
increment. `control.json` records the fixed physical acceptance targets;
`physical_history.json` records each block's independent responses, actual
residuals and acceptance decision. Optimizer status alone never freezes a
step. A failed physical check leaves that increment pending. The same command
resumes it with its saved preceding history. Source predictors use the
supplied final checkpoint as displacement guesses, so its weights are a
required input to these specific experiments.
An unaccepted full-release state can be assessed with `--include-pending`
in the history, weak-residual and tensile readers. This reads its actual
constitutive history without accepting the increment. The supplied
`comparison.json` retains each final state's failed targets. Both supplied
paths accept increments 3--11 and retain increment 12 as pending: their
final wall and weak residuals exceed the common targets. Their source
differences are 6.7% horizontally and 2.4% vertically. The source path's
separate `step12_head.pt` is an unadopted diagnostic, not its final model.
`--polish-evaluations 240` enables an additional bounded scaled L-BFGS
stage after a trust-region block exhausts its evaluation budget; the
physical targets remain unchanged.
When the independent weak test alone remains above target after the response
and boundary checks pass, the runner includes those (56,96) tests in the
training work metric. Displacement coefficients are zero-extended; the
(48,80) displacement projection and all material parameters are unchanged.
If the frozen-source projection floor already exceeds the weak target,
enrichment proceeds before response stabilization once the branch, boundary
and gauge checks pass. Waiting for the original tests to stabilize cannot
remove that unresolved component at the assessed source.
`work_test_enrichment` survives subsequent checkpoint saves and records the
activation increment and the old/new test spaces. For the two supplied
histories, the lost event was recovered from the saved physical assessments:
both enrich at step 10. `first_recorded_evaluations` identifies the first
surviving assessment using that space, rather than an exact activation count.
Earlier accepted increments remain fixed.
The `independent_weak.json` records cover every plastic increment with
(64,112) tests and 128-by-768 integration, outside training and acceptance.
`comparison.json` joins the weak tests to the corresponding sources and
acceptance states. Reconstruct this check with:

```bash
python code/assess_history_resolution.py results/controlled_history/parallel_source/model.pt --output results/reconstructed_source/independent_weak.json --steps 3 4 5 6 7 8 9 10 11 12 --radial-order 64 --angular-modes 112 --radial-quadrature 128 --angular-quadrature 768 --include-pending
python code/assess_tensile_history.py results/controlled_history/parallel_source/model.pt --output results/reconstructed_source/tensile --include-pending
```

Use `parallel_preceding/model.pt` for the second history. The tensile labels
assign each plastic increment by its returned normal stress at that increment's
end. `code/summarize_controlled_history.py` reads the supplied per-history
readouts and regenerates `results/controlled_history/comparison.json`.

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
python code/extract_kinematics_figure.py
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
The whole-space figure locates the fixed stress paths on the final radial-strain
field, with a native-grid B-sector detail, A/B radial profiles, cumulative
deformation through the exact exterior, and the near-wall sign reversal.
`extract_kinematics_figure.py` samples derivatives of the completed displacement
history and its coupled exterior; it also reads the common elastic background
at the two wall locations. The compact
`radial_contributions.csv` contains all twelve states at both locations.
The source analysis keeps signed contributions of actual stepwise plastic
strains to fixed endpoint gauges. `sources/common_regions.csv` and the
`common_region_*` NPZ arrays use the same global side (`|x|>|y|`) and top/bottom
(`|y|>|x|`) sectors in every orientation. They include each source's signed
contribution and accumulated joint-slip activity at all twelve load steps.
The six-panel orientation figure uses these common sectors, with enlarged
final slip-field insets linking each region to its contribution curve.
The insets retain bedding and weak-plane directions and share one colorbar.
The reference perpendicular history gives 0.513745 mm vertical shortening and
0.054013 mm vertical plastic source; the common elastic part is 0.460184 mm.
Side and top/bottom sectors contribute 0.045018 and 0.008995 mm, respectively.
The slip insets query the common a<r<3a window even when the computational
interface lies at 4a.

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
- `code/run_orientation_extension.py`: sequential nonlinear training.
- `code/pilot/joint_return.py`: closed-form local weak-plane return.
- `code/pilot/returned_orientation.py`: displacement-only neural residuals.
- `code/pilot/virtual_work.py`: exterior-coupled virtual-work equilibrium.
- `code/pilot/plasticity.py`: yield surfaces and flow directions.
- `code/analyze_gauge_sources.py`: signed fixed-gauge source integration.
- `code/pilot/discrete_gauge.py` and `code/diagnose_gauge_sources.py`: endpoint
  adjoints in the training work space and separation of weak-equilibrium,
  auxiliary-field and source-quadrature effects.
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
Their initialization weights are supplied in `results/orientation_initialization/`.
The earlier pointwise precursor is in `results/initialization/model.pt`, with
settings in `configs/initialization.json`.

To evaluate the endpoint-source relation from the three saved histories:

```bash
python code/diagnose_gauge_sources.py
```

The output in `results/history_analysis/gauge_diagnostics/` contains the
direct gauges, plastic-source work and adjoint residual contributions, plus
source integrals over common side and top/bottom sectors. Neural displacement
and constitutive histories are evaluated at their saved weights.

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
