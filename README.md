# OES-PINN: anisotropic elastoplastic tunnel unloading

Version 2026.11.25 contains the code and data for **An operator-embedded
spectral PINN for quantifying the contributions of bedding slip and matrix
yielding to tunnel closure**, by Pengcheng Zhu and Tielin Chen.

OES-PINN combines incremental neural displacement fields, a mapped spectral
representation, local matrix/weak-plane plasticity and an exact anisotropic
elastic exterior. The eight-step learned space supports the fixed-configuration
comparison. The six-step forecasts use either a width selected from early
movements or a prescribed width of 64. The selected width also supplies
the starting space for the current-tangent correction. The
full-network calculation retrains the neural displacement at each increment.

## Contents and units

| Location | Content |
| --- | --- |
| `code/` | Learned continuation, full-network training, direct spectral solves, material replay and data analysis |
| `configs/` | Coarse (12,24), refined (18,36), and 24-increment settings, with one configuration per material or interface case |
| `inputs/horseshoe_geometry.json` | Six tangent circular arcs of the reference opening; coordinates in meters |
| `inputs/arch_observations.csv` | Four measured displacement series, in millimeters, with relative monitoring days and original reading order |
| `inputs/specimen_moduli.csv` | Lithology, specimen condition, and initial and linear-branch moduli for 47 specimens, in MPa |
| `inputs/material_moduli.csv` | Linear-branch ranges and specimen counts by reported rock class and condition |
| `inputs/material_campaign.json` | Measurement definition, equal-weight mean and median |
| `inputs/contact_pressure.csv` | Measured maximum stable rock–support pressure and adopted uniform confinement, in MPa |
| `inputs/contact_pressure_readings.csv` | Stable contact pressures at the five positions with readings, in MPa |
| `data/<case>/responses.csv` | Seven displacement histories and their elastic/plastic parts, in millimeters |
| `data/<case>/coefficients.csv` | Dimensionless incremental spectral coefficients for reconstruction of the reported fields |
| `data/<case>/sources.csv` | Signed plastic-displacement contributions by step, measurement, mechanism and region, in millimeters |
| `data/timings.csv` | Recorded solution and source-computation times, in seconds, and active coordinate counts |
| `data/network_optimization_times.csv` | Recorded optimizer blocks and elapsed optimization time by release for the coarse and fine histories used to produce the mechanism maps |
| `data/arch_predictions.csv` | Full-unloading movements used to examine sensitivity to opening size and chord depth |
| `data/cold_start_comparison.csv` | Six-step forecast, coarse direct and separately timed full-network results, including selection, integration and total seconds |
| `data/arch_installation_states.csv` | Remaining movements at the sampled installation states. Table 7 uses both geometries at release 0.75. Columns with the suffix `_direct` use the fine direct history. |
| `data/elastic_comparison.csv` | Seven first-release FE, spectral and full-network movements for the two far-boundary meshes in Section 4.1 |
| `data/bedding_material_path.csv` | Stresses in MPa and dimensionless plastic increments in bedding coordinates, 1 m outside the wall on the 135° ray. The two integrations follow the same strain history at twelve or twenty-four increments. |
| `data/material_and_interface_response.csv` | Terminal closures, mechanism contributions and sampled crossover intervals for the coarse reference and its eleven variations |
| `data/bedding_friction_closures.csv` | Sidewall and vertical closures at the twelve sampled releases in the coarse and fine direct solutions with 22° bedding friction |
| `data/interface_tractions.csv` | Tractions and exterior yield margins on comparing the two interface locations throughout unloading |

The experimental and field inputs concern a deeply buried tunnel in
southwestern China. The material campaign used a point-load device and dial
gauge on irregular slate and carbonaceous-slate specimens. Each force–shortening
curve supplies one linear-branch apparent axial modulus. Equal specimen weights
give a mean of 560.16 MPa and a median of 259.45 MPa.
The model adopts a stiffness scale of 0.56 GPa.
The contact pressures stabilized about 30 days after excavation. The five
readings cover both sidewalls, both arch feet and the invert. The largest
reading, 0.71350 MPa at the left sidewall, sets the scale of the uniform
0.70 MPa traction. The arch file retains repeated observation days; `reading` preserves
their order. Crown settlement and full-span chord closures use the measurement
conventions in the article. Table 7 pairs the day-50 cumulative settlement
and upper-chord closure with the movements remaining after an assumed
installation at three-quarters of unloading. The support-design radius
and the nominal shaft width are compared at the measured 2.5 m chord depth.

`data/histories.json` links each history to its configuration. `full_network`
is the detailed mechanical and measured-response case; `network_coarse` and
`network_steps24` supply its resolution comparisons. `accuracy_reference`
is the fine direct accuracy reference. `learned`, `untrained`, `spectral`,
`spectral_prefix` and `prefix_only` share the first eight direct increments.
`direct_coarse` and `direct_fine` are the complete timed direct calculations.
Table 5 uses `learned`, `full_network` and `accuracy_reference`. Figures 9–11
and the predictions in Table 7 use `full_network`. The two arch mappings
were also evaluated from `accuracy_reference`. Table 3 uses `early_forecast`,
`width_control`, `current_enrichment` and `cold_direct_coarse`.
Its times cover the complete solver and integration processes. The separate
coarse baseline in Table 4 sums setup, the recorded solves for the retained
increments and the integration computation. In Section 4.5 the 5076.5 s total belongs to
`timed_full_network`, including its coarse neural initialization.
The `prefix_only` comparison has displacement data but no regional source
evaluation. Regional sources use 36 Gauss points in each depth band and
768 angles. Mechanism labels `matrix`, `joint` and `tension` denote matrix
plasticity, weak-plane shear and weak-plane tension; `total` is their sum.
Signed contributions may be negative. They refer to each increment and must
be summed over increments for cumulative contributions.

To repeat the material-point comparison in Section 4.1 from the shared
full-network coefficients, run the following at the root of this package.
The principal-axis calculation limits bedding shear by the current normal
traction and solves a scalar equation for matrix flow. The coupled calculation
solves simultaneously for the stress and the three mechanism multipliers.

```sh
python code/compare_horseshoe_material_path.py --output results/bedding_material_path.csv
```

The coefficient table orders radial Chebyshev modes first and the angular
sequence as the constant, cosine modes 1 through M, and sine modes 1 through M;
the constant translation mode is omitted. Its two columns are displacement
components. `reconstruct_shared_history.py` reads this ordering directly.
Displacement scales by `SIGMA_REF * length_scale_m / E_REF`; stress scales by
`SIGMA_REF`, with SI reference values defined in `code/oespinn/config.py`.
The shared coefficients reproduce the reported fields without retraining.
Training commands produce new network weights and numerical outputs locally.

## Installation and table reconstruction

To reconstruct the tables offline, extract the archive and run the
following command from the directory containing this README with Python
3.11 or later.

```sh
python code/analyze_shared_data.py --output results/tables
```

The table command uses Python's standard library. In `forecast_comparison.csv`
(Table 3), each step error is the largest relative error among the seven
movements, calculated against the fine direct history at the same release.
The slip column sums the upper-chord contributions over the entire unloading
path. The measured times come from `data/cold_start_comparison.csv`.
The command also reconstructs all nine rows of Table 6 from the final
movements and incremental mechanism contributions in the case directories.
These results are in `material_comparison.csv`. The other outputs give the
eight-step accuracy and timing comparison, the deep-source fractions and
the four rows of Table 7 in `arch_comparison.csv`.
The two movements for the adjacent installation releases in Section 5.3
are in `data/arch_installation_states.csv`.
Table 3 reports 258.2 s for the frozen width selected from early movements,
352.2 s for prescribed width 64 and 462.5 s for coarse direct equilibrium,
including regional integration. The 192.31 and 296.96 s values in
`feature_comparison.csv` refer to the timed stages of the eight-step learned
and corresponding coarse direct calculations. The paper's timings use a Tesla
PG503-216 GPU and four CPU threads. The package was checked on Windows 10
with a Ryzen Threadripper 3970X, Python 3.13.7 and PyTorch 2.8.0 with CUDA 12.6.
For field reconstruction and new solves, install the versions in
`requirements.txt` with `python -m pip install -r requirements.txt`.
The feature-learning and reduced-space solvers require CUDA. Install a
CUDA-enabled PyTorch 2.8.0 build appropriate to the GPU. Direct solves,
full-network training and data replay also accept `--device cpu`.
Run all commands from the repository root.

## Reconstruct fields and plastic sources

```sh
python code/reconstruct_shared_history.py --case full_network --output results/full_network/model.pt
python code/assess_horseshoe.py results/full_network/model.pt --output results/full_network/independent --nr 64 --nt 512 --test-j 24 --test-m 48 --device cpu
python code/integrate_horseshoe_regions.py results/full_network/model.pt --output results/full_network/regions --nr 36 --nt 768 --device cpu --chunk-size 8192
python code/evaluate_shared_fields.py --case full_network --steps 8 9 10 11 12 --output results/full_network/fields.csv --device cpu
```

Choose any case in `data/histories.json`. The reconstructed file contains
spectral coefficients and physical settings for material replay. The field
export reports position in meters, displacement in millimeters, stress in
MPa and dimensionless plastic strain; `plastic_gamma_xy` is engineering shear
strain. `--radial-points`, `--angular-points` and `--radius-m` control the
wall-fitted sampling grid. The default outer sampling radius is 19 m, inside
the 40.38 m coupling interface. The source command evaluates all seven
measurements by mechanism, sector, radial-depth band and unloading increment.

## Elastic comparison and sensitivity studies

Section 4.1 uses the same six arcs and material constants for two independent
finite-element meshes. The comparison reads the first release from the fine
direct and full-network histories. Reproduce both meshes with

```sh
python code/compare_shared_elastic.py --output results/elastic_comparison.csv
```

For Table 6, each varied case in `data/material_and_interface_response.csv`
has a configuration of the same name in `configs/`. The four tabulated
columns are `horizontal_mm`, `vertical_mm`, `matrix_vertical_minus_horizontal_mm`
and `upper_shear_mm`. The reference row
`cold_direct_coarse` uses `configs/coarse.json`. For example, to calculate the
22-degree bedding-friction case and integrate its mechanism contributions, run

```sh
python code/solve_horseshoe_coefficients.py --config configs/friction22.json --output results/friction22
python code/integrate_horseshoe_regions.py results/friction22/model.pt --output results/friction22/regions --nr 36 --nt 768
```

For the fine solution in Section 5.1, use `friction22_refined` in place of
`friction22` in both commands. The two configurations have the same material,
geometry and loading history. `data/bedding_friction_closures.csv` gives the
wall movements at matching releases for both spatial discretizations.
The `friction22` and `friction22_refined` history directories also contain
their incremental coefficients and the per-mechanism contributions at each release.

The `ratio125`, `ratio175`, `cohesion012` and `pressure056` runs used
`--recover-stalled` on resuming the direct solve after a stalled line search.
This permits an intermediate predictor with the preceding plastic state held
fixed and accepts a stalled endpoint only when the residual work at every
gauge is within 0.15 mm. The other cases used the full residual-norm criterion.
To reproduce the interface comparison from the stored displacement histories,
evaluate the exterior tractions on the same physical circle for both solutions.

```sh
python code/reconstruct_shared_history.py --case cold_direct_coarse --output results/cold_direct_coarse/model.pt
python code/reconstruct_shared_history.py --case interface4 --output results/interface4/model.pt
python code/compare_horseshoe_radius.py results/cold_direct_coarse/model.pt results/interface4/model.pt results/interface_tractions.csv
```

## Learned and direct calculations

The six- and eight-step feature fits use early direct solutions at
orders (18,36), with 48 radial by 384 angular material points and 768
boundary points. Each of the three hidden layers uses tanh and seed 12
initializes the network. An L-BFGS block allows 100 iterations and 125
function evaluations. Fitting ends when the relative elastic-energy error
falls below 3% for every training increment, with up to 80 blocks.
The Cholesky factor of the elastic stiffness weights the candidate basis
before its singular value decomposition. The code keeps singular values
above `1e-10` times the largest. The six-step fixed bases have ranks 199
(width 32) and 391 (width 64). The enriched width-32 run reaches rank 205.

```sh
python code/run_horseshoe_feature_history.py --config configs/refined.json --output results/learned
python code/run_horseshoe_feature_history.py --config configs/refined.json --early-selection --output results/early_forecast
python code/run_horseshoe_feature_history.py --config configs/refined.json --early-selection --current-enrichment --output results/current_enrichment
python code/run_horseshoe_feature_history.py --config configs/refined.json --prefix-step 6 --width 64 --output results/width_control
python code/solve_horseshoe_coefficients.py --config configs/coarse.json --output results/direct_coarse
python code/solve_horseshoe_coefficients.py --config configs/refined.json --output results/direct_fine
python code/solve_horseshoe_reuse.py --method random --output results/untrained
python code/solve_horseshoe_reuse.py --method spectral --output results/spectral
python code/solve_horseshoe_reuse.py --method mixed --output results/spectral_prefix
python code/solve_horseshoe_reuse.py --method snapshots --output results/prefix_only
```

The four trial-space comparisons read `results/learned/prefix.pt`, written by
the first command after it solves the first eight increments. Use `--source`
to select a prefix saved at a different location.

For the six-step enriched calculation, `code/oespinn/neural_reduction.py`
projects the GMRES direction off the learned space in elastic-energy
coordinates. The new component is retained if its norm exceeds the
original direction's norm times the full dimension and machine precision.
The cycle can supply a direction on reaching its iteration limit. The
nonlinear solve determines its amplitude. In `run_horseshoe_feature_history.py`
the plastic state advances after the last reduced solve meets the
projected-residual tolerance at that release.

Use fresh output directories for complete timing: the direct solver resumes
an existing checkpoint. Apply the regional-integration command to each new
history and add its computation time to solution time. The learning command
includes all eight prefix solves, feature fitting, basis construction and the
four reduced solves. The default direct calculation uses the common 0.15 mm
residual-work bound throughout the timed fine calculation in Table 4.
The fine accuracy reference uses a normalized residual tolerance of
0.0001 at steps 9 and 10. Its residual-work bounds are 0.10 mm at
step 11 and 0.15 mm at step 12. The displacement coefficients for all
twelve reference increments are in `data/accuracy_reference/`.

## Full-network calculations

The networks initialized with seed 1729 provide the incremental fields for
the mechanism maps. The timing rows in `data/network_optimization_times.csv`
cover 11 completed coarse predictor steps (5238.5 s) and 94 retained fine
optimization blocks across 12 steps (16,077.2 s), giving a lower bound of
5.92 h. The end-to-end time for that history was not recorded.

```sh
python code/run_horseshoe.py --config configs/coarse.json --output results/network_coarse --optimizer tangent --equilibrium-tolerance-mm 0.05 --response-relative-tolerance 0.0005
python code/run_horseshoe.py --config configs/refined.json --output results/full_network --initialize-from results/network_coarse/model.pt --optimizer tangent --equilibrium-tolerance-mm 0.15 --response-relative-tolerance 0.0005
python code/run_horseshoe.py --config configs/steps24.json --output results/network_steps24 --initialize-from results/network_coarse/model.pt --optimizer tangent --equilibrium-tolerance-mm 0.05 --response-relative-tolerance 0.0005
```

For the separately timed two-stage network, both stages use the constrained
output-weight update and start in fresh output directories:

```sh
python code/run_horseshoe.py --config configs/coarse.json --output results/timed_predictor --optimizer tangent --tangent-evaluations 12 --equilibrium-tolerance-mm 0.15 --response-relative-tolerance 0.0005 --polish-stalled-gauges
python code/run_horseshoe.py --config configs/refined.json --output results/timed_full_network --initialize-from results/timed_predictor/model.pt --optimizer tangent --tangent-evaluations 12 --equilibrium-tolerance-mm 0.15 --response-relative-tolerance 0.0005 --polish-stalled-gauges
```

Each increment optimizes its neural parameters using the consistent material
tangent and scaled trust-region least squares. Initialization supplies a
predictor; all increments are retrained and material states are replayed in
load order. The shared coefficient data preserve the published numerical
realizations, including the histories developed through successive numerical
refinements. A new optimization run can differ within the stated stopping
tolerances. Use coefficient reconstruction for the reported fields and the
training commands to repeat the method.

The earlier circular benchmark code and data remain available in
[the October 2026 version](https://github.com/math-sudu/exact-subdomain-ddpinn/tree/e2916bc51327e65890a414735dbe7b9f3cc30716).
