# OES-PINN: anisotropic elastoplastic tunnel unloading

Code and data for **Anisotropic elastoplastic deformation during tunnel
unloading using an operator-embedded spectral PINN**, by Pengcheng Zhu and
Tielin Chen.

OES-PINN combines incremental neural displacement fields, a mapped spectral
representation, local matrix/weak-plane plasticity and an exact anisotropic
elastic exterior. The learned continuation fits spatial features to eight
early increments and recombines them over the final four increments. The
full-network calculation updates all neural parameters at every increment.
The two routes use the same constitutive and exterior operators.

## Contents and units

| Location | Content |
| --- | --- |
| `code/` | Learned continuation, full-network training, direct spectral solves, material replay and data analysis |
| `configs/` | Coarse (12,24), refined (18,36), and 24-increment settings |
| `inputs/horseshoe_geometry.json` | Six tangent circular arcs of the reference opening; coordinates in meters |
| `inputs/arch_observations.csv` | Four measured displacement series, in millimeters, with relative monitoring days and original reading order |
| `inputs/material_moduli.csv` | Nine reported rock categories and equivalent-modulus ranges, in MPa |
| `inputs/material_campaign.json` | The pooled 47-group specimen summary and measurement method |
| `inputs/contact_pressure.csv` | Measured maximum stable rock–support pressure and adopted uniform confinement, in MPa |
| `data/<case>/responses.csv` | Seven displacement histories and their elastic/plastic parts, in millimeters |
| `data/<case>/coefficients.csv` | Dimensionless incremental spectral coefficients for reconstruction of the reported fields |
| `data/<case>/sources.csv` | Signed plastic-displacement contributions by step, measurement, mechanism and region, in millimeters |
| `data/timings.csv` | Recorded solution and source-computation times, in seconds, and active coordinate counts |
| `data/arch_predictions.csv` | Predicted arch responses under the reference and two scaled geometries |

The experimental and field inputs concern a deeply buried tunnel in
southwestern China. The material campaign used a point-load device and dial
gauge on irregular slate and carbonaceous-slate specimens. Its 47-group pooled
mean is 560.15 MPa; the category file contains the reported ranges, with group
counts where specified. The mean comes from the pooled campaign, rather than
an average of the category endpoints. Contact pressure supplies the confinement
scale. The arch file retains repeated observation days; `reading` preserves
their order. Crown settlement and full-span chord closures use the measurement
conventions in the article. The same day-50 observations are compared both as
cumulative readings and as changes from the first reading. Monitoring days
are not mapped to the model's unloading fraction.

`data/histories.json` links each history to its configuration. `full_network`
is the detailed mechanical and measured-response case; `network_coarse` and
`network_steps24` supply its resolution comparisons. `accuracy_reference`
is the fine direct accuracy reference. `learned`, `untrained`, `spectral`,
`spectral_prefix` and `prefix_only` share the first eight direct increments.
`direct_coarse` and `direct_fine` are the complete timed direct calculations.
The `prefix_only` comparison has displacement data but no regional source
evaluation. Regional sources use 36 Gauss points in each depth band and
768 angles. Mechanism labels `matrix`, `joint` and `tension` denote matrix
plasticity, weak-plane shear and weak-plane tension; `total` is their sum.
Signed contributions may be negative. They refer to each increment and must
be summed over increments for cumulative contributions.

The coefficient table orders radial Chebyshev modes first and the angular
sequence as the constant, cosine modes 1 through M, and sine modes 1 through M;
the constant translation mode is omitted. Its two columns are displacement
components. `reconstruct_shared_history.py` reads this ordering directly.
Displacement scales by `SIGMA_REF * length_scale_m / E_REF`; stress scales by
`SIGMA_REF`, with SI reference values defined in `code/oespinn/config.py`.
The shared coefficients reproduce the reported fields without retraining.
Training commands produce new network weights and numerical outputs locally.

## Installation and table reconstruction

Use Python 3.11 or later and the pinned NumPy, SciPy and PyTorch versions:

```sh
python -m pip install -r requirements.txt
python code/analyze_shared_data.py --output results/tables
```

The table command uses Python's standard library and reconstructs the
displacement/source errors, complete times, arch comparisons and deep-source
fractions directly from the shared data. Its learned and coarse-direct total
times are 192.31 and 296.96 s. The paper's timings use a Tesla PG503-216 GPU
and four CPU threads; rerun timings depend on hardware and software.
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

## Learned and direct calculations

```sh
python code/run_horseshoe_feature_history.py --config configs/refined.json --output results/learned
python code/solve_horseshoe_coefficients.py --config configs/coarse.json --output results/direct_coarse
python code/solve_horseshoe_coefficients.py --config configs/refined.json --output results/direct_fine
python code/solve_horseshoe_reuse.py --method random --source results/learned/prefix.pt --output results/untrained
python code/solve_horseshoe_reuse.py --method spectral --source results/learned/prefix.pt --output results/spectral
python code/solve_horseshoe_reuse.py --method mixed --source results/learned/prefix.pt --output results/spectral_prefix
python code/solve_horseshoe_reuse.py --method snapshots --source results/learned/prefix.pt --output results/prefix_only
```

Use fresh output directories for complete timing: the direct solver resumes
an existing checkpoint. Apply the regional-integration command to each new
history and add its computation time to solution time. The learning command
includes all eight prefix solves, feature fitting, basis construction and the
four reduced solves. The default direct calculation uses the common 0.15 mm
residual-work bound. The shared `accuracy_reference` uses tighter tolerances
before final release and is distinct from the standard-tolerance timed fine
calculation. Their reported values are retained separately in `data/`.

## Full-network calculations

```sh
python code/run_horseshoe.py --config configs/coarse.json --output results/network_coarse --optimizer tangent --equilibrium-tolerance-mm 0.05 --response-relative-tolerance 0.0005
python code/run_horseshoe.py --config configs/refined.json --output results/full_network --initialize-from results/network_coarse/model.pt --optimizer tangent --equilibrium-tolerance-mm 0.15 --response-relative-tolerance 0.0005
python code/run_horseshoe.py --config configs/steps24.json --output results/network_steps24 --initialize-from results/network_coarse/model.pt --optimizer tangent --equilibrium-tolerance-mm 0.05 --response-relative-tolerance 0.0005
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
