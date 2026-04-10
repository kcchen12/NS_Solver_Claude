# NS_Solver

`NS_Solver` is a 2-D incompressible Navier-Stokes solver for box-domain flow. It uses a staggered MAC grid, configurable boundary conditions, optional immersed-boundary cylinder geometry, nonuniform center-focused meshes, MPI decomposition in `y`, and lightweight post-processing helpers.

## Overview

- Solves incompressible flow with a fractional-step / projection method
- Uses SSP-RK3 time integration
- Uses a staggered Cartesian finite-volume layout
- Supports configurable `inflow`, `farfield`, `outflow`, `wall`, and `periodic` boundaries
- Supports optional immersed-boundary cylinder cases
- Supports uniform and nonuniform 2-D grids
- Can generate grid-spacing plots, coefficient-history plots, and aerodynamic reports after a run

Additional notes live in [docs/README.md](docs/README.md).

## Repo Files

- [main.py](/Users/Carolyn/Desktop/NS_Solver_Claude/main.py): main solver entry point
- [config.txt](/Users/Carolyn/Desktop/NS_Solver_Claude/config.txt): run configuration
- [post_config.txt](/Users/Carolyn/Desktop/NS_Solver_Claude/post_config.txt): post-processing configuration
- [pre_generate_grid.py](/Users/Carolyn/Desktop/NS_Solver_Claude/pre_generate_grid.py): standalone prepared-grid generator
- [analyze_aerodynamics.py](/Users/Carolyn/Desktop/NS_Solver_Claude/analyze_aerodynamics.py): aerodynamic coefficient and Strouhal-style analysis
- [view_snapshot_viewer.py](/Users/Carolyn/Desktop/NS_Solver_Claude/view_snapshot_viewer.py): snapshot and coefficient-history plotting

## Quick Start

Typical setup:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

Use an explicit run config and post config:

```bash
python main.py --config config.txt --post-config post_config.txt
```

Run in parallel:

```bash
mpirun -n 4 python main.py
```

## Setup

Requirements:

- Python 3
- `numpy`
- `scipy`
- `matplotlib`
- `h5py`
- `pytest`
- `mpi4py` if you want MPI runs

## Config Files

The project now uses two config files on purpose:

- [config.txt](/Users/Carolyn/Desktop/NS_Solver_Claude/config.txt): how the simulation runs
- [post_config.txt](/Users/Carolyn/Desktop/NS_Solver_Claude/post_config.txt): which derived plots/reports get generated after the run

Examples are provided in [config_example.txt](/Users/Carolyn/Desktop/NS_Solver_Claude/docs/examples/config_example.txt) and [post_config_example.txt](/Users/Carolyn/Desktop/NS_Solver_Claude/docs/examples/post_config_example.txt).

### Run Config

Core grid and physics:

- `nx`, `ny`: cell counts
- `lx`, `ly`: domain size
- `re`: Reynolds number
- `t_end`: end time
- `cfl`: target CFL
- `save_dt`: snapshot interval
- `outdir`: snapshot/grid output directory

Grid control:

- `uniform_grid`: `true` for uniform, `false` for nonuniform
- `grid_beta_x`, `grid_beta_y`: smooth center-density boost in each direction
- `grid_band_fraction_x`, `grid_band_fraction_y`: fraction of the domain refined around the center band

Boundary conditions:

- `bc_left`, `bc_right`, `bc_bottom`, `bc_top`
- `inflow_u`, `inflow_v`, `inflow_w`
- `wall_slip_mode`
- `wall_penetration`
- `wall_normal_velocity`
- `outflow_mode`
- `outflow_speed`

Cylinder / immersed boundary:

- `cylinder`
- `cylinder_center_x`, `cylinder_center_y`
- `cylinder_radius`
- `re_is_cylinder_based`

Initialization and runtime plotting:

- `initial_v_perturbation_pct`: one-time startup perturbation applied to interior `v` as a percent of `inflow_u`
- `plot`: save the standard end-of-run result figure
- `verbose`: print run diagnostics

### Post Config

- `plot_grid`: save the grid-spacing/concentration figure
- `auto_generate_grid_spacing`: automatically generate `results/grid.png`
- `auto_generate_velocity_u`: automatically generate `results/velocity_u.png`
- `auto_generate_velocity_v`: automatically generate `results/velocity_v.png`
- `auto_generate_coeff_history`: automatically generate `results/coeff_history.png`
- `auto_generate_aero_report`: automatically generate `results/aero_report.txt`
- `auto_coeff_t_min`: minimum time for automatic coefficient-history plotting
- `auto_aero_t_min`: minimum time for automatic aerodynamic analysis

## Runtime Notes

### Initial Perturbation

`initial_v_perturbation_pct` is easy to miss but useful for wake development studies and vortex-shedding startup tests.

If:

- `inflow_u = 1.0`
- `initial_v_perturbation_pct = 2.0`

then the solver applies a one-time interior perturbation of:

```text
dv = 0.02 * inflow_u = 0.02
```

This does not permanently change the configured boundary condition values. It only seeds the initial interior field.

### Nonuniform Grid

The current nonuniform grid is a center-band refinement, not a boundary-layer stretch.

- Outside the refined band, spacing stays close to normal uniform spacing
- Inside the band, cell density increases smoothly toward the middle
- `beta_x` and `beta_y` control how much denser the middle becomes
- `band_fraction_x` and `band_fraction_y` control how wide that refined middle region is

Prepared grid metadata is saved automatically into `outdir` as either:

- `uniform_grid.npz`
- `nonuniform_grid.npz`

## Prepared Grid Generation

Generate prepared grid metadata without running the full solve:

```bash
python pre_generate_grid.py
```

Generate a nonuniform prepared grid:

```bash
python pre_generate_grid.py --grid-type nonuniform --beta-x 2.5 --beta-y 2.0 --band-fraction-x 0.33 --band-fraction-y 0.33
```

Common options:

- `--nx`, `--ny`, `--lx`, `--ly`
- `--grid-type uniform|nonuniform`
- `--beta-x`, `--beta-y`
- `--band-fraction-x`, `--band-fraction-y`
- `--focus-x`, `--focus-y`
- `--outdir`
- `--output-name`

## Outputs

Common outputs:

- `output/snap_*.npz`: solution snapshots
- `output/uniform_grid.npz` or `output/nonuniform_grid.npz`: prepared grid metadata
- `results/result.png`: standard flow plot from `main.py` when `plot = true`
- `results/grid.png`: physical grid / spacing plot
- `results/aero.csv`: coefficient and force history
- `results/coeff_history.png`: drag/lift history figure
- `results/aero_report.txt`: aerodynamic summary report

## Post-Processing

### Grid Spacing Plot

Enable in [post_config.txt](/Users/Carolyn/Desktop/NS_Solver_Claude/post_config.txt):

```text
plot_grid = true
auto_generate_grid_spacing = true
```

This produces `results/grid.png` with:

- the physical grid
- point concentration / cell density
- spacing curves

### Velocity Snapshots

Enable in [post_config.txt](/Users/Carolyn/Desktop/NS_Solver_Claude/post_config.txt):

```text
auto_generate_velocity_u = true
auto_generate_velocity_v = true
```

These use the existing snapshot-viewer plotting path and save:

- `results/velocity_u.png`
- `results/velocity_v.png`

### Coefficient History Plot

Generate it manually from the viewer:

```bash
python view_snapshot_viewer.py --plot-coeffs --save coeff_history.png
```

Useful viewer options:

- `--coeff-file`
- `--coeff-indir`
- `--coeff-t-min`
- `--save`

### Aerodynamic Analysis

Run directly:

```bash
python analyze_aerodynamics.py --indir output --config config.txt --use-cylinder-diameter --t-min 1.0 --save-series results/aero.csv --save-report results/aero_report.txt
```

This script computes:

- force histories
- `C_d` and `C_l`
- a dominant lift frequency / Strouhal-style estimate
- summary statistics in a text report

Useful analysis options:

- `--indir`
- `--pattern`
- `--config`
- `--probe-x`, `--probe-y`
- `--u-ref`
- `--length-scale`
- `--use-cylinder-diameter`
- `--cylinder-radius`
- `--t-min`
- `--f-min`, `--f-max`, `--n-freq`
- `--save-series`
- `--save-report`

## Snapshot Viewer

Examples:

```bash
python view_snapshot_viewer.py --list
python view_snapshot_viewer.py -k p --save pressure.png
python view_snapshot_viewer.py output/snap_005.0000.npz -k u --save velocity.png
```

Useful viewer options:

- `-k`, `--key`
- `-s`, `--slice`
- `-c`, `--comp`
- `--save`
- `--plot-coeffs`

## Notes

- The repo uses the split-config workflow: runtime settings in [config.txt](/Users/Carolyn/Desktop/NS_Solver_Claude/config.txt) and derived-output controls in [post_config.txt](/Users/Carolyn/Desktop/NS_Solver_Claude/post_config.txt)
- There is no `evaluate_strouhal.py` in this repo; use [analyze_aerodynamics.py](/Users/Carolyn/Desktop/NS_Solver_Claude/analyze_aerodynamics.py) for current frequency/report analysis
- The current code and examples match the tested solver and post-processing paths in this repository
