# NS_Solver

A 2D incompressible Navier-Stokes solver for modeling fluid flow around objects in a box domain.

## Methods and Capabilities

The solver uses the following methods and has the following capabilities:

1. **Parallel Computing** - Supports MPI for running on Linux clusters
2. **Finite Volume Method** - 2D spatial discretization on a staggered Cartesian grid
3. **Time Integration** - 3rd Order Strong Stability Preserving Runge-Kutta (SSP-RK3)
4. **Pressure-Velocity Coupling** - Fractional-step (projection) method
5. **Boundary Conditions** - Configurable per-face inflow/farfield/outflow/wall/periodic, with selectable wall slip/penetration and outflow mode
6. **Immersed Boundary Method (IBM)** - Support for arbitrary geometries (e.g., cylinders)
7. **Extensible to 3D** - Core algorithms support extension to 3D

## Installation

### Requirements
- Python 3.7+
- Dependencies listed in `requirements.txt`:
  - numpy
  - scipy
  - mpi4py (for parallel runs)
  - matplotlib (for visualization)
  - h5py
  - pytest

### Setup

1. Create and activate a virtual environment:
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
# or
source .venv/bin/activate  # Linux/Mac
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Solver

Run the solver with default configuration:
```bash
cd NS_Solver
python main.py
```

Override config parameters from the command line:
```bash
python main.py --nx 128 --ny 64 --re 200 --cylinder true
```

Run in parallel (4 MPI processes):
```bash
mpirun -n 4 python main.py
```

### Command-Line Arguments

All arguments are optional and override values in `config.txt`:

```
--config FILE       Path to configuration file (default: config.txt)
--nx INT           Grid cells in x direction (default: from config)
--ny INT           Grid cells in y direction (default: from config)
--lx FLOAT         Domain length in x (default: from config)
--ly FLOAT         Domain length in y (default: from config)
--re FLOAT         Reynolds number (default: from config)
--t_end FLOAT      Simulation end time (default: from config)
--cfl FLOAT        CFL number for time stepping (default: from config)
--save_dt FLOAT    Time interval between snapshots (default: from config)
--outdir DIR       Output directory for snapshots (default: from config)
--cylinder BOOL    Enable immersed-boundary cylinder (default: from config)
--plot BOOL        Show plots after simulation (default: from config)
```

## Configuration

Configuration is specified in `config.txt`. See `config_example.txt` for a complete example.

### Grid Parameters
- **nx** - Number of grid cells in x direction (default: 64)
- **ny** - Number of grid cells in y direction (default: 32)

### Domain Dimensions
- **lx** - Physical domain length in x direction (default: 4.0)
- **ly** - Physical domain length in y direction (default: 2.0)

### Physics Parameters
- **re** - Reynolds number ($Re = \frac{U_\infty L}{\nu}$). Higher values = less viscous flow (default: 100.0)

### Boundary Conditions
- **bc_left, bc_right, bc_bottom, bc_top** - Boundary type per face: `inflow`, `farfield`, `outflow`, `wall`, or `periodic`
- **inflow_u, inflow_v, inflow_w** - Velocity components used for inflow/farfield boundaries
- **wall_slip_mode** - Wall tangential behavior: `no-slip` or `free-slip`
- **wall_penetration** - Enable/disable wall-normal penetration at wall boundaries (`true` or `false`)
- **wall_normal_velocity** - Wall-normal velocity value used when `wall_penetration = true`
- **outflow_mode** - Outflow update model: `convective` or `zero-gradient`
- **outflow_speed** - Convective wave speed used by outflow boundaries in `convective` mode

### Time Integration
- **t_end** - Simulation end time (default: 5.0)
- **cfl** - CFL number for adaptive time stepping (default: 0.4). Lower values = smaller time steps = more stability

### I/O Parameters
- **save_dt** - Time interval between snapshot outputs (default: 0.5)
- **outdir** - Directory where snapshots are saved (default: output)

### Features
- **cylinder** - Enable immersed-boundary cylinder at domain center (default: false)
- **plot** - Display matplotlib plots at end of simulation (default: false)
- **verbose** - Print diagnostic information during run (default: true)

### Example: Quick Run
```
nx = 64
ny = 32
lx = 4.0
ly = 2.0
re = 100.0
t_end = 5.0
cfl = 0.4
save_dt = 0.5
outdir = output
cylinder = false
```

### Example: High Resolution
```
nx = 256
ny = 128
lx = 4.0
ly = 2.0
re = 100.0
t_end = 10.0
cfl = 0.4
save_dt = 1.0
outdir = output
cylinder = true
```

## Visualization

Simulation outputs are saved as `.npz` files (NumPy compressed arrays) in the snapshot directory.

### Generating Plots

View snapshots and generate visualizations:

```bash
# List available variables in the latest snapshot
python view_snapshot_viewer.py --list

# Plot pressure field from latest snapshot
python view_snapshot_viewer.py -k p --save pressure.png

# Plot drag/lift coefficient histories
python view_snapshot_viewer.py --plot-coeffs --save coeff_history.png

# Coefficient plots trim startup by default (t >= 0.5).
# Override if needed:
python view_snapshot_viewer.py --plot-coeffs --coeff-t-min 0.25 --save coeff_history.png

# Plot from a specific snapshot file
python view_snapshot_viewer.py output/snap_005.0000.npz -k p --save plot.png
```

### Image Output

By default, plots are automatically saved to the `results/` directory in PNG format at 150 DPI.

### Viewer Options
```
-k, --key KEY              Variable to plot (u, v, p, etc.)
-s, --slice IDX            Slice index for 3D arrays
-c, --comp IDX             Component index for vector fields
--plot-coeffs              Plot drag/lift coefficient histories
--coeff-file FILE.csv      Coefficient CSV path (forces.csv/aero.csv)
--coeff-indir DIR          Directory searched for coefficient CSVs (default: output)
--coeff-t-min FLOAT        Minimum time for coefficient plots (default: 0.5)
--save FILE                Save plot to file (saves to results/ folder)
```

## Strouhal Number Evaluation

Use `evaluate_strouhal.py` to compute dominant frequencies from a point probe
of `u`, `v`, and `p`, then convert to Strouhal number:

$$
St = \frac{fL}{U}
$$

### Example (Cylinder Case)

For the default cylinder setup in `main.py`, the cylinder diameter is
`D = ly/4`. You can use that directly as the characteristic length scale:

```bash
python evaluate_strouhal.py --probe-x 1.5 --probe-y 1.1 --use-cylinder-diameter --t-min 1.0
```

To also save a plain-text summary report (including Combined Strouhal):

```bash
python evaluate_strouhal.py --probe-x 1.5 --probe-y 1.1 --use-cylinder-diameter --t-min 1.0 --save-report results/strouhal_report.txt
```

### Strouhal Script Options

```
--indir DIR                Snapshot directory (default: output)
--pattern GLOB             Snapshot filename pattern (default: snap_*.npz)
--probe-x FLOAT            Probe x coordinate (required)
--probe-y FLOAT            Probe y coordinate (required)
--u-ref FLOAT              Reference velocity U (default: 1.0)
--length-scale FLOAT       Characteristic length L (default: 1.0)
--use-cylinder-diameter    Use L = ly/4 (consistent with default cylinder)
--t-min FLOAT              Ignore data before this time (default: 1.0)
--f-min FLOAT              Min frequency for search (default: 0.05)
--f-max FLOAT              Max frequency for search (default: 2.0)
--n-freq INT               Number of frequency samples (default: 4000)
--save-series FILE.csv     Export probe time series as CSV
--save-report FILE.txt     Export Strouhal summary report as TXT
```

