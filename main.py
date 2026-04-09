"""
main.py — 2-D incompressible Navier-Stokes solver driver.

Simulates **uniform flow in a 2-D box** using:
    - MAC staggered Cartesian grid
    - Finite-volume spatial discretisation
    - SSP-RK3 time integration
    - Fractional-step (projection) pressure-velocity coupling
    - Convective outflow at the right boundary
    - No-slip walls on top and bottom
    - Inflow at the left boundary
    - (Optional) Immersed-boundary cylinder demo

Usage
-----
Serial run (reads from config.txt)::

    python main.py

Parallel run (4 MPI processes)::

    mpirun -n 4 python main.py

Command-line arguments override config file values::

    --config    Path to config file              [default: config.txt]
    --nx        Number of cells in x             [default: from config]
    --ny        Number of cells in y             [default: from config]
    --lx        Domain length in x               [default: from config]
    --ly        Domain length in y               [default: from config]
    --re        Reynolds number                  [default: from config]
    --t_end     End time                         [default: from config]
    --cfl       Target CFL number                [default: from config]
    --save_dt   Interval between snapshots       [default: from config]
    --outdir    Output directory                 [default: from config]
    --cylinder  Add an immersed-boundary cylinder [flag]
    --plot      Show matplotlib plots at the end  [flag]
"""

import argparse
import os
import sys
import numpy as np

from src.grid import CartesianGrid
from src.boundary import BoundaryConfig, BCType
from src.solver import FractionalStepSolver
from src.ibm import ImmersedBoundary
from src.io_utils import save_snapshot, save_grid_metadata, load_grid_metadata
from src.parallel import ParallelDecomposition
from src.config import ConfigParser


def _normalize_bc_type(raw_value: str, default: str) -> str:
    value = str(raw_value).strip().lower()
    valid = {
        BCType.INFLOW,
        BCType.FARFIELD,
        BCType.OUTFLOW,
        BCType.WALL,
        BCType.PERIODIC,
    }
    return value if value in valid else default


def parse_args():
    """
    Parse command-line arguments and merge with config file.

    Command-line arguments take precedence over config file values.
    All paths default to the script directory to ensure consistent output location.
    """
    def str_to_bool(v):
        """Convert string to boolean."""
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_config = os.path.join(script_dir, "config.txt")
    default_outdir = os.path.join(script_dir, "output")

    # First parse just the config file path
    p_pre = argparse.ArgumentParser(add_help=False)
    p_pre.add_argument("--config", type=str, default=default_config,
                       help="Path to configuration file")
    args_pre, remaining = p_pre.parse_known_args()

    # Read config file
    cfg = ConfigParser(args_pre.config)

    # Now parse all arguments with defaults from config file
    p = argparse.ArgumentParser(description="2-D Navier-Stokes solver")
    p.add_argument("--config", type=str, default=default_config,
                   help="Path to configuration file")
    p.add_argument("--nx",       type=int,   default=cfg.get("nx", 64, int))
    p.add_argument("--ny",       type=int,   default=cfg.get("ny", 32, int))
    p.add_argument("--lx",       type=float, default=cfg.get("lx", 4.0, float))
    p.add_argument("--ly",       type=float, default=cfg.get("ly", 2.0, float))
    p.add_argument("--re",       type=float, default=cfg.get("re", 100.0, float),
                   help="Reynolds number (Re = U_inf * L / nu)")
    p.add_argument("--t_end",    type=float,
                   default=cfg.get("t_end", 5.0, float))
    p.add_argument("--cfl",      type=float,
                   default=cfg.get("cfl", 0.4, float))
    p.add_argument("--save_dt",  type=float,
                   default=cfg.get("save_dt", 0.5, float))
    p.add_argument("--outdir",   type=str,
                   default=cfg.get("outdir", default_outdir, str))
    p.add_argument("--cylinder", type=str_to_bool, default=cfg.get("cylinder", False, bool),
                   help="Add an immersed-boundary cylinder at the domain centre")
    p.add_argument("--cylinder-radius", type=float,
                   default=cfg.get("cylinder_radius", -1.0, float),
                   help="Cylinder radius in physical units (<=0 uses default ly/8)")
    p.add_argument("--cylinder-center-x", type=float,
                   default=cfg.get("cylinder_center_x", -1.0, float),
                   help="Cylinder center x-coordinate in physical units (<0 uses default lx/4)")
    p.add_argument("--cylinder-center-y", type=float,
                   default=cfg.get("cylinder_center_y", -1.0, float),
                   help="Cylinder center y-coordinate in physical units (<0 uses default ly/2)")
    p.add_argument("--re-is-cylinder-based", type=str_to_bool,
                   default=cfg.get("re_is_cylinder_based", True, bool),
                   help="Interpret --re as Re_D based on cylinder diameter when cylinder is enabled")
    p.add_argument("--plot",     type=str_to_bool, default=cfg.get("plot", False, bool),
                   help="Show matplotlib plots after simulation")
    p.add_argument("--verbose",  type=str_to_bool, default=cfg.get("verbose", True, bool),
                   help="Print periodic diagnostics during the time loop")

    # Boundary condition configuration
    p.add_argument("--bc-left", type=str,
                   default=cfg.get("bc_left", BCType.INFLOW, str),
                   help="Left boundary type: inflow/farfield/outflow/wall/periodic")
    p.add_argument("--bc-right", type=str,
                   default=cfg.get("bc_right", BCType.OUTFLOW, str),
                   help="Right boundary type: inflow/farfield/outflow/wall/periodic")
    p.add_argument("--bc-bottom", type=str,
                   default=cfg.get("bc_bottom", BCType.WALL, str),
                   help="Bottom boundary type: inflow/farfield/outflow/wall/periodic")
    p.add_argument("--bc-top", type=str,
                   default=cfg.get("bc_top", BCType.WALL, str),
                   help="Top boundary type: inflow/farfield/outflow/wall/periodic")
    p.add_argument("--inflow-u", type=float,
                   default=cfg.get("inflow_u", 1.0, float),
                   help="Inflow/farfield x-velocity component")
    p.add_argument("--inflow-v", type=float,
                   default=cfg.get("inflow_v", 0.0, float),
                   help="Inflow/farfield y-velocity component")
    p.add_argument("--initial-v-perturbation-pct", type=float,
                   default=cfg.get("initial_v_perturbation_pct", 0.0, float),
                   help="One-time initial y-velocity perturbation as a percent of inflow_u")
    p.add_argument("--inflow-w", type=float,
                   default=cfg.get("inflow_w", 0.0, float),
                   help="Inflow/farfield z-velocity component for 3-D")
    p.add_argument("--wall-slip-mode", type=str,
                   default=cfg.get("wall_slip_mode", "no-slip", str),
                   help="Wall tangential model: no-slip or free-slip")
    p.add_argument("--wall-penetration", type=str_to_bool,
                   default=cfg.get("wall_penetration", False, bool),
                   help="Allow non-zero wall-normal velocity on wall boundaries")
    p.add_argument("--wall-normal-velocity", type=float,
                   default=cfg.get("wall_normal_velocity", 0.0, float),
                   help="Wall-normal velocity used when wall_penetration=true")
    p.add_argument("--outflow-mode", type=str,
                   default=cfg.get("outflow_mode", "convective", str),
                   help="Outflow update mode: convective or zero-gradient")
    p.add_argument("--outflow-speed", type=float,
                   default=cfg.get("outflow_speed", 1.0, float),
                   help="Convective outflow wave speed")
    return p.parse_args()


def _grid_metadata_path(args) -> str:
    return os.path.join(args.outdir, "uniform_grid.npz")


def _grid_matches_args(grid, args) -> bool:
    return (
        grid.nx == args.nx and
        grid.ny == args.ny and
        np.isclose(grid.lx, args.lx) and
        np.isclose(grid.ly, args.ly)
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def prepare_uniform_grid(args):
    """Build the runtime grid and write its metadata before the solver starts."""
    grid = CartesianGrid(nx=args.nx, ny=args.ny, lx=args.lx, ly=args.ly)
    os.makedirs(args.outdir, exist_ok=True)
    save_grid_metadata(_grid_metadata_path(args), grid)
    return grid


def get_runtime_grid(args):
    """Load a pre-generated grid when available, otherwise create one."""
    grid_path = _grid_metadata_path(args)
    if os.path.exists(grid_path):
        grid = load_grid_metadata(grid_path)
        if _grid_matches_args(grid, args):
            return grid, True
    return prepare_uniform_grid(args), False


def run(args, grid=None, grid_loaded_from_file=False):
    # ------------------------------------------------------------------
    # MPI setup
    # ------------------------------------------------------------------
    decomp = ParallelDecomposition(args.ny)
    rank = decomp.rank
    is_root = rank == 0

    if is_root and args.verbose:
        print("=" * 60)
        print("  2-D Incompressible Navier-Stokes Solver")
        print("=" * 60)
        print(f"  Config file   : {args.config}")
        print(f"  Grid          : {args.nx} x {args.ny}")
        print(f"  Domain        : {args.lx} x {args.ly}")
        print(f"  Reynolds no.  : {args.re}")
        print(f"  End time      : {args.t_end}")
        print(f"  MPI ranks     : {decomp.size}")
        print(f"  IBM cylinder  : {args.cylinder}")
        print(f"  Grid source   : {'pre-generated file' if grid_loaded_from_file else 'generated at startup'}")
        print("=" * 60)

    # ------------------------------------------------------------------
    # Grid
    # ------------------------------------------------------------------
    if grid is None:
        grid, grid_loaded_from_file = get_runtime_grid(args)

    # ------------------------------------------------------------------
    # Boundary conditions (fully configurable from config/CLI)
    # ------------------------------------------------------------------
    bc_left = _normalize_bc_type(args.bc_left, BCType.INFLOW)
    bc_right = _normalize_bc_type(args.bc_right, BCType.OUTFLOW)
    bc_bottom = _normalize_bc_type(args.bc_bottom, BCType.WALL)
    bc_top = _normalize_bc_type(args.bc_top, BCType.WALL)

    bc = BoundaryConfig(
        left=bc_left,
        right=bc_right,
        bottom=bc_bottom,
        top=bc_top,
        u_inf=args.inflow_u,
        v_inf=args.inflow_v,
        w_inf=args.inflow_w,
        wall_slip_mode=str(args.wall_slip_mode).strip().lower(),
        wall_penetration=bool(args.wall_penetration),
        wall_normal_velocity=args.wall_normal_velocity,
        outflow_mode=str(args.outflow_mode).strip().lower(),
        outflow_speed=args.outflow_speed,
    )

    # ------------------------------------------------------------------
    # Immersed boundary (optional cylinder)
    # ------------------------------------------------------------------
    ibm = ImmersedBoundary(grid)
    r = None
    if args.cylinder:
        cx = args.cylinder_center_x if args.cylinder_center_x >= 0.0 else args.lx / 4.0
        cy = args.cylinder_center_y if args.cylinder_center_y >= 0.0 else args.ly / 2.0
        r = args.cylinder_radius if args.cylinder_radius > 0.0 else args.ly / 8.0
        ibm.add_circle(cx, cy, r)
        if is_root and args.verbose:
            print(f"  IBM cylinder: centre=({cx:.2f},{cy:.2f}), r={r:.4f}")

    # ------------------------------------------------------------------
    # Solver
    # ------------------------------------------------------------------
    if args.cylinder and args.re_is_cylinder_based and r is not None:
        d_cyl = 2.0 * r
        nu = bc.u_inf * d_cyl / args.re
    else:
        nu = 1.0 / args.re

    if is_root and args.verbose and args.cylinder and args.re_is_cylinder_based and r is not None:
        d_cyl = 2.0 * r
        print(
            f"  Re interpretation: Re_D={args.re} with D={d_cyl:.4f} -> nu={nu:.6g}")

    initial_v_perturbation = 0.01 * \
        args.initial_v_perturbation_pct * bc.u_inf
    solver = FractionalStepSolver(grid, bc, nu, ibm=ibm)
    solver.init_fields(
        u0=bc.u_inf,
        v0=bc.v_inf,
        initial_v_perturbation=initial_v_perturbation,
    )
    if is_root and args.verbose and args.initial_v_perturbation_pct != 0.0:
        print(
            "  Initial v perturbation: "
            f"{args.initial_v_perturbation_pct:.3g}% of inflow_u "
            f"-> dv={initial_v_perturbation:.6g}"
        )

    # ------------------------------------------------------------------
    # Output directory
    # ------------------------------------------------------------------
    if is_root:
        os.makedirs(args.outdir, exist_ok=True)

    # ------------------------------------------------------------------
    # Time loop
    # ------------------------------------------------------------------
    t_save_next = 0.0
    step_count = 0

    if is_root and args.verbose:
        print(f"\n  Starting time loop …")

    while solver.t < args.t_end - 1e-12:
        dt = solver.suggest_dt(cfl_target=args.cfl)
        dt = min(dt, args.t_end - solver.t)

        solver.step(dt)
        step_count += 1

        # ---- diagnostics ----
        if is_root and args.verbose and step_count % 50 == 0:
            div_max = np.max(np.abs(solver.divergence()))
            cfl_val = solver.cfl(dt)
            print(f"  t={solver.t:8.4f}  dt={dt:.2e}  "
                  f"|∇·u|_max={div_max:.2e}  CFL={cfl_val:.3f}")

        # ---- save snapshot ----
        if solver.t >= t_save_next - 1e-12:
            if is_root:
                snap_path = os.path.join(
                    args.outdir, f"snap_{solver.t:08.4f}.npz")
                meta = dict(t=solver.t, nx=args.nx, ny=args.ny,
                            lx=args.lx, ly=args.ly, re=args.re,
                            ibm_force_x=solver.last_ibm_force_x,
                            ibm_force_y=solver.last_ibm_force_y)
                save_snapshot(snap_path, solver.u, solver.v, solver.p,
                              solver.t, meta=meta, fmt="numpy")
            t_save_next += args.save_dt

    if is_root and args.verbose:
        print(f"\n  Done.  t_final={solver.t:.4f},  steps={step_count}")

    # ------------------------------------------------------------------
    # Optional plot
    # ------------------------------------------------------------------
    if args.plot and is_root:
        _plot_results(solver, grid, args)

    return solver


# ---------------------------------------------------------------------------
# Plotting helper
# ---------------------------------------------------------------------------

def _plot_results(solver, grid, args):
    try:
        import matplotlib
        matplotlib.use("Agg")          # non-interactive backend for CI
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available – skipping plots.")
        return

    nx, ny = grid.nx, grid.ny

    # Interpolate to cell centres
    u_c = 0.5 * (solver.u[:-1, :] + solver.u[1:, :])
    v_c = 0.5 * (solver.v[:, :-1] + solver.v[:, 1:])
    speed = np.sqrt(u_c**2 + v_c**2)
    # 2-D scalar vorticity: omega_z = dv/dx - du/dy
    edge_x = 2 if grid.nx >= 3 else 1
    edge_y = 2 if grid.ny >= 3 else 1
    dv_dx = np.gradient(v_c, grid.xc, axis=0, edge_order=edge_x)
    du_dy = np.gradient(u_c, grid.yc, axis=1, edge_order=edge_y)
    omega = dv_dx - du_dy

    X, Y = np.meshgrid(grid.xc, grid.yc, indexing="ij")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Vorticity
    wmax = float(np.max(np.abs(omega)))
    wmax = max(wmax, 1e-8)
    levels = np.linspace(-wmax, wmax, 61)
    im0 = axes[0].contourf(X, Y, omega, levels=levels,
                           cmap="RdBu_r", extend="both")
    fig.colorbar(im0, ax=axes[0])
    axes[0].set_title(r"Vorticity $\omega_z$")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].set_aspect("equal")

    # Pressure
    im1 = axes[1].contourf(X, Y, solver.p, 50, cmap="RdBu_r")
    fig.colorbar(im1, ax=axes[1])
    axes[1].set_title("Pressure p")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    axes[1].set_aspect("equal")

    # Streamlines
    axes[2].streamplot(grid.xc, grid.yc,
                       u_c.T, v_c.T,
                       density=1.5, color=speed.T, cmap="plasma")
    axes[2].set_title("Streamlines")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("y")
    axes[2].set_aspect("equal")

    fig.suptitle(f"Re={args.re:.0f},  t={solver.t:.3f}")
    fig.tight_layout()

    results_dir = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "results")
    os.makedirs(results_dir, exist_ok=True)
    plot_path = os.path.join(results_dir, "result.png")
    fig.savefig(plot_path, dpi=150)
    print(f"  Plot saved to {plot_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    runtime_grid, loaded_from_file = get_runtime_grid(args)
    run(args, grid=runtime_grid, grid_loaded_from_file=loaded_from_file)
