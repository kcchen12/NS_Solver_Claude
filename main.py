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
from src.io_utils import save_snapshot
from src.parallel import ParallelDecomposition
from src.config import ConfigParser


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
    p.add_argument("--plot",     type=str_to_bool, default=cfg.get("plot", False, bool),
                   help="Show matplotlib plots after simulation")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(args):
    # ------------------------------------------------------------------
    # MPI setup
    # ------------------------------------------------------------------
    decomp = ParallelDecomposition(args.ny)
    rank = decomp.rank
    is_root = rank == 0

    if is_root:
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
        print("=" * 60)

    # ------------------------------------------------------------------
    # Grid
    # ------------------------------------------------------------------
    grid = CartesianGrid(nx=args.nx, ny=args.ny,
                         lx=args.lx, ly=args.ly)

    # ------------------------------------------------------------------
    # Boundary conditions
    #   LEFT   : inflow  (u = 1, v = 0)
    #   RIGHT  : convective outflow
    #   BOTTOM : no-slip wall
    #   TOP    : no-slip wall
    # ------------------------------------------------------------------
    bc = BoundaryConfig(
        left=BCType.INFLOW,
        right=BCType.OUTFLOW,
        bottom=BCType.WALL,
        top=BCType.WALL,
        u_inf=1.0,
        v_inf=0.0,
    )

    # ------------------------------------------------------------------
    # Immersed boundary (optional cylinder)
    # ------------------------------------------------------------------
    ibm = ImmersedBoundary(grid)
    if args.cylinder:
        cx = args.lx / 4.0        # cylinder centre x
        cy = args.ly / 2.0        # cylinder centre y
        r = args.ly / 8.0        # radius
        ibm.add_circle(cx, cy, r)
        if is_root:
            print(f"  IBM cylinder: centre=({cx:.2f},{cy:.2f}), r={r:.4f}")

    # ------------------------------------------------------------------
    # Solver
    # ------------------------------------------------------------------
    nu = 1.0 / args.re   # kinematic viscosity (U=1, L=1 reference scales)
    solver = FractionalStepSolver(grid, bc, nu, ibm=ibm)
    solver.init_fields(u0=bc.u_inf, v0=bc.v_inf)

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

    if is_root:
        print(f"\n  Starting time loop …")

    while solver.t < args.t_end - 1e-12:
        dt = solver.suggest_dt(cfl_target=args.cfl)
        dt = min(dt, args.t_end - solver.t)

        solver.step(dt)
        step_count += 1

        # ---- diagnostics ----
        if is_root and step_count % 50 == 0:
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
                            lx=args.lx, ly=args.ly, re=args.re)
                save_snapshot(snap_path, solver.u, solver.v, solver.p,
                              solver.t, meta=meta, fmt="numpy")
            t_save_next += args.save_dt

    if is_root:
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

    X, Y = np.meshgrid(grid.xc, grid.yc, indexing="ij")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Speed
    im0 = axes[0].contourf(X, Y, speed, 50, cmap="viridis")
    fig.colorbar(im0, ax=axes[0])
    axes[0].set_title("Speed |u|")
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

    plot_path = os.path.join(args.outdir, "result.png")
    fig.savefig(plot_path, dpi=150)
    print(f"  Plot saved to {plot_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    run(args)
