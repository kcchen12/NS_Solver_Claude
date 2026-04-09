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

from src.grid import CartesianGrid, build_nonuniform_grid_metadata
from src.boundary import BoundaryConfig, BCType
from src.solver import FractionalStepSolver
from src.ibm import ImmersedBoundary
from src.io_utils import (
    save_snapshot,
    save_grid_metadata,
    save_grid_metadata_dict,
    load_prepared_grid,
    load_grid_metadata_dict,
)
from src.parallel import ParallelDecomposition
from src.config import ConfigParser
from analyze_aerodynamics import run_analysis as run_aero_analysis
from view_snapshot_viewer import find_latest_snapshot, plot_coeff_history, plot_snapshot_key


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

    # First parse just the config file paths
    p_pre = argparse.ArgumentParser(add_help=False)
    p_pre.add_argument("--config", type=str, default=default_config,
                       help="Path to configuration file")
    p_pre.add_argument("--post-config", type=str,
                       default=os.path.join(script_dir, "post_config.txt"),
                       help="Path to post-processing configuration file")
    args_pre, remaining = p_pre.parse_known_args()

    # Read config files
    cfg = ConfigParser(args_pre.config)
    post_cfg = ConfigParser(args_pre.post_config)

    # Unified grid controls from config.txt.
    # Backward compatibility:
    # 1) If uniform_grid is set, it overrides string grid type keys.
    # 2) Otherwise grid_type can override legacy runtime_grid_type.
    # 3) grid_beta_x/y can override legacy runtime/pre betas.
    uniform_grid = cfg.get("uniform_grid", None, bool)
    if uniform_grid is None:
        grid_type_default = cfg.get(
            "grid_type", cfg.get("runtime_grid_type", "uniform", str), str
        )
    else:
        grid_type_default = "uniform" if uniform_grid else "nonuniform"

    beta_x_default = cfg.get(
        "grid_beta_x",
        cfg.get("runtime_nonuniform_beta_x", cfg.get(
            "pre_nonuniform_beta_x", 2.0, float), float),
        float,
    )
    beta_y_default = cfg.get(
        "grid_beta_y",
        cfg.get("runtime_nonuniform_beta_y", cfg.get(
            "pre_nonuniform_beta_y", 2.0, float), float),
        float,
    )
    band_fraction_x_default = cfg.get("grid_band_fraction_x", 1.0 / 3.0, float)
    band_fraction_y_default = cfg.get("grid_band_fraction_y", 1.0 / 3.0, float)

    # Now parse all arguments with defaults from config file
    p = argparse.ArgumentParser(description="2-D Navier-Stokes solver")
    p.add_argument("--config", type=str, default=default_config,
                   help="Path to configuration file")
    p.add_argument("--post-config", type=str, default=args_pre.post_config,
                   help="Path to post-processing configuration file")
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
    p.add_argument("--grid-type", type=str,
                   choices=["uniform", "nonuniform"],
                   default=grid_type_default,
                   help="Runtime grid type")
    p.add_argument("--beta-x", type=float,
                   default=beta_x_default,
                   help="x-direction center-density boost for nonuniform grid")
    p.add_argument("--beta-y", type=float,
                   default=beta_y_default,
                   help="y-direction center-density boost for nonuniform grid")
    p.add_argument("--band-fraction-x", type=float,
                   default=band_fraction_x_default,
                   help="Fraction of the x-domain width refined in the nonuniform center band")
    p.add_argument("--band-fraction-y", type=float,
                   default=band_fraction_y_default,
                   help="Fraction of the y-domain width refined in the nonuniform center band")
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
    p.add_argument("--plot-grid", type=str_to_bool,
                   default=post_cfg.get("plot_grid", False, bool),
                   help="Save a physical grid plot showing mesh concentration")
    p.add_argument("--auto-generate-grid-spacing", type=str_to_bool,
                   default=post_cfg.get("auto_generate_grid_spacing", False, bool),
                   help="Automatically save the grid spacing/concentration figure after the run")
    p.add_argument("--auto-generate-coeff-history", type=str_to_bool,
                   default=post_cfg.get("auto_generate_coeff_history", False, bool),
                   help="Automatically save the drag/lift coefficient history figure after the run")
    p.add_argument("--auto-generate-aero-report", type=str_to_bool,
                   default=post_cfg.get("auto_generate_aero_report", False, bool),
                   help="Automatically save the aerodynamic report after the run")
    p.add_argument("--auto-generate-velocity-u", type=str_to_bool,
                   default=post_cfg.get("auto_generate_velocity_u", False, bool),
                   help="Automatically save a plot of the latest u-velocity snapshot")
    p.add_argument("--auto-generate-velocity-v", type=str_to_bool,
                   default=post_cfg.get("auto_generate_velocity_v", False, bool),
                   help="Automatically save a plot of the latest v-velocity snapshot")
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
    p.add_argument("--auto-coeff-t-min", type=float,
                   default=post_cfg.get("auto_coeff_t_min", 0.5, float),
                   help="Minimum time used when auto-generating coefficient history")
    p.add_argument("--auto-aero-t-min", type=float,
                   default=post_cfg.get("auto_aero_t_min", 1.0, float),
                   help="Minimum time used when auto-generating aerodynamic analysis")
    return p.parse_args()


def _grid_metadata_path(args) -> str:
    name = "uniform_grid.npz" if args.grid_type == "uniform" else "nonuniform_grid.npz"
    return os.path.join(args.outdir, name)


def _expected_nonuniform_focus(args) -> tuple[float, float]:
    focus_x = args.cylinder_center_x if args.cylinder_center_x >= 0.0 else args.lx / 2.0
    focus_y = args.cylinder_center_y if args.cylinder_center_y >= 0.0 else args.ly / 2.0
    return focus_x, focus_y


def _expected_nonuniform_band(args) -> tuple[float, float, float, float]:
    center_x, center_y = _expected_nonuniform_focus(args)
    width_x = float(np.clip(args.band_fraction_x, 1e-3, 1.0)) * args.lx
    width_y = float(np.clip(args.band_fraction_y, 1e-3, 1.0)) * args.ly
    start_x = float(np.clip(center_x - 0.5 * width_x, 0.0, args.lx - width_x))
    end_x = start_x + width_x
    start_y = float(np.clip(center_y - 0.5 * width_y, 0.0, args.ly - width_y))
    end_y = start_y + width_y
    return start_x, end_x, start_y, end_y


def _grid_matches_args(metadata: dict, grid, args) -> bool:
    metadata_type = str(metadata.get("grid_type", "uniform")).strip().lower()

    return (
        grid.nx == args.nx and
        grid.ny == args.ny and
        np.isclose(grid.lx, args.lx) and
        np.isclose(grid.ly, args.ly) and
        ((args.grid_type == "uniform" and grid.is_uniform and metadata_type == "uniform") or (
            args.grid_type == "nonuniform" and not grid.is_uniform and metadata_type == "nonuniform"))
    )


def _nonuniform_metadata_matches_args(metadata: dict, args) -> bool:
    # Only accept nonuniform files that were built with the expected mode and parameters.
    mode = str(metadata.get("nonuniform_mode", "")).strip().lower()
    if mode != "center-band":
        return False

    focus_x, focus_y = _expected_nonuniform_focus(args)
    band_start_x, band_end_x, band_start_y, band_end_y = _expected_nonuniform_band(args)
    return (
        np.isclose(float(metadata.get("beta_x", np.nan)), float(args.beta_x)) and
        np.isclose(float(metadata.get("beta_y", np.nan)), float(args.beta_y)) and
        np.isclose(float(metadata.get("focus_x", np.nan)), float(focus_x)) and
        np.isclose(float(metadata.get("focus_y", np.nan)), float(focus_y)) and
        np.isclose(float(metadata.get("band_fraction_x", np.nan)), float(args.band_fraction_x)) and
        np.isclose(float(metadata.get("band_fraction_y", np.nan)), float(args.band_fraction_y)) and
        np.isclose(float(metadata.get("band_start_x", np.nan)), float(band_start_x)) and
        np.isclose(float(metadata.get("band_end_x", np.nan)), float(band_end_x)) and
        np.isclose(float(metadata.get("band_start_y", np.nan)), float(band_start_y)) and
        np.isclose(float(metadata.get("band_end_y", np.nan)), float(band_end_y))
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


def prepare_nonuniform_grid(args):
    """Build the runtime non-uniform grid and write its metadata before startup."""
    focus_x, focus_y = _expected_nonuniform_focus(args)
    metadata = build_nonuniform_grid_metadata(
        nx=args.nx,
        ny=args.ny,
        lx=args.lx,
        ly=args.ly,
        beta_x=args.beta_x,
        beta_y=args.beta_y,
        focus_x=focus_x,
        focus_y=focus_y,
        band_fraction_x=args.band_fraction_x,
        band_fraction_y=args.band_fraction_y,
    )
    os.makedirs(args.outdir, exist_ok=True)
    save_grid_metadata_dict(_grid_metadata_path(args), metadata)
    return CartesianGrid.from_metadata(metadata)


def get_runtime_grid(args):
    """Load a pre-generated runtime grid when available, otherwise create one."""
    grid_path = _grid_metadata_path(args)
    if os.path.exists(grid_path):
        try:
            metadata = load_grid_metadata_dict(grid_path)
            grid = load_prepared_grid(grid_path)
            if _grid_matches_args(metadata, grid, args):
                if args.grid_type == "nonuniform" and not _nonuniform_metadata_matches_args(metadata, args):
                    print(
                        "Warning: Prepared nonuniform grid metadata does not match "
                        "requested beta/band settings; regenerating grid."
                    )
                else:
                    return grid, True
        except (ValueError, OSError, KeyError) as exc:
            print(
                f"Warning: Ignoring incompatible prepared grid at {grid_path}: {exc}"
            )
    if args.grid_type == "nonuniform":
        return prepare_nonuniform_grid(args), False
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
        print(f"  Grid type     : {args.grid_type}")
        print(f"  Domain        : {args.lx} x {args.ly}")
        print(f"  Reynolds no.  : {args.re}")
        print(f"  End time      : {args.t_end}")
        print(f"  MPI ranks     : {decomp.size}")
        print(f"  IBM cylinder  : {args.cylinder}")
        print(
            f"  Grid source   : {'pre-generated file' if grid_loaded_from_file else 'generated at startup'}")
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

    if is_root:
        _run_auto_outputs(grid, args)

    # ------------------------------------------------------------------
    # Optional plot
    # ------------------------------------------------------------------
    if args.plot_grid and is_root:
        _plot_grid(grid, args)
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

    # Velocity visualization. streamplot requires equally spaced coordinates,
    # so fall back to a nonuniform-safe quiver overlay when needed.
    if grid.is_uniform:
        axes[2].streamplot(grid.xc, grid.yc,
                           u_c.T, v_c.T,
                           density=1.5, color=speed.T, cmap="plasma")
        axes[2].set_title("Streamlines")
    else:
        Xf, Yf = np.meshgrid(grid.xf, grid.yf, indexing="ij")
        im2 = axes[2].pcolormesh(Xf, Yf, speed, shading="flat", cmap="plasma")
        fig.colorbar(im2, ax=axes[2])
        stride_x = max(1, grid.nx // 24)
        stride_y = max(1, grid.ny // 16)
        axes[2].quiver(
            X[::stride_x, ::stride_y],
            Y[::stride_x, ::stride_y],
            u_c[::stride_x, ::stride_y],
            v_c[::stride_x, ::stride_y],
            color="white",
            pivot="mid",
            scale_units="xy",
            scale=None,
            width=0.003,
        )
        axes[2].set_title("Velocity Magnitude + Direction")
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


def _plot_grid(grid, args):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection
    except ImportError:
        print("matplotlib not available - skipping grid plot.")
        return

    x_edges = np.asarray(grid.xf, dtype=float)
    y_edges = np.asarray(grid.yf, dtype=float)
    Xf, Yf = np.meshgrid(x_edges, y_edges, indexing="ij")

    dx = np.asarray(grid.dx_cells, dtype=float)
    dy = np.asarray(grid.dy_cells, dtype=float)
    cell_area = dx[:, np.newaxis] * dy[np.newaxis, :]
    density = 1.0 / np.maximum(cell_area, 1e-30)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    mesh_ax = axes[0]
    vertical_segments = [
        [(float(x), 0.0), (float(x), float(grid.ly))]
        for x in x_edges
    ]
    horizontal_segments = [
        [(0.0, float(y)), (float(grid.lx), float(y))]
        for y in y_edges
    ]
    mesh_ax.add_collection(LineCollection(vertical_segments, colors="0.15", linewidths=0.6))
    mesh_ax.add_collection(LineCollection(horizontal_segments, colors="0.15", linewidths=0.6))
    mesh_ax.set_xlim(0.0, grid.lx)
    mesh_ax.set_ylim(0.0, grid.ly)
    mesh_ax.set_aspect("equal")
    mesh_ax.set_title("Physical Grid")
    mesh_ax.set_xlabel("x")
    mesh_ax.set_ylabel("y")

    density_ax = axes[1]
    density_im = density_ax.pcolormesh(Xf, Yf, density, shading="flat", cmap="viridis")
    fig.colorbar(density_im, ax=density_ax, label=r"Cell density $1/(\Delta x \Delta y)$")
    density_ax.set_xlim(0.0, grid.lx)
    density_ax.set_ylim(0.0, grid.ly)
    density_ax.set_aspect("equal")
    density_ax.set_title("Point Concentration")
    density_ax.set_xlabel("x")
    density_ax.set_ylabel("y")

    spacing_ax = axes[2]
    spacing_ax.plot(grid.xc, grid.dx_cells, label=r"$\Delta x$ at $x_c$", color="#d95f02", lw=2.0)
    spacing_ax.plot(grid.yc, grid.dy_cells, label=r"$\Delta y$ at $y_c$", color="#1b9e77", lw=2.0)
    spacing_ax.set_title("Cell Spacing")
    spacing_ax.set_xlabel("Physical coordinate")
    spacing_ax.set_ylabel("Cell width")
    spacing_ax.grid(True, alpha=0.25)
    spacing_ax.legend()

    if args.grid_type == "nonuniform":
        from matplotlib.patches import Rectangle
        band_start_x, band_end_x, band_start_y, band_end_y = _expected_nonuniform_band(args)
        rect_width = band_end_x - band_start_x
        rect_height = band_end_y - band_start_y
        mesh_ax.add_patch(
            Rectangle(
                (band_start_x, band_start_y),
                rect_width,
                rect_height,
                fill=False,
                ec="#c1121f",
                ls="--",
                lw=1.4,
                alpha=0.9,
            )
        )
        density_ax.add_patch(
            Rectangle(
                (band_start_x, band_start_y),
                rect_width,
                rect_height,
                fill=False,
                ec="white",
                ls="--",
                lw=1.2,
                alpha=0.9,
            )
        )

    fig.suptitle(
        f"Grid type={args.grid_type}, nx={grid.nx}, ny={grid.ny}, "
        f"dx_min={grid.dx_min:.4g}, dy_min={grid.dy_min:.4g}"
    )
    fig.tight_layout()

    results_dir = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "results")
    os.makedirs(results_dir, exist_ok=True)
    plot_path = os.path.join(results_dir, "grid.png")
    fig.savefig(plot_path, dpi=180)
    print(f"  Grid plot saved to {plot_path}")
    plt.close(fig)


def _run_auto_outputs(grid, args):
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(results_dir, exist_ok=True)

    if args.auto_generate_grid_spacing:
        _plot_grid(grid, args)

    need_aero_series = args.auto_generate_coeff_history or args.auto_generate_aero_report
    aero_series_path = os.path.join(results_dir, "aero.csv")
    aero_report_path = os.path.join(results_dir, "aero_report.txt")

    if need_aero_series:
        status = run_aero_analysis(
            indir=args.outdir,
            pattern="snap_*.npz",
            config=args.config,
            u_ref=args.inflow_u,
            use_cylinder_diameter=bool(args.re_is_cylinder_based and args.cylinder),
            t_min=args.auto_aero_t_min,
            save_series=aero_series_path,
            save_report=aero_report_path if args.auto_generate_aero_report else None,
        )
        if status != 0:
            print("  Warning: automatic aerodynamic post-processing failed.")
            return

    if args.auto_generate_coeff_history:
        coeff_history_name = "coeff_history.png"
        try:
            plot_coeff_history(
                aero_series_path,
                save_name=coeff_history_name,
                coeff_t_min=args.auto_coeff_t_min,
            )
        except Exception as exc:
            print(f"  Warning: automatic coefficient-history plot failed: {exc}")

    latest_snapshot = None
    if args.auto_generate_velocity_u or args.auto_generate_velocity_v:
        latest_snapshot = find_latest_snapshot(dirpath=args.outdir)
        if latest_snapshot is None:
            print("  Warning: automatic velocity plots skipped because no snapshots were found.")
            return

    if args.auto_generate_velocity_u:
        try:
            plot_snapshot_key(latest_snapshot, "u", save_name="velocity_u.png")
        except Exception as exc:
            print(f"  Warning: automatic u-velocity plot failed: {exc}")

    if args.auto_generate_velocity_v:
        try:
            plot_snapshot_key(latest_snapshot, "v", save_name="velocity_v.png")
        except Exception as exc:
            print(f"  Warning: automatic v-velocity plot failed: {exc}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    runtime_grid, loaded_from_file = get_runtime_grid(args)
    run(args, grid=runtime_grid, grid_loaded_from_file=loaded_from_file)
