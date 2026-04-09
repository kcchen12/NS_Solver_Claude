"""Pre-generate and save prepared grid metadata (uniform or non-uniform)."""

from __future__ import annotations

import argparse
import os

from src.config import ConfigParser
from src.grid import CartesianGrid, build_nonuniform_grid_metadata
from src.io_utils import save_grid_metadata, save_grid_metadata_dict


def parse_args() -> argparse.Namespace:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_config = os.path.join(script_dir, "config.txt")
    default_outdir = os.path.join(script_dir, "output")

    p_pre = argparse.ArgumentParser(add_help=False)
    p_pre.add_argument("--config", type=str, default=default_config,
                       help="Path to configuration file")
    args_pre, remaining = p_pre.parse_known_args()

    cfg = ConfigParser(args_pre.config)

    p = argparse.ArgumentParser(
        description="Pre-generate prepared grid metadata for post-processing and setup",
    )
    p.add_argument("--config", type=str, default=default_config,
                   help="Path to configuration file")
    p.add_argument("--nx", type=int, default=cfg.get("nx", 64, int))
    p.add_argument("--ny", type=int, default=cfg.get("ny", 32, int))
    p.add_argument("--lx", type=float, default=cfg.get("lx", 4.0, float))
    p.add_argument("--ly", type=float, default=cfg.get("ly", 2.0, float))
    p.add_argument("--outdir", type=str,
                   default=cfg.get("outdir", default_outdir, str))

    p.add_argument(
        "--grid-type",
        type=str,
        choices=["uniform", "nonuniform"],
        default=cfg.get("pre_grid_type", "uniform", str),
        help="Prepared-grid type to generate",
    )
    p.add_argument(
        "--beta-x",
        type=float,
        default=cfg.get("pre_nonuniform_beta_x", 2.0, float),
        help="x-direction tanh stretch beta for nonuniform grid (<=0 gives uniform spacing)",
    )
    p.add_argument(
        "--beta-y",
        type=float,
        default=cfg.get("pre_nonuniform_beta_y", 2.0, float),
        help="y-direction tanh stretch beta for nonuniform grid (<=0 gives uniform spacing)",
    )
    p.add_argument(
        "--focus-x",
        type=float,
        default=cfg.get("cylinder_center_x", -1.0, float),
        help="Focus x-coordinate for piecewise-cylinder mode (<0 uses lx/4)",
    )
    p.add_argument(
        "--focus-y",
        type=float,
        default=cfg.get("cylinder_center_y", -1.0, float),
        help="Focus y-coordinate for piecewise-cylinder mode (<0 uses ly/2)",
    )
    p.add_argument(
        "--output-name",
        type=str,
        default=None,
        help="Optional output filename override (default depends on grid type)",
    )

    return p.parse_args(remaining)


def _default_output_name(grid_type: str) -> str:
    return "uniform_grid.npz" if grid_type == "uniform" else "nonuniform_grid.npz"


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    output_name = args.output_name or _default_output_name(args.grid_type)
    output_path = os.path.join(args.outdir, output_name)
    display_path = output_path.replace("\\", "/")

    if args.grid_type == "uniform":
        grid = CartesianGrid(nx=args.nx, ny=args.ny, lx=args.lx, ly=args.ly)
        save_grid_metadata(output_path, grid)
        print(
            "Saved uniform prepared grid metadata to "
            f"{display_path} for nx={grid.nx}, ny={grid.ny}, lx={grid.lx}, ly={grid.ly}"
        )
    else:
        focus_x = args.focus_x if args.focus_x >= 0.0 else args.lx / 4.0
        focus_y = args.focus_y if args.focus_y >= 0.0 else args.ly / 2.0
        metadata = build_nonuniform_grid_metadata(
            nx=args.nx,
            ny=args.ny,
            lx=args.lx,
            ly=args.ly,
            beta_x=args.beta_x,
            beta_y=args.beta_y,
            focus_x=focus_x,
            focus_y=focus_y,
        )
        save_grid_metadata_dict(output_path, metadata)
        print(
            "Saved nonuniform prepared grid metadata to "
            f"{display_path} for nx={args.nx}, ny={args.ny}, lx={args.lx}, ly={args.ly}, "
            f"beta_x={args.beta_x}, beta_y={args.beta_y}, "
            f"mode=piecewise-cylinder, focus=({focus_x}, {focus_y})"
        )
