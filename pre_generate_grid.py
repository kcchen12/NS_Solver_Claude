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

    # Unified grid controls from config.txt.
    # Backward compatibility:
    # 1) If uniform_grid is set, it overrides string grid type keys.
    # 2) Otherwise grid_type can override legacy pre_grid_type.
    # 3) grid_beta_x/y can override legacy pre betas.
    uniform_grid = cfg.get("uniform_grid", None, bool)
    if uniform_grid is None:
        grid_type_default = cfg.get("grid_type", cfg.get(
            "pre_grid_type", "uniform", str), str)
    else:
        grid_type_default = "uniform" if uniform_grid else "nonuniform"

    beta_x_default = cfg.get("grid_beta_x", cfg.get(
        "pre_nonuniform_beta_x", 2.0, float), float)
    beta_y_default = cfg.get("grid_beta_y", cfg.get(
        "pre_nonuniform_beta_y", 2.0, float), float)
    band_fraction_x_default = cfg.get("grid_band_fraction_x", 1.0 / 3.0, float)
    band_fraction_y_default = cfg.get("grid_band_fraction_y", 1.0 / 3.0, float)

    p = argparse.ArgumentParser(
        description="Pre-generate prepared grid metadata for post-processing and setup",
    )
    p.add_argument("--config", type=str, default=default_config,
                   help="Path to configuration file")
    p.add_argument("--nx", type=int, default=cfg.get("nx", 64, int))
    p.add_argument("--ny", type=int, default=cfg.get("ny", 32, int))
    p.add_argument("--lx", type=float, default=cfg.get("lx", 4.0, float))
    p.add_argument("--ly", type=float, default=cfg.get("ly", 2.0, float))
    p.add_argument("--x-min", type=float, default=cfg.get("x_min", None, float),
                   help="Domain lower bound in x (optional; defaults to 0)")
    p.add_argument("--x-max", type=float, default=cfg.get("x_max", None, float),
                   help="Domain upper bound in x (optional; inferred from x_min+lx)")
    p.add_argument("--y-min", type=float, default=cfg.get("y_min", None, float),
                   help="Domain lower bound in y (optional; defaults to 0)")
    p.add_argument("--y-max", type=float, default=cfg.get("y_max", None, float),
                   help="Domain upper bound in y (optional; inferred from y_min+ly)")
    p.add_argument("--outdir", type=str,
                   default=cfg.get("outdir", default_outdir, str))

    p.add_argument(
        "--grid-type",
        type=str,
        choices=["uniform", "nonuniform"],
        default=grid_type_default,
        help="Prepared-grid type to generate",
    )
    p.add_argument(
        "--beta-x",
        type=float,
        default=beta_x_default,
        help="x-direction center-density boost for nonuniform grid (<=0 gives uniform spacing)",
    )
    p.add_argument(
        "--beta-y",
        type=float,
        default=beta_y_default,
        help="y-direction center-density boost for nonuniform grid (<=0 gives uniform spacing)",
    )
    p.add_argument(
        "--band-fraction-x",
        type=float,
        default=band_fraction_x_default,
        help="Fraction of the x-domain width refined in the nonuniform center band",
    )
    p.add_argument(
        "--band-fraction-y",
        type=float,
        default=band_fraction_y_default,
        help="Fraction of the y-domain width refined in the nonuniform center band",
    )
    p.add_argument(
        "--focus-x",
        type=float,
        default=cfg.get("cylinder_center_x", -1.0, float),
        help="Center x-coordinate for the refined band (<0 uses domain midpoint)",
    )
    p.add_argument(
        "--focus-y",
        type=float,
        default=cfg.get("cylinder_center_y", -1.0, float),
        help="Center y-coordinate for the refined band (<0 uses domain midpoint)",
    )
    p.add_argument(
        "--output-name",
        type=str,
        default=None,
        help="Optional output filename override (default depends on grid type)",
    )

    args = p.parse_args(remaining)
    args.x_min = 0.0 if args.x_min is None else float(args.x_min)
    args.y_min = 0.0 if args.y_min is None else float(args.y_min)

    if args.x_max is None:
        args.x_max = args.x_min + float(args.lx)
    else:
        args.x_max = float(args.x_max)
    if args.y_max is None:
        args.y_max = args.y_min + float(args.ly)
    else:
        args.y_max = float(args.y_max)

    if not args.x_max > args.x_min:
        p.error("Require x_max > x_min")
    if not args.y_max > args.y_min:
        p.error("Require y_max > y_min")

    args.lx = float(args.x_max - args.x_min)
    args.ly = float(args.y_max - args.y_min)
    return args


def _default_output_name(grid_type: str) -> str:
    return "uniform_grid.npz" if grid_type == "uniform" else "nonuniform_grid.npz"


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    output_name = args.output_name or _default_output_name(args.grid_type)
    output_path = os.path.join(args.outdir, output_name)
    display_path = output_path.replace("\\", "/")

    if args.grid_type == "uniform":
        grid = CartesianGrid(nx=args.nx, ny=args.ny, lx=args.lx, ly=args.ly,
                             x_min=args.x_min, y_min=args.y_min)
        save_grid_metadata(output_path, grid)
        print(
            "Saved uniform prepared grid metadata to "
            f"{display_path} for nx={grid.nx}, ny={grid.ny}, lx={grid.lx}, ly={grid.ly}"
        )
    else:
        focus_x = args.focus_x if args.focus_x >= 0.0 else args.x_min + args.lx / 2.0
        focus_y = args.focus_y if args.focus_y >= 0.0 else args.y_min + args.ly / 2.0
        metadata = build_nonuniform_grid_metadata(
            nx=args.nx,
            ny=args.ny,
            lx=args.lx,
            ly=args.ly,
            beta_x=args.beta_x,
            beta_y=args.beta_y,
            x_min=args.x_min,
            y_min=args.y_min,
            focus_x=focus_x,
            focus_y=focus_y,
            band_fraction_x=args.band_fraction_x,
            band_fraction_y=args.band_fraction_y,
        )
        save_grid_metadata_dict(output_path, metadata)
        print(
            "Saved nonuniform prepared grid metadata to "
            f"{display_path} for nx={args.nx}, ny={args.ny}, lx={args.lx}, ly={args.ly}, "
            f"beta_x={args.beta_x}, beta_y={args.beta_y}, "
            f"band_fraction_x={args.band_fraction_x}, band_fraction_y={args.band_fraction_y}, "
            f"mode=center-band, center=({focus_x}, {focus_y})"
        )
