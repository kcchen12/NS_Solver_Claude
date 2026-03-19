#!/usr/bin/env python3
"""Simple viewer for `.npz` snapshot files produced by this project.

Usage examples:
  # show keys and shapes in the latest snapshot
  python view_snapshot_viewer.py --list

  # plot a named variable from a snapshot file
  python view_snapshot_viewer.py output/snap_005.0000.npz -k velocity --comp 0

  # plot the latest snapshot, component 1, save to PNG
  python view_snapshot_viewer.py -k pressure --save out.png

  # plot with velocity arrows overlaid
  python view_snapshot_viewer.py -k pressure --show-quiver

  # plot velocity field with streamlines
  python view_snapshot_viewer.py -k velocity --streamlines

The script attempts to guess array layouts but accepts explicit --slice and --comp parameters.
"""

from __future__ import annotations
import argparse
import glob
import os
import sys
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


def ensure_images_dir(images_dir: str = "images") -> str:
    """Ensure the images directory exists."""
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
    return images_dir


def find_latest_snapshot(dirpath: str = "output", pattern: str = "snap_*.npz") -> Optional[str]:
    paths = sorted(glob.glob(os.path.join(dirpath, pattern)))
    return paths[-1] if paths else None


def extract_velocity_field(data: dict) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Try to find and extract velocity components from the data dictionary."""
    # Look for 'velocity' key directly
    if 'velocity' in data:
        vel = np.array(data['velocity'])
        if vel.ndim == 3 and vel.shape[2] == 2:
            return vel[:, :, 0], vel[:, :, 1]
        elif vel.ndim == 3 and vel.shape[0] == 2:
            return vel[0, :, :], vel[1, :, :]
        elif vel.ndim == 4 and vel.shape[3] == 2:  # 3D data
            return vel[:, :, :, 0], vel[:, :, :, 1]
        elif vel.ndim == 4 and vel.shape[0] == 2:  # 3D data
            return vel[0, :, :, :], vel[1, :, :, :]

    # Look for 'u' and 'v' keys
    if 'u' in data and 'v' in data:
        u = np.array(data['u'])
        v = np.array(data['v'])
        if u.ndim == 3 and u.shape[0] <= 4:  # (comp, ny, nx) format
            return u[0, :, :] if u.shape[0] == 1 else u, v[0, :, :] if v.shape[0] == 1 else v
        return u, v

    return None


def is_velocity_field(key: str) -> bool:
    """Check if a key is likely a velocity field."""
    key_lower = key.lower()
    return 'velocity' in key_lower or 'vel' in key_lower or key_lower in ['u', 'v', 'w']


def inspect_npz(path: str):
    with np.load(path, allow_pickle=True) as data:
        keys = list(data.keys())
        print(f"File: {path}")
        if not keys:
            print("(no arrays found)")
            return
        for k in keys:
            print(f"- {k}: shape={data[k].shape}, dtype={data[k].dtype}")


def pick_slice_and_component(arr: np.ndarray, slice_idx: Optional[int], comp_idx: Optional[int]):
    # Try to handle common shapes: (ny,nx), (ny,nx,comp), (comp,ny,nx), (nz,ny,nx), (comp,nz,ny,nx)
    ndim = arr.ndim
    if ndim == 2:
        return arr
    if ndim == 3:
        # If last dim reasonably small, treat as components
        if arr.shape[2] <= 4:
            c = 0 if comp_idx is None else comp_idx
            return arr[:, :, c]
        # If first dim small, treat as (comp,ny,nx)
        if arr.shape[0] <= 4:
            c = 0 if comp_idx is None else comp_idx
            return arr[c, :, :]
        # otherwise treat as (nz,ny,nx) -> take slice
        z = arr.shape[0] // 2 if slice_idx is None else slice_idx
        return arr[z, :, :]
    if ndim == 4:
        # common: (comp,nz,ny,nx) or (nz,ny,nx,comp)
        if arr.shape[0] <= 4:
            c = 0 if comp_idx is None else comp_idx
            z = arr.shape[1] // 2 if slice_idx is None else slice_idx
            return arr[c, z, :, :]
        if arr.shape[3] <= 4:
            c = 0 if comp_idx is None else comp_idx
            z = arr.shape[0] // 2 if slice_idx is None else slice_idx
            return arr[z, :, :, c]
    raise ValueError(f"Unsupported array shape: {arr.shape}")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="View .npz snapshot files with velocity visualization (matplotlib viewer)")
    parser.add_argument("file", nargs="?",
                        help=".npz snapshot file (default: latest in output/)")
    parser.add_argument("-l", "--list", action="store_true",
                        help="list keys and shapes then exit")
    parser.add_argument(
        "-k", "--key", help="variable name (npz archive key) to plot")
    parser.add_argument("-s", "--slice", type=int,
                        help="slice index for 3D arrays (nz dimension)")
    parser.add_argument("-c", "--comp", type=int,
                        help="component index for vector fields")
    parser.add_argument("--cmap", default="viridis",
                        help="matplotlib colormap")
    parser.add_argument("--vmin", type=float, help="color scale vmin")
    parser.add_argument("--vmax", type=float, help="color scale vmax")
    parser.add_argument(
        "--save", help="save plot to this path instead of showing")
    parser.add_argument("--show-quiver", action="store_true",
                        help="if data is a 2-component vector field, show quiver (subsampled)")
    parser.add_argument("--streamlines", action="store_true",
                        help="show streamlines instead of quiver arrows (for velocity fields)")
    parser.add_argument("--add-pressure", action="store_true",
                        help="overlay velocity vectors on pressure field")
    parser.add_argument("--no-velocity-overlay", action="store_true",
                        help="don't automatically overlay velocity on main plot")
    parser.add_argument("--quiver-density", type=int, default=16,
                        help="quiver arrow density (default: 16, smaller=denser)")
    args = parser.parse_args(argv)

    if args.file is None:
        latest = find_latest_snapshot()
        if latest is None:
            print("No snapshot file specified and none found in 'output/'",
                  file=sys.stderr)
            parser.print_help()
            sys.exit(1)
        args.file = latest

    if not os.path.exists(args.file):
        print(f"File not found: {args.file}", file=sys.stderr)
        sys.exit(2)

    with np.load(args.file, allow_pickle=True) as data:
        keys = list(data.keys())
        if args.list:
            inspect_npz(args.file)
            return
        if not keys:
            print("No arrays found in file", file=sys.stderr)
            sys.exit(3)

        # Choose which variable to plot
        key = args.key or keys[0]
        if key not in keys:
            print(f"Key '{key}' not found. Available: {keys}", file=sys.stderr)
            sys.exit(4)
        arr = np.array(data[key])

        # Try to extract velocity for overlay
        vel_data = extract_velocity_field(data)

    # Extract main variable to plot
    if args.show_quiver or args.streamlines:
        # Quiver/streamlines mode: interpret as vector field
        if arr.ndim == 3 and arr.shape[2] == 2:
            vx = arr[:, :, 0]
            vy = arr[:, :, 1]
            img = np.hypot(vx, vy)  # magnitude
        elif arr.ndim == 3 and arr.shape[0] == 2:
            vx = arr[0, :, :]
            vy = arr[1, :, :]
            img = np.hypot(vx, vy)  # magnitude
        else:
            # fall back to scalar selection
            try:
                img = pick_slice_and_component(arr, args.slice, args.comp)
            except Exception as e:
                print("Cannot interpret array for vector plot:",
                      e, file=sys.stderr)
                sys.exit(5)
            vx = vy = None
    else:
        try:
            img = pick_slice_and_component(arr, args.slice, args.comp)
        except Exception as e:
            print(f"Error selecting slice/component: {e}", file=sys.stderr)
            print("Available array shape:", arr.shape)
            sys.exit(6)
        vx = vy = None

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(img, origin="lower", cmap=args.cmap,
                   vmin=args.vmin, vmax=args.vmax)
    ax.set_title(f"{os.path.basename(args.file)} : {key}",
                 fontsize=12, fontweight='bold')
    fig.colorbar(im, ax=ax, label=key)

    # Add velocity visualization
    if args.show_quiver and vx is not None and vy is not None:
        # Overlay quiver plot (subsampled for clarity)
        X = np.arange(0, img.shape[1])
        Y = np.arange(0, img.shape[0])
        XI, YI = np.meshgrid(X, Y)
        step = max(1, args.quiver_density)

        try:
            ax.quiver(XI[::step, ::step], YI[::step, ::step],
                      vx[::step, ::step], vy[::step, ::step],
                      scale=1, scale_units='xy', angles='xy',
                      alpha=0.6, width=0.003, color='white',
                      edgecolor='none', label='Velocity')
            ax.legend(loc='upper right', fontsize=10)
        except Exception as e:
            print(f"Warning: Could not overlay quiver: {e}", file=sys.stderr)

    elif args.streamlines and vx is not None and vy is not None:
        # Overlay streamlines for smooth velocity visualization
        X = np.arange(0, img.shape[1])
        Y = np.arange(0, img.shape[0])
        XI, YI = np.meshgrid(X, Y)

        try:
            # Create streamlines with colored arrows
            speed = np.hypot(vx, vy)
            lw = 2 * speed / speed.max()  # Line width varies with speed
            strm = ax.streamplot(XI[0, :], YI[:, 0], vx, vy,
                                 color=speed, cmap='hot', linewidth=lw,
                                 density=1.5, arrowsize=1.5)
        except Exception as e:
            print(
                f"Warning: Could not overlay streamlines: {e}", file=sys.stderr)

    elif vel_data is not None and not args.no_velocity_overlay and not args.show_quiver and not args.streamlines:
        # Automatically overlay velocity field if plotting a scalar and velocity is available
        vx_auto, vy_auto = vel_data

        # Handle 3D velocity data - take same slice as main variable
        if vx_auto.ndim == 3 and img.ndim == 2:
            z = img.shape[0] // 2 if args.slice is None else args.slice
            vx_auto = vx_auto[z, :, :]
            vy_auto = vy_auto[z, :, :]

        if vx_auto.shape == img.shape and vy_auto.shape == img.shape:
            X = np.arange(0, img.shape[1])
            Y = np.arange(0, img.shape[0])
            XI, YI = np.meshgrid(X, Y)
            step = max(1, args.quiver_density)

            try:
                ax.quiver(XI[::step, ::step], YI[::step, ::step],
                          vx_auto[::step, ::step], vy_auto[::step, ::step],
                          scale=1, scale_units='xy', angles='xy',
                          alpha=0.5, width=0.002, color='red',
                          edgecolor='none')
            except Exception:
                pass  # Silently skip if overlay fails

    ax.set_xlabel('x', fontsize=10)
    ax.set_ylabel('y', fontsize=10)

    if args.save:
        images_dir = ensure_images_dir()
        save_path = os.path.join(images_dir, args.save)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure: {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
