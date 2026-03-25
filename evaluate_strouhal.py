#!/usr/bin/env python3
"""Evaluate Strouhal number from pointwise probe measurements.

This script extracts u, v, and p at a single physical probe location from
saved snapshot files and estimates dominant frequencies using Lomb-Scargle
spectral analysis (robust to uneven time spacing).

Definitions
-----------
Strouhal number:
    St = f * L / U
where:
    f : dominant shedding frequency
    L : characteristic length scale
    U : reference velocity scale

For the default cylinder setup in main.py:
    cylinder radius r = ly / 8
    cylinder diameter D = ly / 4
so use --use-cylinder-diameter to set L = D automatically.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from scipy.signal import lombscargle


@dataclass
class SpectralResult:
    freq: float
    peak_power: float
    st: float


@dataclass(frozen=True)
class BilinearPlan:
    i: int
    j: int
    w11: float
    w21: float
    w12: float
    w22: float


def _safe_scalar(data: np.lib.npyio.NpzFile, key: str) -> Optional[float]:
    if key not in data.files:
        return None
    return float(np.array(data[key]).item())


def _time_from_filename(path: str) -> Optional[float]:
    """Try extracting snapshot time from filename pattern snap_<time>.npz."""
    name = os.path.basename(path)
    m = re.match(r"^snap_([-+0-9.eE]+)\.npz$", name)
    if m is None:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def _collect_snapshots(indir: str, pattern: str) -> List[Tuple[float, str]]:
    """Return unique snapshot list as (time, path), sorted by time."""
    candidates = sorted(glob.glob(os.path.join(indir, pattern)))
    if not candidates:
        return []

    # Keep one file per timestamp (last one wins).
    by_time: Dict[float, str] = {}
    for path in candidates:
        t = _time_from_filename(path)
        if t is None:
            with np.load(path) as data:
                t = float(data["t"])
        by_time[t] = path

    return [(t, by_time[t]) for t in sorted(by_time.keys())]


def _build_bilinear_plan(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    x: float,
    y: float,
) -> BilinearPlan:
    """Precompute cell indices and bilinear weights for one fixed probe."""
    i = int(np.searchsorted(x_grid, x) - 1)
    j = int(np.searchsorted(y_grid, y) - 1)

    i = int(np.clip(i, 0, len(x_grid) - 2))
    j = int(np.clip(j, 0, len(y_grid) - 2))

    x0, x1 = x_grid[i], x_grid[i + 1]
    y0, y1 = y_grid[j], y_grid[j + 1]

    tx = 0.0 if x1 == x0 else (x - x0) / (x1 - x0)
    ty = 0.0 if y1 == y0 else (y - y0) / (y1 - y0)

    w11 = (1.0 - tx) * (1.0 - ty)
    w21 = tx * (1.0 - ty)
    w12 = (1.0 - tx) * ty
    w22 = tx * ty

    return BilinearPlan(i=i, j=j, w11=w11, w21=w21, w12=w12, w22=w22)


def _apply_bilinear_plan(values: np.ndarray, plan: BilinearPlan) -> float:
    """Apply precomputed bilinear interpolation weights to a field array."""
    i = plan.i
    j = plan.j
    return float(
        plan.w11 * values[i, j]
        + plan.w21 * values[i + 1, j]
        + plan.w12 * values[i, j + 1]
        + plan.w22 * values[i + 1, j + 1]
    )


def _dominant_frequency(
    t: np.ndarray,
    signal: np.ndarray,
    t_min: float,
    f_min: float,
    f_max: float,
    n_freq: int,
) -> Optional[Tuple[float, float]]:
    """Return (f_peak, peak_power) from Lomb-Scargle spectrum."""
    mask = t >= t_min
    ts = t[mask]
    ys = signal[mask]

    if ts.size < 8:
        return None

    ys = ys - np.mean(ys)
    sigma = float(np.std(ys))
    if sigma < 1e-12:
        return None

    freq = np.linspace(f_min, f_max, n_freq)
    omega = 2.0 * np.pi * freq
    power = lombscargle(ts, ys, omega, precenter=False, normalize=True)
    idx = int(np.argmax(power))

    return float(freq[idx]), float(power[idx])


def _is_edge_frequency(f: float, f_min: float, f_max: float) -> bool:
    """Return True if f is effectively at the search-window edge."""
    width = max(f_max - f_min, 1e-12)
    tol = 1e-3 * width
    return (f - f_min) <= tol or (f_max - f) <= tol


def _read_config(config_path: str) -> Dict[str, float]:
    """Parse config file and return dictionary of key-value pairs."""
    config = {}
    if not os.path.exists(config_path):
        return config

    try:
        with open(config_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if '=' in line:
                    key, val = line.split('=', 1)
                    key = key.strip()
                    val = val.strip().lower()

                    # Handle boolean values
                    if val in ('true', 'yes', '1'):
                        config[key] = 1.0
                    elif val in ('false', 'no', '0'):
                        config[key] = 0.0
                    else:
                        try:
                            config[key] = float(val)
                        except ValueError:
                            pass  # Skip non-numeric values
    except Exception:
        pass  # Silently ignore config read errors

    return config


def _estimate_scales(
    first_path: str,
    length_scale: Optional[float],
    use_cylinder_diameter: bool,
    u_ref: float,
    config_path: Optional[str] = None,
) -> Tuple[float, float, int, int, float, float]:
    """Read metadata and return (L, U, nx, ny, lx, ly)."""
    with np.load(first_path) as data:
        p = data["p"]
        nx = int(p.shape[0])
        ny = int(p.shape[1])
        lx = _safe_scalar(data, "meta_lx")
        ly = _safe_scalar(data, "meta_ly")

    if lx is None or ly is None:
        raise ValueError(
            "Snapshot metadata does not include meta_lx/meta_ly. "
            "Please provide snapshots written by main.py with metadata."
        )

    if length_scale is not None:
        l_char = float(length_scale)
    elif use_cylinder_diameter:
        l_char = float(ly / 4.0)
    else:
        # Try to read actual cylinder dimensions from config
        config = _read_config(config_path) if config_path else {}

        if 'cylinder' in config and config['cylinder'] != 0:
            # Cylinder is enabled; get radius and compute diameter
            cylinder_radius = config.get('cylinder_radius', ly / 8.0)
            if cylinder_radius <= 0:
                cylinder_radius = ly / 8.0
            l_char = 2.0 * cylinder_radius  # diameter
        else:
            # No cylinder or config not available; default to 1.0
            l_char = 1.0

    if l_char <= 0.0:
        raise ValueError("Characteristic length must be positive.")
    if u_ref <= 0.0:
        raise ValueError("Reference velocity must be positive.")

    return l_char, float(u_ref), nx, ny, float(lx), float(ly)


def _extract_probe_series(
    snapshots: Iterable[Tuple[float, str]],
    nx: int,
    ny: int,
    lx: float,
    ly: float,
    probe_x: float,
    probe_y: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return arrays: t, u_probe, v_probe, p_probe."""
    snapshots_list = list(snapshots)
    n = len(snapshots_list)

    xf = np.linspace(0.0, lx, nx + 1)
    xc = 0.5 * (xf[:-1] + xf[1:])
    yf = np.linspace(0.0, ly, ny + 1)
    yc = 0.5 * (yf[:-1] + yf[1:])

    u_plan = _build_bilinear_plan(xf, yc, probe_x, probe_y)
    v_plan = _build_bilinear_plan(xc, yf, probe_x, probe_y)
    p_plan = _build_bilinear_plan(xc, yc, probe_x, probe_y)

    times = np.empty(n, dtype=float)
    u_vals = np.empty(n, dtype=float)
    v_vals = np.empty(n, dtype=float)
    p_vals = np.empty(n, dtype=float)

    for k, (t, path) in enumerate(snapshots_list):
        with np.load(path) as data:
            u = data["u"]
            v = data["v"]
            p = data["p"]

        u_probe = _apply_bilinear_plan(u, u_plan)
        v_probe = _apply_bilinear_plan(v, v_plan)
        p_probe = _apply_bilinear_plan(p, p_plan)

        times[k] = t
        u_vals[k] = u_probe
        v_vals[k] = v_probe
        p_vals[k] = p_probe

    return times, u_vals, v_vals, p_vals


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate Strouhal number from probe u/v/p signals.",
    )
    parser.add_argument("--indir", type=str, default="output",
                        help="Snapshot directory (default: output)")
    parser.add_argument("--pattern", type=str, default="snap_*.npz",
                        help="Snapshot filename pattern (default: snap_*.npz)")
    parser.add_argument("--config", type=str, default="config.txt",
                        help="Configuration file for reading cylinder dimensions (default: config.txt)")

    parser.add_argument("--probe-x", type=float, required=True,
                        help="Probe x coordinate in physical units")
    parser.add_argument("--probe-y", type=float, required=True,
                        help="Probe y coordinate in physical units")

    parser.add_argument("--u-ref", type=float, default=1.0,
                        help="Reference velocity U in St = fL/U (default: 1.0)")
    parser.add_argument("--length-scale", type=float, default=None,
                        help="Characteristic length L (default: read from config or 1.0)")
    parser.add_argument("--use-cylinder-diameter", action="store_true",
                        help="Use L = ly/4, consistent with cylinder setup in main.py (overrides config)")

    parser.add_argument("--t-min", type=float, default=1.0,
                        help="Ignore data before this time when fitting frequency (default: 1.0)")
    parser.add_argument("--f-min", type=float, default=0.05,
                        help="Minimum search frequency (default: 0.05)")
    parser.add_argument("--f-max", type=float, default=2.0,
                        help="Maximum search frequency (default: 2.0)")
    parser.add_argument("--n-freq", type=int, default=4000,
                        help="Number of candidate frequencies (default: 4000)")

    parser.add_argument("--save-series", type=str, default=None,
                        help="Optional CSV path for exported probe time series")
    parser.add_argument("--save-report", type=str, default=None,
                        help="Optional TXT path for Strouhal summary report")

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    snapshots = _collect_snapshots(args.indir, args.pattern)
    if not snapshots:
        print(
            f"No snapshots found in {args.indir!r} with pattern {args.pattern!r}.")
        return 1

    l_char, u_ref, nx, ny, lx, ly = _estimate_scales(
        snapshots[0][1],
        length_scale=args.length_scale,
        use_cylinder_diameter=args.use_cylinder_diameter,
        u_ref=args.u_ref,
        config_path=args.config,
    )

    t, u_probe, v_probe, p_probe = _extract_probe_series(
        snapshots,
        nx=nx,
        ny=ny,
        lx=lx,
        ly=ly,
        probe_x=args.probe_x,
        probe_y=args.probe_y,
    )

    if args.save_series:
        out = np.column_stack((t, u_probe, v_probe, p_probe))
        header = "t,u_probe,v_probe,p_probe"
        np.savetxt(args.save_series, out, delimiter=",",
                   header=header, comments="")
        print(f"Saved probe series: {args.save_series}")

    dt = np.diff(t)
    dt_median = float(np.median(dt)) if dt.size else np.nan
    nyquist_est = 0.5 / \
        dt_median if np.isfinite(dt_median) and dt_median > 0 else np.nan

    results: Dict[str, Optional[SpectralResult]] = {}
    for name, sig in (("u", u_probe), ("v", v_probe), ("p", p_probe)):
        out = _dominant_frequency(
            t,
            sig,
            t_min=args.t_min,
            f_min=args.f_min,
            f_max=args.f_max,
            n_freq=args.n_freq,
        )
        if out is None:
            results[name] = None
            continue
        f_peak, peak_power = out
        results[name] = SpectralResult(
            freq=f_peak,
            peak_power=peak_power,
            st=f_peak * l_char / u_ref,
        )

    print("=" * 66)
    print("Strouhal analysis from point probe")
    print("=" * 66)
    print(f"Snapshots         : {len(snapshots)}")
    print(f"Time span         : [{t.min():.4f}, {t.max():.4f}]")
    print(f"Probe             : x={args.probe_x:.6g}, y={args.probe_y:.6g}")
    print(f"Scales            : L={l_char:.6g}, U={u_ref:.6g}")
    print(f"Frequency window  : [{args.f_min:.4f}, {args.f_max:.4f}]")
    if np.isfinite(nyquist_est):
        print(
            f"Median dt         : {dt_median:.6g} (Nyquist approx {nyquist_est:.6g})")

    for key in ("u", "v", "p"):
        res = results[key]
        if res is None:
            print(
                f"{key} peak         : unavailable (insufficient variation/samples)")
        else:
            print(
                f"{key} peak         : f={res.freq:.6g}, St={res.st:.6g}, "
                f"power={res.peak_power:.6g}"
            )

    # For bluff-body shedding, v often carries the fundamental and u/p the
    # second harmonic. Use v, u/2, p/2 to form a robust combined estimate.
    f_candidates: List[float] = []
    if results["v"] is not None and not _is_edge_frequency(results["v"].freq, args.f_min, args.f_max):
        f_candidates.append(results["v"].freq)
    if results["u"] is not None and not _is_edge_frequency(results["u"].freq, args.f_min, args.f_max):
        f_candidates.append(0.5 * results["u"].freq)
    if results["p"] is not None and not _is_edge_frequency(results["p"].freq, args.f_min, args.f_max):
        f_candidates.append(0.5 * results["p"].freq)

    combined_f0: Optional[float] = None
    combined_st: Optional[float] = None

    if f_candidates:
        f0 = float(np.median(np.array(f_candidates)))
        st0 = f0 * l_char / u_ref
        combined_f0 = f0
        combined_st = st0
        print("-" * 66)
        print(f"Combined f0        : {f0:.6g}")
        print(f"Combined Strouhal  : {st0:.6g}")

        if np.isfinite(nyquist_est) and f0 > 0.8 * nyquist_est:
            print("WARNING: Estimated f0 is close to the Nyquist limit.")
            print("         Use smaller save_dt or a larger output rate for confidence.")

    else:
        print("-" * 66)
        print("Combined estimate  : unavailable")

    if args.save_report:
        report_dir = os.path.dirname(args.save_report)
        if report_dir:
            os.makedirs(report_dir, exist_ok=True)

        lines: List[str] = [
            "Strouhal analysis from point probe",
            "=" * 66,
            f"Snapshots         : {len(snapshots)}",
            f"Time span         : [{t.min():.4f}, {t.max():.4f}]",
            f"Probe             : x={args.probe_x:.6g}, y={args.probe_y:.6g}",
            f"Scales            : L={l_char:.6g}, U={u_ref:.6g}",
            f"Frequency window  : [{args.f_min:.4f}, {args.f_max:.4f}]",
        ]

        if np.isfinite(nyquist_est):
            lines.append(
                f"Median dt         : {dt_median:.6g} "
                f"(Nyquist approx {nyquist_est:.6g})"
            )

        for key in ("u", "v", "p"):
            res = results[key]
            if res is None:
                lines.append(
                    f"{key} peak         : unavailable "
                    "(insufficient variation/samples)"
                )
            else:
                lines.append(
                    f"{key} peak         : f={res.freq:.6g}, St={res.st:.6g}, "
                    f"power={res.peak_power:.6g}"
                )

        lines.append("-" * 66)
        if combined_f0 is not None and combined_st is not None:
            lines.append(f"Combined f0        : {combined_f0:.6g}")
            lines.append(f"Combined Strouhal  : {combined_st:.6g}")
        else:
            lines.append("Combined estimate  : unavailable")

        with open(args.save_report, "w", encoding="utf-8") as fout:
            fout.write("\n".join(lines) + "\n")

        print(f"Saved Strouhal report: {args.save_report}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
