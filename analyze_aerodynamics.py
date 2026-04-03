#!/usr/bin/env python3
"""Comprehensive aerodynamic analysis from snapshot data.

This script combines Strouhal number and drag/lift coefficient analysis.
It extracts a point probe signal for frequency analysis and computes forces
on an immersed boundary object (e.g., cylinder), providing complete aerodynamic
metrics in one unified report.

Outputs:
    - Time series CSV with probe values, forces, and coefficients
    - Comprehensive text report with all statistics
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


DEFAULT_RESULTS_DIR = "results"


@dataclass
class SpectralResult:
    freq: float
    peak_power: float
    st: float


@dataclass
class ForceResult:
    time: float
    f_x: float
    f_y: float
    c_d: float
    c_l: float


@dataclass
class ProbePoint:
    time: float
    u: float
    v: float
    p: float


@dataclass(frozen=True)
class BilinearPlan:
    i: int
    j: int
    w11: float
    w21: float
    w12: float
    w22: float


@dataclass(frozen=True)
class CylinderGeometry:
    center_x: float
    center_y: float
    radius: float


@dataclass(frozen=True)
class SurfaceForcePlan:
    normals_x: np.ndarray
    normals_y: np.ndarray
    arc_length: float
    bilinear_plans: Tuple[BilinearPlan, ...]


def _time_from_filename(path: str) -> Optional[float]:
    """Extract snapshot time from filename pattern snap_<time>.npz."""
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

    by_time: Dict[float, str] = {}
    for path in candidates:
        t = _time_from_filename(path)
        if t is None:
            with np.load(path) as data:
                t = float(data["t"])
        by_time[t] = path

    return [(t, by_time[t]) for t in sorted(by_time.keys())]


def _safe_scalar(data: np.lib.npyio.NpzFile, key: str) -> Optional[float]:
    if key not in data.files:
        return None
    return float(np.array(data[key]).item())


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

                    if val in ('true', 'yes', '1'):
                        config[key] = 1.0
                    elif val in ('false', 'no', '0'):
                        config[key] = 0.0
                    else:
                        try:
                            config[key] = float(val)
                        except ValueError:
                            pass
    except Exception:
        pass

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

    config = _read_config(config_path) if config_path else {}
    cfg_length_scale = config.get("aero_length_scale", None)
    cfg_use_cyl_d = bool(config.get("aero_use_cylinder_diameter", 0.0))

    if length_scale is not None:
        l_char = float(length_scale)
    elif cfg_length_scale is not None and float(cfg_length_scale) > 0.0:
        l_char = float(cfg_length_scale)
    elif use_cylinder_diameter:
        l_char = float(ly / 4.0)
    elif cfg_use_cyl_d:
        l_char = float(ly / 4.0)
    else:
        if 'cylinder' in config and config['cylinder'] != 0:
            cylinder_radius = config.get('cylinder_radius', ly / 8.0)
            if cylinder_radius <= 0:
                cylinder_radius = ly / 8.0
            l_char = 2.0 * cylinder_radius
        else:
            l_char = 1.0

    if l_char <= 0.0:
        raise ValueError("Characteristic length must be positive.")
    if u_ref <= 0.0:
        raise ValueError("Reference velocity must be positive.")

    return l_char, float(u_ref), nx, ny, float(lx), float(ly)


def _estimate_cylinder_geometry(
    first_path: str,
    config_path: Optional[str] = None,
    cylinder_radius: Optional[float] = None,
) -> Tuple[int, int, float, float, CylinderGeometry]:
    """Read metadata and return (nx, ny, lx, ly, cylinder_geometry)."""
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

    if cylinder_radius is not None:
        r = float(cylinder_radius)
    else:
        config = _read_config(config_path) if config_path else {}
        if 'cylinder_radius' in config:
            r = float(config['cylinder_radius'])
        else:
            r = float(ly / 8.0)

    center_x = float(lx / 2.0)
    center_y = float(ly / 2.0)

    geom = CylinderGeometry(
        center_x=center_x,
        center_y=center_y,
        radius=r,
    )

    return nx, ny, float(lx), float(ly), geom


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

    if probe_x is None or probe_y is None:
        times = np.array([t for t, _ in snapshots_list], dtype=float)
        nan_vals = np.full(n, np.nan, dtype=float)
        return times, nan_vals.copy(), nan_vals.copy(), nan_vals.copy()

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


def _build_surface_force_plan(
    xc: np.ndarray,
    yc: np.ndarray,
    geom: CylinderGeometry,
    n_samples: int = 720,
) -> SurfaceForcePlan:
    """Precompute interpolation plans for a line integral on the cylinder surface."""
    theta = np.linspace(0.0, 2.0 * np.pi, n_samples, endpoint=False)
    x_surf = geom.center_x + geom.radius * np.cos(theta)
    y_surf = geom.center_y + geom.radius * np.sin(theta)
    plans = tuple(
        _build_bilinear_plan(xc, yc, float(x), float(y))
        for x, y in zip(x_surf, y_surf)
    )
    return SurfaceForcePlan(
        normals_x=np.cos(theta),
        normals_y=np.sin(theta),
        arc_length=2.0 * np.pi * geom.radius / float(n_samples),
        bilinear_plans=plans,
    )


def _compute_pressure_forces(
    p: np.ndarray,
    force_plan: SurfaceForcePlan,
) -> Tuple[float, float]:
    """Compute pressure forces on the cylinder via a contour integral."""
    pressure_samples = np.array(
        [_apply_bilinear_plan(p, plan) for plan in force_plan.bilinear_plans],
        dtype=float,
    )
    pressure_samples -= np.mean(pressure_samples)

    # Force on the body is - integral(p * n ds) over the body surface.
    f_x = -force_plan.arc_length * np.sum(
        pressure_samples * force_plan.normals_x
    )
    f_y = -force_plan.arc_length * np.sum(
        pressure_samples * force_plan.normals_y
    )
    return float(f_x), float(f_y)


def _compute_forces(
    snapshot_path: str,
    force_plan: SurfaceForcePlan,
) -> Tuple[float, float]:
    """Compute x and y forces from a single snapshot."""
    with np.load(snapshot_path) as data:
        fx_meta = _safe_scalar(data, "meta_ibm_force_x")
        fy_meta = _safe_scalar(data, "meta_ibm_force_y")
        if fx_meta is not None and fy_meta is not None:
            return float(fx_meta), float(fy_meta)
        p = data["p"]
    return _compute_pressure_forces(p, force_plan)


def _compute_coefficients(
    f_x: float,
    f_y: float,
    u_ref: float,
    char_length: float,
    rho: float = 1.0,
) -> Tuple[float, float]:
    """Convert forces to non-dimensional coefficients."""
    if u_ref <= 0:
        return 0.0, 0.0

    q = 0.5 * rho * u_ref**2
    area = char_length
    c_d = 2.0 * f_x / (q * area) if area > 0 else 0.0
    c_l = 2.0 * f_y / (q * area) if area > 0 else 0.0

    return c_d, c_l


def _extract_combined_series(
    snapshots: List[Tuple[float, str]],
    nx: int,
    ny: int,
    lx: float,
    ly: float,
    probe_x: float,
    probe_y: float,
    geom: CylinderGeometry,
    u_ref: float,
    char_length: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract probe series and forces for all snapshots.

    Returns:
        (t, u_probe, v_probe, p_probe, f_x, f_y, c_d, c_l)
    """
    t, u_probe, v_probe, p_probe = _extract_probe_series(
        snapshots, nx, ny, lx, ly, probe_x, probe_y
    )
    xf = np.linspace(0.0, lx, nx + 1)
    xc = 0.5 * (xf[:-1] + xf[1:])
    yf = np.linspace(0.0, ly, ny + 1)
    yc = 0.5 * (yf[:-1] + yf[1:])
    force_plan = _build_surface_force_plan(xc, yc, geom)

    n = len(snapshots)
    f_x_arr = np.empty(n, dtype=float)
    f_y_arr = np.empty(n, dtype=float)
    c_d_arr = np.empty(n, dtype=float)
    c_l_arr = np.empty(n, dtype=float)

    for k, (_, path) in enumerate(snapshots):
        f_x, f_y = _compute_forces(path, force_plan)
        c_d, c_l = _compute_coefficients(f_x, f_y, u_ref, char_length)
        f_x_arr[k] = f_x
        f_y_arr[k] = f_y
        c_d_arr[k] = c_d
        c_l_arr[k] = c_l

    return t, u_probe, v_probe, p_probe, f_x_arr, f_y_arr, c_d_arr, c_l_arr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Comprehensive aerodynamic analysis: Strouhal and drag/lift coefficients.",
    )
    parser.add_argument("--indir", type=str, default="output",
                        help="Snapshot directory (default: output)")
    parser.add_argument("--pattern", type=str, default="snap_*.npz",
                        help="Snapshot filename pattern (default: snap_*.npz)")
    parser.add_argument("--config", type=str, default="config.txt",
                        help="Configuration file (default: config.txt)")

    parser.add_argument("--probe-x", type=float, default=None,
                        help="Optional probe x coordinate in physical units")
    parser.add_argument("--probe-y", type=float, default=None,
                        help="Optional probe y coordinate in physical units")

    parser.add_argument("--u-ref", type=float, default=1.0,
                        help="Reference velocity U (default: 1.0)")
    parser.add_argument("--length-scale", type=float, default=None,
                        help="Characteristic length L (overrides config; default: read from config or 1.0)")
    parser.add_argument("--use-cylinder-diameter", action="store_true",
                        help="Use L = ly/4 (diameter for default cylinder setup)")
    parser.add_argument("--cylinder-radius", type=float, default=None,
                        help="Cylinder radius (default: read from config or ly/8)")

    parser.add_argument("--t-min", type=float, default=1.0,
                        help="Ignore data before this time for frequency fit (default: 1.0)")
    parser.add_argument("--f-min", type=float, default=0.05,
                        help="Min search frequency (default: 0.05)")
    parser.add_argument("--f-max", type=float, default=2.0,
                        help="Max search frequency (default: 2.0)")
    parser.add_argument("--n-freq", type=int, default=4000,
                        help="Number of frequency samples (default: 4000)")

    parser.add_argument("--save-series", type=str, default=None,
                        help="Optional CSV path for combined time series")
    parser.add_argument("--save-report", type=str,
                        default=os.path.join(
                            DEFAULT_RESULTS_DIR, "aero_report.txt"),
                        help="TXT path for comprehensive report (default: results/aero_report.txt)")

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    snapshots = _collect_snapshots(args.indir, args.pattern)
    if not snapshots:
        print(
            f"No snapshots found in {args.indir!r} with pattern {args.pattern!r}.")
        return 1

    # Estimate common scales
    l_char, u_ref, nx, ny, lx, ly = _estimate_scales(
        snapshots[0][1],
        length_scale=args.length_scale,
        use_cylinder_diameter=args.use_cylinder_diameter,
        u_ref=args.u_ref,
        config_path=args.config,
    )

    # Estimate cylinder geometry
    _, _, _, _, geom = _estimate_cylinder_geometry(
        snapshots[0][1],
        config_path=args.config,
        cylinder_radius=args.cylinder_radius,
    )

    # Extract combined series
    t, u_probe, v_probe, p_probe, f_x, f_y, c_d, c_l = _extract_combined_series(
        snapshots,
        nx=nx,
        ny=ny,
        lx=lx,
        ly=ly,
        probe_x=args.probe_x,
        probe_y=args.probe_y,
        geom=geom,
        u_ref=u_ref,
        char_length=l_char,
    )

    # Save combined time series
    if args.save_series:
        out = np.column_stack(
            (t, u_probe, v_probe, p_probe, f_x, f_y, c_d, c_l))
        header = "t,u_probe,v_probe,p_probe,f_x,f_y,c_d,c_l"
        np.savetxt(args.save_series, out, delimiter=",",
                   header=header, comments="")
        print(f"Saved combined series: {args.save_series}")

    # Compute Strouhal from lift coefficient history.
    dt = np.diff(t)
    dt_median = float(np.median(dt)) if dt.size else np.nan
    nyquist_est = 0.5 / \
        dt_median if np.isfinite(dt_median) and dt_median > 0 else np.nan

    lift_strouhal: Optional[SpectralResult] = None
    out = _dominant_frequency(
        t,
        c_l,
        t_min=args.t_min,
        f_min=args.f_min,
        f_max=args.f_max,
        n_freq=args.n_freq,
    )
    if out is not None:
        f_peak, peak_power = out
        lift_strouhal = SpectralResult(
            freq=f_peak,
            peak_power=peak_power,
            st=f_peak * l_char / u_ref,
        )

    # Force statistics
    c_d_mean = float(np.mean(c_d))
    c_d_std = float(np.std(c_d))
    c_l_mean = float(np.mean(c_l))
    c_l_std = float(np.std(c_l))
    c_l_rms = float(np.sqrt(np.mean(c_l**2)))

    # Print comprehensive report
    print("=" * 70)
    print("COMPREHENSIVE AERODYNAMIC ANALYSIS")
    print("=" * 70)
    print(f"Snapshots         : {len(snapshots)}")
    print(f"Time span         : [{t.min():.4f}, {t.max():.4f}]")
    if args.probe_x is not None and args.probe_y is not None:
        print(f"Probe location    : ({args.probe_x:.6g}, {args.probe_y:.6g})")
    print(f"Cylinder center   : ({geom.center_x:.6g}, {geom.center_y:.6g})")
    print(f"Cylinder radius   : {geom.radius:.6g}")
    print(f"Char. length (L)  : {l_char:.6g}")
    print(f"Ref. velocity (U) : {u_ref:.6g}")
    print()
    print("-" * 70)
    print("STROUHAL NUMBER ANALYSIS")
    print("-" * 70)
    print("Signal used       : C_l")
    print(f"Frequency window  : [{args.f_min:.4f}, {args.f_max:.4f}]")
    if np.isfinite(nyquist_est):
        print(
            f"Median dt         : {dt_median:.6g} (Nyquist approx {nyquist_est:.6g})")

    if lift_strouhal is None:
        print("C_l peak         : unavailable (insufficient variation/samples)")
    else:
        edge_note = ""
        if _is_edge_frequency(lift_strouhal.freq, args.f_min, args.f_max):
            edge_note = " [edge]"
        print(
            f"C_l peak         : f={lift_strouhal.freq:.6g}, "
            f"St={lift_strouhal.st:.6g}, "
            f"power={lift_strouhal.peak_power:.6g}{edge_note}"
        )
        print("-" * 70)
        print(f"Lift f0           : {lift_strouhal.freq:.6g}")
        print(f"Lift Strouhal     : {lift_strouhal.st:.6g}")

        if np.isfinite(nyquist_est) and lift_strouhal.freq > 0.8 * nyquist_est:
            print("WARNING: Estimated f0 is close to Nyquist limit.")
            print("         Use smaller save_dt for confidence.")

    print()
    print("-" * 70)
    print("DRAG AND LIFT COEFFICIENT ANALYSIS")
    print("-" * 70)
    print(f"C_d mean          : {c_d_mean:.6g}")
    print(f"C_d std           : {c_d_std:.6g}")
    print(f"C_l mean          : {c_l_mean:.6g}")
    print(f"C_l std           : {c_l_std:.6g}")
    print(f"C_l rms           : {c_l_rms:.6g}")

    # Save comprehensive report
    if args.save_report:
        report_dir = os.path.dirname(args.save_report)
        if report_dir:
            os.makedirs(report_dir, exist_ok=True)

        lines: List[str] = [
            "COMPREHENSIVE AERODYNAMIC ANALYSIS",
            "=" * 70,
            f"Snapshots         : {len(snapshots)}",
            f"Time span         : [{t.min():.4f}, {t.max():.4f}]",
            f"Cylinder center   : ({geom.center_x:.6g}, {geom.center_y:.6g})",
            f"Cylinder radius   : {geom.radius:.6g}",
            f"Char. length (L)  : {l_char:.6g}",
            f"Ref. velocity (U) : {u_ref:.6g}",
            "",
            "-" * 70,
            "STROUHAL NUMBER ANALYSIS",
            "-" * 70,
            "Signal used       : C_l",
            f"Frequency window  : [{args.f_min:.4f}, {args.f_max:.4f}]",
        ]

        if args.probe_x is not None and args.probe_y is not None:
            lines.insert(
                4,
                f"Probe location    : ({args.probe_x:.6g}, {args.probe_y:.6g})",
            )

        if np.isfinite(nyquist_est):
            lines.append(
                f"Median dt         : {dt_median:.6g} "
                f"(Nyquist approx {nyquist_est:.6g})"
            )

        if lift_strouhal is None:
            lines.append(
                "C_l peak         : unavailable (insufficient variation/samples)"
            )
        else:
            edge_note = ""
            if _is_edge_frequency(lift_strouhal.freq, args.f_min, args.f_max):
                edge_note = " [edge]"
            lines.append(
                f"C_l peak         : f={lift_strouhal.freq:.6g}, "
                f"St={lift_strouhal.st:.6g}, "
                f"power={lift_strouhal.peak_power:.6g}{edge_note}"
            )
            lines.append("")
            lines.append(f"Lift f0           : {lift_strouhal.freq:.6g}")
            lines.append(f"Lift Strouhal     : {lift_strouhal.st:.6g}")

        lines.extend([
            "",
            "-" * 70,
            "DRAG AND LIFT COEFFICIENT ANALYSIS",
            "-" * 70,
            f"C_d mean          : {c_d_mean:.6g}",
            f"C_d std           : {c_d_std:.6g}",
            f"C_l mean          : {c_l_mean:.6g}",
            f"C_l std           : {c_l_std:.6g}",
            f"C_l rms           : {c_l_rms:.6g}",
        ])

        with open(args.save_report, "w", encoding="utf-8") as fout:
            fout.write("\n".join(lines) + "\n")

        print(f"\nSaved comprehensive report: {args.save_report}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
