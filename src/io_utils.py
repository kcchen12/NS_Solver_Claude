"""
I/O utilities — save and load simulation snapshots.

Supported formats
-----------------
HDF5  (via ``h5py``)  — recommended for large datasets / parallel I/O.
NumPy (via ``numpy``) — simple, no extra dependencies.

Snapshot format
---------------
Each snapshot stores:
    u    : x-velocity  (nx+1, ny)
    v    : y-velocity  (nx,   ny+1)
    p    : pressure    (nx,   ny)
    t    : simulation time (scalar)
    meta : dict with grid parameters and solver settings
"""

import os
import numpy as np

from src.grid import CartesianGrid

try:
    import h5py
    _HDF5_AVAILABLE = True
except ImportError:
    _HDF5_AVAILABLE = False


# ---------------------------------------------------------------------------
# HDF5
# ---------------------------------------------------------------------------

def save_hdf5(path: str, u: np.ndarray, v: np.ndarray, p: np.ndarray,
              t: float, meta: dict = None, extra: dict = None) -> None:
    """Save a snapshot to an HDF5 file."""
    if not _HDF5_AVAILABLE:
        raise ImportError("h5py is required for HDF5 output.")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset("u", data=u, compression="gzip")
        f.create_dataset("v", data=v, compression="gzip")
        f.create_dataset("p", data=p, compression="gzip")
        if extra:
            for key, value in extra.items():
                f.create_dataset(str(key), data=np.array(
                    value), compression="gzip")
        f.attrs["t"] = t
        if meta:
            for k, v_ in meta.items():
                f.attrs[k] = v_


def load_hdf5(path: str):
    """Load a snapshot from an HDF5 file.

    Returns (u, v, p, t, meta).
    """
    if not _HDF5_AVAILABLE:
        raise ImportError("h5py is required for HDF5 input.")
    with h5py.File(path, "r") as f:
        u = f["u"][:]
        v = f["v"][:]
        p = f["p"][:]
        t = float(f.attrs["t"])
        meta = dict(f.attrs)
    return u, v, p, t, meta


# ---------------------------------------------------------------------------
# NumPy
# ---------------------------------------------------------------------------

def save_numpy(path: str, u: np.ndarray, v: np.ndarray, p: np.ndarray,
               t: float, meta: dict = None, extra: dict = None) -> None:
    """Save a snapshot to a compressed NumPy archive (.npz)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    kwargs = dict(u=u, v=v, p=p, t=np.array(t))
    if meta:
        for k, v_ in meta.items():
            kwargs[f"meta_{k}"] = np.array(v_)
    if extra:
        for k, v_ in extra.items():
            kwargs[str(k)] = np.array(v_)
    np.savez_compressed(path, **kwargs)


def load_numpy(path: str):
    """Load a snapshot from a NumPy archive.

    Returns (u, v, p, t, meta).
    """
    with np.load(path, allow_pickle=False) as data:
        u = data["u"].copy()
        v = data["v"].copy()
        p = data["p"].copy()
        t = float(data["t"])
        meta = {}
        for key in data.files:
            if not key.startswith("meta_"):
                continue
            value = data[key]
            meta[key[5:]] = value.item() if np.ndim(
                value) == 0 else value.copy()
    return u, v, p, t, meta


def save_grid_metadata(path: str, grid) -> None:
    """Save a prepared grid description to a compressed NumPy archive."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez_compressed(path, **grid.to_metadata())


def save_grid_metadata_dict(path: str, metadata: dict) -> None:
    """Save arbitrary prepared-grid metadata to a compressed NumPy archive."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez_compressed(path, **metadata)


def load_grid_metadata_dict(path: str) -> dict:
    """Load prepared-grid metadata as a raw dictionary."""
    with np.load(path, allow_pickle=False) as data:
        metadata = {}
        for key in data.files:
            value = data[key]
            if np.ndim(value) == 0:
                metadata[key] = value.item()
            else:
                metadata[key] = value.copy()
    return metadata


def load_grid_metadata(path: str) -> CartesianGrid:
    """Load a prepared uniform grid description from a compressed NumPy archive."""
    metadata = load_grid_metadata_dict(path)
    grid_type = str(metadata.get("grid_type", "uniform"))
    if grid_type != "uniform":
        raise ValueError(
            "load_grid_metadata only supports uniform grids. "
            "Use load_grid_metadata_dict for non-uniform prepared-grid metadata."
        )
    return CartesianGrid.from_metadata(metadata)


def load_prepared_grid(path: str) -> CartesianGrid:
    """Load prepared-grid metadata and construct a CartesianGrid."""
    metadata = load_grid_metadata_dict(path)
    return CartesianGrid.from_metadata(metadata)


# ---------------------------------------------------------------------------
# Generic save / load (auto-detect format)
# ---------------------------------------------------------------------------

def save_snapshot(path: str, u, v, p, t: float,
                  meta: dict = None, extra: dict = None,
                  fmt: str = "auto") -> None:
    """
    Save a snapshot.

    Parameters
    ----------
    path : str
        Output file path (extension determines format if *fmt* == 'auto').
    fmt : {'auto', 'hdf5', 'numpy'}
    """
    if fmt == "auto":
        fmt = "hdf5" if path.endswith((".h5", ".hdf5")) else "numpy"
    if fmt == "hdf5":
        save_hdf5(path, u, v, p, t, meta, extra)
    else:
        save_numpy(path, u, v, p, t, meta, extra)


def load_snapshot(path: str, fmt: str = "auto"):
    """Load a snapshot (auto-detect format from extension)."""
    if fmt == "auto":
        fmt = "hdf5" if path.endswith((".h5", ".hdf5")) else "numpy"
    if fmt == "hdf5":
        return load_hdf5(path)
    return load_numpy(path)
