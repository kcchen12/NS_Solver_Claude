"""
MPI parallel utilities for domain decomposition.

Strategy
--------
The domain is split into horizontal slabs (decomposed in the y-direction).
Each MPI rank owns ``ny_local`` rows of cells:

    rank 0 : rows  0          .. ny_local[0]  - 1
    rank 1 : rows  ny_local[0].. ny_local[0+1]- 1
    ...

Ghost-cell exchange
-------------------
One layer of ghost cells is maintained above and below each local slab so
that the finite-volume stencils (which reach ±1 cell) can be evaluated
without further communication.

Pressure Poisson (simple parallel implementation)
--------------------------------------------------
For moderate problem sizes the global Poisson problem is gathered on
rank 0, solved with the serial sparse solver, then scattered back.
For large problems a fully parallel iterative solver (e.g. via
``petsc4py``) should replace this.

Fallback
--------
If ``mpi4py`` is not available or the run is serial (``size == 1``),
all helper functions behave as no-ops.
"""

import numpy as np

try:
    from mpi4py import MPI
    _MPI_AVAILABLE = True
except Exception:
    MPI = None
    _MPI_AVAILABLE = False


class ParallelDecomposition:
    """
    1-D domain decomposition in the y-direction.

    Parameters
    ----------
    ny_global : int
        Total number of cells in the y-direction.
    comm : MPI.Comm, optional
        MPI communicator. Defaults to ``MPI.COMM_WORLD`` if available,
        otherwise a dummy serial communicator is used.
    """

    def __init__(self, ny_global: int, comm=None):
        self.ny_global = ny_global

        if _MPI_AVAILABLE:
            self.comm = comm if comm is not None else MPI.COMM_WORLD
            self.rank = self.comm.Get_rank()
            self.size = self.comm.Get_size()
        else:
            self.comm = None
            self.rank = 0
            self.size = 1

        # Distribute rows as evenly as possible
        base, remainder = divmod(ny_global, self.size)
        self.ny_local_list = [
            base + (1 if r < remainder else 0) for r in range(self.size)
        ]
        self.ny_local = self.ny_local_list[self.rank]

        # Global start/end row index for this rank
        self.j_start = sum(self.ny_local_list[:self.rank])
        self.j_end = self.j_start + self.ny_local  # exclusive

    @property
    def is_parallel(self) -> bool:
        return self.size > 1

    # ------------------------------------------------------------------
    # Ghost-cell exchange for velocity
    # ------------------------------------------------------------------

    def exchange_ghost_u(self, u_local: np.ndarray) -> np.ndarray:
        """
        Exchange one layer of ghost cells in the y-direction for u.

        Parameters
        ----------
        u_local : np.ndarray, shape (nx+1, ny_local)

        Returns
        -------
        u_ext : np.ndarray, shape (nx+1, ny_local+2)
            u with one ghost row prepended and appended.
        """
        if not self.is_parallel or not _MPI_AVAILABLE:
            return u_local  # no-op in serial

        from mpi4py import MPI
        comm = self.comm
        rank = self.rank
        size = self.size
        nx1 = u_local.shape[0]
        ny_l = u_local.shape[1]

        ghost_bot = np.empty(nx1, dtype=u_local.dtype)
        ghost_top = np.empty(nx1, dtype=u_local.dtype)

        # Send bottom row to rank-1, receive from rank+1 (upward exchange)
        if rank < size - 1:
            comm.Send(u_local[:, -1].copy(), dest=rank + 1, tag=0)
        if rank > 0:
            comm.Recv(ghost_bot, source=rank - 1, tag=0)
        else:
            ghost_bot[:] = u_local[:, 0]  # reflect at bottom (Neumann)

        if rank > 0:
            comm.Send(u_local[:, 0].copy(), dest=rank - 1, tag=1)
        if rank < size - 1:
            comm.Recv(ghost_top, source=rank + 1, tag=1)
        else:
            ghost_top[:] = u_local[:, -1]  # reflect at top

        u_ext = np.concatenate(
            [ghost_bot[:, np.newaxis], u_local, ghost_top[:, np.newaxis]],
            axis=1
        )
        return u_ext

    # ------------------------------------------------------------------
    # Gather / scatter helpers for Poisson solver
    # ------------------------------------------------------------------

    def gather_to_root(self, field_local: np.ndarray) -> np.ndarray:
        """
        Gather a (nx, ny_local) field to rank 0 as (nx, ny_global).

        Returns the full array on rank 0, None on other ranks.
        """
        if not self.is_parallel or not _MPI_AVAILABLE:
            return field_local

        from mpi4py import MPI
        nx = field_local.shape[0]
        recvbuf = None
        if self.rank == 0:
            recvbuf = np.empty((nx, self.ny_global), dtype=field_local.dtype)

        counts = [nx * n for n in self.ny_local_list]
        displs = [sum(counts[:r]) for r in range(self.size)]

        self.comm.Gatherv(
            field_local.ravel(),
            [recvbuf, counts, displs, MPI.DOUBLE] if self.rank == 0 else None,
            root=0
        )
        if self.rank == 0:
            return recvbuf
        return None

    def scatter_from_root(self, field_global: np.ndarray,
                          nx: int, dtype=np.float64) -> np.ndarray:
        """
        Scatter (nx, ny_global) from rank 0 to all ranks as (nx, ny_local).
        """
        if not self.is_parallel or not _MPI_AVAILABLE:
            return field_global

        from mpi4py import MPI
        counts = [nx * n for n in self.ny_local_list]
        displs = [sum(counts[:r]) for r in range(self.size)]

        sendbuf = field_global.ravel() if self.rank == 0 else None
        recvbuf = np.empty(nx * self.ny_local, dtype=dtype)
        self.comm.Scatterv(
            [sendbuf, counts, displs, MPI.DOUBLE],
            recvbuf,
            root=0
        )
        return recvbuf.reshape((nx, self.ny_local))

    def allreduce_max(self, value: float) -> float:
        """Global maximum across all ranks."""
        if not self.is_parallel or not _MPI_AVAILABLE:
            return value
        from mpi4py import MPI
        return self.comm.allreduce(value, op=MPI.MAX)

    def barrier(self) -> None:
        """MPI barrier (no-op in serial)."""
        if self.is_parallel and _MPI_AVAILABLE:
            self.comm.Barrier()
