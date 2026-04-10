"""Tests for serial MPI decomposition helpers."""

import numpy as np

from src.parallel import ParallelDecomposition


class TestParallelDecompositionSerial:
    def test_row_partition_covers_domain(self):
        decomp = ParallelDecomposition(11)
        assert decomp.ny_local == 11
        assert decomp.j_start == 0
        assert decomp.j_end == 11
        assert not decomp.is_parallel

    def test_serial_exchange_is_noop(self):
        decomp = ParallelDecomposition(4)
        u_local = np.arange(15, dtype=float).reshape(5, 3)
        exchanged = decomp.exchange_ghost_u(u_local)
        assert exchanged is u_local

    def test_serial_gather_and_scatter_are_noops(self):
        decomp = ParallelDecomposition(4)
        field = np.arange(12, dtype=float).reshape(3, 4)
        gathered = decomp.gather_to_root(field)
        scattered = decomp.scatter_from_root(field, nx=3)
        assert gathered is field
        assert scattered is field
