"""Row-index groups used by the joint symmetric graph projection."""
import numpy as np
from .backend import to_host


def make_row_groups(indptr, xp):
    # Bucket by powers of two: padding stays below twice the number of edges.
    # Indices are built once, outside the repeated projection loop.
    ptr = np.asarray(to_host(indptr), dtype=np.int64)
    degree = np.diff(ptr)
    buckets = np.ceil(np.log2(degree)).astype(int)
    groups = []
    for bucket in np.unique(buckets):
        rows = np.flatnonzero(buckets == bucket)
        width = 1 << int(bucket)
        slot = np.arange(width)[None, :]
        valid = slot < degree[rows, None]
        indices = np.where(valid, ptr[rows, None] + slot, 0)
        positions = np.flatnonzero(valid.ravel())
        edges = indices.ravel()[positions]
        groups.append((xp.asarray(indices), xp.asarray(valid),
                       xp.asarray(positions), xp.asarray(edges),
                       xp.arange(1, width + 1, dtype=xp.float64)[None, :]))
    return groups
