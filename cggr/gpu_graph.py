"""CUDA graph solver: sparse edge storage, batched projection, warm spectra.

Graph and eigenvector arithmetic stays on the selected GPU in float64.
Support feasibility LP and exact graph component labels use CPU boundaries.
The candidate support is initialized once and fixed during optimization.
"""
from dataclasses import dataclass

import cupy as cp
import numpy as np
from scipy import sparse as host_sparse
from cupyx.scipy import sparse
from cupyx.scipy.sparse.linalg import LinearOperator, cg

from .graph import Support as HostSupport, feasible_initial_support, rank_state
from .projection import make_row_groups


# Edges and CSR pointers are int32; weights are float64. These kernels avoid
# constructing full sparse matrix products just to sample their support.
_KERNELS = cp.RawModule(code=r"""
extern "C" __global__
void sample_csr(const int* ptr, const int* col, const double* val,
                const int* rows, const int* cols, double* out, int edges) {
    int e = blockDim.x * blockIdx.x + threadIdx.x;
    if (e >= edges) return;
    int left = ptr[rows[e]], right = ptr[rows[e] + 1], stop = right;
    while (left < right) {
        int mid = left + (right - left) / 2;
        if (col[mid] < cols[e]) left = mid + 1; else right = mid;
    }
    out[e] = (left < stop && col[left] == cols[e]) ? val[left] : 0.0;
}
extern "C" __global__
void sparse_distance(const int* ptr, const int* col, const double* val,
                     const int* rows, const int* cols, double* out, int edges) {
    int e = blockDim.x * blockIdx.x + threadIdx.x;
    if (e >= edges) return;
    int a = ptr[rows[e]], ae = ptr[rows[e] + 1];
    int b = ptr[cols[e]], be = ptr[cols[e] + 1];
    double total = 0.0;
    while (a < ae || b < be) {
        double delta;
        if (b >= be || (a < ae && col[a] < col[b])) delta = val[a++];
        else if (a >= ae || col[b] < col[a]) delta = -val[b++];
        else { delta = val[a++] - val[b++]; }
        total += delta * delta;
    }
    out[e] = total;
}
extern "C" __global__
void dense_distance(const double* x, const int* rows, const int* cols,
                    double* out, int edges, int dimensions) {
    int e = blockDim.x * blockIdx.x + threadIdx.x;
    if (e >= edges) return;
    double total = 0.0;
    for (int j = 0; j < dimensions; ++j) {
        double delta = x[(long long)rows[e] * dimensions + j]
                     - x[(long long)cols[e] * dimensions + j];
        total += delta * delta;
    }
    out[e] = total;
}
extern "C" __global__
void restricted_laplacian(const int* ptr, const int* col, const int* rows,
                          const int* reverse, const double* c,
                          const double* degree, const double* x,
                          double* out, int edges) {
    int e = blockDim.x * blockIdx.x + threadIdx.x;
    if (e >= edges) return;
    int i = rows[e], j = col[e];
    int a = ptr[i], ae = ptr[i + 1], b = ptr[j], be = ptr[j + 1];
    double dot = 0.0;
    while (a < ae && b < be) {
        if (col[a] < col[b]) ++a;
        else if (col[b] < col[a]) ++b;
        else { dot += c[a] * x[reverse[b]]; ++a; ++b; }
    }
    out[e] = degree[i] * x[e] - dot;
}
""")


def _launch(name, size, arguments):
    _KERNELS.get_function(name)(((size + 255) // 256,), (256,),
                               (*arguments, np.int32(size)))


class Support(HostSupport):
    @classmethod
    def from_host(cls, host):
        if len(host.rows) >= np.iinfo(np.int32).max:
            raise ValueError("GPU support exceeds int32 CSR capacity")
        result = cls(host.n, *(cp.asarray(x, dtype=cp.int32) for x in
                     (host.rows, host.cols, host.indptr, host.reverse)))
        result._groups = make_row_groups(host.indptr, cp)
        return result

    def matrix(self, values):
        return sparse.csr_matrix((values, self.cols, self.indptr), shape=(self.n, self.n))

    def sample(self, matrix):
        matrix = matrix.tocsr()
        if (matrix.indices.data.ptr == self.cols.data.ptr
                and matrix.indptr.data.ptr == self.indptr.data.ptr):
            return matrix.data
        matrix.sort_indices()
        out = cp.empty(len(self.rows), dtype=cp.float64)
        _launch("sample_csr", len(out), (
            matrix.indptr.astype(cp.int32, copy=False),
            matrix.indices.astype(cp.int32, copy=False), matrix.data,
            self.rows, self.cols, out))
        return out

    def feasible_projection(self, values, cfg, state=None):
        from .symmetric_projection import project
        return project(self, values, cfg, state)


def local_scale_initialization(z, cfg):
    n, k = len(z), cfg.neighbors
    norms = cp.einsum("ij,ij->i", z, z)
    sigma = cp.empty(n, dtype=cp.float64)
    # Exact, blocked all-pairs distances: no resident dense N-by-N graph.
    for start in range(0, n, cfg.distance_block_rows):
        stop = min(n, start + cfg.distance_block_rows)
        d = cp.maximum(norms[start:stop, None] + norms[None, :] - 2 * z[start:stop] @ z.T, 0)
        d[cp.arange(stop - start), cp.arange(start, stop)] = 0
        sigma[start:stop] = cp.maximum(cp.sqrt(cp.partition(d, k, axis=1)[:, k]), 1e-6)
    all_cols = cp.empty((n, k), dtype=cp.int32)
    all_weights = cp.empty((n, k), dtype=cp.float64)
    for start in range(0, n, cfg.distance_block_rows):
        stop = min(n, start + cfg.distance_block_rows)
        d = cp.maximum(norms[start:stop, None] + norms[None, :] - 2 * z[start:stop] @ z.T, 0)
        d /= sigma[start:stop, None] * sigma[None, :] + 1e-12
        d[cp.arange(stop - start), cp.arange(start, stop)] = cp.inf
        order = cp.argsort(d, axis=1)[:, :k + 1]  # CuPy argsort is stable.
        selected = cp.take_along_axis(d, order, axis=1)
        denom = k * selected[:, k] - selected[:, :k].sum(axis=1)
        all_cols[start:stop] = order[:, :k]
        all_weights[start:stop] = cp.where(
            denom[:, None] < 1e-12, 1.0 / k,
            (selected[:, k, None] - selected[:, :k]) / cp.maximum(denom[:, None], 1e-12))
    # The existing exact LP is a one-time CPU boundary, outside training.
    rows = np.repeat(np.arange(n), k)
    cols = cp.asnumpy(all_cols).ravel()
    mask = host_sparse.csr_matrix((np.ones(len(cols)), (rows, cols)), shape=(n, n))
    host = feasible_initial_support(mask, cfg.clusters, lambda: cp.asnumpy(z))
    initial = host_sparse.csr_matrix((cp.asnumpy(all_weights).ravel(), (rows, cols)), shape=(n, n))
    initial = 0.5 * (initial + initial.T)
    support = Support.from_host(host)
    values = support.feasible_projection(cp.asarray(host.sample(initial)), cfg)
    return support, support.matrix(values), sigma


def laplacian(graph):
    symmetric = (0.5 * (graph + graph.T)).tocsr()
    return sparse.diags(cp.asarray(symmetric.sum(axis=1)).ravel()) - symmetric


def edge_distances(x, support, batch=8192):
    out = cp.empty(len(support.rows), dtype=cp.float64)
    if sparse.issparse(x):
        x = x.tocsr()
        x.sort_indices()
        _launch("sparse_distance", len(out), (
            x.indptr.astype(cp.int32, copy=False), x.indices.astype(cp.int32, copy=False),
            x.data, support.rows, support.cols, out))
    else:
        x = cp.ascontiguousarray(x, dtype=cp.float64)
        _KERNELS.get_function("dense_distance")(((len(out) + 255) // 256,), (256,),
            (x, support.rows, support.cols, out, np.int32(len(out)), np.int32(x.shape[1])))
    return out


@dataclass
class ADMMState:
    x: object
    dual: object
    penalty: float
    projection_dual: object = None

    @classmethod
    def initialize(cls, graph, support, cfg):
        x = support.sample(graph).copy()
        return cls(x, cp.zeros_like(x), cfg.penalty_initial)


def solve_graph(a_operator, b, graph, support, state, cfg, diagonal=None, preconditioner_diagonal=None):
    z = support.sample(graph).copy()
    cg_iterations = projection_iterations = projection_cg_iterations = 0
    cg_warning_count = 0
    cg_residual = 0.0
    projection_error = 0.0
    primal = dual = 0.0
    for _ in range(cfg.graph_inner_steps):
        mu = state.penalty
        rhs = b + mu * z - state.dual
        if diagonal is not None:
            # (2/N + mu) I: exact ADMM x update, no CG needed.
            state.x = rhs / (2 * diagonal + mu)
        else:
            def count_iteration(_):
                nonlocal cg_iterations
                cg_iterations += 1
            operator = LinearOperator((len(z), len(z)), dtype=cp.float64,
                                      matvec=lambda x: 2 * a_operator(x) + mu * x)
            inverse_diagonal = (1.0 / mu if preconditioner_diagonal is None
                                else 1.0 / (2 * preconditioner_diagonal + mu))
            preconditioner = LinearOperator((len(z), len(z)), dtype=cp.float64,
                                           matvec=lambda x: inverse_diagonal * x)
            state.x, info = cg(operator, rhs, x0=state.x, tol=cfg.cg_rtol,
                              atol=0.0, maxiter=cfg.cg_maxiter, callback=count_iteration, M=preconditioner)
            if info < 0:
                raise RuntimeError(f"GPU graph CG breakdown (info={info})")
            if info > 0:
                residual = float(cp.linalg.norm(operator @ state.x - rhs) /
                    cp.maximum(cp.linalg.norm(rhs), 1e-30))
                if not np.isfinite(residual):
                    raise RuntimeError("Non-finite GPU graph CG residual")
                cg_residual = max(cg_residual, residual)
                cg_warning_count += 1
                print(f"GPU graph CG warning: residual={residual:.3e}; "
                      "using bounded iterate and enforcing graph projection", flush=True)
        if not bool(cp.isfinite(state.x).all()):
            raise RuntimeError("Non-finite GPU graph solve")
        previous = z
        z = support.feasible_projection(state.x + state.dual / mu, cfg, state)
        projection_iterations += support.last_projection_iterations
        projection_cg_iterations += support.last_projection_cg_iterations
        projection_error = max(projection_error, support.last_projection_error)
        primal = float(cp.linalg.norm(state.x - z) / cp.maximum(1.0, cp.linalg.norm(z)))
        dual = float(mu * cp.linalg.norm(z - previous) / cp.maximum(1.0, cp.linalg.norm(state.dual)))
        state.dual += mu * (state.x - z)
        if primal > 10 * dual:
            state.penalty = min(mu * cfg.penalty_growth, cfg.penalty_max)
        elif dual > 10 * primal:
            state.penalty = max(mu / cfg.penalty_growth, cfg.penalty_initial * 1e-6)
    return support.matrix(z), {"primal": primal, "dual": dual, "mu": state.penalty,
                              "cg_iterations": cg_iterations,
                              "cg_warning_count": cg_warning_count,
                              "cg_warning_residual": cg_residual,
                              "projection_iterations": projection_iterations,
                              "projection_cg_iterations": projection_cg_iterations,
                              "projection_error": projection_error}


def update_s(z_distance, h, c, s, support, state, cfg):
    n, k = support.n, cfg.clusters
    c_values = support.sample(c)
    degree = cp.bincount(support.rows, weights=c_values, minlength=n)
    b = -z_distance / (2 * n) - cfg.alpha * edge_distances(h, support) / (2 * k)
    b += 2 * c_values / n

    def operator(x):
        out = cp.empty_like(x)
        _launch("restricted_laplacian", len(x), (
            support.indptr, support.cols, support.rows, support.reverse,
            c_values, degree, x, out))
        return (out + x) / n

    return solve_graph(operator, b, s, support, state, cfg,
        preconditioner_diagonal=(degree[support.rows] + 1.0) / n)


def update_c(s, q, xi, c, support, state, cfg):
    n = support.n
    distance = edge_distances(s, support)
    support._distance_graph, support._distance_values = s, distance
    b = -distance / (2 * n)
    b += 2 * support.sample(s) / n - xi * edge_distances(q, support) / 2
    return solve_graph(None, b, c, support, state, cfg, diagonal=1.0 / n)


def regression_energy(s, c, support):
    distance = (support._distance_values if getattr(support, "_distance_graph", None) is s
                else edge_distances(s, support))
    return float(support.sample(c) @ distance) / 2


def bottom_spectrum(graph, cfg, rng, initial=None, workspace=None):
    from .gpu_spectrum import solve
    return solve(graph, laplacian(graph), cfg, rng, initial, workspace)


def graph_residuals(graph):
    delta = graph - graph.T
    return {"row_sum": float(cp.max(cp.abs(cp.asarray(graph.sum(axis=1)).ravel() - 1))),
            "symmetry": float(cp.linalg.norm(delta.data) / cp.maximum(1e-12, cp.linalg.norm(graph.data))),
            "minimum": min(0.0, float(graph.data.min())),
            "diagonal": float(cp.max(cp.abs(graph.diagonal())))}
