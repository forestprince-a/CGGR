"""Fixed-support graph learning; convex quadratic ADMM with feasible outputs.

Graph variables use directed edges in a symmetric, zero-diagonal support.
No dense N-by-N Gram matrix is constructed.
"""
from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.optimize import linprog
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import ArpackNoConvergence, LinearOperator, cg, eigsh, lobpcg
from sklearn.neighbors import NearestNeighbors

class InfeasibleSupportError(ValueError):
    """A proven structural failure, distinct from an LP solver failure."""


@dataclass
class Support:
    n: int
    rows: np.ndarray
    cols: np.ndarray
    indptr: np.ndarray
    reverse: np.ndarray

    @classmethod
    def from_graph(cls, graph):
        mask = graph.astype(bool).astype(np.float64).tocsr()
        mask.setdiag(0)
        mask.eliminate_zeros()
        mask = mask.maximum(mask.T).tocsr()
        mask.sort_indices()
        rows = np.repeat(np.arange(mask.shape[0]), np.diff(mask.indptr))
        cols = mask.indices.copy()
        keys = rows.astype(np.int64) * mask.shape[0] + cols
        reverse = np.searchsorted(keys, cols.astype(np.int64) * mask.shape[0] + rows)
        if np.any(np.diff(mask.indptr) == 0):
            raise ValueError("Fixed support has an isolated cell")
        return cls(mask.shape[0], rows, cols, mask.indptr.copy(), reverse)

    def matrix(self, values):
        return sparse.csr_matrix((values, self.cols, self.indptr), shape=(self.n, self.n))

    def sample(self, matrix):
        return np.asarray(matrix[self.rows, self.cols]).reshape(-1)

    def feasible_projection(self, values, cfg, state=None):
        from .symmetric_projection import project
        return project(self, values, cfg, state)

    def check_feasible(self, clusters):
        mask = self.matrix(np.ones(len(self.rows)))
        components = connected_components(mask, directed=False, return_labels=False)
        if components > clusters:
            raise InfeasibleSupportError(f"Support has {components} components > K={clusters}; cannot join them")
        undirected = np.flatnonzero(self.rows < self.cols)
        edge_ids = np.arange(len(undirected))
        incidence = sparse.coo_matrix((np.ones(2 * len(edge_ids)),
            (np.r_[self.rows[undirected], self.cols[undirected]], np.r_[edge_ids, edge_ids])),
            shape=(self.n, len(edge_ids))).tocsr()
        result = linprog(np.zeros(len(edge_ids)), A_eq=incidence, b_eq=np.ones(self.n),
                         bounds=(0.0, None), method="highs")
        if not result.success:
            if result.status == 2:
                raise InfeasibleSupportError(
                    f"Fixed support lacks a feasible symmetric stochastic graph: {result.message}")
            raise RuntimeError(f"Support feasibility solver failed: {result.message}")
        # This checks stochastic feasibility only, not exact K-component feasibility.


def feasible_initial_support(mask, clusters, load_embedding):
    """Keep feasible kNN support; repair impossible support once before training.

    A cycle through all cells admits weight 1/2 on each of its two incident
    edges, proving symmetric, nonnegative, zero-diagonal row-stochastic
    feasibility. Its union with kNN adds at most N undirected edges and is
    connected. Training still uses a fixed sparse support. This does NOT prove
    that an exactly K-component weighted graph exists on that support.
    """
    support = Support.from_graph(mask)
    try:
        support.check_feasible(clusters)
        return support
    except InfeasibleSupportError as exc:
        reason = str(exc)
    z = np.asarray(load_embedding(), dtype=np.float64)
    if z.ndim != 2 or len(z) != support.n or z.shape[1] == 0 or not np.isfinite(z).all():
        raise ValueError("Support repair requires a finite cell embedding")
    if support.n < 3:
        raise ValueError("Support cycle repair requires at least three cells")
    # Deterministic and label-free; nearby coordinates in the most variable
    # embedding dimension are linked. Stable ties retain input cell order.
    axis = int(np.argmax(np.var(z, axis=0)))
    order = np.argsort(z[:, axis], kind="stable")
    cycle = sparse.csr_matrix((np.ones(support.n), (order, np.roll(order, -1))),
                              shape=(support.n, support.n))
    repaired = Support.from_graph(support.matrix(np.ones(len(support.rows))) + cycle + cycle.T)
    repaired.check_feasible(clusters)
    added = (len(repaired.rows) - len(support.rows)) // 2
    print(f"INIT support repaired: added_undirected_edges={added}, "
          f"policy=embedding_order_cycle; reason={reason}", flush=True)
    return repaired


def local_scale_initialization(z, cfg):
    n, k = len(z), cfg.neighbors
    raw, _ = NearestNeighbors(n_neighbors=k + 1, metric="euclidean").fit(z).kneighbors(z)
    sigma = np.maximum(raw[:, k], 1e-6)
    norms = np.einsum("ij,ij->i", z, z)
    all_cols, all_weights = [], []
    for start in range(0, n, cfg.distance_block_rows):
        stop = min(n, start + cfg.distance_block_rows)
        d = np.maximum(norms[start:stop, None] + norms[None, :] - 2 * z[start:stop] @ z.T, 0)
        d /= sigma[start:stop, None] * sigma[None, :] + 1e-12
        d[np.arange(stop - start), np.arange(start, stop)] = np.inf
        # Stable sorting makes tied-neighbor selection reproducible.
        order = np.argsort(d, axis=1, kind="stable")[:, :k + 1]
        selected = np.take_along_axis(d, order, axis=1)
        for indices, distances in zip(order, selected):
            denom = k * distances[k] - distances[:k].sum()
            weight = np.full(k, 1.0 / k) if denom < 1e-12 else (distances[k] - distances[:k]) / denom
            all_cols.extend(indices[:k])
            all_weights.extend(weight)
    rows = np.repeat(np.arange(n), k)
    cols = np.asarray(all_cols)
    mask = sparse.csr_matrix((np.ones(len(cols)), (rows, cols)), shape=(n, n))
    support = feasible_initial_support(mask, cfg.clusters, lambda: z)
    initial = sparse.csr_matrix((all_weights, (rows, cols)), shape=(n, n))
    initial = 0.5 * (initial + initial.T)
    weights = support.feasible_projection(support.sample(initial), cfg)
    return support, support.matrix(weights), sigma


def laplacian(graph):
    symmetric = (0.5 * (graph + graph.T)).tocsr()
    return sparse.diags(np.asarray(symmetric.sum(axis=1)).ravel()) - symmetric


def edge_distances(x, support, batch=8192):
    result = np.empty(len(support.rows))
    for a in range(0, len(result), batch):
        b = min(a + batch, len(result))
        delta = x[support.rows[a:b]] - x[support.cols[a:b]]
        if sparse.issparse(delta):
            result[a:b] = np.asarray(delta.multiply(delta).sum(axis=1)).ravel()
        else:
            result[a:b] = np.einsum("ij,ij->i", delta, delta)
    return result


@dataclass
class ADMMState:
    x: np.ndarray
    dual: np.ndarray
    penalty: float
    projection_dual: object = None

    @classmethod
    def initialize(cls, graph, support, cfg):
        x = support.sample(graph).copy()
        return cls(x, np.zeros_like(x), cfg.penalty_initial)


def solve_graph(a_operator, b, graph, support, state, cfg, diagonal=None):
    """min x^T A x - b^T x + indicator_feasible(z), subject to x=z.

    (2A + mu I)x = b + mu z - lambda;
    z = projection(x + lambda/mu); lambda += mu(x-z).
    Exactly five ADMM updates; no claim of exact subproblem convergence.
    """
    z = support.sample(graph).copy()
    primal = dual = 0.0
    for _ in range(cfg.graph_inner_steps):
        mu = state.penalty
        rhs = b + mu * z - state.dual
        if diagonal is not None:
            state.x, info = rhs / (2 * diagonal + mu), 0
        else:
            operator = LinearOperator((len(z), len(z)), dtype=np.float64,
                matvec=lambda x: 2 * support.sample(a_operator(support.matrix(x))) + mu * x)
            state.x, info = cg(operator, rhs, x0=state.x, rtol=cfg.cg_rtol,
                              atol=0.0, maxiter=cfg.cg_maxiter)
        if info != 0 or not np.isfinite(state.x).all():
            raise RuntimeError(f"Graph quadratic solve failed (CG info={info})")
        previous = z.copy()
        z = support.feasible_projection(state.x + state.dual / mu, cfg, state)
        primal = float(np.linalg.norm(state.x - z) / max(1.0, np.linalg.norm(z)))
        dual = float(mu * np.linalg.norm(z - previous) / max(1.0, np.linalg.norm(state.dual)))
        state.dual += mu * (state.x - z)
        # Residual balancing avoids blindly saturating mu over 1500 updates.
        if primal > 10 * dual:
            state.penalty = min(mu * cfg.penalty_growth, cfg.penalty_max)
        elif dual > 10 * primal:
            state.penalty = max(mu / cfg.penalty_growth, cfg.penalty_initial * 1e-6)
    return support.matrix(z), {"primal": primal, "dual": dual, "mu": state.penalty}


def update_s(z_distance, h, c, s, support, state, cfg):
    n, k = support.n, cfg.clusters
    lc = laplacian(c)
    b = -z_distance / (2 * n) - cfg.alpha * edge_distances(h, support) / (2 * k)
    b += 2 * support.sample(c) / n
    return solve_graph(lambda x: (lc @ x + x) / n,
                       b, s, support, state, cfg)


def update_c(s, q, xi, c, support, state, cfg):
    n = support.n
    b = -edge_distances(s, support) / (2 * n)
    b += 2 * support.sample(s) / n - xi * edge_distances(q, support) / 2
    return solve_graph(None, b, c, support, state, cfg, diagonal=1.0 / n)


def bottom_spectrum(graph, cfg, rng, initial=None, workspace=None):
    """K+1 eigenpairs distinguish fewer/exactly/more than K numerical zeros."""
    lap = laplacian(graph)
    n, count = graph.shape[0], cfg.clusters + 1
    if n <= max(64, count + 1):
        values, vectors = np.linalg.eigh(lap.toarray())
        values, vectors = values[:count], vectors[:, :count]
    else:
        initial_vector = rng.standard_normal(n)
        try:
            values, vectors = eigsh(
                lap,
                k=count,
                which="SA",
                v0=initial_vector,
                tol=cfg.eigen_solver_tolerance,
                maxiter=cfg.eigen_maxiter,
            )
        except ArpackNoConvergence as first_error:
            # Repeated or tightly clustered zero eigenvalues can stall ARPACK
            # on graph Laplacians. Keep any complete partial result; otherwise
            # retry with a larger Krylov subspace and a practical tolerance.
            if (first_error.eigenvalues is not None
                    and first_error.eigenvectors is not None
                    and len(first_error.eigenvalues) >= count):
                values = first_error.eigenvalues
                vectors = first_error.eigenvectors
            else:
                retry_tolerance = max(cfg.eigen_solver_tolerance, 1e-7)
                retry_ncv = min(n, max(4 * count + 1, 40))
                try:
                    values, vectors = eigsh(
                        lap,
                        k=count,
                        which="SM",
                        v0=initial_vector,
                        ncv=retry_ncv,
                        tol=retry_tolerance,
                        maxiter=cfg.eigen_maxiter,
                    )
                except ArpackNoConvergence as retry_error:
                    if (retry_error.eigenvalues is not None
                            and retry_error.eigenvectors is not None
                            and len(retry_error.eigenvalues) >= count):
                        values = retry_error.eigenvalues
                        vectors = retry_error.eigenvectors
                    else:
                        initial_block = rng.standard_normal((n, count))
                        initial_block[:, 0] = 1.0 / np.sqrt(n)
                        initial_block, _ = np.linalg.qr(initial_block, mode="reduced")
                        lobpcg_tolerance = max(
                            retry_tolerance,
                            0.5 * cfg.eigen_zero_tolerance,
                        )
                        values, vectors = lobpcg(
                            lap,
                            initial_block,
                            largest=False,
                            tol=lobpcg_tolerance,
                            maxiter=cfg.eigen_maxiter,
                        )
        order = np.argsort(values)
        values, vectors = values[order][:count], vectors[:, order][:, :count]
    # Use the same rank-accuracy target as the CUDA solver. A finite but
    # unresolved spectrum must not change xi or be reported as rank-valid.
    tolerance = min(1e-8, 0.01 * cfg.eigen_zero_tolerance)
    components = -1
    if abs(values[-1]) <= tolerance:
        mask = graph.copy().tocsr()
        mask.eliminate_zeros()
        components, labels = connected_components(mask, directed=False)
        if components >= count:
            sizes = np.bincount(labels)
            constant = np.sqrt(sizes / n)[:, None]
            candidates = rng.standard_normal((components, count - 1))
            basis, _ = np.linalg.qr(np.concatenate((constant, candidates), axis=1), mode="reduced")
            vectors = basis[labels] / np.sqrt(sizes[labels])[:, None]
            values = np.zeros(count)
    if not np.isfinite(values).all() or not np.isfinite(vectors).all() or values[0] < -tolerance:
        raise RuntimeError("Invalid Laplacian spectrum")
    residual = np.linalg.norm(lap @ vectors - vectors * values, axis=0).max()
    orthogonality = np.linalg.norm(vectors.T @ vectors - np.eye(count))
    if not np.isfinite(residual + orthogonality) or orthogonality > 1e-7:
        raise RuntimeError(
            "Invalid eigenpair state: "
            f"residual={residual:.3e}, "
            f"orthogonality={orthogonality:.3e}"
        )
    if workspace is not None:
        workspace["basis"] = vectors
        workspace["diagnostics"] = {
            "residual": float(residual), "orthogonality": float(orthogonality),
            "accuracy_target": tolerance, "accuracy_satisfied": bool(residual <= tolerance),
            "exact_components": int(components),
        }
    return vectors[:, :cfg.clusters], values


def rank_state(values, cfg):
    k, tol = cfg.clusters, cfg.eigen_zero_tolerance
    if values[k - 1] > tol:
        return "too_few"
    if values[k] <= tol:
        return "too_many"
    return "satisfied"


def graph_residuals(graph):
    delta = graph - graph.T
    denom = max(1e-12, np.linalg.norm(graph.data))
    return {"row_sum": float(np.max(np.abs(np.asarray(graph.sum(axis=1)).ravel() - 1))),
            "symmetry": float(np.linalg.norm(delta.data) / denom),
            "minimum": float(min(0, graph.data.min())),
            "diagonal": float(np.max(np.abs(graph.diagonal())))}
