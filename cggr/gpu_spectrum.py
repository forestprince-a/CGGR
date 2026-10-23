"""GPU bottom eigenspace with exact nullspace deflation and bounded retries."""
import warnings

import cupy as cp
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from cupyx.scipy.sparse.linalg import lobpcg


def _guarded_lobpcg(*args, **kwargs):
    """Route CuPy's null residual-block failure to the existing retry solver."""
    try:
        return lobpcg(*args, **kwargs)
    except AttributeError as exc:
        # Match the observed CuPy internal failure only. Unrelated attribute
        # errors and CUDA runtime/OOM errors must still propagate.
        trace = exc.__traceback__
        internal = False
        while trace is not None:
            filename = trace.tb_frame.f_code.co_filename.replace("\\", "/")
            internal |= filename.endswith("cupyx/scipy/sparse/linalg/_lobpcg.py")
            trace = trace.tb_next
        if str(exc) != "'NoneType' object has no attribute 'ndim'" or not internal:
            raise
        raise ValueError("CuPy LOBPCG null residual block: " + str(exc)) from exc


def _orthogonalize(block, against=None):
    if against is not None and against.shape[1]:
        # Twice for numerical stability when warm directions become null modes.
        for _ in range(2):
            block = block - against @ (against.T @ block)
    norms = cp.linalg.norm(block, axis=0)
    block = block[:, norms > 1e-12]
    norms = norms[norms > 1e-12]
    if not block.shape[1]:
        return block
    # Thin SVD exposes dependent columns instead of factoring a singular Gram.
    u, sigma, _ = cp.linalg.svd(block / norms, full_matrices=False)
    return u[:, sigma > 1e-10 * sigma[0]]


def _ritz(lap, basis, count):
    product = lap @ basis
    small = basis.T @ product
    values, rotation = cp.linalg.eigh(0.5 * (small + small.T))
    vectors = basis @ rotation[:, :count]
    return values[:count], vectors


def _errors(lap, values, vectors):
    residual = float(cp.linalg.norm(lap @ vectors - vectors * values, axis=0).max())
    orth = float(cp.linalg.norm(vectors.T @ vectors - cp.eye(vectors.shape[1])))
    return residual, orth


def _component_vectors(labels_gpu, sizes, count, rng, cached=None):
    """A zero eigenspace covering every component, with no zero cell rows."""
    components = len(sizes)
    constant = cp.sqrt(sizes / sizes.sum())[:, None]
    warm = constant[:, :0]
    if cached is not None:
        coordinates = cp.stack([cp.bincount(labels_gpu, weights=cached[:, j],
            minlength=components) for j in range(cached.shape[1])], axis=1)
        coordinates /= cp.sqrt(sizes)[:, None]
        warm = _orthogonalize(coordinates, constant)[:, :count - 1]
    basis = cp.concatenate((constant, warm), axis=1)
    if basis.shape[1] < count:
        random = _orthogonalize(cp.asarray(rng.standard_normal((components, count))), basis)
        basis = cp.concatenate((basis, random[:, :count - basis.shape[1]]), axis=1)
    if basis.shape[1] != count:
        raise RuntimeError("Insufficient component eigenspace directions")
    # Separately orthogonalized warm/random blocks can retain a small constant
    # component, amplified when nearly dependent warm columns are normalized.
    # Reorthogonalize the COMPLETE basis in component coordinates. The first
    # column stays constant, so every cell still has a nonzero embedding row.
    # This changes only the basis inside the exact zero eigenspace, not C.
    basis, _ = cp.linalg.qr(basis, mode="reduced")
    return basis[labels_gpu] / cp.sqrt(sizes[labels_gpu])[:, None]


def _block_retry(lap, block, null, needed, cfg, tolerance):
    """Locally optimal block Rayleigh-Ritz using rank-revealing orthogonalization.

    This retries with the best available block, not a random scalar Krylov
    vector (which cannot span a repeated eigenvalue from one starting vector).
    Memory is bounded by three blocks, independent of the iteration count.
    """
    width = block.shape[1]
    previous = block[:, :0]
    best_values, best_block, best_error = None, block, float("inf")
    values, block = _ritz(lap, block, width)
    for iteration in range(cfg.gpu_eigen_maxiter + 1):
        residual = lap @ block - block * values
        error = float(cp.linalg.norm(residual[:, :needed], axis=0).max())
        orth = float(cp.linalg.norm(block.T @ block - cp.eye(block.shape[1])))
        usable = np.isfinite(error + orth) and orth <= 1e-7
        if usable and error < best_error:
            best_values, best_block, best_error = values, block, error
        if usable and error <= tolerance:
            return values, block, iteration
        if not usable or iteration == cfg.gpu_eigen_maxiter:
            break
        try:
            basis = _orthogonalize(cp.concatenate((block, residual, previous), axis=1), null)
            if basis.shape[1] < needed:
                break
            values, new = _ritz(lap, basis, min(width, basis.shape[1]))
        except np.linalg.LinAlgError:
            # A later small factorization must not discard an earlier valid Q.
            break
        previous = new - block @ (block.T @ new)
        block = new
    if best_values is None:
        raise RuntimeError("Non-finite block spectrum during retry")
    return best_values, best_block, iteration




def solve(graph, lap, cfg, rng, initial=None, workspace=None):
    workspace = {} if workspace is None else workspace
    n, count = graph.shape[0], cfg.clusters + 1
    # Accuracy is diagnostic after the bounded solves; it is not an early-stop
    # condition for the 300 outer iterations. Invalid numeric state still fails.
    tolerance = min(1e-8, 0.01 * cfg.eigen_zero_tolerance)
    solve_tolerance = min(cfg.eigen_solver_tolerance, 0.1 * tolerance)
    retries = iterations = warning_count = 0
    backend_failure = ""
    components = -1  # Not computed in the small dense branch.
    if n <= max(cfg.gpu_dense_eigen_limit, 5 * count):
        values, vectors = cp.linalg.eigh(lap.toarray())
        values, vectors = values[:count], vectors[:, :count]
        mode = "dense_gpu"
        if float(cp.abs(values[-1])) <= tolerance:
            mask = graph.get().tocsr(copy=True)
            mask.data = (mask.data > 0).astype(np.float64)
            mask.eliminate_zeros()
            components, labels = connected_components(mask, directed=False)
            if components >= count:
                vectors = _component_vectors(cp.asarray(labels), cp.asarray(np.bincount(labels), dtype=cp.float64),
                    count, rng, workspace.get("basis", initial))
                values = cp.zeros(count, dtype=cp.float64)
                mode = "exact_component_nullspace"
        workspace["basis"] = vectors
    else:
        # Only exact positive edges determine components. Never threshold small
        # positive weights or alter C just to obtain a desired rank.
        matrix = graph.tocsr()
        if "host_indices" not in workspace:
            workspace["host_indices"] = cp.asnumpy(matrix.indices)
            workspace["host_indptr"] = cp.asnumpy(matrix.indptr)
        mask = csr_matrix((cp.asnumpy(matrix.data > 0),
            workspace["host_indices"], workspace["host_indptr"]), shape=(n, n), copy=True)
        # eliminate_zeros mutates CSR pointers; never let it modify the cached
        # fixed-support structure used by the next epoch.
        mask.eliminate_zeros()
        components, labels = connected_components(mask, directed=False)
        labels_gpu = cp.asarray(labels)
        sizes = cp.asarray(np.bincount(labels), dtype=cp.float64)
        zero_count = min(components, count)
        null = (labels_gpu[:, None] == cp.arange(zero_count)[None, :]).astype(cp.float64)
        null /= cp.sqrt(sizes[:zero_count])[None, :]
        if components >= count:
            # Selecting just the first K component indicators leaves all other
            # components with zero embeddings. Span the nullspace across ALL
            # components, retaining the constant vector and warm directions.
            values = cp.zeros(count, dtype=cp.float64)
            vectors = _component_vectors(labels_gpu, sizes, count, rng, workspace.get("basis", initial))
            mode = "exact_component_nullspace"
            workspace["basis"] = vectors
        else:
            needed = count - components
            width = min(needed + 4, (n - components) // 5)
            cached = workspace.get("basis", initial)
            candidates = cp.asarray(rng.standard_normal((n, width)))
            # Preserve the useful warm subspace before appending random probes.
            warm = _orthogonalize(cached, null) if cached is not None else candidates[:, :0]
            random = _orthogonalize(candidates, cp.concatenate((null, warm), axis=1))
            block = cp.concatenate((warm, random), axis=1)
            positive, block = _ritz(lap, block, min(width, block.shape[1]))
            if block.shape[1] < needed:
                raise RuntimeError("Insufficient independent spectral starting directions")
            best = block
            baseline_values = positive
            baseline_error, baseline_orth = _errors(lap, positive[:needed], best[:, :needed])
            mode = "deflated_lobpcg"
            try:
                with warnings.catch_warnings(record=True) as caught:
                    _, candidate, history = _guarded_lobpcg(lap, block.copy(), Y=null,
                        largest=False, tol=solve_tolerance, maxiter=cfg.gpu_eigen_maxiter,
                        retResidualNormsHistory=True)
                warning_count = len(caught)
                iterations = len(history)
                candidate = _orthogonalize(candidate, null)
                if candidate.shape[1] >= needed:
                    candidate_values, candidate = _ritz(lap, candidate, candidate.shape[1])
                    candidate_error, candidate_orth = _errors(lap, candidate_values[:needed], candidate[:, :needed])
                    if (np.isfinite(candidate_error + candidate_orth) and candidate_orth <= 1e-7
                            and (baseline_orth > 1e-7 or candidate_error <= baseline_error)):
                        positive, best = candidate_values, candidate
            except (np.linalg.LinAlgError, ValueError) as exc:
                # CuPy 13 reports some singular Gram factorizations as ValueError.
                # The bounded independent block solver still has to pass the
                # full residual/orthogonality guard. CUDA runtime/OOM propagate.
                backend_failure = str(exc)
                positive, best = baseline_values, block
            residual, orth = _errors(lap, positive[:needed], best[:, :needed])
            if backend_failure or not np.isfinite(residual + orth) or residual > tolerance or orth > 1e-7:
                retries = 1
                mode = "deflated_block_ritz"
                try:
                    retry_values, retry_block, extra = _block_retry(lap, best, null, needed, cfg, tolerance)
                    retry_error, retry_orth = _errors(lap, retry_values[:needed], retry_block[:, :needed])
                    if (np.isfinite(retry_error + retry_orth) and retry_orth <= 1e-7
                            and (not np.isfinite(residual + orth) or orth > 1e-7 or retry_error <= residual)):
                        positive, best = retry_values, retry_block
                    iterations += extra
                except np.linalg.LinAlgError as exc:
                    backend_failure = (backend_failure + "; " + str(exc)).strip("; ")
            workspace["basis"] = cp.concatenate((null, best), axis=1)
            values = cp.concatenate((cp.zeros(components), positive[:needed]))
            vectors = cp.concatenate((null, best[:, :needed]), axis=1)
    residual, orth = _errors(lap, values, vectors)
    workspace["diagnostics"] = {"mode": mode, "residual": residual,
        "orthogonality": orth, "exact_components": int(components),
        "iterations": iterations, "retries": retries, "warnings": warning_count}
    workspace["diagnostics"].update({"accuracy_satisfied": residual <= tolerance,
        "accuracy_target": tolerance})
    workspace["diagnostics"]["backend_failure"] = backend_failure
    if (not np.isfinite(residual + orth) or orth > 1e-7
            or not bool(cp.isfinite(values).all()) or float(values.min()) < -tolerance):
        workspace["failure"] = {"values": values, "vectors": vectors,
            "residual": residual, "orthogonality": orth}
        raise RuntimeError(f"Invalid GPU spectrum: mode={mode}, residual={residual:.3e}, "
                           f"orthogonality={orth:.3e}, residual_target={tolerance:.3e}, "
                           "orthogonality_limit=1.000e-07")
    workspace.pop("failure", None)
    if residual > tolerance:
        print(f"GPU spectrum warning: residual={residual:.3e}, target={tolerance:.3e}; "
              "using best finite orthogonal iterate, continuing training", flush=True)
    return vectors[:, :cfg.clusters], values
