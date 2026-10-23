"""Warm dual Newton projection onto the SAME fixed-support graph polytope.

For symmetric a, w_ij = max(a_ij - u_i - u_j, 0). The dual gradient
is 1 - sum_j w_ij; its generalized Hessian is the active signless
Laplacian. No constraint or objective penalty is relaxed here.
"""
import numpy as np

from .backend import array_module
from .projection import make_row_groups


def project(support, values, cfg, state=None):
    xp = array_module(values)
    if xp is np:
        from scipy.sparse.linalg import LinearOperator, cg
    else:
        from cupyx.scipy.sparse.linalg import LinearOperator, cg
    rows, cols, n = support.rows, support.cols, support.n
    a = 0.5 * (values + values[support.reverse])
    if not bool(xp.isfinite(a).all()):
        raise RuntimeError("Non-finite input to graph projection")
    warm = getattr(state, "projection_dual", None)
    u = xp.zeros(n, dtype=xp.float64) if warm is None else warm.copy()
    groups = getattr(support, "_groups", None)
    if groups is None:
        support._groups = groups = make_row_groups(support.indptr, xp)

    def row_sum(w):
        return xp.bincount(rows, weights=w, minlength=n)

    def coordinate_direction():
        # Exact one-coordinate dual minimizers. Half a simultaneous Jacobi
        # step is a descent step by convexity (each edge has two endpoints).
        # Used only for cold starts / a degenerate Newton active set.
        threshold = xp.empty(n, dtype=xp.float64)
        shifted = a - u[cols]
        for indices, valid, _, _, ranks in groups:
            ordered = xp.sort(xp.where(valid, shifted[indices], -xp.inf), axis=1)[:, ::-1]
            sums = xp.cumsum(xp.where(xp.isfinite(ordered), ordered, 0.0), axis=1) - 1.0
            rho = xp.sum(ordered - sums / ranks > 0, axis=1) - 1
            threshold[rows[indices[:, 0]]] = sums[xp.arange(len(indices)), rho] / (rho + 1)
        return 0.5 * (threshold - u)

    # A sorted coordinate sweep jumps across large inactive gaps in C when
    # xi/mu is large. Pure gradient descent would cross those gaps very slowly.
    if warm is None:
        u += coordinate_direction()
    cg_total = coordinate_steps = 0
    error = float("inf")
    for iteration in range(cfg.projection_maxiter + 1):
        raw = a - (u[rows] + u[cols])
        w = xp.maximum(raw, 0.0)
        gradient = 1.0 - row_sum(w)
        error = float(xp.max(xp.abs(gradient)))
        if error <= cfg.projection_tolerance:
            support.last_projection_iterations = iteration + 1
            support.last_projection_error = error
            support.last_projection_cg_iterations = cg_total
            support.last_projection_coordinate_steps = coordinate_steps
            if state is not None:
                state.projection_dual = u
            if hasattr(support, "last_projection_failure"):
                del support.last_projection_failure
            return w
        if not np.isfinite(error) or iteration == cfg.projection_maxiter:
            break
        active = (raw > 0).astype(xp.float64)
        degree = row_sum(active)
        if bool((degree == 0).any()):
            u += coordinate_direction()
            coordinate_steps += 1
            continue
        # Regularization is ONLY in the Newton search direction. The final
        # solution is accepted using the original, unregularized KKT equations.
        damping = max(1e-10, min(1e-2, error * 1e-2))
        diagonal = degree + damping
        operator = LinearOperator((n, n), dtype=xp.float64,
            matvec=lambda x: diagonal * x + row_sum(active * x[cols]))
        preconditioner = LinearOperator((n, n), dtype=xp.float64,
            matvec=lambda x: x / diagonal)
        def count(_):
            nonlocal cg_total
            cg_total += 1
        tolerance = min(0.1, max(1e-6, 0.1 * error))
        kwargs = {"rtol" if xp is np else "tol": tolerance}
        direction, info = cg(operator, -gradient, M=preconditioner,
            atol=0.0, maxiter=cfg.cg_maxiter, callback=count, **kwargs)
        slope = float(gradient @ direction)
        accepted = False
        if info >= 0 and np.isfinite(slope) and slope < 0:
            edge_direction = direction[rows] + direction[cols]
            step = 1.0
            for _ in range(24):
                shift = step * edge_direction
                trial_raw = raw - shift
                # Stable dual difference, avoiding subtraction of enormous
                # objective values at large xi. This is the exact Bregman term.
                remainder = xp.where(raw > 0,
                    xp.where(trial_raw > 0, shift * shift, 2 * raw * shift - raw * raw),
                    xp.maximum(trial_raw, 0.0) ** 2)
                delta = step * slope + 0.25 * float(remainder.sum())
                if np.isfinite(delta) and delta <= 1e-4 * step * slope:
                    u += step * direction
                    accepted = True
                    break
                step *= 0.5
        if not accepted:
            u += coordinate_direction()
            coordinate_steps += 1
    support.last_projection_failure = {"input": values.copy(), "dual": u.copy(),
        "row_error": error, "iterations": iteration + 1}
    raise RuntimeError(f"Fixed-support projection failed: row_error={error:.3e}, "
                       f"iterations={iteration + 1}, cg_iterations={cg_total}")
