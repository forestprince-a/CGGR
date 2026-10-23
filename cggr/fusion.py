import numpy as np
from .backend import array_module

from .graph import laplacian


def polar(matrix):
    xp = array_module(matrix)
    u, _, vt = xp.linalg.svd(matrix, full_matrices=False)
    return u @ vt


def qr_retraction(matrix):
    xp = array_module(matrix)
    q, r = xp.linalg.qr(matrix, mode="reduced")
    signs = xp.sign(xp.diag(r))
    signs[signs == 0] = 1
    return q * signs[None, :]


def fusion_errors(hs, rotations, f):
    xp = array_module(f)
    return xp.stack([xp.sum((h @ r - f) ** 2) / f.shape[1]
                     for h, r in zip(hs, rotations)])


def update_weights(errors, floor_mass=0.1):
    """Exact active-set solution of min sum e_v*w_v^2, w_v>=0.1/V.

    No epsilon is added to the objective. Exact-zero errors are handled
    explicitly; inverse-error weights are NOT just clipped and renormalized.
    """
    xp = array_module(errors)
    errors = xp.asarray(errors, dtype=xp.float64)
    if errors.ndim != 1 or not xp.isfinite(errors).all() or xp.any(errors < 0):
        raise ValueError("Invalid fusion errors")
    v = len(errors)
    floor = floor_mass / v
    weights = xp.full(v, floor)
    zero = errors == 0
    if zero.any():
        weights[zero] += (1 - floor_mass) / zero.sum()
        return weights
    free = xp.ones(v, dtype=bool)
    while free.any():
        mass = 1 - floor * (~free).sum()
        inverse = errors[free].min() / errors[free]
        candidate = mass * inverse / inverse.sum()
        below = candidate < floor
        if not below.any():
            weights[free] = candidate
            return weights
        free[xp.flatnonzero(free)[below]] = False
    raise RuntimeError("Weight active-set solver exhausted the feasible set")


def initialize_fusion(hs):
    xp = array_module(hs[0])
    rotations = [xp.eye(hs[0].shape[1])]
    rotations.extend(polar(h.T @ hs[0]) for h in hs[1:])
    weights = xp.full(len(hs), 1 / len(hs))
    f = polar(sum(w * w * h @ r for w, h, r in zip(weights, hs, rotations)))
    return rotations, weights, f


def update_h(h, s, r, f, weight, views, cfg):
    xp = array_module(h)
    if xp is np:
        lap = laplacian(s)
    else:
        from .gpu_graph import laplacian as gpu_laplacian
        lap = gpu_laplacian(s)
    a = cfg.alpha / (views * cfg.clusters)
    b = cfg.beta * weight ** 2 / cfg.clusters

    def objective(x):
        return float(a * xp.sum(x * (lap @ x)) + b * xp.sum((x @ r - f) ** 2))

    gradient = 2 * a * (lap @ h) + 2 * b * ((h @ r - f) @ r.T)
    inner = h.T @ gradient
    tangent = gradient - h @ (0.5 * (inner + inner.T))
    norm_sq = float(xp.sum(tangent ** 2))
    if norm_sq <= 1e-24:
        return h, 0.0, False
    current, step = objective(h), cfg.h_initial_step
    if cfg.h_adaptive_step:
        # For a nonnegative symmetric graph, ||L|| <= 2*max(row degree).
        # A scale-aware trial step avoids 300 almost-zero updates from 1e-3.
        # Armijo still checks the SAME objective after the Stiefel retraction.
        degree_max = float(xp.asarray(s.sum(axis=1)).max())
        step = max(step, 1.0 / (4 * a * degree_max + 2 * b))
    for _ in range(cfg.max_backtracks):
        trial = qr_retraction(h - step * tangent)
        value = objective(trial)
        if xp.isfinite(value) and value <= current - cfg.armijo * step * norm_sq:
            return trial, step, False
        step *= cfg.backtracking
    # A failed line search does not change the graph or end the 300 epochs.
    return h, 0.0, True
