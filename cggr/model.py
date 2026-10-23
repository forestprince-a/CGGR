"""Joint continuous model. Labels appear only in the final K-means call."""
from dataclasses import dataclass, field
from time import perf_counter

import numpy as np
from sklearn.cluster import KMeans

from .config import ModelConfig
from .backend import select_device, synchronize, device_info, to_host
from . import graph as cpu_graph
from .graph import ADMMState
from .fusion import fusion_errors, initialize_fusion, polar, update_h, update_weights


@dataclass
class ViewState:
    support: object
    s: object
    c: object
    q: np.ndarray
    h: np.ndarray
    sigma: np.ndarray
    z_distance: np.ndarray
    s_admm: ADMMState
    c_admm: ADMMState
    xi: float
    spectrum: np.ndarray
    spectral_workspace: dict = field(default_factory=dict)


class CGGR:
    def __init__(self, config: ModelConfig):
        self.config = config

    def _losses(self):
        xp = self.xp_
        laplacian = self.graph_.laplacian
        cfg, v = self.config, len(self.states_)
        n, k = self.states_[0].support.n, cfg.clusters
        raw_data = raw_regression = raw_consistency = bridge = 0.0
        for state in self.states_:
            raw_data += float(state.support.sample(state.s) @ state.z_distance) / (2 * n * v)
            if hasattr(self.graph_, "regression_energy"):
                raw_regression += self.graph_.regression_energy(state.s, state.c, state.support) / (n * v)
            else:
                raw_regression += float(state.s.multiply(laplacian(state.c) @ state.s).sum()) / (n * v)
            difference = state.s - state.c
            raw_consistency += float(difference.multiply(difference).sum()) / (n * v)
            bridge += float(xp.sum(state.h * (laplacian(state.s) @ state.h))) / (k * v)
        fusion = float(self.weights_ ** 2 @ fusion_errors(
            [s.h for s in self.states_], self.rotations_, self.consensus_))
        graph = raw_data + raw_regression + raw_consistency
        return {"geometry": raw_data, "regression": raw_regression,
                "consistency": raw_consistency, "graph": graph,
                "bridge": bridge, "fusion": fusion,
                "weighted_bridge": cfg.alpha * bridge,
                "weighted_fusion": cfg.beta * fusion,
                "total": graph + cfg.alpha * bridge + cfg.beta * fusion}

    def fit(self, views, callback=None):
        """views are preprocessed, paired N-by-p arrays; no y parameter.

        callback(epoch, diagnostics, model) runs after every complete epoch.
        It may persist state; it must not make label-dependent decisions.
        """
        fit_started = perf_counter()
        self.active_stage_ = "input_validation"
        self.timings_ = {}
        if len(views) not in (2, 3):
            raise ValueError("Two or three modalities required")
        self.xp_ = xp = select_device(self.config.device)
        self.backend_info_ = device_info(xp)
        if xp is np:
            self.graph_ = cpu_graph
        else:
            from . import gpu_graph
            self.graph_ = gpu_graph
        graph_ops = self.graph_
        arrays = [xp.asarray(x, dtype=xp.float64) for x in views]
        print(f"TRAIN backend={self.backend_info_['backend']} "
              f"device={self.backend_info_['device']} dtype=float64", flush=True)
        if any(x.ndim != 2 for x in arrays):
            raise ValueError("All embeddings must be matrices")
        n = len(arrays[0])
        self.config.validate(n, len(arrays))
        if any(len(x) != n or x.shape[1] == 0 or not xp.isfinite(x).all() for x in arrays):
            raise ValueError("Embeddings must be finite and have the same cell order and count")
        if any(not bool(xp.allclose(xp.linalg.norm(x, axis=1), 1.0,
                                   rtol=0, atol=1e-8)) for x in arrays):
            raise ValueError("Each preprocessed modality must have unit row norms")
        cfg, rng = self.config, np.random.default_rng(self.config.random_state)
        self.states_, self.history_ = [], []
        self.epochs_completed_ = 0
        self.spectrum_warning_calls_ = 0
        # Clear fitted outputs if an existing estimator is reused.
        for attr in ("labels_", "cluster_embedding_"):
            if hasattr(self, attr):
                delattr(self, attr)
        for view_index, z in enumerate(arrays):
            self.active_stage_ = f"initialization/view{view_index}"
            support, s, sigma = graph_ops.local_scale_initialization(z, cfg)
            distance = graph_ops.edge_distances(z, support)
            c = s.copy()
            spectral_workspace = {}
            h, spectrum = graph_ops.bottom_spectrum(s, cfg, rng, workspace=spectral_workspace)
            self.spectrum_warning_calls_ += int(not spectral_workspace.get(
                "diagnostics", {}).get("accuracy_satisfied", False))
            self.states_.append(ViewState(support, s, c, h.copy(), h, sigma,
                distance, graph_ops.ADMMState.initialize(s, support, cfg),
                graph_ops.ADMMState.initialize(c, support, cfg), cfg.rank_penalty_initial,
                spectrum, spectral_workspace))
        self.rotations_, self.weights_, self.consensus_ = initialize_fusion([s.h for s in self.states_])
        synchronize(xp)
        self.timings_["initialization_seconds"] = perf_counter() - fit_started
        optimization_started = perf_counter()
        for epoch in range(1, cfg.epochs + 1):
            epoch_started = perf_counter()
            diagnostics = {"epoch": epoch}
            for v, state in enumerate(self.states_):
                self.active_stage_ = f"epoch{epoch}/view{v}/S"
                stage_started = perf_counter()
                state.s, ds = graph_ops.update_s(state.z_distance, state.h, state.c, state.s,
                                       state.support, state.s_admm, cfg)
                synchronize(xp)
                diagnostics[f"v{v}_s_seconds"] = perf_counter() - stage_started
                stage_started = perf_counter()
                self.active_stage_ = f"epoch{epoch}/view{v}/C"
                state.c, dc = graph_ops.update_c(state.s, state.q, state.xi, state.c,
                                       state.support, state.c_admm, cfg)
                synchronize(xp)
                diagnostics[f"v{v}_c_seconds"] = perf_counter() - stage_started
                stage_started = perf_counter()
                self.active_stage_ = f"epoch{epoch}/view{v}/spectrum"
                state.q, state.spectrum = graph_ops.bottom_spectrum(state.c, cfg, rng,
                    initial=state.q, workspace=state.spectral_workspace)
                synchronize(xp)
                diagnostics[f"v{v}_spectrum_seconds"] = perf_counter() - stage_started
                diagnostics.update({f"v{v}_spectrum_{key}": value
                    for key, value in state.spectral_workspace.get("diagnostics", {}).items()})
                spectrum_accurate = state.spectral_workspace.get(
                    "diagnostics", {}).get("accuracy_satisfied", False)
                self.spectrum_warning_calls_ += int(not spectrum_accurate)
                # Continue the epoch with the finite orthogonal approximation.
                # Do not change xi based on an unverified near-zero spectrum.
                status = graph_ops.rank_state(state.spectrum, cfg) if spectrum_accurate else "unresolved"
                if status == "too_few":
                    state.xi = min(state.xi * 2, cfg.rank_penalty_max)
                elif status == "too_many":
                    state.xi = max(state.xi / 2, cfg.rank_penalty_min)
                diagnostics[f"v{v}_xi_next"] = state.xi
                for name, graph, solver in (("s", state.s, ds), ("c", state.c, dc)):
                    errors = graph_ops.graph_residuals(graph)
                    if max(errors["row_sum"], errors["symmetry"], errors["diagonal"],
                           -errors["minimum"]) > cfg.projection_tolerance:
                        raise RuntimeError(f"Invalid feasible graph at epoch {epoch}, modality {v}, {name}")
                    diagnostics.update({f"v{v}_{name}_{key}": value for key, value in (errors | solver).items()})
            stage_started = perf_counter()
            self.active_stage_ = f"epoch{epoch}/fusion"
            for v, state in enumerate(self.states_):
                state.h, step, stalled = update_h(state.h, state.s, self.rotations_[v],
                    self.consensus_, self.weights_[v], len(arrays), cfg)
                diagnostics[f"v{v}_h_step"] = step
                diagnostics[f"v{v}_h_stalled"] = stalled
            self.rotations_ = [polar(s.h.T @ self.consensus_) for s in self.states_]
            self.consensus_ = polar(sum(w ** 2 * s.h @ r for w, s, r in
                                        zip(self.weights_, self.states_, self.rotations_)))
            self.weights_ = update_weights(fusion_errors([s.h for s in self.states_],
                                            self.rotations_, self.consensus_), cfg.weight_floor_mass)
            synchronize(xp)
            diagnostics["fusion_seconds"] = perf_counter() - stage_started
            stage_started = perf_counter()
            ortho = [xp.linalg.norm(self.consensus_.T @ self.consensus_ - xp.eye(cfg.clusters))]
            ortho += [xp.linalg.norm(s.h.T @ s.h - xp.eye(cfg.clusters)) for s in self.states_]
            ortho += [xp.linalg.norm(r.T @ r - xp.eye(cfg.clusters)) for r in self.rotations_]
            ortho += [xp.linalg.norm(s.q.T @ s.q - xp.eye(cfg.clusters)) for s in self.states_]
            diagnostics["orthogonality_max"] = float(max(ortho))
            diagnostics["weight_sum_error"] = float(abs(self.weights_.sum() - 1))
            if max(ortho) > 1e-6 or abs(self.weights_.sum() - 1) > 1e-8:
                raise RuntimeError("Fusion constraints violated")
            if self.weights_.min() < cfg.weight_floor_mass / len(arrays) - 1e-10:
                raise RuntimeError("Weight lower bound violated")
            diagnostics.update({f"weight_{v}": float(w) for v, w in enumerate(self.weights_)})
            diagnostics.update(self._losses())
            numeric = [value for value in diagnostics.values() if isinstance(value, (float, int))]
            if not np.isfinite(numeric).all():
                raise RuntimeError("Non-finite training diagnostics")
            synchronize(xp)
            diagnostics["diagnostics_seconds"] = perf_counter() - stage_started
            diagnostics["epoch_seconds"] = perf_counter() - epoch_started
            diagnostics["fit_elapsed_seconds"] = perf_counter() - fit_started
            self.history_.append(diagnostics)
            self.epochs_completed_ = epoch
            if callback is not None:
                callback(epoch, diagnostics, self)
        self.timings_["optimization_seconds"] = perf_counter() - optimization_started
        kmeans_started = perf_counter()
        self.active_stage_ = "final_kmeans"
        norms = xp.linalg.norm(self.consensus_, axis=1, keepdims=True)
        if xp.any(norms < 1e-12):
            raise RuntimeError("Final consensus contains zero rows")
        self.cluster_embedding_ = self.consensus_ / norms
        # One device-to-host boundary preserves the existing sklearn K-means
        # protocol and exported NumPy labels; the 300-epoch loop stays on GPU.
        embedding_host = to_host(self.cluster_embedding_)
        self.labels_ = KMeans(n_clusters=cfg.clusters, init="k-means++", n_init=cfg.kmeans_n_init,
                             random_state=cfg.random_state).fit_predict(embedding_host) + 1
        if np.unique(self.labels_).size != cfg.clusters:
            raise RuntimeError("Final K-means did not produce K nonempty clusters")
        self.timings_["kmeans_seconds"] = perf_counter() - kmeans_started
        self.timings_["runtime_seconds"] = perf_counter() - fit_started
        return self
