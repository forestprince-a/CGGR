"""Explicit HDF5 adapters and paired, label-free count preprocessing."""
import json
from time import perf_counter
from pathlib import Path

import h5py
import numpy as np
from scipy import sparse
from sklearn.decomposition import PCA, TruncatedSVD


def validate_view_spec(spec):
    """Reject preprocessing overrides outside Supplementary S1.1."""
    kind = spec["kind"]
    if kind not in ("rna", "adt", "atac", "gene_activity"):
        raise ValueError(f"Unknown modality {kind}")
    expected = {"min_cells_per_feature": 3 if kind == "rna" else 1,
                "min_features_per_cell": 1, "components": 50}
    if kind in ("rna", "gene_activity"):
        expected["hvg"] = 2000
    for key, value in expected.items():
        if key in spec and spec[key] != value:
            raise ValueError(f"The Method fixes {kind}.{key}={value}")
    if kind == "atac" and spec.get("drop_first_lsi", "auto") != "auto":
        raise ValueError("The Method selects LSI-1 using absolute depth correlation >= 0.7")


def read_matrix(node):
    if isinstance(node, h5py.Dataset):
        if node.ndim != 2:
            raise ValueError(f"{node.name} is not a matrix")
        blocks = [sparse.csr_matrix(node[a:a + 256]) for a in range(0, node.shape[0], 256)]
        return sparse.vstack(blocks, format="csr")
    shape = tuple(int(x) for x in (node["shape"][()] if "shape" in node else node.attrs["shape"]))
    kind = node.attrs.get("encoding-type", node.attrs.get("format", "csr"))
    if isinstance(kind, bytes):
        kind = kind.decode()
    if kind not in ("csr", "csr_matrix", "csc", "csc_matrix"):
        raise ValueError(f"Unsupported sparse encoding {kind}")
    constructor = sparse.csc_matrix if str(kind).startswith("csc") else sparse.csr_matrix
    return constructor((node["data"][()], node["indices"][()], node["indptr"][()]), shape=shape).tocsr()


def text_vector(node):
    values = node[()].reshape(-1)
    return np.array([x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in values])


def load_views(spec):
    """Never reads the label key. Dataset orientations are mandatory."""
    if spec.get("status", "ready") != "ready":
        raise ValueError(spec.get("reason", "Dataset needs semantic configuration"))
    views = []
    with h5py.File(spec["path"], "r") as handle:
        for view in spec["views"]:
            validate_view_spec(view)
            x = read_matrix(handle[view["key"]])
            axis = view["cells_axis"]
            if axis not in (0, 1):
                raise ValueError("cells_axis must be explicitly 0 or 1")
            if axis == 1:
                x = x.T.tocsr()
            x = x.astype(np.float64)
            x.sum_duplicates()
            x.eliminate_zeros()
            if not np.isfinite(x.data).all() or np.any(x.data < 0):
                raise ValueError(f"{view['key']} must contain finite nonnegative counts/scores")
            views.append(x)
        n = views[0].shape[0]
        if len(views) not in (2, 3) or any(x.shape[0] != n for x in views):
            raise ValueError("This model requires paired modalities with the same cell count")
        id_keys = spec.get("cell_id_keys", [])
        if id_keys:
            columns = [text_vector(handle[key]) for key in id_keys]
            if any(len(col) != n for col in columns):
                raise ValueError("Cell ID length differs from cell count")
            ids = np.array([json.dumps(list(row), ensure_ascii=False) for row in zip(*columns)])
            if len(np.unique(ids)) != n:
                raise ValueError("Cell IDs are not unique; specify batch-qualified identifiers")
        else:
            ids = np.array([f"row:{i}" for i in range(n)])
    return views, ids


def scale_embedding(x):
    x = np.asarray(x, dtype=np.float64)
    sd = x.std(axis=0)
    x = (x - x.mean(axis=0)) / np.maximum(sd, 1e-12)
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    if not np.isfinite(x).all() or np.any(norm < 1e-12):
        raise ValueError("Preprocessing produced non-finite or zero embedding rows")
    return x / norm


def paired_qc(views, view_specs):
    """Label-blind minimal QC; same row mask is applied to every modality.

    Fixed defaults remove empty cells and features present in fewer than
    three cells (RNA) or one cell (ADT, ATAC, gene activity).
    Iterate masks to a fixed point so feature removal cannot leave zero cells.
    """
    if len(views) != len(view_specs) or len(views) not in (2, 3):
        raise ValueError("Provide one specification per paired modality")
    for spec in view_specs:
        validate_view_spec(spec)
    n = views[0].shape[0]
    if any(x.shape[0] != n for x in views):
        raise ValueError("Modalities must have the same paired cell count")
    keep = np.ones(n, dtype=bool)
    feature_masks = [np.ones(x.shape[1], dtype=bool) for x in views]
    while True:
        old_keep = keep.copy()
        old_features = [mask.copy() for mask in feature_masks]
        for i, (x, spec) in enumerate(zip(views, view_specs)):
            minimum = 3 if spec["kind"] == "rna" else 1
            counts = np.asarray((x[keep] > 0).sum(axis=0)).ravel()
            feature_masks[i] &= counts >= minimum
            if not feature_masks[i].any():
                raise ValueError(f"QC removed all features from {spec['key']}")
            detected = np.asarray((x[:, feature_masks[i]] > 0).sum(axis=1)).ravel()
            keep &= detected >= 1
        if keep.sum() < 4:
            raise ValueError("Insufficient cells after paired QC")
        if np.array_equal(keep, old_keep) and all(np.array_equal(a, b) for a, b in zip(feature_masks, old_features)):
            break
    return [x[keep][:, mask].tocsr() for x, mask in zip(views, feature_masks)], keep, feature_masks


def preprocess_rna(x, spec, seed):
    validate_view_spec(spec)
    import anndata as ad
    import scanpy as sc

    data = ad.AnnData(x.copy())
    sc.pp.normalize_total(data, target_sum=1e4)
    sc.pp.log1p(data)
    top = min(2000, x.shape[1])
    if top < x.shape[1]:
        sc.pp.highly_variable_genes(data, flavor="seurat", n_top_genes=top, n_bins=20)
        selected = data.var["highly_variable"].to_numpy()
        if selected.sum() > top:
            # Dispersion ties must not exceed the Method's 2,000-gene cap.
            indices = np.flatnonzero(selected)
            scores = np.nan_to_num(data.var["dispersions_norm"].to_numpy()[indices], nan=-np.inf)
            chosen = indices[np.argsort(-scores, kind="stable")[:top]]
            selected = np.zeros(x.shape[1], dtype=bool)
            selected[chosen] = True
    else:
        selected = np.ones(x.shape[1], dtype=bool)
    reduced = data.X[:, selected].toarray()
    reduced = (reduced - reduced.mean(axis=0)) / np.maximum(reduced.std(axis=0), 1e-12)
    dimensions = min(50, reduced.shape[0] - 1, reduced.shape[1])
    if dimensions < 1:
        raise ValueError("No RNA dimensions remain after HVG selection")
    z = PCA(n_components=dimensions, svd_solver="randomized", random_state=seed).fit_transform(reduced)
    return scale_embedding(z), {"hvg_indices_after_feature_qc": np.flatnonzero(selected).tolist(),
                               "hvg_flavor": "seurat", "components": dimensions}


def preprocess_adt(x, spec, seed):
    validate_view_spec(spec)
    # Explicit CLR: log(x+1) minus its within-cell mean across proteins.
    z = np.log1p(x.toarray())
    z -= z.mean(axis=1, keepdims=True)
    components = 50
    if components is not None and z.shape[1] > components:
        d = min(components, z.shape[0] - 1, z.shape[1])
        z = PCA(n_components=d, svd_solver="randomized", random_state=seed).fit_transform(z)
    return scale_embedding(z), {"clr": "log(count+1), center across proteins within each cell",
                               "components": z.shape[1]}


def preprocess_atac(x, spec, seed):
    validate_view_spec(spec)
    depth = np.asarray(x.sum(axis=1)).ravel()
    df = np.asarray((x > 0).sum(axis=0)).ravel()
    tfidf = sparse.diags(1.0 / depth) @ x @ sparse.diags(np.log1p(x.shape[0] / np.maximum(df, 1)))
    requested = 50
    dimensions = min(requested + 1, x.shape[0] - 1, x.shape[1] - 1)
    if dimensions < 2:
        raise ValueError("ATAC input too small for LSI")
    z = TruncatedSVD(n_components=dimensions, random_state=seed).fit_transform(tfidf)
    log_depth = np.log1p(depth)
    corr = 0.0 if min(np.std(z[:, 0]), np.std(log_depth)) < 1e-12 else float(np.corrcoef(z[:, 0], log_depth)[0, 1])
    drop = abs(corr) >= 0.7
    z = z[:, 1:requested + 1] if drop else z[:, :requested]
    return scale_embedding(z), {"lsi1_log_depth_correlation": corr, "drop_first_lsi": bool(drop),
                               "components": z.shape[1], "depth_correlation_threshold": 0.7}


def prepare_dataset(spec, output, seed=0):
    started = perf_counter()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    views, ids = load_views(spec)
    filtered, mask, features = paired_qc(views, spec["views"])
    arrays, notes = {}, []
    for i, (x, view_spec) in enumerate(zip(filtered, spec["views"])):
        kind = view_spec["kind"]
        function = {"rna": preprocess_rna, "adt": preprocess_adt, "atac": preprocess_atac,
                    "gene_activity": preprocess_rna}.get(kind)
        if function is None:
            raise ValueError(f"Unknown or unconfirmed modality {kind}")
        z, note = function(x, view_spec, seed)
        arrays[f"Z{i}"] = z
        np.save(output / f"feature_indices_view{i}.npy", np.flatnonzero(features[i]))
        notes.append({"kind": kind, "raw_shape": list(views[i].shape), "qc_shape": list(x.shape),
                      "embedding_shape": list(z.shape), **note})
    arrays["cell_ids"] = ids[mask]
    arrays["row_indices"] = np.flatnonzero(mask)
    np.savez_compressed(output / "embeddings.npz", **arrays)
    stat = Path(spec["path"]).stat()
    manifest = {"dataset": spec, "seed": seed, "retained_cells": int(mask.sum()),
                "removed_cells": int((~mask).sum()), "source_size": stat.st_size,
                "source_mtime_ns": stat.st_mtime_ns, "modalities": notes,
                "alignment": "shared identifiers" if spec.get("cell_id_keys") else "assumed original paired file row order",
                "labels_used_in_preprocessing": False,
                "preprocessing_seconds": perf_counter() - started}
    (output / "preprocessing.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return output / "embeddings.npz"
