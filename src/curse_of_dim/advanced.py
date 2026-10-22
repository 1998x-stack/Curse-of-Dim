"""Advanced geometry experiments: nonlinear manifolds, anisotropy, and rank.

Only synthetic data is used. All PCA transforms are fitted on database/training
samples, never on held-out queries. The figures are saved without a display.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from sklearn.neighbors import NearestNeighbors


def effective_rank(x: np.ndarray) -> dict[str, float]:
    """Spectral effective rank and participation ratio of centered data.

    Eigenvalues are those of the sample covariance. Both definitions depend on
    the measurement units and need not estimate manifold/intrinsic dimension.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 2 or not np.isfinite(x).all():
        raise ValueError("Expected finite matrix with >=2 samples")
    s = np.linalg.svd(x - x.mean(axis=0), compute_uv=False)
    lam = s**2 / (x.shape[0] - 1)
    total = lam.sum()
    if total < 1e-24:
        return {"entropy_rank": 0.0, "participation_ratio": 0.0,
                "top1_share": 0.0}
    p = lam[lam > 0] / total
    return {"entropy_rank": float(np.exp(-(p * np.log(p)).sum())),
            "participation_ratio": float(total**2 / np.sum(lam**2)),
            "top1_share": float(lam[0] / total)}


def knn_overlap(reference: np.ndarray, query: np.ndarray,
                projected_reference: np.ndarray, projected_query: np.ndarray,
                k: int = 10) -> float:
    """Fraction of ground-truth kNN IDs recovered by a candidate representation."""
    if k <= 0 or k > len(reference):
        raise ValueError("k must be between 1 and the database size")
    oracle = NearestNeighbors(n_neighbors=k).fit(reference).kneighbors(query, return_distance=False)
    pred = NearestNeighbors(n_neighbors=k).fit(projected_reference).kneighbors(
        projected_query, return_distance=False)
    return float(np.mean([len(set(a) & set(b)) / k for a, b in zip(oracle, pred)]))


def swiss_roll(n: int, ambient: int = 64, seed: int = 2026) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Swiss-roll points embedded isometrically from R³ to R^ambient.

    oracle=[arc length along spiral, height] gives local/geodesic coordinates.
    Note: not all geodesics on finite Swiss roll follow exact Euclidean oracle.
    """
    if ambient < 3 or n < 10:
        raise ValueError("ambient >=3 and n >=10 required")
    rng = np.random.default_rng(seed)
    t = rng.uniform(1.5 * np.pi, 4.5 * np.pi, n)
    h = rng.uniform(-10.0, 10.0, n)
    x3 = np.column_stack([t * np.cos(t), h, t * np.sin(t)])
    # Arc length of (t cos t, t sin t): ds/dt = sqrt(1+t²).
    arc = .5 * (t * np.sqrt(1 + t**2) + np.arcsinh(t))
    oracle = np.column_stack([arc, h])
    q, _ = np.linalg.qr(rng.normal(size=(ambient, 3)))
    return x3 @ q.T, oracle, t


def embedding_transformations(docs: np.ndarray, queries: np.ndarray,
                              whiten_components: int = 16) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Fit de-biasing transforms on docs ONLY; apply to held-out queries.

    The PC-removal variant removes the dominant centered PC, not the mean.
    Mean-centering changes cosine similarities, but not Euclidean distances.
    Whitening is truncated/ridge-stabilized and is NOT guaranteed to help.
    """
    docs = np.asarray(docs, dtype=float)
    queries = np.asarray(queries, dtype=float)
    if (docs.ndim != 2 or queries.ndim != 2 or docs.shape[1] != queries.shape[1]
            or len(docs) < 3 or not np.isfinite(docs).all() or not np.isfinite(queries).all()):
        raise ValueError("Require finite 2D docs and queries with matching dimensions")
    center = docs.mean(axis=0)
    dc, qc = docs - center, queries - center
    # Full rank PCA once; with deterministic full SVD and no query leakage.
    pca = PCA(n_components=min(docs.shape[0]-1, docs.shape[1]), svd_solver="full")
    pca.fit(dc)
    dominant = pca.components_[0]
    dt = dc - np.outer(dc @ dominant, dominant)
    qt = qc - np.outer(qc @ dominant, dominant)
    width = min(whiten_components, len(pca.components_))
    components = pca.components_[:width]
    scale = np.sqrt(np.maximum(pca.explained_variance_[:width], 1e-5))
    return {
        "raw": (docs, queries),
        "centered": (dc, qc),
        "remove_top_pc": (dt, qt),
        "whiten_top_components": (dc @ components.T / scale, qc @ components.T / scale),
    }


def retrieval_metrics(docs: np.ndarray, queries: np.ndarray,
                      doc_labels: np.ndarray | None = None,
                      query_labels: np.ndarray | None = None,
                      k: int = 5) -> dict[str, float]:
    """Cosine retrieval with hubness and optional label-based precision/hit.

    precision@k is fraction of top-k documents sharing the query label;
    hit@k is fraction of queries finding at least one such document.
    """
    if not (1 <= k <= len(docs)):
        raise ValueError("k must be >=1 and <= number of documents")
    docs = np.asarray(docs, dtype=np.float64)
    queries = np.asarray(queries, dtype=np.float64)
    dn = docs / np.maximum(np.linalg.norm(docs, axis=1, keepdims=True), 1e-12)
    qn = queries / np.maximum(np.linalg.norm(queries, axis=1, keepdims=True), 1e-12)
    similarities = qn @ dn.T
    # kth ordered indices, then sort each candidate list.
    idx = np.argpartition(-similarities, kth=k-1, axis=1)[:, :k]
    order = np.argsort(-np.take_along_axis(similarities, idx, axis=1), axis=1)
    idx = np.take_along_axis(idx, order, axis=1)
    appearances = np.bincount(idx.ravel(), minlength=len(docs))
    sorted_counts = np.sort(appearances)
    gini = float(np.sum((2 * np.arange(1,len(docs)+1)-len(docs)-1) * sorted_counts)
                 / (len(docs) * max(int(appearances.sum()), 1)))
    out = {"hubness_gini": gini, "max_neighbor_occurrences": int(appearances.max())}
    if (doc_labels is None) != (query_labels is None):
        raise ValueError("Both document and query labels must be provided together")
    if doc_labels is not None:
        dl, ql = np.asarray(doc_labels), np.asarray(query_labels)
        if len(dl) != len(docs) or len(ql) != len(queries):
            raise ValueError("Labels must match corresponding row counts")
        matches = dl[idx] == ql[:, None]
        out.update({"precision_at_k": float(matches.mean()),
                    "hit_at_k": float(matches.any(axis=1).mean()),
                    "top1_accuracy": float(matches[:,0].mean())})
    return out


def synthetic_embeddings(n_docs: int = 540, n_queries: int = 180,
                         ambient: int = 96, seed: int = 902,
                         scenario: str = "nuisance") -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Known-label retrieval benchmark. Shared mean + nuisance vs top-PC signal."""
    if ambient < 16:
        raise ValueError("ambient must be at least 16")
    if scenario not in {"nuisance", "signal"}:
        raise ValueError("Unknown scenario")
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(ambient, 9)))
    basis, v = q[:, :8], q[:,8]
    n_classes = 12 if scenario == "nuisance" else 2
    dl = np.arange(n_docs) % n_classes
    ql = np.arange(n_queries) % n_classes
    rng.shuffle(dl)
    rng.shuffle(ql)
    if scenario == "nuisance":
        centers = rng.normal(size=(n_classes, 8)) * 1.5
        def encode(labels):
            z = centers[labels] + rng.normal(size=(len(labels), 8)) * .75
            return (z @ basis.T + rng.normal(size=(len(labels), 1)) * 9 * v
                    + 9 * v + rng.normal(size=(len(labels), ambient)) * .1)
    else:
        def encode(labels):
            # PC-1 carries the class boundary; removing it is destructive.
            semantic = (labels[:, None] * 2 - 1) * 5.0 * v
            return (semantic + rng.normal(size=(len(labels), 8)) @ basis.T * 1.0
                    + rng.normal(size=(len(labels), ambient)) * .12)
    return encode(dl), encode(ql), dl, ql


def _save(fig: plt.Figure, out: Path, name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / name, dpi=145, bbox_inches="tight")
    plt.close(fig)


def experiment_08_nonlinear_manifold(out: str | Path = "outputs") -> dict:
    """Swiss roll: PCA 2D cannot generally unfold a curved intrinsic-2D manifold."""
    x, oracle, color = swiss_roll(780, 64, 808)
    doc, query = np.arange(610), np.arange(610, 780)
    pca = PCA(n_components=2).fit(x[doc])
    emb_pca = pca.transform(x)
    isomap = Isomap(n_neighbors=12, n_components=2).fit(x[doc])
    emb_iso = np.vstack([isomap.embedding_, isomap.transform(x[query])])
    embeds = {"ambient_64d": x, "pca_2d": emb_pca, "isomap_2d": emb_iso}
    scores = {key: knn_overlap(oracle[doc], oracle[query], emb[doc], emb[query], k=10)
              for key, emb in embeds.items()}
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.25), constrained_layout=True)
    for a, (name, emb) in zip(ax[:2], [("PCA", emb_pca), ("Isomap", emb_iso)]):
        a.scatter(emb[doc,0], emb[doc,1], c=color[doc], cmap="viridis", s=7, alpha=.75)
        a.set(title=f"{name}: projected coordinates (color = spiral t)", xlabel="axis 1", ylabel="axis 2")
    ax[2].bar(list(scores), list(scores.values()))
    ax[2].set(title="10-NN overlap with intrinsic-geometry oracle", ylabel="overlap@10", ylim=(0,1))
    ax[2].tick_params(axis="x", rotation=18)
    fig.suptitle("08 | Ambient 64D is a 3D isometry of a curved 2D manifold", fontsize=13)
    _save(fig, Path(out), "08_nonlinear_manifold.png")
    return {"oracle_neighbor_overlap": scores, "ambient_dimension": 64,
            "intrinsic_dimension": 2, "split": {"train": len(doc), "test": len(query)}}


def experiment_09_anisotropic_embeddings(out: str | Path = "outputs") -> dict:
    """Cosine retrieval: nuisance-PC removal can help; signal-PC removal can hurt."""
    metrics = {}
    for scenario in ("nuisance", "signal"):
        d, q, dl, ql = synthetic_embeddings(scenario=scenario)
        metrics[scenario] = {name: retrieval_metrics(dd, qq, dl, ql)
                             for name, (dd, qq) in embedding_transformations(d, q).items()}
        metrics[scenario]["spectral"] = effective_rank(d)
    # Separate distribution panel illustrates that mean direction dominates cosine.
    d, q, _, _ = synthetic_embeddings()
    transforms = embedding_transformations(d, q)
    rng = np.random.default_rng(19)
    pairs = rng.integers(0, len(d), size=(2200,2))
    fig, ax = plt.subplots(2, 2, figsize=(12.4, 8.5), constrained_layout=True)
    for name in ("raw", "centered", "remove_top_pc"):
        a, _ = transforms[name]
        norm = a / np.maximum(np.linalg.norm(a,axis=1,keepdims=True),1e-12)
        cos = np.sum(norm[pairs[:,0]] * norm[pairs[:,1]], axis=1)
        ax[0,0].hist(cos, bins=np.linspace(-1,1,48), density=True, alpha=.45, label=name)
    ax[0,0].set(title="Random document pair cosine", xlabel="cosine", ylabel="density")
    ax[0,0].legend(fontsize=8)
    pc = PCA(n_components=18).fit(d - d.mean(axis=0))
    ax[0,1].semilogy(np.arange(1,19), pc.explained_variance_ratio_, "o-")
    ax[0,1].set(title="Nuisance scenario: centered covariance spectrum", xlabel="PC rank", ylabel="variance share (log)")
    for a, scenario in zip(ax[1], ("nuisance", "signal")):
        m = metrics[scenario]
        names = list(transforms)
        xx = np.arange(len(names))
        a.bar(xx-.18, [m[n]["top1_accuracy"] for n in names], .36, label="top-1 label accuracy")
        a.bar(xx+.18, [m[n]["precision_at_k"] for n in names], .36, label="precision@5")
        a.set(xticks=xx, xticklabels=names, ylim=(0,1.05), title=f"Scenario: {scenario}")
        a.tick_params(axis="x", rotation=15)
        a.legend(fontsize=8)
    fig.suptitle("09 | Anisotropy, nuisance directions, and the top-PC counterexample", fontsize=13)
    _save(fig, Path(out), "09_anisotropy_retrieval.png")
    return metrics


def experiment_10_effective_dimension(out: str | Path = "outputs") -> dict:
    """Effective rank and neighbor stability vary with ambient noise."""
    rng = np.random.default_rng(1001)
    d, rank, n, nq = 128, 6, 480, 140
    z = rng.normal(size=(n+nq,rank))
    basis, _ = np.linalg.qr(rng.normal(size=(d, rank)))
    noise = rng.normal(size=(n+nq,d))
    levels = [0.0, .02, .05, .1, .2, .35, .6]
    metrics = []
    for scale in levels:
        x = z @ basis.T + scale * noise
        pca = PCA(n_components=rank, svd_solver="full").fit(x[:n])
        proj = pca.transform(x)
        row = {"ambient_noise_sd": float(scale), **effective_rank(x[:n]),
               "raw_overlap_at_10": knn_overlap(z[:n], z[n:], x[:n], x[n:], 10),
               "pca_overlap_at_10": knn_overlap(z[:n], z[n:], proj[:n], proj[n:], 10)}
        metrics.append(row)
    fig, ax = plt.subplots(1, 3, figsize=(14.0, 4.25), constrained_layout=True)
    noise_sd = [m["ambient_noise_sd"] for m in metrics]
    for key in ("entropy_rank", "participation_ratio"):
        ax[0].plot(noise_sd, [m[key] for m in metrics], "o-", label=key)
    ax[0].axhline(rank, color="gray", ls="--", label="true latent dimension")
    ax[0].set(title="Covariance effective rank", xlabel="per-coordinate Gaussian noise SD", ylabel="estimated rank")
    ax[0].legend(fontsize=8)
    for key,label in [("raw_overlap_at_10","ambient 128D"),("pca_overlap_at_10","PCA to 6D")]:
        ax[1].plot(noise_sd, [m[key] for m in metrics], "o-", label=label)
    ax[1].set(title="True latent kNN preserved?", xlabel="noise SD", ylabel="neighbor overlap@10", ylim=(0,1.05))
    ax[1].legend(fontsize=8)
    ax[2].plot(noise_sd, [m["top1_share"] for m in metrics], "o-")
    ax[2].set(title="Top covariance eigenvalue share", xlabel="noise SD", ylabel="PC1 variance share")
    fig.suptitle("10 | Same true 6D latent coordinates, growing 128D isotropic noise", fontsize=13)
    _save(fig, Path(out), "10_effective_dimension.png")
    return {"ambient_dimension":d,"true_latent_dimension":rank,"sweep":metrics}


def experiment_11_autoencoder(out: str | Path = "outputs") -> dict:
    """Optional torch bottleneck AE vs PCA and Isomap on nonlinear Swiss roll."""
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise RuntimeError("Install optional dependencies: pip install -e '.[deep]'") from exc
    from sklearn.preprocessing import StandardScaler
    torch.manual_seed(1101)
    torch.set_num_threads(2)
    x, oracle, color = swiss_roll(800, 64, 1101)
    n = 640
    scaler = StandardScaler().fit(x[:n])
    standardized = scaler.transform(x).astype("float32")
    pca = PCA(n_components=2).fit(standardized[:n])
    latent_pca = pca.transform(standardized)
    pca_mse = float(np.mean((pca.inverse_transform(latent_pca[n:]) - standardized[n:])**2))
    iso = Isomap(n_neighbors=12,n_components=2).fit(standardized[:n])
    latent_iso = np.vstack([iso.embedding_, iso.transform(standardized[n:])])

    class BottleneckAE(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.encoder = nn.Sequential(nn.Linear(dim,64),nn.GELU(),nn.Linear(64,32),
                                         nn.GELU(),nn.Linear(32,2))
            self.decoder = nn.Sequential(nn.Linear(2,32),nn.GELU(),nn.Linear(32,64),
                                         nn.GELU(),nn.Linear(64,dim))
        def forward(self, tensor):
            latent = self.encoder(tensor)
            return self.decoder(latent), latent

    model = BottleneckAE(standardized.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=.003, weight_decay=1e-5)
    train = torch.from_numpy(standardized[:n])
    test = torch.from_numpy(standardized[n:])
    history = []
    rng = np.random.default_rng(1101)
    for epoch in range(220):
        model.train()
        for batch in np.array_split(rng.permutation(n), 8):
            pred, _ = model(train[batch])
            loss = nn.functional.mse_loss(pred, train[batch])
            opt.zero_grad()
            loss.backward()
            opt.step()
        if epoch % 5 == 0 or epoch == 219:
            model.eval()
            with torch.no_grad():
                history.append((epoch+1,float(nn.functional.mse_loss(model(test)[0],test))))
    model.eval()
    with torch.no_grad():
        recon, latent = model(torch.from_numpy(standardized))
        latent_ae = latent.numpy()
        ae_mse = float(nn.functional.mse_loss(recon[n:],test))
    overlaps = {
        "PCA": knn_overlap(oracle[:n],oracle[n:],latent_pca[:n],latent_pca[n:]),
        "Isomap": knn_overlap(oracle[:n],oracle[n:],latent_iso[:n],latent_iso[n:]),
        "Autoencoder": knn_overlap(oracle[:n],oracle[n:],latent_ae[:n],latent_ae[n:]),
    }
    fig,ax=plt.subplots(2,2,figsize=(11,9),constrained_layout=True)
    for a,(name,emb) in zip([ax[0,0],ax[0,1],ax[1,0]],
                            [("PCA",latent_pca),("Isomap",latent_iso),("Autoencoder",latent_ae)]):
        a.scatter(emb[n:,0],emb[n:,1],c=color[n:],cmap="viridis",s=11,alpha=.8)
        a.set(title=f"{name} (held-out queries colored by spiral position)",xlabel="latent 1",ylabel="latent 2")
    ax[1,1].plot([e for e,_ in history],[v for _,v in history],label="AE held-out reconstruction MSE")
    ax[1,1].axhline(pca_mse,ls="--",color="gray",label="PCA held-out reconstruction MSE")
    ax[1,1].set(title="Reconstruction != semantic/geodesic neighborhood",xlabel="epoch",ylabel="MSE")
    ax[1,1].legend(fontsize=8)
    fig.suptitle("11 | Optional Torch AE bottleneck (2D) vs PCA / Isomap",fontsize=13)
    _save(fig,Path(out),"11_autoencoder_manifold.png")
    return {"heldout_reconstruction_mse":{"pca":pca_mse,"autoencoder":ae_mse},
            "oracle_neighbor_overlap_at_10":overlaps,"epochs":220,
            "note":"AE reconstruction loss does not guarantee manifold unfolding"}
