"""
Assembly-level KPI surrogate.

Predicts six system KPIs (TPH, total mass, energy/ton, wear part life,
bearing L10, unit cost) from the AssemblyFeatures produced by
`loop.assembly_features.extract`.

Architecture: permutation-invariant set network.
  - per-node MLP encoder, then mean-pool across nodes
  - per-edge MLP encoder, then mean-pool across edges
  - global feature MLP encoder
  - concatenate and feed an MLP head to produce six KPI values
  - normalisation: features z-scored at training time; stats saved with
    the model so inference reproduces the exact normalisation

This is graph-aware via permutation invariance (mean-pooling) but does
not perform message passing. For our 13-node single-toggle assembly the
inductive gain of message passing is marginal — the dominant signal is
already captured by what each part is (one-hot) and what it's connected
to (edge counts after pooling). The simpler architecture means PyTorch
Geometric is not required.

Save format is a torch checkpoint dict: model weights, schema, target
keys, x/y normalisation stats. `load()` reconstructs the same model.
"""
from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

from .assembly_features import AssemblyFeatures, FeatureSchema


def _require_torch():
    import torch
    return torch


def _to_tensor(features: AssemblyFeatures, torch_module):
    """Convert an AssemblyFeatures to (x_nodes, x_edges, x_global) tensors,
    each with a leading batch dimension of 1."""
    x_n = torch_module.tensor(features.node_features, dtype=torch_module.float32)
    if features.edge_features:
        x_e = torch_module.tensor(features.edge_features, dtype=torch_module.float32)
    else:
        x_e = torch_module.zeros((1, features.schema.edge_dim),
                                  dtype=torch_module.float32)
    x_g = torch_module.tensor(features.global_features, dtype=torch_module.float32)
    return x_n.unsqueeze(0), x_e.unsqueeze(0), x_g.unsqueeze(0)


class SystemKPISurrogate:
    """Train + predict assembly-level system KPIs.

    Usage:
        surrogate = SystemKPISurrogate(schema, target_keys=[...])
        surrogate.fit(samples, epochs=300)
        kpis = surrogate.predict(features)
        surrogate.save("runs/surrogates/system_v1.pt")
        loaded = SystemKPISurrogate.load("runs/surrogates/system_v1.pt")
    """

    def __init__(self, schema: FeatureSchema, target_keys: list[str],
                 hidden: int = 128, dropout: float = 0.1):
        torch = _require_torch()
        nn = torch.nn

        self.schema = schema
        self.target_keys = list(target_keys)
        self.hidden = hidden
        self.dropout = dropout

        class _Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.node_encoder = nn.Sequential(
                    nn.Linear(schema.node_dim, hidden), nn.GELU(),
                    nn.Linear(hidden, hidden),
                )
                self.edge_encoder = nn.Sequential(
                    nn.Linear(schema.edge_dim, hidden), nn.GELU(),
                )
                self.global_encoder = nn.Sequential(
                    nn.Linear(schema.global_dim, hidden), nn.GELU(),
                )
                self.head = nn.Sequential(
                    nn.Linear(3 * hidden, hidden), nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden, hidden), nn.GELU(),
                    nn.Linear(hidden, len(target_keys)),
                )

            def forward(self, x_n, x_e, x_g):
                h_n = self.node_encoder(x_n).mean(dim=1)
                h_e = self.edge_encoder(x_e).mean(dim=1)
                h_g = self.global_encoder(x_g)
                return self.head(torch.cat([h_n, h_e, h_g], dim=-1))

        self._net = _Net()
        self._torch = torch
        self._x_mean: Any = None
        self._x_std: Any = None
        self._y_mean: Any = None
        self._y_std: Any = None
        self._history: dict[str, list[float]] = {"train": [], "val": []}

    # ------------------------------------------------------------------ training

    def fit(self, samples: Iterable[tuple[AssemblyFeatures, dict[str, float]]],
            *, epochs: int = 300, lr: float = 1e-3, batch_size: int = 16,
            val_split: float = 0.2, weight_decay: float = 1e-5,
            seed: int = 0) -> dict[str, Any]:
        torch = self._torch
        sample_list = list(samples)
        if not sample_list:
            raise ValueError("no samples provided")
        for feats, _ in sample_list:
            if feats.schema != self.schema:
                raise ValueError("feature schema mismatch with surrogate schema")

        # Materialise tensors. All samples in this run share graph topology
        # (single assembly type), so node/edge counts are constant — we can
        # stack into dense tensors.
        n_nodes = len(sample_list[0][0].node_features)
        n_edges = max(1, len(sample_list[0][0].edge_features))
        X_n = torch.tensor([s[0].node_features for s in sample_list],
                            dtype=torch.float32)
        edge_feats = [s[0].edge_features or
                      [[0.0] * self.schema.edge_dim] * n_edges
                      for s in sample_list]
        X_e = torch.tensor(edge_feats, dtype=torch.float32)
        X_g = torch.tensor([s[0].global_features for s in sample_list],
                            dtype=torch.float32)
        Y = torch.tensor([[s[1][k] for k in self.target_keys]
                           for s in sample_list], dtype=torch.float32)

        # Normalise (z-score). Save stats so inference matches.
        self._x_mean = (X_n.mean(dim=(0, 1)), X_e.mean(dim=(0, 1)),
                        X_g.mean(dim=0))
        self._x_std = (X_n.std(dim=(0, 1)).clamp(min=1e-6),
                        X_e.std(dim=(0, 1)).clamp(min=1e-6),
                        X_g.std(dim=0).clamp(min=1e-6))
        self._y_mean = Y.mean(dim=0)
        self._y_std = Y.std(dim=0).clamp(min=1e-6)

        Xn_norm = (X_n - self._x_mean[0]) / self._x_std[0]
        Xe_norm = (X_e - self._x_mean[1]) / self._x_std[1]
        Xg_norm = (X_g - self._x_mean[2]) / self._x_std[2]
        Y_norm = (Y - self._y_mean) / self._y_std

        rng = torch.Generator().manual_seed(seed)
        n_total = len(sample_list)
        n_val = max(1, int(n_total * val_split)) if n_total >= 5 else 0
        perm = torch.randperm(n_total, generator=rng)
        train_idx = perm[n_val:]
        val_idx = perm[:n_val] if n_val else torch.tensor([], dtype=torch.long)

        opt = torch.optim.AdamW(self._net.parameters(),
                                 lr=lr, weight_decay=weight_decay)
        loss_fn = torch.nn.MSELoss()

        self._history = {"train": [], "val": []}
        for epoch in range(epochs):
            self._net.train()
            perm_t = train_idx[torch.randperm(len(train_idx), generator=rng)]
            ep_train_loss = 0.0
            n_batches = 0
            for i in range(0, len(perm_t), batch_size):
                idx = perm_t[i:i + batch_size]
                pred = self._net(Xn_norm[idx], Xe_norm[idx], Xg_norm[idx])
                loss = loss_fn(pred, Y_norm[idx])
                opt.zero_grad(); loss.backward(); opt.step()
                ep_train_loss += loss.item()
                n_batches += 1
            self._history["train"].append(ep_train_loss / max(n_batches, 1))

            if len(val_idx):
                self._net.eval()
                with torch.no_grad():
                    pred = self._net(Xn_norm[val_idx], Xe_norm[val_idx],
                                      Xg_norm[val_idx])
                    val_loss = float(loss_fn(pred, Y_norm[val_idx]))
                self._history["val"].append(val_loss)
            else:
                self._history["val"].append(float("nan"))

        return {
            "epochs": epochs,
            "final_train_loss": self._history["train"][-1],
            "final_val_loss": self._history["val"][-1],
            "n_train": int(len(train_idx)),
            "n_val": int(len(val_idx)),
        }

    # ------------------------------------------------------------------ inference

    def predict(self, features: AssemblyFeatures) -> dict[str, float]:
        if self._y_mean is None:
            raise RuntimeError("surrogate not yet trained")
        if features.schema != self.schema:
            raise ValueError("feature schema mismatch")
        torch = self._torch
        self._net.eval()
        x_n, x_e, x_g = _to_tensor(features, torch)
        xn = (x_n - self._x_mean[0]) / self._x_std[0]
        xe = (x_e - self._x_mean[1]) / self._x_std[1]
        xg = (x_g - self._x_mean[2]) / self._x_std[2]
        with torch.no_grad():
            y_norm = self._net(xn, xe, xg)
        y = y_norm * self._y_std + self._y_mean
        return {k: float(y[0, i]) for i, k in enumerate(self.target_keys)}

    def predict_batch(self,
                       feature_list: list[AssemblyFeatures]
                       ) -> list[dict[str, float]]:
        return [self.predict(f) for f in feature_list]

    # ------------------------------------------------------------------ I/O

    def save(self, path: str | Path) -> Path:
        torch = self._torch
        if self._y_mean is None:
            raise RuntimeError("save() called before fit() — nothing to save")
        out = Path(path).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": self._net.state_dict(),
            "schema": asdict(self.schema),
            "target_keys": self.target_keys,
            "hidden": self.hidden,
            "dropout": self.dropout,
            "x_mean": [t for t in self._x_mean],
            "x_std": [t for t in self._x_std],
            "y_mean": self._y_mean,
            "y_std": self._y_std,
            "history": self._history,
        }, out)
        return out

    @classmethod
    def load(cls, path: str | Path) -> "SystemKPISurrogate":
        torch = _require_torch()
        ckpt = torch.load(Path(path), map_location="cpu", weights_only=False)
        schema = FeatureSchema(
            part_classes=tuple(ckpt["schema"]["part_classes"]),
            materials=tuple(ckpt["schema"]["materials"]),
            numeric_param_pad=ckpt["schema"]["numeric_param_pad"],
            edge_kinds=tuple(ckpt["schema"]["edge_kinds"]),
            criticality_vocab=tuple(ckpt["schema"]["criticality_vocab"]),
        )
        inst = cls(schema=schema, target_keys=ckpt["target_keys"],
                    hidden=ckpt["hidden"], dropout=ckpt["dropout"])
        inst._net.load_state_dict(ckpt["state_dict"])
        inst._x_mean = tuple(ckpt["x_mean"])
        inst._x_std = tuple(ckpt["x_std"])
        inst._y_mean = ckpt["y_mean"]
        inst._y_std = ckpt["y_std"]
        inst._history = ckpt.get("history", {"train": [], "val": []})
        return inst
