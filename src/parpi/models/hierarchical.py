"""Hierarchical pathway-attention network (gene -> pathway -> cell).

What is kept from the original idea, and what is fixed
------------------------------------------------------
* Kept: a biological hierarchy (genes feed pathways, pathways feed the cell
  representation) and attention weights that say which pathways matter.
* Fixed: the unit of learning is the *cell* (thousands of samples), not the
  dose graph (nine samples). Pathway definitions are identical for every cell
  and dataset. Labels are never model inputs. The reconstruction target is
  real data (pathway scores), not the model's own embeddings. Dose is a
  supervised target here, not an input slot, so the model can be applied to
  cells whose dose is unknown.

Layers
------
1. Gene -> pathway: for each pathway and head, a softmax over *its own* member
   genes (learned within-pathway gene weights). Initialised uniform, so at
   step 0 the activations equal the rank-based pathway score baseline.
2. Pathway -> cell: pathway tokens (activation projection + learned pathway
   embedding) are pooled by attention with a learned query. These attention
   weights are the per-cell pathway importances.
3. Heads: resistance score (sigmoid), optional cell-state classifier, and a
   pathway-score reconstruction head used as a regulariser.
"""

from __future__ import annotations

import logging
import math

import numpy as np
import scipy.sparse as sp

from parpi.config import HierarchicalConfig
from parpi.models.base import Inputs, Prediction, Targets

log = logging.getLogger(__name__)

try:
    import torch
    from torch import nn
    from torch.nn import functional as F  # noqa: N812
except ImportError as err:  # pragma: no cover - exercised only without torch
    raise ImportError(
        "The hierarchical model needs PyTorch: pip install 'parpi-resistance[torch]'"
    ) from err


class GenePathwayLayer(nn.Module):
    """Masked, per-pathway softmax pooling of gene ranks into pathway activations."""

    def __init__(self, membership: sp.csr_matrix, heads: int) -> None:
        super().__init__()
        coo = membership.tocoo()
        self.n_genes, self.n_sets = membership.shape
        self.heads = heads
        self.register_buffer("gene_idx", torch.as_tensor(coo.row, dtype=torch.long))
        self.register_buffer("set_idx", torch.as_tensor(coo.col, dtype=torch.long))
        self.edge_logit = nn.Parameter(torch.zeros(coo.nnz, heads))

    def edge_weights(self) -> torch.Tensor:
        """Softmax over the member genes of each pathway, per head -> (E, H)."""
        logit = self.edge_logit - self.edge_logit.max(dim=0, keepdim=True).values.detach()
        w = logit.exp()
        z = torch.zeros(self.n_sets, self.heads, device=w.device, dtype=w.dtype)
        z = z.index_add(0, self.set_idx, w)
        return w / z[self.set_idx].clamp_min(1e-12)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, G) centred ranks -> (B, K, H) pathway activations."""
        # Work on the membership edges only (gather genes, weight, sum per pathway):
        # cost is O(batch x edges), ~100x cheaper than a dense genes x pathways
        # product for Reactome-sized libraries, and index_add is deterministic on CPU.
        w = self.edge_weights()
        xg = x[:, self.gene_idx]  # (B, E)
        outs = []
        for h in range(self.heads):
            acc = torch.zeros(x.shape[0], self.n_sets, device=x.device, dtype=x.dtype)
            outs.append(acc.index_add(1, self.set_idx, xg * w[:, h]))
        return torch.stack(outs, dim=-1)


class HiPathNet(nn.Module):
    def __init__(self, membership: sp.csr_matrix, cfg: HierarchicalConfig, n_states: int) -> None:
        super().__init__()
        k = membership.shape[1]
        self.gene_pathway = GenePathwayLayer(membership, cfg.heads)
        self.token_proj = nn.Sequential(
            nn.Linear(cfg.heads, cfg.dim), nn.GELU(), nn.Linear(cfg.dim, cfg.dim)
        )
        self.pathway_emb = nn.Parameter(torch.randn(k, cfg.dim) * 0.02)
        self.norm = nn.LayerNorm(cfg.dim)
        self.query = nn.Parameter(torch.randn(cfg.dim) * 0.02)
        self.drop = nn.Dropout(cfg.dropout)
        self.trunk = nn.Sequential(nn.Linear(cfg.dim, cfg.dim), nn.GELU(), nn.Dropout(cfg.dropout))
        self.score_head = nn.Linear(cfg.dim, 1)
        self.state_head = nn.Linear(cfg.dim, n_states) if n_states > 1 else None
        self.recon_head = nn.Linear(cfg.dim, k)
        # Direct (ridge-like) path from standardised pathway activity to the score,
        # so the network is never weaker than the linear baseline; starts at zero.
        self.skip = nn.Linear(k, 1)
        nn.init.zeros_(self.skip.weight)
        self.scale = 1.0 / math.sqrt(cfg.dim)
        # Per-pathway centring/scaling of activations, set from training data.
        self.register_buffer("act_mean", torch.zeros(k))
        self.register_buffer("act_std", torch.ones(k))

    def set_scaling(self, pathways: np.ndarray, score_mean: float) -> None:
        """Standardise inputs and start the score at the training mean."""
        self.act_mean.copy_(torch.as_tensor(pathways.mean(0), dtype=torch.float32))
        self.act_std.copy_(torch.as_tensor(pathways.std(0) + 1e-6, dtype=torch.float32))
        p = float(np.clip(score_mean, 1e-3, 1 - 1e-3))
        with torch.no_grad():
            self.score_head.bias.fill_(math.log(p / (1 - p)))

    def standardise(self, pathways: torch.Tensor) -> torch.Tensor:
        return (pathways - self.act_mean) / self.act_std

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        act = self.gene_pathway(x)  # (B, K, H)
        # Raw activations differ between cells by only ~0.01-0.1, so standardise
        # each pathway before projecting, or the signal is swamped by biases.
        act = (act - self.act_mean[:, None]) / self.act_std[:, None]
        tokens = self.norm(self.token_proj(act) + self.pathway_emb)  # (B, K, D)
        attn = torch.softmax((tokens @ self.query) * self.scale, dim=1)  # (B, K)
        z = self.trunk(self.drop((attn.unsqueeze(-1) * tokens).sum(1)))  # (B, D)
        logit = self.score_head(z) + self.skip(self.drop(act.mean(-1)))
        out = {
            "score": torch.sigmoid(logit).squeeze(-1),
            "attention": attn,
            "recon": self.recon_head(z),
        }
        if self.state_head is not None:
            out["state_logits"] = self.state_head(z)
        return out


def _set_determinism(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - torch DataLoader-free loop, only seeding globals
    torch.use_deterministic_algorithms(True, warn_only=True)


class HierarchicalModel:
    name = "hierarchical"

    def __init__(self, cfg: HierarchicalConfig, membership: sp.csr_matrix) -> None:
        self.cfg = cfg
        self.membership = sp.csr_matrix(membership)
        self.net: HiPathNet | None = None
        self.classes_: np.ndarray | None = None
        self._importance: np.ndarray | None = None
        self.history: list[dict[str, float]] = []

    # ------------------------------------------------------------------ train
    def fit(self, x: Inputs, y: Targets, seed: int) -> HierarchicalModel:
        _set_determinism(seed)
        cfg = self.cfg
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state_idx = None
        if y.state is not None:
            labelled = y.state != ""
            if len(np.unique(y.state[labelled])) > 1:
                self.classes_ = np.unique(y.state[labelled])
                # Unlabelled cells get -1 and are ignored by the state loss.
                state_idx = np.where(labelled, np.searchsorted(self.classes_, y.state), -1)
        n_states = 0 if self.classes_ is None else len(self.classes_)
        self.net = HiPathNet(self.membership, cfg, n_states)
        self.net.set_scaling(x.pathways, float(y.score.mean()))
        self.net = self.net.to(device)

        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(x))
        n_val = max(1, int(0.15 * len(x)))
        val, tr = perm[:n_val], perm[n_val:]

        def tens(a: np.ndarray, dtype: torch.dtype = torch.float32) -> torch.Tensor:
            return torch.as_tensor(a, dtype=dtype, device=device)

        X = tens(x.ranks - 0.5)
        P = tens(x.pathways)
        Y = tens(y.score)
        S = tens(state_idx, torch.long) if state_idx is not None else None

        opt = torch.optim.AdamW(self.net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        best, best_state, bad = float("inf"), None, 0
        g = torch.Generator().manual_seed(seed)
        for epoch in range(cfg.epochs):
            self.net.train()
            order = torch.as_tensor(tr)[torch.randperm(len(tr), generator=g)]
            total = 0.0
            for start in range(0, len(order), cfg.batch_size):
                b = order[start : start + cfg.batch_size].to(device)
                loss = self._loss(X[b], P[b], Y[b], None if S is None else S[b])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt.step()
                total += float(loss.detach()) * len(b)
            self.net.eval()
            with torch.no_grad():
                vb = torch.as_tensor(val, device=device)
                vloss = float(self._loss(X[vb], P[vb], Y[vb], None if S is None else S[vb]))
            self.history.append({"epoch": epoch, "train": total / len(tr), "val": vloss})
            log.debug("epoch %d train %.4f val %.4f", epoch, total / len(tr), vloss)
            if vloss < best - 1e-5:
                best, bad = vloss, 0
                best_state = {k: v.detach().clone() for k, v in self.net.state_dict().items()}
            else:
                bad += 1
                if bad >= cfg.patience:
                    break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        self.net = self.net.cpu().eval()
        sample = rng.choice(len(x), min(len(x), 2000), replace=False)
        self._importance = self.attention(x.take(sample)).mean(0)
        return self

    def _loss(
        self, x: torch.Tensor, p: torch.Tensor, y: torch.Tensor, s: torch.Tensor | None
    ) -> torch.Tensor:
        assert self.net is not None
        out = self.net(x)
        loss = F.mse_loss(out["score"], y)
        target = self.net.standardise(p)
        loss = loss + self.cfg.recon_weight * F.mse_loss(out["recon"], target)
        if s is not None and "state_logits" in out and bool((s >= 0).any()):
            ce = F.cross_entropy(out["state_logits"], s, ignore_index=-1)
            loss = loss + self.cfg.state_weight * ce
        return loss

    # ---------------------------------------------------------------- predict
    @torch.no_grad()
    def _forward_np(self, x: Inputs, batch: int = 1024) -> dict[str, np.ndarray]:
        assert self.net is not None, "fit first"
        self.net.eval()
        chunks: dict[str, list[np.ndarray]] = {}
        for start in range(0, len(x), batch):
            xb = torch.as_tensor(x.ranks[start : start + batch] - 0.5, dtype=torch.float32)
            for k, v in self.net(xb).items():
                chunks.setdefault(k, []).append(v.numpy())
        return {k: np.concatenate(v) for k, v in chunks.items()}

    def predict(self, x: Inputs) -> Prediction:
        out = self._forward_np(x)
        proba = None
        if "state_logits" in out:
            logits = out["state_logits"]
            e = np.exp(logits - logits.max(1, keepdims=True))
            proba = e / e.sum(1, keepdims=True)
        return Prediction(
            score=out["score"],
            score_std=np.zeros(len(x)),
            state_proba=proba,
            state_classes=self.classes_,
        )

    def attention(self, x: Inputs) -> np.ndarray:
        """Per-cell pathway attention weights (cells x gene sets)."""
        return self._forward_np(x)["attention"]

    def gene_weights(self) -> sp.csr_matrix:
        """Learned within-pathway gene weights (genes x sets), averaged over heads."""
        assert self.net is not None
        layer = self.net.gene_pathway
        with torch.no_grad():
            w = layer.edge_weights().mean(1).numpy()
        return sp.csr_matrix(
            (w, (layer.gene_idx.numpy(), layer.set_idx.numpy())),
            shape=(layer.n_genes, layer.n_sets),
        )

    def pathway_importance(self) -> np.ndarray:
        assert self._importance is not None, "fit first"
        return self._importance
