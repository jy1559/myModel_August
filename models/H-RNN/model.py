"""
models/HRNN/model.py
~~~~~~~~~~~~~~~~~~~~
Hierarchical RNN (HRNN Init variant) for session based recommendation, as
proposed in *Personalizing Session based Recommendations with Hierarchical
Recurrent Neural Networks* (Quadrana et al., RecSys 2017).

The implementation follows the **seqrec** framework already used in your
project:
    • `forward()` returns token level hidden representations and loss masks so
      that the existing `mymodel_loss()` (negative sampling CE) can be reused
      unchanged.
    • Weightied dot product scoring is handled externally by
      `loss._calc_logits()`; we therefore only expose an `id_emb` member inside
      `self.embed` for compatibility.
"""

import types
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# 1.  Low‑level GRU helpers
# ---------------------------------------------------------------------------
class SessionGRU(nn.Module):
    """GRU encoder that copes with variable‑length (padded) sequences."""

    def __init__(self, emb_dim: int, hid_dim: int, n_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.gru = nn.GRU(
            emb_dim,
            hid_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )

    def forward(
        self,
        x: torch.Tensor,            # [B,L,D_emb]
        lengths: torch.Tensor,      # [B]
        h0: torch.Tensor | None = None,  # [num_layers,B,D_hid]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return `(outputs, last_hidden)`.

        * `outputs` : [B,L,D_hid]  right‑padded to the max length in the batch.
        * `last_hidden` : [B,D_hid]  (top‑layer hidden state at true last token)
        """
        # Run GRU over padded sequence directly (batch_first=True)
        out, h_n = self.gru(x, h0)     # out: [B,L,D]

        # Gather last hidden for each sequence according to *lengths*
        idx = (lengths - 1).clamp(min=0).unsqueeze(-1).unsqueeze(-1)
        last = out.gather(1, idx.expand(-1, -1, out.size(-1))).squeeze(1)  # [B,D_hid]
        return out, last

# ---------------------------------------------------------------------------
# 2.  Main model
# ---------------------------------------------------------------------------
class SeqRecModel(nn.Module):
    """Hierarchical RNN – *Init* variant.

    Input batch keys (same as other models):
        • item_id   : [B,S,I]
        • eval_from : [B]          (#sessions to ignore at the left)
        • u_type    : [B]

    Output dict (for `mymodel_loss`):
        • reps       : [B,S,L,D]
        • loss_masks : [B,S,L]     (1: valid token, 0: padding)
        • u_type     : 그대로 전달
    """

    def __init__(self, n_items: int, cfg: Dict):
        super().__init__()
        self.D_emb: int = cfg.get("embed_dim", 64)
        self.D_hid: int = cfg.get("hidden_dim", 100)
        n_layers: int = cfg.get("n_layers", 1)
        dropout: float = cfg.get("dropout", 0.0)
        self.device = torch.device(cfg.get("device", "cpu"))

        # ── modules ───────────────────────────────────────────────────────
        self.id_emb = nn.Embedding(n_items + 1, self.D_emb, padding_idx=0)
        self.session_enc = SessionGRU(self.D_emb, self.D_hid, n_layers, dropout)
        self.user_gru = nn.GRU(self.D_hid, self.D_hid, batch_first=True)

        # Expose id_emb the way loss helpers expect (embed.id_emb)
        self.embed = types.SimpleNamespace(id_emb=self.id_emb)

    # ------------------------------------------------------------------
    def forward(self, batch: Dict):
        item_ids: torch.Tensor = batch["item_id"].to(self.device)  # [B,S,I]
        u_type: torch.Tensor = batch["u_type"]  # 그대로 CPU/GPU 따라감

        B, S, I = item_ids.shape

        reps_list: List[torch.Tensor] = []  # each [B,L,D_hid]
        mask_list: List[torch.Tensor] = []

        user_state = torch.zeros(1, B, self.D_hid, device=self.device)  # [1,B,D]

        for s in range(S):
            seq: torch.Tensor = item_ids[:, s]            # [B,I]
            lengths: torch.Tensor = (seq != 0).sum(-1)    # [B]
            max_len = lengths.max().item()
            if max_len == 0:
                # 빈 세션: pad 1‑length dummy to keep alignment
                reps_list.append(torch.zeros(B, 1, self.D_hid, device=self.device))
                mask_list.append(torch.zeros(B, 1, dtype=torch.bool, device=self.device))
                continue

            seq = seq[:, :max_len]
            lengths_clamped = lengths.clamp(min=1)
            emb = self.id_emb(seq.clamp_min(0))                       # [B,L,D_emb]

            n_layers_sess = self.session_enc.gru.num_layers
            h0_sess = user_state.repeat(n_layers_sess, 1, 1)   # [n_layers,B,D]
            out_s, last_h = self.session_enc(emb, lengths_clamped, h0_sess)
            reps_list.append(out_s)                      # variable L
            mask_list.append((seq > 0))

            # HRNN‑Init: last hidden ➜ user state GRU (single step)
            _, user_state = self.user_gru(last_h.unsqueeze(1), user_state)

        # ─────────────────── Stack / pad to uniform shape ───────────────────
        L_max = max(t.size(1) for t in reps_list)
        pad_r = lambda t: F.pad(t, (0, 0, 0, L_max - t.size(1)))
        pad_m = lambda t: F.pad(t, (0, L_max - t.size(1)))

        reps = torch.stack([pad_r(t) for t in reps_list], dim=1)   # [B,S,L_max,D]
        masks = torch.stack([pad_m(m) for m in mask_list], dim=1)  # [B,S,L_max]

        return {
            "reps": reps,           # [B,S,L,D]
            "loss_masks": masks,   # [B,S,L]
            "u_type": u_type,
        }
