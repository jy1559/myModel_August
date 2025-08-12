# models/GAG/model.py
# ~~~~~~~~~~~~~~~~~~~
# Global-Attributed Graph (GAG) – **static** variant (no reservoir)
# Compatible with the existing seqrec pipeline:
#   • forward() returns {"reps": [N,D], "target": [N], "u_type": [N]}
#   • loss.py → narm_loss() 재사용
#
# Paper: Jiang et al., “GAG: Global Attributed Graph Neural Network for
#        Streaming Session-based Recommendation” (SIGIR 2020)

from __future__ import annotations
import math, types
from typing import Dict, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# 1.  GAG Layer (node ↔ global attribute update, Eq.(4)–(7) of the paper)
# ---------------------------------------------------------------------------
class GAGLayer(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.d = d
        # message MLPs for in-edges / out-edges
        self.msg_in  = nn.Sequential(nn.Linear(2 * d, d), nn.ReLU(), nn.Linear(d, d))
        self.msg_out = nn.Sequential(nn.Linear(2 * d, d), nn.ReLU(), nn.Linear(d, d))
        # GRU-style node update
        self.lin_node = nn.Linear(2 * d, d, bias=False)
        self.lin_gate = nn.Linear(d, d)
        # global attribute update
        self.lin_glb = nn.Linear(3 * d, d)

    def forward(
        self,
        h: torch.Tensor,   # [B,L,D] node features
        A: torch.Tensor,   # [B,L,2L] in/out adjacency
        u: torch.Tensor,   # [B,D]   global attribute (user state)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L, D = h.size()
        # ─ Edge-to-node messages ───────────────────────────────────────────
        u_exp = u.unsqueeze(1).expand(-1, L, -1)                  # [B,L,D]
        m_in  = torch.bmm(A[:, :, :L],  self.msg_in (torch.cat([h, u_exp], -1)))
        m_out = torch.bmm(A[:, :, L:], self.msg_out(torch.cat([h, u_exp], -1)))
        m = m_in + m_out                                           # [B,L,D]

        # ─ GRU-style node update ──────────────────────────────────────────
        z = torch.sigmoid(self.lin_gate(m))                        # update-gate
        h_new = (1 - z) * h + z * torch.tanh(self.lin_node(torch.cat([m, h], -1)))

        # ─ Attention-based global update ─────────────────────────────────
        ht   = h_new[:, -1]                                        # 마지막 클릭
        alpha = torch.softmax((h_new * ht.unsqueeze(1)).sum(-1), dim=1)  # [B,L]
        c    = torch.bmm(alpha.unsqueeze(1), h_new).squeeze(1)     # [B,D]
        u_new = F.relu(self.lin_glb(torch.cat([u, c, ht], -1)))    # [B,D]
        return h_new, u_new

# ---------------------------------------------------------------------------
# 2.  SeqRecModel – plug-and-play with train.py / loss.py
# ---------------------------------------------------------------------------
class SeqRecModel(nn.Module):
    """GAG (static) model for session-based recommendation."""

    def __init__(self, n_items: int, cfg: Dict):
        super().__init__()
        self.D = cfg.get("embed_dim", 200)          # paper uses 200
        self.L_gag = cfg.get("gag_layers", 1)       # keep inside model (no CLI arg)
        self.device = torch.device(cfg.get("device", "cpu"))

        # ─ Embedding ------------------------------------------------------
        self.id_emb = nn.Embedding(n_items + 1, self.D, padding_idx=0)
        self.embed  = types.SimpleNamespace(id_emb=self.id_emb)   # for _calc_logits()

        # ─ GAG stack ------------------------------------------------------
        self.gag_layers = nn.ModuleList([GAGLayer(self.D) for _ in range(self.L_gag)])

    # ----------------------------------------------------------------------
    @staticmethod
    def _build_graph(seq: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Return normalized in/out adjacency  A ∈ ℝ^{B×L×2L}  (SR-GNN style)."""
        B, L = seq.shape
        A = torch.zeros(B, L, 2 * L, device=seq.device)
        valid = mask[:, 1:] & mask[:, :-1]                       # edge exists
        if valid.any():
            b_idx, pos = valid.nonzero(as_tuple=True)
            u = pos                     # source position
            v = pos + 1                 # target position
            # in-edges
            A[b_idx, v, u] = 1.0
            # out-edges (shifted block)
            A[b_idx, u, v + L] = 1.0
            deg = A[:, :, :L].sum(-1).clamp(min=1)               # in-degree
            A[:, :, :L] = A[:, :, :L] / deg.unsqueeze(-1)
            A[:, :, L:] = A[:, :, L:] / deg.unsqueeze(-1)
        return A

    # ----------------------------------------------------------------------
    def forward(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """
        Input
        -----
        batch["item_id"] : [B,S,L]  (-1→PAD, we clamp to 0)
        batch["eval_from"]: [B]     (loss 계산 시 좌측 세션 skip)
        batch["u_type"]   : [B]
        """
        ids = batch["item_id"].clamp_min(0).to(self.device)   # PAD = 0
        eval_from = batch["eval_from"].to(self.device)        # [B]
        B, S, L = ids.shape

        # ─ eval_from < s 세션만 학습/평가 (하지만 user-state 업데이트는 모두) ─
        sess_idx = torch.arange(S, device=self.device)        # [S]
        allow_sess = sess_idx.unsqueeze(0) >= eval_from.unsqueeze(1)  # [B,S]

        # ─ iterate over sessions (user-state carried by global attr) ─
        dtype = self.id_emb.weight.dtype
        user_state = torch.zeros(B, self.D,
                         device=self.device,
                         dtype=dtype) 
        reps, tgt, u_t = [], [], []

        for s in range(S):
            seq = ids[:, s]                          # [B,L]
            mask = seq != 0
            lengths = mask.sum(-1)                   # [B]
            valid = lengths >= 2                     # len ≥2

            if valid.any():
                # 1) target = last item
                target = seq[valid, lengths[valid] - 1]
                # 2) input sequence (without last)
                L_in = (lengths[valid] - 1).max().item()
                seq_in = seq[valid, :L_in]
                in_mask = (torch.arange(L_in, device=self.device)
                           .unsqueeze(0) < (lengths[valid] - 1).unsqueeze(1))
                seq_in = seq_in * in_mask.long()

                # 3) graph & embedding
                h = self.id_emb(seq_in)                              # [B',L_in,D]
                A = self._build_graph(seq_in, in_mask).to(h.dtype)
                u = user_state[valid]                                # [B',D]
                for layer in self.gag_layers:
                    h, u = layer(h, A, u)                            # update
                u = u.to(user_state.dtype)   # ⇦ dtype 통일 (fp32)
                c = u                                                # session rep

                # 4) store reps/targets for allowed sessions only
                use_for_loss = allow_sess[valid, s]
                reps.append(c[use_for_loss])
                tgt.append(target[use_for_loss])
                u_t.append(batch["u_type"][valid][use_for_loss])

                # 5) user state update for *all* valid sessions
                user_state[valid] = u

        if len(reps) == 0:
            # edge case: no valid session for loss (e.g., all filtered out)
            dummy = torch.zeros(0, self.D, device=self.device)
            return {"reps": dummy, "target": dummy, "u_type": dummy}

        reps  = torch.cat(reps, 0)          # [N,D]
        tgt   = torch.cat(tgt, 0)           # [N]
        utype = torch.cat(u_t, 0)           # [N]

        return {"reps": reps, "target": tgt, "u_type": utype}
