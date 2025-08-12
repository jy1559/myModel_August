from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# 1. GGNN Cell (공식 SR-GNN 구현 그대로)
# ---------------------------------------------------------------------------
class GNN(nn.Module):
    def __init__(self, hidden_size: int, step: int = 1):
        super().__init__()
        self.hidden_size = hidden_size
        self.step = step
        gate_size = 3 * hidden_size
        input_size = 2 * hidden_size
        self.w_ih = nn.Parameter(torch.Tensor(gate_size, input_size))
        self.w_hh = nn.Parameter(torch.Tensor(gate_size, hidden_size))
        self.b_ih = nn.Parameter(torch.Tensor(gate_size))
        self.b_hh = nn.Parameter(torch.Tensor(gate_size))
        self.b_iah = nn.Parameter(torch.Tensor(hidden_size))
        self.b_oah = nn.Parameter(torch.Tensor(hidden_size))
        self.edge_in  = nn.Linear(hidden_size, hidden_size, bias=True)
        self.edge_out = nn.Linear(hidden_size, hidden_size, bias=True)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1 / math.sqrt(self.hidden_size)
        for p in self.parameters():
            p.data.uniform_(-stdv, stdv)

    def _cell(self, A: torch.Tensor, h: torch.Tensor):
        B, L, H = h.shape
        a_in, a_out = A[:, :, :L], A[:, :, L:]
        m_in  = torch.matmul(a_in,  self.edge_in(h))  + self.b_iah
        m_out = torch.matmul(a_out, self.edge_out(h)) + self.b_oah
        inputs = torch.cat([m_in, m_out], -1)                 # [B,L,2H]
        gi = F.linear(inputs, self.w_ih, self.b_ih)           # [B,L,3H]
        gh = F.linear(h,      self.w_hh, self.b_hh)           # [B,L,3H]
        i_r, i_i, i_n = gi.chunk(3, -1)
        h_r, h_i, h_n = gh.chunk(3, -1)
        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate   = torch.tanh(i_n + resetgate * h_n)
        return newgate + inputgate * (h - newgate)

    def forward(self, A, h):
        for _ in range(self.step):
            h = self._cell(A, h)
        return h

# ---------------------------------------------------------------------------
# 2. SR-GNN Model (seqrec-compatible)
# ---------------------------------------------------------------------------
class SeqRecModel(nn.Module):
    """SR-GNN adapted to the seqrec framework (NARM-style output)."""

    def __init__(self, n_items: int, cfg: Dict):
        super().__init__()
        self.hidden_size = cfg.get("embed_dim", 64)
        self.gnn_step   = cfg.get("gnn_step", 1)
        self.nonhybrid  = cfg.get("srgnn_nonhybrid", False)
        self.device     = cfg.get("device", "cpu")

        self.embedding = nn.Embedding(n_items+1, self.hidden_size, padding_idx=0)
        self.gnn = GNN(self.hidden_size, step=self.gnn_step)
        self.linear_one   = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear_two   = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear_three = nn.Linear(self.hidden_size, 1, bias=False)
        self.linear_trans = nn.Linear(self.hidden_size * 2, self.hidden_size)
        # loss._calc_logits() 요구사항: emb / b
        self.emb = self.embedding
        self.b   = nn.Identity()
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1 / math.sqrt(self.hidden_size)
        for p in self.parameters():
            p.data.uniform_(-stdv, stdv)

    # ------------------------------------------------------------------
    @staticmethod
    def _build_graph(seq: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """adjacency A ∈ ℝ^{B×L×2L}, out-degree 정규화"""
        B, L = seq.shape
        A = torch.zeros(B, L, 2 * L, device=seq.device)
        src = seq[:, :-1]
        dst = seq[:, 1:]
        valid = mask[:, 1:] & mask[:, :-1]
        if valid.any():
            b_idx, pos = valid.nonzero(as_tuple=True)
            u = pos                # source position
            v = pos + 1            # target position
            # in-edges
            A[b_idx, v, u] = 1.0
            # out-edges (shifted block)
            A[b_idx, u, v + L] = 1.0
            deg = A[:, :, :L].sum(-1).clamp(min=1)
            A[:, :, :L] = A[:, :, :L] / deg.unsqueeze(-1)
            A[:, :, L:] = A[:, :, L:] / deg.unsqueeze(-1)
        return A

    # ------------------------------------------------------------------
    def forward(self, batch: Dict):
        # 0) Flatten (B,S,I) → [B*S, I] and keep len ≥ 2
        ids = batch["item_id"].clamp_min(0).to(self.device)  # [B,S,I]
        eval_from = batch["eval_from"].to(self.device) 
        
        # ─ eval_from 이전 세션은 전부 0 패딩으로 처리 ─
        sess_idx = torch.arange(ids.size(1), device=self.device)  # [S]
        allow_sess = sess_idx.unsqueeze(0) >= eval_from.unsqueeze(1)   # [B,S]
        ids = ids * allow_sess.unsqueeze(-1)                 # [B,S,I]

        B, S, I = ids.shape
        sess = ids.view(B * S, I)                             # [B*S, I]
        lengths_full = (sess != 0).sum(1)                     # [B*S]
        valid = lengths_full >= 2
        sess = sess[valid]
        lengths_full = lengths_full[valid]
        if sess.size(0) == 0:
            raise RuntimeError("SR-GNN: No valid sessions")
        N = sess.size(0)

        # target (last real item)
        target = sess[torch.arange(N, device=self.device), lengths_full - 1]

        # 1) input sequence = session without last item
        lengths_in = lengths_full - 1                         # ≥1
        L_max_in = lengths_in.max().item()
        seq_in = sess[:, :L_max_in]
        mask = torch.arange(L_max_in, device=self.device).unsqueeze(0) < lengths_in.unsqueeze(1)
        seq = seq_in * mask.long()

        # 2) GGNN
        A = self._build_graph(seq, mask)
        h = self.emb(seq)
        h = self.gnn(A, h)

        # 3) Attention pooling
        last_idx = lengths_in - 1                             # 0-based
        ht = h[torch.arange(N, device=self.device), last_idx]
        q1 = self.linear_one(ht).unsqueeze(1)
        q2 = self.linear_two(h)
        alpha = self.linear_three(torch.sigmoid(q1 + q2)).squeeze(-1)
        alpha = alpha * mask.float()
        alpha = alpha / (alpha.sum(1, keepdim=True) + 1e-8)
        a = torch.bmm(alpha.unsqueeze(1), h).squeeze(1)
        s = a if self.nonhybrid else self.linear_trans(torch.cat([a, ht], -1))

        return {"reps": s, "target": target,
                "u_type": batch["u_type"].repeat_interleave(S)[valid],}
