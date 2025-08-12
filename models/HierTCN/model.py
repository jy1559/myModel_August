from __future__ import annotations
"""
models/HierTCN/model.py
~~~~~~~~~~~~~~~~~~~~~~
Hierarchical **Temporal Convolutional Network** for session‑aware
recommendation (HierTCN, Quadrana+ 2019).

Key differences vs. HRNN implementation:
    • Session encoder: dilated causal **TCN** instead of GRU.
    • User RNN input: **mean‑pooled item embeddings** of previous session.
    • BOS token (id = n_items) prepended ⇒ 첫 interaction까지 예측 대상.
    • `eval_from` 은 **loss 마스크**에서만 처리; forward 는 모든 세션을
      통과시켜 user_state 누적.
    • 모델 전용 하이퍼파라미터 (`tcn_layers`, `tcn_kernel`) 은 cfg 에서만
      읽고, train.py 공용 인자 추가 없이 사용할 수 있도록 기본값을 둡니다.
"""

import types
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# 1.  Causal TCN block
# ---------------------------------------------------------------------------
class CausalConvBlock(nn.Module):
    """Dilated causal Conv1d + residual layer."""

    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float = 0.0):
        super().__init__()
        pad = (kernel_size - 1) * dilation  # left padding only (causal)
        self.conv = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            padding=pad,
            dilation=dilation,
        )
        self.dropout = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_normal_(self.conv.weight, nonlinearity="relu")
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: [B,C,L]
        out = self.conv(x)
        # remove padding on the right (causal)
        out = out[..., : x.size(-1)]
        return torch.relu(out + x)


class SessionTCN(nn.Module):
    """Stack of causal TCN blocks."""

    def __init__(self, channels: int, n_layers: int = 3, kernel: int = 3, dropout: float = 0.2):
        super().__init__()
        dilations = [2 ** i for i in range(n_layers)]
        blocks = [
            CausalConvBlock(channels, kernel, d, dropout=dropout) for d in dilations
        ]
        self.net = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: [B,L,C]
        x = x.transpose(1, 2)  # -> [B,C,L]
        out = self.net(x)
        return out.transpose(1, 2)  # back to [B,L,C]


# ---------------------------------------------------------------------------
# 2.  HierTCN model
# ---------------------------------------------------------------------------
class SeqRecModel(nn.Module):
    """HierTCN adapted to seqrec loss interface."""

    def __init__(self, n_items: int, cfg: Dict):
        super().__init__()
        D = cfg.get("embed_dim", 128)
        self.device = torch.device(cfg.get("device", "cpu"))
        # ── special tokens ───────────────────────────────────────────
        self.PAD_ID = 0
        self.BOS_ID = n_items  # reserve last index for BOS

        # ── item embedding ───────────────────────────────────────────
        self.id_emb = nn.Embedding(n_items + 1, D, padding_idx=self.PAD_ID)
        # expose for loss helper
        self.embed = types.SimpleNamespace(id_emb=self.id_emb)

        # ── session encoder (TCN) ────────────────────────────────────
        tcn_layers = cfg.get("tcn_layers", 3)
        tcn_kernel = cfg.get("tcn_kernel", 3)
        dropout = cfg.get("dropout", 0.2)
        self.session_tcn = SessionTCN(D, tcn_layers, tcn_kernel, dropout)

        # ── user updater (GRU) ───────────────────────────────────────
        self.user_gru = nn.GRU(D, D, batch_first=True)

        # ── fusion (concat + linear) ────────────────────────────────
        self.fusion = nn.Linear(D * 2, D)

    # ------------------------------------------------------------------
    def forward(self, batch: Dict):
        """Return token‑level reps [B,S,L,D] and loss_masks [B,S,L]."""
        item_ids = batch["item_id"].to(self.device)     # [B,S,I]  (‑1 pad)
        eval_from = batch["eval_from"].to(self.device)   # [B]
        u_type = batch["u_type"]

        # clamp negative padding to 0 (PAD_ID)
        item_ids = item_ids.clamp_min(0)

        B, S, I = item_ids.shape
        reps_list: List[torch.Tensor] = []
        mask_list: List[torch.Tensor] = []

        # user state: 1‑layer [1,B,D]
        user_state = torch.zeros(1, B, self.id_emb.embedding_dim, device=self.device)

        for s in range(S):
            seq = item_ids[:, s]                        # [B,I]
            lengths = (seq != self.PAD_ID).sum(-1)      # [B]
            max_len = lengths.max().item()
            # prepend BOS token
            bos_col = torch.full((B, 1), self.BOS_ID, device=self.device)
            seq = torch.cat([bos_col, seq[:, :max_len]], dim=1)       # [B,L']
            mask_seq = seq != self.PAD_ID                            # [B,L']
            emb = self.id_emb(seq)                                   # [B,L',D]
            tc_out = self.session_tcn(emb)                           # [B,L',D]

            user_b = user_state.squeeze(0)                            # [B,D]

            # ─ fusion: concat user rep to each token ────────────────
            fusion_in = torch.cat(
                [tc_out, user_b.unsqueeze(1).expand_as(tc_out)], dim=-1
            )                                                        # [B,L',2D]
            reps = self.fusion(fusion_in)                            # [B,L',D]

            reps_list.append(reps)
            for b in range(B):
                if lengths[b] > 0:
                    # mask_seq의 길이는 max_len+1 이므로, 
                    # 실제 시퀀스 길이에 해당하는 위치가 마지막 유효 토큰
                    last_valid_idx = lengths[b] 
                    mask_seq[b, last_valid_idx] = False
            mask_list.append(mask_seq)
            
            # ─ user GRU update with mean‑pooled item emb (pad 제외) ─
            mean_emb = (
                emb * mask_seq.unsqueeze(-1).float()
            ).sum(1) / mask_seq.sum(1).clamp(min=1).unsqueeze(-1)     # [B,D]
            _, user_state = self.user_gru(mean_emb.unsqueeze(1), user_state)

        # ── pad & stack to uniform L_max ─────────────────────────────
        L_max = max(t.size(1) for t in reps_list)
        pad_r = lambda t: F.pad(t, (0, 0, 0, L_max - t.size(1)))
        pad_m = lambda t: F.pad(t, (0, L_max - t.size(1)))
        reps_stack = torch.stack([pad_r(t) for t in reps_list], dim=1)   # [B,S,L,D]
        mask_stack = torch.stack([pad_m(m) for m in mask_list], dim=1)   # [B,S,L]

        # ── loss mask: remove eval_from 이전 세션 & pure‑pad rows ────
        allow_sess = torch.arange(S, device=self.device).unsqueeze(0) >= eval_from.unsqueeze(1)
        loss_masks = mask_stack & allow_sess.unsqueeze(-1)

        return {
            "reps": reps_stack,
            "loss_masks": loss_masks,
            "u_type": u_type,
        }
