from __future__ import annotations

import math
from typing import Dict, List, Tuple
from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

# ---------------------------------------------------------------------------
# Positional + Sinusoidal Time‑interval Embedding helpers
# ---------------------------------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        self.pos_emb = nn.Embedding(max_len + 1, d_model)

    def forward(self, length: int, device: torch.device):
        idx = torch.arange(length, device=device).clamp_max(self.pos_emb.num_embeddings-1)
        return self.pos_emb(idx)            # [L,d]


def sinusoidal_time_embedding(delta: torch.Tensor, d_model: int) -> torch.Tensor:
    """delta: [B,L,L] float → return [B,L,L,d]"""
    device = delta.device
    half = d_model // 2
    denom = torch.exp(torch.arange(half, device=device, dtype=torch.float) *
                       -(math.log(10000.0) / (half - 1)))   # [half]
    sinusoid = delta.unsqueeze(-1) * denom                  # [B,L,L,half]
    sin = torch.sin(sinusoid)
    cos = torch.cos(sinusoid)
    return torch.cat([sin, cos], dim=-1)                    # [B,L,L,d]

# ---------------------------------------------------------------------------
# Time‑aware self‑attention layer
# ---------------------------------------------------------------------------
# ───────────────────────── Δt → log-bucket helper ──────────────────────────
def bucketize_time(delta: Tensor,
                   num_buckets: int = 32,
                   max_delta: float = 3600 * 24 * 30) -> Tensor:
    """
    Δt(초) → 0 … num_buckets-1  (log-scale, 양수/음수 모두 지원)
      bucket 0 : |Δt| ≤ 1 s
      bucket N-1: |Δt| ≥ max_delta
      나머지    : log2 균등 분할
    """
    dt = delta.abs().clamp(min=1.0)                     # 0 → 1  ,  음수 → 양수
    log2 = torch.log2(dt)
    log2_max = math.log2(max_delta)
    idx = (log2 / (log2_max / (num_buckets - 2))).long() + 1   # 1 … N-2
    idx = idx.clamp(1, num_buckets - 2)
    idx[dt <= 1.0]      = 0
    idx[dt >= max_delta] = num_buckets - 1
    return idx.to(torch.long)                            # delta 와 동일 shape


# ─────────────────────── Time-aware Self-Attention ──────────────────────────
class TimeAwareSelfAttention(nn.Module):
    """
    메모리 절약형 TiSASRec Attention
      • Δt 32-bucket + scalar bias (per head)
      • causal mask + padding mask 지원
      • PyTorch ≥ 2.1 의 Flash-Attention(SDPA) 사용, 없으면 기본 matmul
    """
    def __init__(self, d_model: int, n_heads: int,
                 dropout: float = 0.1,
                 num_dt_buckets: int = 32,
                 max_dt: float = 3600*24*30):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model  = d_model
        self.n_heads  = n_heads
        self.d_head   = d_model // n_heads
        self.num_dt_buckets = num_dt_buckets
        self.max_dt  = max_dt

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        # Δt bucket → scalar bias  (per head)
        self.time_bias = nn.Embedding(num_dt_buckets, n_heads)
        self.out_proj  = nn.Linear(d_model, d_model)
        self.dropout   = nn.Dropout(dropout)

    # -------- Δt → log-bucket ------------------------------------------------
    @staticmethod
    def _bucket(delta, nb=32, mx=3600*24*30):
        dt = delta.abs().clamp(min=1.)
        log2 = torch.log2(dt); log2_max = math.log2(mx)
        idx  = (log2 / (log2_max/(nb-2))).long()+1
        idx.clamp_(1, nb-2); idx[dt<=1.] = 0; idx[dt>=mx] = nb-1
        return idx

    def forward(self, x, ts, pad_mask=None):
        B,L,_ = x.shape; H=self.n_heads; dh=self.d_head; dev=x.device
        Q = self.W_Q(x).view(B,L,H,dh).transpose(1,2)
        K = self.W_K(x).view(B,L,H,dh).transpose(1,2)
        V = self.W_V(x).view(B,L,H,dh).transpose(1,2)
        Qh,Kh,Vh=[t.reshape(B*H,L,dh) for t in (Q,K,V)]

        # ─ score & Δt bias ---------------------------------------------------
        score = (Qh @ Kh.transpose(1,2)) / math.sqrt(dh)          # [B*H,L,L]
        dt_id = self._bucket(ts.unsqueeze(2)-ts.unsqueeze(1),
                             self.time_bias.num_embeddings, self.max_dt)
        tb    = self.time_bias(dt_id).permute(0,3,1,2)            # [B,H,L,L]
        score = score + tb.reshape(B*H, L, L)

        # ─ mask --------------------------------------------------------------
        causal = torch.tril(torch.ones(L,L,device=dev,dtype=torch.bool))
        causal = causal.unsqueeze(0).expand(B, L, L) # [B,L,L]
        if pad_mask is not None:
            valid = pad_mask.bool()                        # [B, L]
            valid = valid.unsqueeze(2) & valid.unsqueeze(1)  # [B, L, L]
            keep  = causal & valid                         # True = keep
        else:
            keep  = causal
        score.masked_fill_(~causal.repeat_interleave(H,0), -1e4)

        # ─ softmax / dropout / V mix ----------------------------------------
        attn = self.dropout(torch.softmax(score, dim=-1))
        out  = (attn @ Vh).view(B,H,L,dh).transpose(1,2).reshape(B,L,-1)
        return self.out_proj(out)
# ---------------------------------------------------------------------------
# Feed‑Forward
# ---------------------------------------------------------------------------
class FFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.act = nn.ReLU()
        self.dp  = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.dp(self.act(self.fc1(x))))

# ---------------------------------------------------------------------------
# TiSASRec main model (SeqRecModel)
# ---------------------------------------------------------------------------
class SeqRecModel(nn.Module):
    """Time‑interval aware SASRec for seqrec framework (NARM‑style output)."""

    def __init__(self, n_items: int, cfg: Dict):
        super().__init__()
        self.d_model = cfg.get("embed_dim", 64)
        self.n_heads = cfg.get("n_heads", 2)
        self.n_layers = cfg.get("n_layers", 2)
        self.max_len = cfg.get("max_len", 512)
        self.dropout = cfg.get("dropout", 0.1)
        self.device  = cfg.get("device", "cpu")

        self.item_emb = nn.Embedding(n_items + 1, self.d_model, padding_idx=0)  # 0 pad
        self.pos_enc  = PositionalEncoding(self.max_len, self.d_model)
        self.layers   = nn.ModuleList([
            nn.ModuleDict({
                "attn": TimeAwareSelfAttention(self.d_model, self.n_heads, self.dropout),
                "ffn": FFN(self.d_model, self.d_model * 4, self.dropout),
                "norm1": nn.LayerNorm(self.d_model),
                "norm2": nn.LayerNorm(self.d_model),
            }) for _ in range(self.n_layers)
        ])
        self.dp_in = nn.Dropout(self.dropout)

        # for loss._calc_logits() compatibility (ID_64 path)
        self.emb = self.item_emb
        self.b   = nn.Identity()

    # ------------------------------------------------------------------
    def _flatten_sessions(self, item_ids: torch.Tensor, delta_ts: torch.Tensor, allow_sess: torch.Tensor):
        """Compact sessions by **removing intra‑session padding** and concatenate.

        Args
        ----
        item_ids, delta_ts : [B,S,I] (0‑pad or ‑1 pad)

        Returns
        -------
        seq  : LongTensor [B,L_max]   (0 pad)
        ts   : FloatTensor [B,L_max]  cumulative timestamp
        mask : BoolTensor  [B,L_max]  True for valid tokens
        """
        B, S, I = item_ids.shape
        device = item_ids.device
        seq_list: List[torch.Tensor] = []
        ts_list:  List[torch.Tensor] = []
        allow_list:  List[torch.Tensor] = []
        lens: List[int] = []

        for b in range(B):
            flat_items = item_ids[b].view(-1)             # [S*I]
            flat_dt    = delta_ts[b].view(-1)
            valid      = flat_items > 0
            seq_b      = flat_items[valid]                # [L_b]
            dt_b       = flat_dt[valid]
            ts_b       = torch.cumsum(dt_b, dim=0)        # absolute ts

            allow_tok = torch.cat([
                allow_sess[b, s].repeat((item_ids[b, s] > 0).sum())
                for s in range(S)
            ])
            seq_list.append(seq_b)
            ts_list.append(ts_b)
            allow_list.append(allow_tok)
            lens.append(seq_b.size(0))

        L_max = max(lens)
        zeros_L = torch.zeros(B, L_max, device=device)
        seq_pad = torch.zeros(B, L_max, dtype=torch.long,  device=device)
        ts_pad  = torch.zeros(B, L_max, dtype=torch.float, device=device)
        mask_pad= torch.zeros(B, L_max, dtype=torch.bool,  device=device)
        allow_pad = torch.zeros(B, L_max, dtype=torch.bool,  device=device)
        for b in range(B):
            ln = lens[b]
            seq_pad[b, :ln] = seq_list[b]
            ts_pad [b, :ln] = ts_list[b]
            mask_pad[b, :ln] = True
            allow_pad[b, :ln] = allow_list[b]
        return seq_pad, ts_pad, mask_pad, allow_pad

    # ------------------------------------------------------------------
    def forward(self, batch: Dict):
        item_ids = batch["item_id"].clamp_min(0).to(self.device)
        delta_ts = batch["delta_ts"].to(self.device)

        eval_from = batch["eval_from"].to(self.device) #[B]
        u_type = batch["u_type"].to(self.device)
        S = item_ids.size(1)
        sess_idx = torch.arange(S, device=self.device).view(1,S)
        allow_sess = sess_idx >= eval_from.unsqueeze(1)                  # [B,S]
        #item_ids = item_ids * allow.unsqueeze(-1) # 0패딩
        #delta_ts = delta_ts * allow.unsqueeze(-1)

        seq, ts, mask, allow_tok = self._flatten_sessions(item_ids, delta_ts, allow_sess)
        B, L = seq.shape
        # embedding + positional
        pos_emb = self.pos_enc(L, self.device).unsqueeze(0)          # [1,L,D]
        x = self.item_emb(seq) + pos_emb
        x = self.dp_in(x)

        for layer in self.layers:
            attn_out = layer["attn"](x, ts, mask)
            x = layer["norm1"](x + attn_out)
            ffn_out  = layer["ffn"](x)
            x = layer["norm2"](x + ffn_out)

        # loss mask: valid positions except first token
        loss_mask = allow_tok & mask
        loss_mask[:, 0] = False
        utype_lists = [
            torch.full((ln,), batch["u_type"][b], dtype=torch.long, device=self.device)
            for b, ln in enumerate(mask.sum(1).tolist())
        ]
        u_type_pad = torch.zeros_like(seq)
        for b, ln in enumerate(mask.sum(1).tolist()):
            u_type_pad[b, :ln] = utype_lists[b]

        return {"reps": x, "loss_masks": loss_mask, 'seq':seq, "u_type": u_type_pad}
