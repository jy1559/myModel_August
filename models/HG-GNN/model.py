# models/HGGNN/model.py
# -----------------------------------------------------------
# Heterogeneous Global Graph Neural Network  ─ seqrec-compatible
#   • 전역 그래프 파일(global_graph.pt)을 **데이터셋 폴더**에서 자동 로드
#   • narm_loss() 포맷으로 forward 반환

from __future__ import annotations
import math, types, torch, pathlib, json
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple

# ─────────────────────────────────────────────────────────────
# 1.  Heterogeneous GNN Layer
# ─────────────────────────────────────────────────────────────
class HGNNLayer(nn.Module):
    def __init__(self, d: int, n_rel: int):
        super().__init__()
        self.lin_r = nn.ModuleList([nn.Linear(d, d, bias=False) for _ in range(n_rel)])
        self.up_item = nn.Linear(d, d)
        self.up_user = nn.Linear(d, d)

    def forward(
        self,
        h_item: torch.Tensor,           # [V,D]
        h_user: torch.Tensor,           # [U,D]
        A: List[torch.sparse.FloatTensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # concat once for speed
        h_all = torch.cat([h_item, h_user], 0)        # [V+U,D]
        msg_item, msg_user = 0, 0
        for r, A_r in enumerate(A):
            m = torch.sparse.mm(A_r, self.lin_r[r](h_all))
            msg_item = msg_item + m[: h_item.size(0)]
            msg_user = msg_user + m[h_item.size(0) :]
        h_item = F.relu(self.up_item(msg_item + h_item))
        h_user = F.relu(self.up_user(msg_user + h_user))
        return h_item, h_user

# ─────────────────────────────────────────────────────────────
# 2.  Session GGNN (SR-GNN reuse)
# ─────────────────────────────────────────────────────────────
class SessionGNN(nn.Module):
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
# ─────────────────────────────────────────────────────────────
# 3.  HG-GNN Main Model
# ─────────────────────────────────────────────────────────────
class SeqRecModel(nn.Module):
    def __init__(self, n_items: int, cfg: Dict):
        super().__init__()
        self.d = cfg.get("embed_dim", 128)
        self.device = torch.device(cfg.get("device", "cpu"))

        # ─ Load global graph ------------------------------------------------
        root = (
            pathlib.Path(cfg["dataset_folder"])
            / cfg["dataset_name"]
            / "timesplit"
            / str(cfg["sampling_N"])
        )
        gpath = root / "global_graph.pt"
        data = torch.load(gpath, map_location="cpu", weights_only=True)

        self.A_rels = [A.to(self.device) for A in data["rels"]]
        self.uid_map: dict[int, int] = data["uid_map"]          # raw uid → graph idx
        self.num_item: int = data["num_item"]   # includes PAD(0)
        self.num_user: int = data["num_user"]

        # ─ Embedding tables (size ≥ full ID range) --------------------------
        self.id_emb = nn.Embedding(
            max(n_items, self.num_item), self.d, padding_idx=0
        )
        self.user_emb = nn.Embedding(
            max(cfg.get("n_users", 0), self.num_user), self.d
        )
        self.embed = types.SimpleNamespace(id_emb=self.id_emb)

        # ─ HGNN layers ------------------------------------------------------
        K = cfg.get("hgnn_layers", 2)
        self.hgnn_layers = nn.ModuleList(
            [HGNNLayer(self.d, len(self.A_rels)) for _ in range(K)]
        )

        # ─ Session-GGNN & attention ----------------------------------------
        self.gnn = SessionGNN(self.d, step=1)
        self.lin_q1 = nn.Linear(self.d, self.d, bias=False)
        self.lin_q2 = nn.Linear(self.d, self.d, bias=False)
        self.lin_att = nn.Linear(self.d, 1, bias=False)
        self.fuse = nn.Linear(self.d * 2, self.d)

    # =========================================================
    # helper: build bidirectional session graph (SR-GNN style)
    # =========================================================
    @staticmethod
    def _build_graph(seq: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, L = seq.shape
        A = torch.zeros(B, L, 2 * L, device=seq.device)
        valid = mask[:, 1:] & mask[:, :-1]
        if valid.any():
            b, pos = valid.nonzero(as_tuple=True)
            u, v = pos, pos + 1
            A[b, v, u] = 1.0
            A[b, u, v + L] = 1.0
            deg = A[:, :, :L].sum(-1).clamp(min=1)
            A[:, :, :L] /= deg.unsqueeze(-1)
            A[:, :, L:] /= deg.unsqueeze(-1)
        return A

    # =========================================================
    # helper: global HGNN forward (no grad)
    # =========================================================
    def _global_forward(self):
        # ① pad 포함 아이템·유저 임베딩 슬라이스 -- FP32 로 변환
        h_item = self.id_emb.weight[: self.num_item].float()
        h_user = self.user_emb.weight[: self.num_user].float()

        # ② sparse.mm 를 FP32 로 돌리도록 autocast 비활성
        with torch.amp.autocast(device_type="cuda", enabled=False):
            for layer in self.hgnn_layers:
                h_item, h_user = layer(h_item, h_user, self.A_rels)

        # ③ 끝나고 다시 모델 기본 dtype(fp16/32)로 캐스팅
        target_dtype = self.id_emb.weight.dtype        # fp16 or fp32
        return h_item.to(target_dtype), h_user.to(target_dtype)

    # =========================================================
    # forward =================================================
    # =========================================================
    def forward(self, batch: Dict):
        ids = batch["item_id"].clamp_min(0).to(self.device)    # [B,S,L]
        eval_from = batch["eval_from"].to(self.device)         # [B]
        B, S, L = ids.shape
        sess_idx = torch.arange(S, device=self.device)
        allow = sess_idx.unsqueeze(0) >= eval_from.unsqueeze(1)
        ids = ids * allow.unsqueeze(-1)

        g_item, g_user = self._global_forward()                # [V,D] [U,D]
        reps, tgt, ut = [], [], []

        for s in range(S):
            seq = ids[:, s]
            mask = seq.ne(0)
            lens = mask.sum(-1)
            valid = lens.ge(2)
            if not valid.any():
                continue

            target = seq[valid, lens[valid] - 1]               # [B']
            L_in = (lens[valid] - 1).max().item()
            seq_in = seq[valid, :L_in]
            m_in = (
                torch.arange(L_in, device=self.device).unsqueeze(0)
                < (lens[valid] - 1).unsqueeze(1)
            )
            seq_in = seq_in * m_in.long()

            A = self._build_graph(seq_in, m_in)
            h = self.id_emb(seq_in)
            h = self.gnn(A, h)

            last_idx = (lens[valid] - 2).clamp(min=0)
            ht = h[torch.arange(h.size(0), device=self.device), last_idx]
            q1 = self.lin_q1(ht).unsqueeze(1)
            q2 = self.lin_q2(h)
            alpha = self.lin_att(torch.sigmoid(q1 + q2)).squeeze(-1)
            alpha = alpha * m_in.float()
            alpha = alpha / (alpha.sum(1, keepdim=True) + 1e-8)
            c_cur = torch.bmm(alpha.unsqueeze(1), h).squeeze(1)  # [B',D]

            # ─ historical preference (user global)
            if "uid" in batch and self.uid_map:
                idx = [
                    self.uid_map.get(int(u), self.num_user - 1)
                    for u in batch["uid"][valid].tolist()
                ]
                c_hist = g_user[torch.tensor(idx, device=self.device)]
            else:
                c_hist = torch.zeros_like(c_cur)

            c_t = self.fuse(torch.cat([c_cur, c_hist], -1))
            use = allow[valid, s]
            reps.append(c_t[use])
            tgt.append(target[use])
            ut.append(batch["u_type"][valid][use])

        if not reps:
            dummy = torch.zeros(0, self.d, device=self.device)
            return {"reps": dummy, "target": dummy, "u_type": dummy}

        return {
            "reps": torch.cat(reps, 0),
            "target": torch.cat(tgt, 0),
            "u_type": torch.cat(ut, 0),
        }