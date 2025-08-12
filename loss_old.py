"""
loss.py
=======
Utility functions to compute masked cross‑entropy loss with negative sampling
for the sequential‑recommendation model.

Main entry point
----------------
    >>> loss = compute_loss(batch, model_out, model, sampler, cfg)

* `model_out` is the dictionary returned by `SeqRecModel.forward` containing
  `reps : Tensor[B,S,L,D]` and `loss_masks : Tensor[B,S,L]`.
* `batch` must include at least `item_id : Tensor[B,S,I]` (ground‑truth ids).
* `sampler` is a `Sampler` instance that returns (pos+neg) candidate ids.
* `cfg` needs keys `strategy` (str) and `num_neg` (int).
"""

from __future__ import annotations

import os
import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F
from typing import Dict
# ─── item-frequency loader (global cache) ───
_item_freq = None
def _load_item_freq(ds_name):
    global _item_freq
    if _item_freq is None:
        import numpy as np, pathlib
        p = pathlib.Path("/home/jy1559/Datasets")/ds_name/"item_stats.npz"
        _item_freq = np.load(p)["train_counts"]      # [n_items+1]
    return _item_freq

def make_position_masks(
    valid_mask: torch.Tensor,
    top_k: int = 10,                 # 1~top_k 개별, 그 이후 ‘plus’
) -> Dict[str, torch.Tensor]:
    """
    label 예시
    ----------
    sess_1  sess_2 … sess_10  sess_11plus
    inter_1 inter_2 … inter_10 inter_11plus
    """
    B, S, L = valid_mask.shape
    dev = valid_mask.device

    sess_idx  = torch.arange(S, device=dev)[None, :, None]  # [1,S,1]
    inter_idx = torch.arange(L, device=dev)[None, None, :]  # [1,1,L]

    masks = {}
    masks["all"] = valid_mask
    for s in range(top_k):
        for i in range(top_k):
            masks[f"s{s+1}_i{i+1}"] = valid_mask & (sess_idx == s) & (inter_idx == i)
        masks[f"s{s+1}_i{top_k+1}+"] = valid_mask & (sess_idx == s) & (inter_idx >= top_k)
        masks[f"s{s+1}_all"] = valid_mask & (sess_idx == s)
    for i in range(top_k):
        masks[f"s{top_k+1}+_i{i+1}"] = valid_mask & (sess_idx >= top_k) & (inter_idx == i)
        masks[f"all_i{i+1}"] = valid_mask & (inter_idx == i)
    masks[f"s{top_k+1}+_i{top_k+1}+"] = valid_mask & (sess_idx >= top_k) & (inter_idx >= top_k)
    masks[f"all_i{top_k+1}+"] = valid_mask & (inter_idx >= top_k)
    masks[f"s{top_k+1}+_all"] = valid_mask & (sess_idx >= top_k) 
    return {k: m for k, m in masks.items() if m.any()}

def make_seq_position_masks(
    valid_mask: torch.Tensor,
    *,
    top_k_abs: int = 20,        # 절대 위치 개별 마스크 (1~K)
    ratio_step: float = 0.05,   # 상대 위치 bin 폭
    len_bins: list[int] = (10, 30, 60)  # 길이 경계 값
) -> Dict[str, torch.Tensor]:
    """
    Parameters
    ----------
    valid_mask : BoolTensor [B, L]
        True  → 실제 아이템 위치
        False → padding
    Returns
    -------
    Dict[label → mask]  (mask shape 동일 [B,L])
        * abs_1 … abs_20 abs_21plus
        * rel_0.05 … rel_0.95
        * len_0to10 / len_11to30 / len_31to60 / len_61plus
        * 빈(bin) 마스크는 자동 제거
    """
    B, L = valid_mask.shape
    dev  = valid_mask.device
    masks: Dict[str, torch.Tensor] = {}

    # ── 절대 위치 bin ────────────────────────────────────────────
    abs_idx = torch.arange(L, device=dev)  # [L]
    for i in range(top_k_abs):
        masks[f"abs_{i+1}"] = valid_mask & (abs_idx == i)
    masks[f"abs_{top_k_abs+1}plus"] = valid_mask & (abs_idx >= top_k_abs)

    # ── 상대 위치 bin (0.00~0.99, ratio_step 간격) ────────────────
    # 길이별 ratio 계산 -> [B,L]  (pad 위치는 0)
    lengths = valid_mask.sum(1).clamp(min=1)            # [B]
    rel = torch.arange(L, device=dev).unsqueeze(0) / lengths.unsqueeze(1)  # [B,L] float
    rel = (rel * (1/ratio_step)).round() * ratio_step   # 0.05 단위 반올림
    for r in torch.arange(ratio_step, 1.0, ratio_step):
        lbl = f"rel_{r:.2f}".replace("0.", "")          # rel_005, rel_010, ...
        masks[lbl] = valid_mask & (rel == r)

    # ── 시퀀스 길이 bin ───────────────────────────────────────────
    # 사용자-별 길이로 마스크 만드는 방식
    length_bins = []
    prev = 0
    for th in len_bins:
        length_bins.append((prev, th))
        prev = th + 1
    length_bins.append((prev, 99999))  # plus bin

    seq_len = valid_mask.sum(1)        # [B]
    for lo, hi in length_bins:
        lbl = f"len_{lo}to{hi if hi!=99999 else 'plus'}"
        flag = (seq_len >= lo) & (seq_len <= hi)
        masks[lbl] = valid_mask & flag.unsqueeze(1)

    # ── 빈(bin) 제거 ───────────────────────────────────────────────
    return {k: m for k, m in masks.items() if m.any()}
"""
아래는 개별로 안하고 1 vs 2~, 1~2 vs. 3~ 이렇게 저장한 코드
def make_position_masks(valid_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
    Return dict(label → mask) where mask has same shape as *valid_mask*.
    Bins:
      sess_F        : 첫 세션        sess_1to2     : 세션 1~2       sess_1to3    : 세션 1~3      sess_1to10    : 세션 1~10
      sess_2plus    : 세션 2부터     sess_3plus    : 세션 3부터     sess_4plus    : 세션 4부터    sess_11plus   : 세션 11부터

      inter_1       : 인터랙션 1      inter_1to2    : 인터랙션 1~2      inter_1to3    : 인터랙션 1~3      inter_1to10   : 인터랙션 1~10
      inter_2plus   : 인터랙션 2부터      inter_3plus   : 인터랙션 3부터      inter_10plus  : 인터랙션 10부터
    B, S, L = valid_mask.shape
    device = valid_mask.device
    sess_idx  = torch.arange(S, device=device)[None, :, None]  # [1,S,1]
    inter_idx = torch.arange(L, device=device)[None, None, :]  # [1,1,L]

    masks: Dict[str, torch.Tensor] = {
        # session‑level bins
        "sess_F":       valid_mask & (sess_idx == 0), "sess_1to3":    valid_mask & (sess_idx < 3), "sess_1to10":   valid_mask & (sess_idx < 10),
        "sess_2plus":   valid_mask & (sess_idx >= 1), "sess_4plus":   valid_mask & (sess_idx >= 3), "sess_11plus":  valid_mask & (sess_idx >= 10),
        "sess_3to5":  valid_mask & (sess_idx < 5) & (sess_idx >= 2), "sess_5to10":  valid_mask & (sess_idx < 10) & (sess_idx >= 4), "sess_4to20":  valid_mask & (sess_idx < 20) & (sess_idx >= 3),
        # interaction‑level bins
        "inter_1":       valid_mask & (inter_idx == 0), "inter_1to2":    valid_mask & (inter_idx < 2), "inter_1to3":    valid_mask & (inter_idx < 3),
        "inter_1to10":   valid_mask & (inter_idx < 10), "inter_2plus":   valid_mask & (inter_idx >= 1), "inter_3plus":   valid_mask & (inter_idx >= 2), "inter_10plus":  valid_mask & (inter_idx >= 9),
    }
    # drop empty bins to avoid 0‑division later
    return {k: v for k, v in masks.items() if v.any()}
    """



# ---------------------------------------------------------------------------
# 2. Mask → selection helpers
# ---------------------------------------------------------------------------
@torch.no_grad()
def _every_sess_last_inter(mask: torch.Tensor) -> torch.Tensor:
    # mask [B,S,L]  (1==valid)
    idx_last = mask.sum(-1, keepdim=True) - 1  # [B,S,1]
    sel = torch.zeros_like(mask)
    sel.scatter_(-1, idx_last.clamp(min=0), 1)
    return sel & mask.bool()

@torch.no_grad()
def _every_sess_except_first(mask: torch.Tensor) -> torch.Tensor:
    sel = mask.clone()
    sel[..., 0] = 0
    return sel

@torch.no_grad()
def _last_sess_all(mask: torch.Tensor) -> torch.Tensor:
    valid_sess = mask.sum(-1) > 0        # [B,S]
    last_idx = valid_sess.cumsum(1) == valid_sess.sum(1, keepdim=True)
    return mask * last_idx.unsqueeze(-1)

@torch.no_grad()
def _last_sess_last(mask: torch.Tensor) -> torch.Tensor:
    sel = _every_sess_last_inter(mask)
    valid_sess = mask.sum(-1) > 0
    is_last = valid_sess.cumsum(1) == valid_sess.sum(1, keepdim=True)
    return sel * is_last.unsqueeze(-1)

_STRATEGY_FN = {
    "everysess_allinter": lambda m: m,
    "everysess_lastinter": _every_sess_last_inter,
    "everysess_exceptfirst": _every_sess_except_first,
    "lastsess_allinter": _last_sess_all,
    "lastsess_lastinter": _last_sess_last,
}

# ---------------------------------------------------------------------------
# 3. Logit computation (weight‑tied dot‑product)
# ---------------------------------------------------------------------------
def _calc_logits(
    reps: torch.Tensor,
    sample_ids: torch.Tensor,
    model,
    candidate_emb: str = "ID_64",
    model_name: str = "myModel",  # for type hinting
    *,
    dt_sel: torch.Tensor | None = None,    # [N]
    add_sel: torch.Tensor | None = None,   # [N,A] or None
    pos_sel=None,
    k_neg: int = 64,
) -> torch.Tensor:
    """
    reps: [N,D], sample_ids: [N,k]  → logits [N,k]
    *ID_64*   : 기존 ID embedding(64-d) 사용
    *LLM_128* : LLM 384/250-d → proj128 → dot(reps, cand)
               llm_emb requires `pad_mask`; all candidates are real tokens,
               so pad_mask == 1 (True).
    """
    cand_type = candidate_emb.upper()
    sample_ids = sample_ids.clamp_min(0)  # [N,k] (0은 padding)
    if cand_type == "ID_64":
        if model_name == "myModel" or model_name == "GAG" or model_name == "HG-GNN" or model_name == "H-RNN" or model_name == "HierTCN":
            # [N,k,64]
            cand_vec = model.embed.id_emb(sample_ids)
        elif model_name == "NARM":
            # sample_ids: [N,k]
            cand_vec = model.emb(sample_ids)
            # cand_vec: [N,k,emb_dim]
            cand_vec = model.b(cand_vec)
            # cand_vec: [N,k,2*H]
        elif model_name == "SR-GNN":
            cand_vec = model.emb(sample_ids)
        elif model_name == "TiSASRec":
            cand_vec = model.item_emb(sample_ids)

    elif cand_type == "LLM_128":
        # ─ 1) LLM 384-d lookup ──────────────────────────────────────────────
        pad_mask = torch.ones_like(sample_ids, dtype=torch.bool)  # no padding in candidates
        cand_vec  = model.embed.llm_emb(sample_ids, pad_mask)       # [N,k,128]
    elif cand_type == "INPUT_256":
        # dt_sel: [N] , pos_sel: [N]
        dt_cand  = dt_sel.unsqueeze(1).expand(-1, k_neg)          # [N,k]
        pos_cand = pos_sel.unsqueeze(1).expand(-1, k_neg)         # [N,k]
        add_cand = None
        if add_sel is not None:
            add_cand = add_sel.unsqueeze(1).expand(-1, k_neg, -1) # [N,k,A]

        mask_cand = torch.ones_like(sample_ids, dtype=torch.bool)

        # ─ 1) 내용 concat → proj256 ─
        cand_vec = model.embed(
            sample_ids, dt_cand, mask_cand, add_cand,
            pos_idx=pos_cand        # ✨ (embed() 시그니처에 optional 인자로 추가)
        )                           # [N,k,256]
    else:
        raise ValueError(f"Unknown candidate_emb '{candidate_emb}'. Use 'ID_64' or 'LLM_128'.")
    
    reps = reps.unsqueeze(1)               # [N,1,D]
    logits = (reps * cand_vec).sum(-1)          # [N,k]
    return logits

@torch.no_grad()
def _calc_metrics(logits: torch.Tensor, targets: torch.Tensor, ks=(1,5,10)):
    """
    logits  : [N, k]   (k = 1+num_neg)
    targets : [N]      (정답 always 0)
    Returns : { 'HR@1':..., 'NDCG@5':..., ... }
    """
    max_k = max(ks)
    topk = logits.topk(max_k, dim=1, largest=True, sorted=True).indices  # [N,max_k]
    # rank of positive (0) in each row, -1 if not in top-max_k
    hits = (topk == 0).nonzero(as_tuple=False)
    ranks = torch.full((logits.size(0),), -1, device=logits.device, dtype=torch.long)
    ranks[hits[:,0]] = hits[:,1]

    out = {}
    for k in ks:
        in_topk = ranks >= 0
        hit_k   = (ranks < k) & in_topk
        hr      = hit_k.float().mean()
        ndcg    = ((1.0 / torch.log2(ranks[hit_k].float() + 2.0)).sum() / logits.size(0))
        out[f"HR@{k}"]   = hr
        out[f"NDCG@{k}"] = ndcg
    return out
# ---------------------------------------------------------------------------
# 4. Public API
# ---------------------------------------------------------------------------

def compute_loss(
    batch: Dict,
    model_out: Dict,
    model,
    cfg: Dict,
    *,
    strategy: str = "everysess_allinter",
    log_pos_metrics: bool = False,
    candidate_emb: str = "ID_64",
) -> tuple[torch.Tensor, Dict[str, Dict[str, torch.Tensor]]]:
    """
    Return
    -------
    loss_bp : torch.Tensor
        CE loss (back-prop 대상) – strategy 마스크만 사용
    metrics : Dict[label → Dict]
        label ∈ {'ALL', 'sess_F', ...}  ─ HR@1/5/10 + loss
    """
    if cfg["model"] == "myModel" or cfg["model"] == "H-RNN" or cfg["model"] == "HierTCN":
        return mymodel_loss(
            batch, model_out, model, cfg,
            strategy=strategy, log_pos_metrics=log_pos_metrics,
            candidate_emb=candidate_emb,
            model_name = cfg["model"]
        )
    elif cfg["model"] == "NARM" or cfg["model"] == "SR-GNN" or cfg["model"] == "GAG" or cfg["model"] == "HG-GNN":
        return narm_loss(
            batch, model_out, model, cfg,
            candidate_emb=candidate_emb,
            model_name = cfg["model"]
        )
    elif cfg["model"] == "TiSASRec":
        return tisasrec_loss(
            batch, model_out, model, cfg,
            strategy=strategy, candidate_emb=candidate_emb,
            model_name = cfg["model"]
        )
    else:
        raise ValueError(f"Unknown model_name '{cfg['model']}'. Use 'myModel'.")
    
def narm_loss(
    batch: Dict,
    model_out: Dict,
    model,
    cfg: Dict,
    log_pos_metrics: bool = False,
    candidate_emb: str = "ID_64",
    model_name: str = "NARM",
) -> tuple[torch.Tensor, Dict[str, Dict[str, torch.Tensor]]]:
    """
    NARM forward 가 반환한
        model_out = {"reps": c_t,           # [B', D_rep]
                     "target": target_id}   # [B']
    을 받아 Cross-Entropy + HR/NDCG 계산.

    * strategy/bin 개념 없이 "ALL" 한 묶음만 리턴.
    """
    reps    = model_out["reps"]          # [B', D]
    tgt_id  = model_out["target"]        # [B']
    device  = reps.device
    freq_t = cfg["item_freq"].to(device) 
    threshold = cfg["pop_threshold"].to(device)
    # ───────────────────── 1. Negative 샘플 생성 ──────────────────────
    k_neg = cfg['num_neg']

    sampler = cfg["negSampler"]
    cand_ids = sampler.sample(tgt_id, k=k_neg)      # [B', k]

    # ───────────────────── 2. 로짓 계산  ─────────────────────────────
    logits = _calc_logits(
        reps, cand_ids, model, candidate_emb,
        model_name=model_name,  # type hinting
        k_neg=k_neg     # dt_sel·add_sel·pos_sel 필요 X
    ).float()                                            # [B', k]

    # 타깃은 항상 cand_ids[:,0] 이므로 index 0
    target_pos = torch.zeros_like(tgt_id, dtype=torch.long)

    # CE loss (back-prop 대상)
    loss_bp = F.cross_entropy(logits, target_pos, reduction="mean")

    # ───────────────────── 3. HR/NDCG 메트릭 ─────────────────────────
    metric_all = _calc_metrics(logits, target_pos, ks=(1, 5, 10))
    metric_all["loss"] = loss_bp.detach()
    
    # ───────── 추가 HR 분리 로깅 ─────────
    metrics_dict = {"ALL": metric_all}

    if log_pos_metrics:
        # ① u_type 별 HR ------------------------------------------------
        u_type_flat = model_out.get("u_type", None)  # [B'] 또는 None
        if u_type_flat is not None:
            for t in torch.unique(u_type_flat):
                mask = u_type_flat == t
                if mask.any():
                    metrics_dict[f"u{int(t)}"] = _calc_metrics(
                        logits[mask], target_pos[mask], ks=(1, 5, 10)
                    )

        # ② 아이템 인기별 HR (상위 20% = hot 예시) -----------------------
        item_pop = freq_t[tgt_id]                         # [B']
        mask_hot, mask_cold = item_pop >= threshold, item_pop < threshold
        if mask_hot.any():
            metrics_dict["hot"] = _calc_metrics(
                logits[mask_hot], target_pos[mask_hot], ks=(1, 5, 10)
            )
        if mask_cold.any():
            metrics_dict["cold"] = _calc_metrics(
                logits[mask_cold], target_pos[mask_cold], ks=(1, 5, 10)
            )
    
    return loss_bp, metrics_dict

def mymodel_loss(
    batch: Dict,
    model_out: Dict,
    model,
    cfg: Dict,
    *,
    strategy: str = "everysess_allinter",
    log_pos_metrics: bool = False,
    candidate_emb: str = "ID_64",
    model_name: str = "myModel",  # type hinting
) -> tuple[torch.Tensor, Dict[str, Dict[str, torch.Tensor]]]:
    # ───────────────────────── 0. 기본 꺼내기 ──────────────────────────
    reps      = model_out["reps"]                   # [B,S,L,D_rep]
    mask_v    = model_out["loss_masks"].bool()      # [B,S,L]
    item_ids  = batch["item_id"]                    # [B,S,I]
    delta_ts  = batch["delta_ts"]                    # [B,S,I]
    add_info  = batch.get("add_info", None)         # [B,S,I,A] or None
    u_type = batch["u_type"] 
    freq_t = cfg["item_freq"].to(reps.device)
    pop_th = cfg["pop_threshold"].to(reps.device)

    B, S, I = item_ids.shape
    L       = reps.size(2)
    device  = reps.device

    # pad to L (only if I < L)
    pad_cols = (0, L - I)
    gts  = F.pad(item_ids, pad_cols)                # [B,S,L]
    dts  = F.pad(delta_ts, pad_cols)
    if add_info is not None:
        add_info = F.pad(add_info, (0, 0, *pad_cols))

    # positional idx tensor [B,S,L]
    pos_full = torch.arange(L, device=device).view(1, 1, L).expand(B, S, L)
    if cfg["model"] == "H-RNN":
        # 1) next-item 타깃으로 시프트
        gts      = gts.roll(-1, dims=2)
        dts      = dts.roll(-1, dims=2)
        pos_full = pos_full.roll(-1, dims=2)
        if add_info is not None:
            add_info = add_info.roll(-1, dims=2)

        # 2) 마지막 컬럼에는 next-item이 없으므로 loss 마스크에서 제외
        mask_v = mask_v & mask_v.roll(-1, dims=2)
    # ───────────────────────── 1. 마스크 준비 ─────────────────────────
    eval_from = batch["eval_from"].to(device)             # [B]
    sess_idx   = torch.arange(S, device=device).view(1, S, 1)  
    allow_sess = sess_idx >= eval_from.view(B, 1, 1)      # [B,S,1]
    mask_v     = mask_v & allow_sess                      # [B,S,L]
    train_mask = _STRATEGY_FN[strategy](mask_v)

    pos_masks  = make_position_masks(mask_v) if log_pos_metrics else {}

    # ───────────────────────── 2. Sampler ────────────────────────────
    k_neg    = cfg['num_neg']
    sampler = cfg["negSampler"]

    # ───────────────────────── helper view ───────────────────────────
    def _flat(t):  # view keep contiguous
        return t.reshape(-1, *t.shape[3:]) if t.dim() > 2 else t.reshape(-1)

    # ───────────────────────── 3. 학습용 (backward) ───────────────────
    idx_tr   = train_mask.view(-1)
    reps_tr  = _flat(reps)[idx_tr]
    tgt_tr   = _flat(gts)[idx_tr]
    dt_tr    = _flat(dts)[idx_tr]
    pos_tr   = _flat(pos_full)[idx_tr]
    add_tr   = _flat(add_info)[idx_tr] if add_info is not None else None

    neg_tr   = sampler.sample(tgt_tr, k_neg)        # [N,k]
    logit_tr = _calc_logits(
        reps_tr, neg_tr, model, candidate_emb,
        model_name = model_name,
        dt_sel=dt_tr, add_sel=add_tr, pos_sel=pos_tr, k_neg=k_neg
    ).float()

    targets_tr = torch.zeros(logit_tr.size(0), dtype=torch.long, device=device)
    loss_bp    = F.cross_entropy(logit_tr, targets_tr, reduction="mean")

    # ───────────────────────── 4. metrics (ALL + bins) ───────────────
    metrics: Dict[str, Dict[str, torch.Tensor]] = {}

    def _make_metrics(idx_mask, label: str):
        reps_x = _flat(reps)[idx_mask]
        tgt_x  = _flat(gts)[idx_mask]
        dt_x   = _flat(dts)[idx_mask]
        pos_x  = _flat(pos_full)[idx_mask]
        add_x  = _flat(add_info)[idx_mask] if add_info is not None else None

        neg_x  = sampler.sample(tgt_x, k_neg)
        log_x  = _calc_logits(
            reps_x, neg_x, model, candidate_emb,
            model_name = model_name,
            dt_sel=dt_x, add_sel=add_x, pos_sel=pos_x, k_neg=k_neg
        ).float()

        tar_x  = torch.zeros_like(tgt_x, dtype=torch.long)
        md     = _calc_metrics(log_x, tar_x, ks=(1, 5, 10))
        md["loss"] = F.cross_entropy(log_x, tar_x).detach()
        metrics[label] = md

    # ALL
    _make_metrics(mask_v.view(-1), "ALL")

    # 위치-bin
    if log_pos_metrics:
        for lbl, msk in pos_masks.items():
            if msk.any():
                _make_metrics(msk.view(-1), lbl)
        # ── ① u_type 별 HR ───────────────────────────
        u_full = u_type.view(-1, 1, 1).expand_as(mask_v)   # [B,S,L]
        u_flat = u_full.reshape(-1)
        for t in torch.unique(u_type):
            m = mask_v.view(-1) & (u_flat == t)
            if m.any(): 
                _make_metrics(m, f"u{int(t)}")

        # ── ② 아이템 인기별 HR (상위 20% = hot 예시) ─────
        gt_flat = _flat(gts)            # [B*S*L]
        pop = freq_t[gt_flat]           # [N]
        mask_hot  = mask_v.view(-1) & (pop >= pop_th)
        mask_cold = mask_v.view(-1) & (pop <  pop_th)
        if mask_hot.any():
            _make_metrics(mask_hot,  "hot")
        if mask_cold.any():
            _make_metrics(mask_cold, "cold")

    return loss_bp, metrics
def nan_check(tensor: torch.Tensor, name: str):
    """Check if tensor contains NaN values and print a warning."""
    if torch.isnan(tensor).any():
        print(f"Warning: NaN detected in {name} tensor!")
        print(tensor[torch.isnan(tensor)])  # Print the NaN values for debugging
        return True
    return False
def tisasrec_loss(
    batch: Dict,
    model_out: Dict,
    model,
    cfg: Dict,
    *,
    strategy: str = "allseq",      # "allseq" | "lastonly"
    log_pos_metrics: bool = False, # 위치·길이 bin HR/NDCG 로깅 여부
    candidate_emb: str = "ID_64",
    model_name: str = "TiSASRec",  # type hinting
):
    """
    TiSASRec forward 가 반환한
        reps       : [B, L, D]
        loss_masks : [B, L]  (1 = 유효 토큰, 첫 토큰은 0)
    로부터 h_t → item_{t+1} 예측을 학습/평가.
    위치·길이별 메트릭(log_pos_metrics=True)도 함께 계산.
    """
    # ────────── 0. 기본 꺼내기 ────────────────────────────────────
    reps   = model_out["reps"]                   # [B,L,D]
    mask_v = model_out["loss_masks"].bool()      # [B,L]
    u_type = model_out["u_type"]     # [B,L]
    seq    = model_out["seq"] 
    device = reps.device
    freq_t = cfg["item_freq"].to(device) 
    threshold = cfg["pop_threshold"].to(device) # scalar
    B, L, D = reps.shape

    # ── 예측용 mask  (h_t 로 t+1 ID 예측) ─────────────────────────────
    sel_mask = mask_v.clone().roll(-1, dims=1)  # True: t+1 아이템 존재
    tgt_mask = mask_v.clone()            # True: h_t 사용
    #sel_mask &= tgt_mask                 # 마지막 토큰 제외
    if strategy == "lastonly":
        sel_mask.zero_()
        tgt_mask.zero_()
        last = mask_v.sum(1) - 2         # 마지막-1 위치
        sel_mask[torch.arange(B), last] = True
        tgt_mask[torch.arange(B), last+1] = True   # 타깃은 마지막 아이템
    flat   = lambda x: x.view(-1)               # 편의 람다
    reps_sel   = reps.view(-1, D)[flat(sel_mask)]   # [N,D]
    tgt_sel    =   seq.view(-1)[flat(tgt_mask)]     # [N]
    u_type_sel = u_type.view(-1)[flat(sel_mask)]    # [N]  ← metrics용

    # ────────── 2. flatten & 로짓 계산 ───────────────────────────
    k_neg    = cfg['num_neg']
    sampler = cfg["negSampler"]
    cand_id = sampler.sample(tgt_sel, k=k_neg)              # [N,k]
    logits  = _calc_logits(
        reps_sel, cand_id, model, "ID_64", model_name=model_name,
        k_neg=k_neg
    ).float()

    targets = torch.zeros_like(tgt_sel, dtype=torch.long)   # 정답은 cand_id[:,0]
    loss_bp = F.cross_entropy(logits, targets, reduction="mean")

    # ────────── 3. 기본 메트릭 ────────────────────────────────────
    metrics = {"ALL": _calc_metrics(logits, targets, ks=(1, 5, 10))}
    metrics["ALL"]["loss"] = loss_bp.detach()

    # ────────── 4. 위치·길이 bin 메트릭 (선택) ────────────────────
    log_pos_metrics = True
    if log_pos_metrics:
        # 길이-버킷 flag  [B]
        len_bins = [(0,2), (3,5), (6,10), (11, 20), (21, 1_000_000)]
        len_labels = ["all"] + [f"len_{lo}to{hi if hi!=1_000_000 else 'plus'}"
                                for lo,hi in len_bins]
        len_flags = [torch.ones(B, dtype=torch.bool, device=device)]  # 'all'
        seq_len   = mask_v.sum(1)                                     # [B]
        for lo,hi in len_bins:
            len_flags.append((seq_len>=lo)&(seq_len<=hi))

        # 절대 위치 idx [L] , 상대 위치 ratio [B,L]
        idx = torch.arange(L, device=device)
        rel = idx.unsqueeze(0) / seq_len.unsqueeze(1).clamp(min=1)    # [B,L]
        rel = (rel / 0.05).round() * 0.05

        for len_lbl, len_flag in zip(len_labels, len_flags):
            base_mask = sel_mask & len_flag.unsqueeze(1)             # [B,L]
            if base_mask.sum()==0:
                continue

            # ─ 절대 위치 bin (1-10, 11+)
            for i in range(10):
                m = base_mask & (idx==i)
                if m.any():
                    _add_metric(metrics, reps, seq, m, sampler, model,
                                 candidate_emb, k_neg,
                                 f"{len_lbl}_abs_{i+1}")
            m = base_mask & (idx>=10)
            if m.any():
                _add_metric(metrics, reps, seq, m, sampler, model,
                             candidate_emb, k_neg,
                             f"{len_lbl}_abs_11plus")
            m = base_mask
            if m.any():
                _add_metric(metrics, reps, seq, m, sampler, model,
                             candidate_emb, k_neg,
                             f"{len_lbl}_all")
            # ─ 상대 위치 bin (0.05 간격 0.05~0.95)
            for step in torch.arange(0.05, 1.0, 0.05, device=device):
                lbl = f"{len_lbl}_rel_{step:.2f}".replace("0.","")
                m = base_mask & (rel==step)
                if m.any():
                    _add_metric(metrics, reps, seq, m, sampler, model,
                                 candidate_emb, k_neg, lbl)
        # ────────── 3-B  u_type / 인기별 HR ────────
        if u_type_sel is not None:
            for t in torch.unique(u_type_sel):
                m = u_type_sel == t
                if m.any():
                    metrics[f"u{int(t)}"] = _calc_metrics(logits[m], targets[m], ks=(1,5,10))

        item_pop = freq_t[targets]        # [N]
        hot_mask  = item_pop >= threshold
        cold_mask = item_pop <  threshold
        if hot_mask.any():
            metrics["hot"]  = _calc_metrics(logits[hot_mask],  targets[hot_mask],  ks=(1,5,10))
        if cold_mask.any():
            metrics["cold"] = _calc_metrics(logits[cold_mask], targets[cold_mask], ks=(1,5,10))
    return loss_bp, metrics


# ──────────  helper : selected-mask metric  ──────────
def _add_metric(metrics_dict, reps_full, seq_full, sel_mask,
                sampler, model, cand_emb, k_neg, label):
    reps_b = reps_full[sel_mask]
    tgt_b  = seq_full.roll(-1,1)[sel_mask]
    cand_b = sampler.sample(tgt_b, k=k_neg)
    log_b  = _calc_logits(reps_b, cand_b, model, cand_emb, model_name='TiSASRec', k_neg=k_neg).float()
    tar_b  = torch.zeros_like(tgt_b, dtype=torch.long)
    md     = _calc_metrics(log_b, tar_b, ks=(1,5,10))
    md["loss"] = F.cross_entropy(log_b, tar_b).detach()
    metrics_dict[label] = md