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

# ────── item 인기(Train frequency) 버킷 ──────
FREQ_BINS = {
    "0"       :( -1,      0),
    "1"       :(  0,      1),
    "2-4"     :(  1,      4),
    "5-10"    :(  4,     10),
    "11-30"   :( 10,     30),
    "31-100"  :( 30,    100),
    "101-300" :(100,    300),
    "301-1000":(300,   1000),
    "1001+"   :(1000,  np.inf),
}
U_TYPE_REV = {         # 6-type 정수 코드 (원하는 순서대로 조정 가능)
    0:"cold_bad" ,
    1:"cold_good",
    2:"few_bad"  ,
    3:"few_good" ,
    4:"warm_bad" ,
    5:"warm_good",
}
def freq_bin(c):                     # c = train frequency
    for lbl,(lo,hi) in FREQ_BINS.items():
        if lo < c <= hi: return lbl
    return "0"

# ────── 세션/인터랙션 위치 버킷 ──────
BIN_LABELS = ["1","2","3","4","5","6","7","8","9","10",
              "11-30","31-100","101-500","500+"]
def pos_bin(idx):
    print(idx)
    if idx < 10:               return str(idx+1)
    if idx < 30:               return "11-30"
    if idx < 100:              return "31-100"
    if idx < 500:              return "101-500"
    return "500+"
_BOUNDS = torch.tensor([1,2,3,4,5,6,7,8,9,10,30,100,500], dtype=torch.long)

def bucket_labels(idxs: torch.Tensor) -> torch.Tensor:
    """
    idxs : [N]  (long)  ─ 세션 번호 또는 인터랙션 위치
    return : [N]  (long)  ─ BIN_LABELS 의 인덱스(0~len-1)
    """
    # torch.bucketize : 주어진 경계보다 '작은' 버킷 인덱스 반환
    # ex) idx=0 → 0, idx=5→4, idx=12→10, idx=600→13
    return torch.bucketize(idxs, _BOUNDS.to(idxs.device), right=True)

def collect_metrics(
    logits:  torch.Tensor,            # [N,k]
    target:  torch.Tensor,            # [N] (정답=0)
    u_type:  torch.Tensor | None,     # [N] or None
    item_id: torch.Tensor | None,     # [N] (tgt id) or None
    item_freq: torch.Tensor | None,   # [max_id+1]
    sess_idx: torch.Tensor | None,    # [N] 세션 번호 or None (0-base)
    intr_idx: torch.Tensor | None,    # [N] 세션 내 위치 or None (0-base)
    ks=(1,5,10)
) -> dict[str, dict]:
    """필요한 입력만 None 아니게 넣어주면 해당 버킷별 HR/NDCG 반환."""
    N         = logits.size(0)
    base      = _calc_metrics(logits, target, ks)
    base["loss"] = F.cross_entropy(logits, target).detach()
    outputs   = {"ALL": base}

    # ── helper ─────────────────────────────────────────────────────
    def _add(mask: torch.Tensor, name: str):
        if mask.any():
            outputs[name] = _calc_metrics(logits[mask], target[mask], ks)

    # 1. u_type
    if u_type is not None:
        for t in torch.unique(u_type):
            _add(u_type==t, f"{U_TYPE_REV[int(t)]}")

    # 2. item popularity
    if item_id is not None and item_freq is not None:
        freq_vals = item_freq[item_id]            # [N]
        for lbl,(lo,hi) in FREQ_BINS.items():
            m = (freq_vals>lo)&(freq_vals<=hi)
            _add(m, f"freq_{lbl}")

    # 3. session idx (multi-session 모델만)
    if sess_idx is not None:
        sess_bins = bucket_labels(sess_idx)   # [N] 0~13
        for i, lbl in enumerate(BIN_LABELS):
            _add(sess_bins == i, f"sess_{lbl}")
    # 4. intra-session pos
    if intr_idx is not None:
        pos_bins  = bucket_labels(intr_idx)
        for i, lbl in enumerate(BIN_LABELS):
            _add(pos_bins == i, f"pos_{lbl}")
    return outputs

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
            log_pos_metrics = log_pos_metrics,
            candidate_emb=candidate_emb,
            model_name = cfg["model"]
        )
    elif cfg["model"] == "TiSASRec":
        return tisasrec_loss(
            batch, model_out, model, cfg,
            log_pos_metrics=log_pos_metrics,
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
    if log_pos_metrics:
        metrics_dict = collect_metrics(
        logits, target_pos,
        u_type      = model_out.get("u_type"),        # [N] or None
        item_id     = tgt_id,
        item_freq   = cfg["item_freq"].to(device),
        sess_idx    = None,                           # single-session
        intr_idx    = None                            # 1 step 예측
    )
    else:
        metrics_dict = {"ALL": metric_all}
    
    return loss_bp, metrics_dict

# ─────────────────────────────────────────────────────────────────────────────
def mymodel_loss(
    batch: Dict, model_out: Dict, model, cfg: Dict,
    *, strategy: str = "everysess_allinter",
       log_pos_metrics: bool = False,
       candidate_emb: str = "ID_64",
       model_name: str = "myModel",
) -> tuple[torch.Tensor, Dict[str, Dict[str, torch.Tensor]]]:

    # ── 0. tensor 꺼내기 ──────────────────────────────────────────────
    reps      = model_out["reps"]            # [B,S,L,D]
    loss_mask = model_out["loss_masks"].bool()   # [B,S,L]
    item_ids  = batch["item_id"]             # [B,S,I]
    delta_ts  = batch["delta_ts"]
    add_info  = batch.get("add_info", None)  # [B,S,I,A] or None
    u_type    = batch["u_type"]              # [B,S]
    device    = reps.device
    B, S, I   = item_ids.shape
    L         = reps.size(2)
    k_neg     = cfg["num_neg"]
    sampler   = cfg["negSampler"]

    # ── 1. pad → [B,S,L]  (I ≤ L) ───────────────────────────────────
    pad_cols  = (0, L - I)
    gts       = F.pad(item_ids, pad_cols)
    dts       = F.pad(delta_ts, pad_cols)
    if add_info is not None:
        add_info = F.pad(add_info, (0, 0, *pad_cols))

    # ── 2. next-item shift for H-RNN ─────────────────────────────────
    if cfg["model"] == "H-RNN":
        #print("H-RNN: shifting item_ids, delta_ts, add_info, loss_mask")
        #print(f"item_ids.shape: {gts.shape}, delta_ts.shape: {dts.shape}, add_info.shape: {add_info.shape if add_info is not None else None}, loss_mask.shape: {loss_mask.shape}")
        #print(f"item_ids before shift: {gts[0,0,:10]},{gts[0, 0, -10:]}, delta_ts before shift: {dts[0,0,:10]},{dts[0,0,-10:]}, loss_mask before shift: {loss_mask[0,0,:10]},{loss_mask[0,0,-10:]}")
        gts = gts.roll(-1, dims=2)
        dts = dts.roll(-1, dims=2)
        if add_info is not None:
            add_info = add_info.roll(-1, dims=2)
        loss_mask &= loss_mask.roll(-1, dims=2)
        loss_mask[..., -1] = False
        gts[... , -1] = -1
        dts[... , -1] = 0
        #print(f"item_ids after shift: {gts[0,0,:10]},{gts[0, 0, -10:]}, delta_ts after shift: {dts[0,0,:10]},{dts[0,0,-10:]}, loss_mask after shift: {loss_mask[0,0,:10]},{loss_mask[0,0,-10:]}")

    # ── 3. strategy별 학습 마스크 ────────────────────────────────────
    eval_from = batch["eval_from"].to(device)     # [B]
    sess_idx  = torch.arange(S, device=device).view(1, S, 1)
    allow     = sess_idx >= eval_from.view(B, 1, 1)
    loss_mask &= allow                           # [B,S,L]
    train_mask = _STRATEGY_FN[strategy](loss_mask)   # [B,S,L]

    # ── 4. flatten util ─────────────────────────────────────────────
    flat = lambda t: t.reshape(-1, *t.shape[3:]) if t.dim() > 2 else t.reshape(-1)

    idx_tr   = train_mask.view(-1)
    reps_tr  = flat(reps)[idx_tr]
    tgt_tr   = flat(gts)[idx_tr]
    dt_tr    = flat(dts)[idx_tr]
    pos_tr   = torch.arange(L, device=device).repeat(B, S).view(-1)[idx_tr]
    add_tr   = flat(add_info)[idx_tr] if add_info is not None else None

    # ── 5. 학습 로짓 & CE ───────────────────────────────────────────
    neg_tr   = sampler.sample(tgt_tr, k=k_neg)
    logit_tr = _calc_logits(
        reps_tr, neg_tr, model, candidate_emb,
        model_name = model_name,
        dt_sel=dt_tr, add_sel=add_tr, pos_sel=pos_tr, k_neg=k_neg
    ).float()

    loss_bp  = F.cross_entropy(logit_tr,
                               torch.zeros_like(tgt_tr, dtype=torch.long),
                               reduction="mean")

    # ── 6. 메트릭 계산 (collect_metrics 활용) ───────────────────────
    metrics: Dict[str, Dict[str, torch.Tensor]]

    if not log_pos_metrics:
        # 전체 성능만
        metrics = {
            "ALL": {
                **_calc_metrics(logit_tr,
                                torch.zeros_like(tgt_tr, dtype=torch.long),
                                ks=(1,5,10)),
                "loss": loss_bp.detach()
            }
        }
        return loss_bp, metrics

    # (1) flatten 전체 valid 위치
    idx_all  = loss_mask.view(-1)
    reps_all = flat(reps)[idx_all]
    tgt_all  = flat(gts)[idx_all]
    dt_all   = flat(dts)[idx_all]
    pos_all  = torch.arange(L, device=device).repeat(B, S).view(-1)[idx_all]
    add_all  = flat(add_info)[idx_all] if add_info is not None else None

    neg_all  = sampler.sample(tgt_all, k=k_neg)
    logits   = _calc_logits(
        reps_all, neg_all, model, candidate_emb,
        model_name=model_name,
        dt_sel=dt_all, add_sel=add_all, pos_sel=pos_all, k_neg=k_neg
    ).float()

    targets  = torch.zeros_like(tgt_all, dtype=torch.long)

    # (2) 세션 번호·인터랙션 위치 인덱스 준비
    sess_full = sess_idx.expand(B, S, L).reshape(-1)[idx_all]  # [N]
    intr_full = pos_all                                        # [N]
    u_full    = u_type.view(-1,1,1).expand(B, S, L).reshape(-1)[idx_all]

    metrics = collect_metrics(
        logits, targets,
        u_type    = u_full,
        item_id   = tgt_all,
        item_freq = cfg["item_freq"].to(device),
        sess_idx  = sess_full,
        intr_idx  = intr_full,
    )

    return loss_bp, metrics


def nan_check(tensor: torch.Tensor, name: str):
    """Check if tensor contains NaN values and print a warning."""
    if torch.isnan(tensor).any():
        print(f"Warning: NaN detected in {name} tensor!")
        print(tensor[torch.isnan(tensor)])  # Print the NaN values for debugging
        return True
    return False

def tisasrec_loss(
    batch: Dict, model_out: Dict, model, cfg: Dict,
    *, strategy: str = "allseq",          # "allseq" | "lastonly"
       log_pos_metrics: bool = False,     # 위치·길이 bin 로깅 여부
       candidate_emb: str = "ID_64",
       model_name: str = "TiSASRec",
) -> tuple[torch.Tensor, Dict[str, Dict[str, torch.Tensor]]]:

    # ─────────────────── 0. 기본 꺼내기 ─────────────────────────────
    reps     = model_out["reps"]               # [B,L,D]
    mask_v   = model_out["loss_masks"].bool()  # [B,L]
    u_type   = model_out["u_type"]             # [B,L]
    seq_full = model_out["seq"]                # [B,L]
    device   = reps.device
    B, L, D  = reps.shape
    ks       = (1, 5, 10)

    # ─────────────────── 1. 선택·타깃 마스크 ────────────────────────
    sel_mask = mask_v.roll(-1, 1)              # h_t  (예측에 사용)
    tgt_mask = mask_v                          # item_{t+1} (정답)
    if strategy == "lastonly":
        sel_mask.zero_(); tgt_mask.zero_()
        last = mask_v.sum(1) - 2               # 마지막-1 위치
        sel_mask[torch.arange(B), last]     = True
        tgt_mask[torch.arange(B), last + 1] = True

    # ─────────────────── 2. flatten 선택 샘플 ───────────────────────
    flat = lambda x: x.view(-1)
    reps_sel = reps.view(-1, D)[flat(sel_mask)]          # [N,D]
    tgt_sel  = seq_full.view(-1)[flat(tgt_mask)]         # [N]
    u_sel    = u_type.view(-1)[flat(sel_mask)]           # [N]
    idx_full = torch.arange(L, device=device).repeat(B, 1)  # [B,L]
    idx_sel  = idx_full[sel_mask]                        # [N] ← 위치 인덱스

    # ─────────────────── 3. 로짓 & CE ──────────────────────────────
    k_neg   = cfg["num_neg"]
    sampler = cfg["negSampler"]
    cand_id = sampler.sample(tgt_sel, k=k_neg)           # [N,k]
    logits  = _calc_logits(
        reps_sel, cand_id, model, candidate_emb,
        model_name=model_name, k_neg=k_neg
    ).float()
    targets = torch.zeros_like(tgt_sel, dtype=torch.long)
    loss_bp = F.cross_entropy(logits, targets, reduction="mean")

    # ─────────────────── 4. 메트릭 집계 ────────────────────────────
    if not log_pos_metrics:                        # 전체만 필요
        metrics = {"ALL": _calc_metrics(logits, targets, ks)}
        metrics["ALL"]["loss"] = loss_bp.detach()
        return loss_bp, metrics

    # log_pos_metrics == True → collect_metrics 사용
    metrics = collect_metrics(
        logits, targets,
        u_type    = u_sel,                         # [N]
        item_id   = tgt_sel,                       # [N]
        item_freq = cfg["item_freq"].to(device),   # [V]
        sess_idx  = None,                          # 세션 개념 없음
        intr_idx  = idx_sel.long(),                # [N] (0-base 위치)
    )
    metrics["ALL"]["loss"] = loss_bp.detach()
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