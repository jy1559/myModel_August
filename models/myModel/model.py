from __future__ import annotations

import torch
import torch.nn as nn
from typing import Dict, List, Iterable
from time import time
import torch.nn.functional as F
# ---------------------------------------------------------------------------
# 0.  external helper: optional token‑wise MLP
# ---------------------------------------------------------------------------

def build_token_mlp(d_model: int, d_out: int) -> nn.Module:
    """Return a lightweight 2‑layer MLP (ReLU) that preserves dimension."""
    if d_out > 128:
        return nn.Sequential(
        nn.Linear(d_model, 256), nn.ReLU(inplace=True), nn.Linear(256, d_out)
    )
    else:
        return nn.Sequential(
        nn.Linear(d_model, 128), nn.ReLU(inplace=True), nn.Linear(128, d_out)
    )

    
# ---------------------------------------------------------------------------
# 1. local imports
# ---------------------------------------------------------------------------
from .input_embedding import InputEmbedding
from .session_encoder import SessionEncoder
from .user_update import UserStateUpdater

# ---------------------------------------------------------------------------
# 2.  Main Model
# ---------------------------------------------------------------------------
class SeqRecModel(nn.Module):
    """Session‑aware sequential recommender that outputs **token‑level hidden reps**.
    Item‑similarity / loss calculation is done externally.
    """

    # ---------------------------------------------------------------------
    def __init__(self, n_items: int, cfg: Dict):
        super().__init__()
        self.cfg = cfg
        d_model = cfg.get("embed_dim", 256)
        device  = cfg.get("device", "cpu")

        # ---- sub‑modules --------------------------------------------------
        self.embed   = InputEmbedding(n_items, **cfg)  # [B,S,I,D]
        self.encoder = SessionEncoder(
            d_model=d_model,
            n_heads=cfg.get("n_heads", 8),
            d_ff=cfg.get("d_ff", d_model * 4),
            n_layers=cfg.get("n_layers", 2),
            user_embedding=cfg.get("user_embedding", "none"),
            user_embedding_first=cfg.get("user_embedding_first", True),
            sum_token=cfg.get("sum_token", True),
            cls_token=cfg.get("cls_token", False),
            aggregation=cfg.get("aggregation", "last"),
            other_information=cfg.get("other_information", "none"),
            dropout=cfg.get("dropout", 0.1),
        )
        self.updater = UserStateUpdater(
            d_session=d_model,
            hidden_size=d_model,
            method=cfg.get("user_update_method", "default"),
            rnn=cfg.get("update_rnn", "GRU"),
            dt_emb_module=getattr(self.embed, "dt_emb", None),
            device=device,
        )

        # optional extra transformation before external scorer ----------------
        self.use_mlp = cfg.get("use_mlp", True)
        cand_emb = cfg.get("candidate_emb", "ID_64")
        d_out = int(cand_emb.split('_')[-1]) if "INPUT" not in cand_emb else d_model
        if self.use_mlp:
            self.token_mlp = build_token_mlp(d_model, d_out)

        # ------------------------------------------------------------------
        # parameter groups registry (freeze / unfreeze control)
        # ------------------------------------------------------------------
        self._param_groups: Dict[str, List[nn.Parameter]] = {}
        self._register_group("id_emb", self.embed.id_emb.parameters())
        if getattr(self.embed, "use_llm", True):
            self._register_group("llm_emb", self.embed.llm_emb.parameters())
        if getattr(self.embed, "use_dt", False):
            self._register_group("dt_emb", self.embed.dt_emb.parameters())
        if getattr(self.embed, "use_add", False):
            self._register_group("add_info_emb", self.embed.add_emb.parameters())
        self._register_group("encoder", self.encoder.parameters())
        self._register_group("user_rnn", (p for n, p in self.updater.named_parameters() if n != "initial_state"))
        self._register_group("user_init", (self.updater.initial_state,))
        if self.use_mlp:
            self._register_group("token_mlp", self.token_mlp.parameters())

    # ------------------------------------------------------------------
    # helper: param‑group registry
    # ------------------------------------------------------------------
    def _register_group(self, tag: str, params: Iterable[nn.Parameter]):
        params = list(params)
        if params:
            self._param_groups[tag] = params

    # ------------------------------------------------------------------
    # freeze / unfreeze utilities
    # ------------------------------------------------------------------
    def freeze(self, *tags: str, except_mode: bool = False):
        target = (set(self._param_groups) - set(tags)) if except_mode else set(tags)
        for tag in target:
            for p in self._param_groups.get(tag, []):
                p.requires_grad = False

    def unfreeze(self, *tags: str):
        for tag in tags:
            for p in self._param_groups.get(tag, []):
                p.requires_grad = True

    def set_train_strategy(self, name: str):
        name = name.lower()
        if name == "all":
            self.unfreeze(*self._param_groups)
        elif name == "user_only":
            self.freeze("user_rnn", "user_init", except_mode=True)
        elif name == "init_only":
            self.freeze("user_init", except_mode=True)
        elif name == "encoder_only":
            self.freeze("encoder", except_mode=True)
        elif name == "mlp_only":
            self.freeze("token_mlp", except_mode=True)
        else:
            raise ValueError(f"unknown strategy: {name}")

    def trainable_parameters(self):
        return (p for p in self.parameters() if p.requires_grad)

    # ------------------------------------------------------------------
    # forward pass  (returns hidden reps + loss masks)
    # ------------------------------------------------------------------
    def forward(self, batch: Dict, *, return_extra: bool = False):
        item_ids   = batch["item_id"]              # [B,S,I]
        delta_ts   = batch["delta_ts"]             # [B,S,I]
        int_mask   = batch["interaction_mask"]     # [B,S,I]
        add_info   = batch.get("add_info", None)   # [B,S,I,F]
        sess_gap   = delta_ts[:, :, 0]   # [B,S]
        sess_mask  = batch.get("session_mask", (int_mask.sum(-1) > 0).long())
        B, S, _ = item_ids.shape
        device  = item_ids.device

        sc = time()
        user_state = self.updater.reset_state(B, device)
        reps:  List[torch.Tensor] = []   # hidden representation per session (token‑level)
        masks: List[torch.Tensor] = []
        vecs:  List[torch.Tensor] = []   # session‑level vector (optional)
        input_emb = self.embed(
                item_ids, delta_ts, int_mask,
                None if add_info is None else add_info
            )
        """print(f'input_emb.time: {time() - sc:.3f}s')
        sc = time()"""
        for s in range(S):
            if sess_mask[:, s].sum() == 0:
                continue

            emb_s = input_emb[:, s]
            tok_hid, sess_vec, loss_mask = self.encoder(
                emb_s, int_mask[:, s],
                usr_embedding=user_state[0] if isinstance(user_state, tuple) else user_state[0],
            )
            """print(f'session {s+1}_encoder.time: {time() - sc:.3f}s')
            sc = time()"""
            if self.use_mlp:
                tok_hid = self.token_mlp(tok_hid)

            reps.append(tok_hid)   # [B,L,64] (self.use_mlp=True) or [B,L,256] 
            masks.append(loss_mask)
            vecs.append(sess_vec)

            gap = sess_gap[:, s] if sess_gap is not None else None
            lead_pad = int_mask[:, s, 0]
            """print(f'session {s+1}_additional.time: {time() - sc:.3f}s')
            sc = time()"""
            user_state = self.updater(sess_vec, user_state, dt_sec=gap, pad_mask=lead_pad)
            """print(f'session {s+1}_updater.time: {time() - sc:.3f}s')
            sc = time()"""

        L_max = max(t.size(1) for t in reps)
        pad   = lambda x,L: F.pad(x, (0,0,0,L-x.size(1)))         # right-pad
        reps  = torch.stack([pad(t,L_max) for t in reps], dim=1)   # [B,S,L,64] or [B,S,L,256]
        masks = torch.stack([pad(m,L_max) for m in masks], dim=1)  # [B,S,L]
        vecs  = torch.stack(vecs, dim=1)                          # [B,S,D]
        out   = {"reps": reps, "loss_masks": masks, "u_type": batch["u_type"]}
        if return_extra: out["session_repr"] = vecs
        return out
