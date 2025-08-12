# models/embedding.py
import math, torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
import numpy as np
import os

# ------------------------------------------------
# 0.  유틸: pad_mask 곱으로 패딩 0 보장
# ------------------------------------------------
def masked_linear(x: torch.Tensor, linear: nn.Linear, pad_mask4: torch.Tensor) -> torch.Tensor:
    """
    x : [..., in_dim]   pad_mask4 : [..., 1]  (0/1)
    bias=False 권장. bias=True라도 마지막에 곱해 0 → grad 차단.
    """
    out = linear(x)              # [.., out_dim]
    return out * pad_mask4.unsqueeze(-1)       # pad 위치 grad = 0

# ------------------------------------------------
# 1.  서브-임베더 클래스들
# ------------------------------------------------
class IDEmb(nn.Module):
    """아이템 ID 임베딩 (padding_id=0)"""
    def __init__(self, n_items:int, d:int=64):
        super().__init__()
        self.table = nn.Embedding(n_items+1, d, padding_idx=0)  # ← init

    def forward(self, item_ids: torch.Tensor) -> torch.Tensor:  # [B,S,I] → [...,d]
        return self.table(item_ids.clamp_min(0))               # pad=-1 → 0

class DeltaTEmb(nn.Module):
    """
    Δt 임베딩
    method = 'bucket' (로그 버킷) | 'linear' | 'time2vec'
    """
    def __init__(self, d:int=16, method:str='bucket',
                 num_bucket:int=32, bucket_size:float=1.0):
        super().__init__()
        self.method = method
        if method == 'bucket':
            self.bucket_size = bucket_size
            self.table = nn.Embedding(num_bucket, d, padding_idx=0)
        elif method == 'linear':
            self.proj  = nn.Linear(1, d, bias=False)           # ←
        elif method == 'time2vec':
            self.freq  = nn.Parameter(torch.randn(d//2))       # ←
            self.phase = nn.Parameter(torch.randn(d//2))
        else:
            raise ValueError(method)

    def forward(self, dt: torch.Tensor, pad_mask4: torch.Tensor) -> torch.Tensor:
        """
        dt: [B,S,I] float (0 for pad)
        pad_mask4: [B,S,I]
        """
        if self.method == 'bucket':
            bucket = (dt / self.bucket_size).clamp_min(1).log2().floor().clamp_min(0).clamp_max(self.table.num_embeddings-1).long()
            return self.table(bucket) * pad_mask4.unsqueeze(-1)
        elif self.method == 'linear':
            return masked_linear(dt.unsqueeze(-1), self.proj, pad_mask4)
        else:  # time2vec
            # sin/cos 주기
            x = dt.unsqueeze(-1)                                  # [...,1]
            sin = torch.sin(x * self.freq + self.phase)
            cos = torch.cos(x * self.freq + self.phase)
            out = torch.cat([sin, cos], -1)                       # [...,d]
            return out * pad_mask4

class AddInfoEmb(nn.Module):
    """다중 add-info (categorical or float) → 합산 벡터"""
    def __init__(self, specs:Tuple[Tuple[str,int]], d:int=16):
        """
        specs = (('cat', card1), ('num', 1), ...)
        """
        super().__init__()
        self.encoders = nn.ModuleList()
        for typ, sz in specs:
            if typ == 'cat':
                self.encoders.append(nn.Embedding(sz+1, d, padding_idx=0))
            elif typ == 'num':
                self.encoders.append(nn.Linear(1, d, bias=False))
            else:
                raise ValueError
        self.d = d

    def forward(self, add_info: list, pad_mask4: torch.Tensor) -> torch.Tensor:
        """
        add_info : [B,S,I,F]  (F == len(specs))
        """
        outs = []
        for i,(enc) in enumerate(self.encoders):
            if isinstance(enc, nn.Embedding):
                idx_max = add_info[..., i].max().item()
                feat = enc(add_info[..., i].long())
            else:  # Linear
                feat = masked_linear(add_info[...,i:i+1], enc, pad_mask4)
            outs.append(feat)
        return torch.stack(outs, -1).sum(-1) * pad_mask4.unsqueeze(-1)        # [...,d]

class LLMBasedEmb(nn.Module):
    """사전 계산된 384-d → Linear proj128"""
    def __init__(self, npz_path:str, d:int=128, device='cpu'):
        super().__init__()
        item_embedding = np.load(npz_path)["embeds"] 
        data = torch.as_tensor(item_embedding, device=device)  # {'embedding_tensor':Tensor[N+1,384]}
        self.llm_tbl = nn.Embedding.from_pretrained(data, freeze=True, padding_idx=0)
        input_dim = 384
        self.proj = nn.Linear(input_dim, d, bias=False)      # ←

    def forward(self, item_ids: torch.Tensor, pad_mask4: torch.Tensor) -> torch.Tensor:
        vec = self.llm_tbl(item_ids.clamp_min(0))
        out = self.proj(vec)              # [B,S,I,128]
        return out * pad_mask4.unsqueeze(-1)  # pad 위치 grad = 0

# ------------------------------------------------
# 2. Positional Encoding 모듈
# ------------------------------------------------
class IntraPos(nn.Module):
    def __init__(self, d:int=256, max_len:int=256, method:str='learn'):
        super().__init__()
        self.method = method
        self.max_len = max_len
        if method == 'learn':
            self.pe = nn.Embedding(max_len, d, padding_idx=0)
        else:   # sinusoidal
            pe = torch.zeros(max_len, d)
            pos = torch.arange(0,max_len).unsqueeze(1)
            div = torch.exp(torch.arange(0,d,2)*(-math.log(1e4)/d))
            pe[:,0::2] = torch.sin(pos*div); pe[:,1::2]=torch.cos(pos*div)
            self.register_buffer('pe_buf', pe)         # [L,d]

    def forward(self, idx_or_len: torch.Tensor | int, device) -> torch.Tensor:  # -> [length,d]
        if isinstance(idx_or_len, int):
            pos_idx = torch.arange(idx_or_len, device=device or 'cpu')  # [L]
        else:                       # tensor 이미 들어온 경우
            pos_idx = idx_or_len.to(device or idx_or_len.device)        # [N] or [...]

        if self.method=='learn':
            pos_idx = pos_idx.clamp_max(self.max_len - 1)  # 0 pad
            pe = self.pe(pos_idx)
            return pe
        else:
            assert "아직 Sinusoida은 고쳐야 함. 원래 length만 받았는데 loss구할 때 candidate의 PE 위해 각 다른 pos에 대해 구해야 할 때가 있음"
            return self.pe(pos_idx) 

# ------------------------------------------------
# 3.  최종 입력 임베딩 Composer
# ------------------------------------------------
class InputEmbedding(nn.Module):
    """
    Args
    ----
    cfg : Dict      실험 설정
       cfg['use_llm']: bool
       cfg['add_info_specs']: [('cat',card),('num',1),...]
    """
    def __init__(self, n_items:int, **cfg):
        super().__init__()
        self.id_emb = IDEmb(n_items, d=64)

        # Δt
        self.use_dt = bool(cfg.get('use_dt', True))
        self.dt_emb = DeltaTEmb(d=16, method=cfg.get('dt_method','bucket'),
                                num_bucket=cfg.get('num_bucket',32),
                                bucket_size=cfg.get('bucket_size',1.0))

        # add_info
        self.use_add = bool(cfg.get('use_add_info', False))
        if self.use_add:
            self.add_emb = AddInfoEmb(cfg['add_info_specs'], d=16)

        # LLM
        dataset_folder = cfg.get('dataset_folder', '/home/jy1559/Datasets')
        self.use_llm = cfg.get('use_llm', True)
        if self.use_llm:
            self.llm_emb = LLMBasedEmb(os.path.join(dataset_folder, cfg['dataset_name'], 'timesplit', cfg['sampling_N'], "item_embeddings.npz"), d=128, device=cfg['device'])

        # Projection
        d_concat = 64 + (16 if self.use_dt else 0) + (16 if self.use_add else 0) + (128 if self.use_llm else 0)
        self.proj = nn.Linear(d_concat, cfg.get('embed_dim',256), bias=False)

        # Positional
        self.pos_enc = IntraPos(d=cfg.get('embed_dim',256),
                                max_len=cfg.get('max_len',128),
                                method=cfg.get('pe_method','learn'))

    # ---------- forward ----------
    def forward(self,
                item_ids: torch.Tensor,       # [B,S,I]  (-1 pad)
                delta_ts: torch.Tensor,       # [B,S,I]  (float, 0 pad)
                interaction_mask: torch.Tensor,
                add_info: Optional[list]=None,  # [B,S,I,F] or None
                pos_idx=None
               ) -> torch.Tensor:
        """
        return : embed [B,S,I, d_model]  (pad 위치 0)
        """
        if pos_idx == None:
            B,S,I = item_ids.shape
        else: I = pos_idx[:, 0].max().item() + 1
        device= item_ids.device
        pad_mask4 = interaction_mask

        emb_list = [
        ]
        emb_list.append(self.id_emb(item_ids))
        if self.use_dt:
            emb_list.append(self.dt_emb(delta_ts, pad_mask4))         # 16
        if self.use_add:
            emb_list.append(self.add_emb(add_info, pad_mask4))     # 16
        if self.use_llm:
            emb_list.append(self.llm_emb(item_ids, pad_mask4))      #128

        concat = torch.cat(emb_list, dim=-1)          # [B,S,I,d_concat]
        proj   = self.proj(concat)       # [B,S,I,256] pad=0

        # Positional: broadcast Add
        if pos_idx is None:
            pos = self.pos_enc(I, device).view(1,1,I,-1)
        else:
            pos = self.pos_enc(I, device)[pos_idx]
        proj = (proj + pos) * pad_mask4.unsqueeze(-1)  # 다시 pad=0 보장
        return proj
