
class self_attention:
    def __init__(self, **kwargs):
        """
        method["user_embedding"]: 'none', 'additive_bias', 'cross_attention', 'FiLM'
            이게 addtiva_bias나 cross_attention이면 attention에서 user embedding을 활용
            additive_bias면 forward에서만 input의 kwargs에서 user_embedding 받아서 bias로 사용
            cross_attnetion이면 일부 head에서 user embedding을 query로 사용 -> 따로 Wq를 미리 정의해야 함, forward에서도 kwargs 받아서 활용용
        method["other_information"]: 'none', 'TiSASRec', 'DIF-SR'
            이게 'TiSASRec'면 attention에서 delta_t를 Key에 learnable matrix 곱해서 더해줌 -> init에서 Wk_delta_t 정의 필요 + forard에서 input의 kwargs에서 delta_t 받아서 곱해줌
            'DIF-SR'면 attention에서 Key에 add_info를 concat해준 다음 Wk와 곱함 -> forward에서 add_info 받아서 concat해줘야함
        """
        pass
    def forward(self, x):
        """
        Applies self-attention to the input tensor.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_length, embedding_dim].
        
        Returns:
            torch.Tensor: Output tensor after applying self-attention.
        """
        # Placeholder for self-attention logic
        return x

class FFN:
    def __init__(self, d_model, d_ff, dropout=0.1):
        """
        d_model: input dimension
        d_ff: feed-forward dimension
        dropout: dropout rate
        """
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = dropout

    def forward(self, x):
        """
        Applies feed-forward network to the input tensor.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_length, d_model].
        
        Returns:
            torch.Tensor: Output tensor after applying feed-forward network.
        """
        # Placeholder for FFN logic
        return x
    
class aggregator:
    def __init__(self, method):
        """
        Method 따라 input을 처리하는 함수.
        method: 'last', 'sum_attn_out', 'sum_ffn_out', 'weighted', 'adaptive_attn', 'adaptive_ffn'
        코드 복잡해질 수 있으니까  'adaptive_attn', 'adaptive_ffn'은 나중에 사용할 때 구현현
        """
        self.method = method

    def forward(self, attention_out, ffn_out, valid_mask):
        return 
    

def preprocess_encoder_input(session, usr_embedding, user_embedding_first = True, sum_token = None):
    """
    user_embedding_first: user embedding을 session 앞에 추가할 지 여부
    session: [B, I, d]
    usr_embedding: [B, d]
    sum_token: [d] or None
    user_embedding 은 user_embedding_first가 True면 session 앞에 추가
    sum_token 있으면 마지막에 추가

    [B, I+0~2, d] 형태로 반환 (조건에 따라 I+0~2)
    """
    

class session_encoder:
    def __init__(self, 
                 method = {
                        'user_embedding': 'first',
                        'sum_token': True,
                        'user_embedding_first': True,
                        'aggregation': 'last',
                        'other_information': None
                 }, 
                 **kwargs):
        """
        method
        user_embedding: user embedding 활용 방법
            none: 따로 추가적인 방법 사용X
            additive_bias: attention에서 additive bias로 사용
            cross_attention: cross attention 사용 (일부 head에서 user embedding을 query로 사용)
            FiLM: MLP 2개 (a, b) 생성 후 a(u)⊙output + b(u)로 결과 게이팅
        sum_token: session 마지막에 sum token 추가할 지
        user_embedding_first: session 앞에 user embedding 추가할 지
        aggregation: session aggregation 방법
            last: attention_out에서 마지막 valid token 선택
            sum_attn_out: attention_out의 valid token 합산
            sum_ffn_out: ffn_out의 valid token 합산
            weighted: 마지막 valid token에 높은 가중치 부여하여 attention_out 가중합
            adaptive_attn: attention_out의 valid token에 adaptive 가중치 부여하여 가중합
            adaptive_ffn: ffn_out의 valid token에 adaptive 가중치 부여하여 가중합
        other_information: 다른 추가적인 정보(delt_t, add_info 등)
            None: 사용하지 않음
            TiSASRec: attention에서 delta_t를 Key에 learnable matrix 곱해서 더해줌
            DIF-SR: attention에서 Key에 add_info를 concat해준 다음 Wk와 곱함
        """
        self.user_embedding = method.get('user_embedding', 'first')
        self.sum_token = method.get('sum_token', True)
        self.aggregation = method.get('aggregation', 'last')

        self.aggregator = aggregator()
        self.attention_block = self_attention()

    def forward(self, session, valid_mask, usr_embedding, **kwargs):
        """
        Encodes a session using the provided model.
        
        Args:
            session (tensor): [B, I, d] tensor representing the session. (B: batch size, I: number of interaction, d: embedding dimension)
            valid_mask (tensor): [B, I] (1: valid interaction, 0: padding), ([[1, 1, 1, ..., 1, 0, 0, 0], [1, 1, ...]] 형태) (B: batch size, I: number of interaction))
        Returns:
            list: Encoded representation of the session.
        """
        
        return 
    

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict, List, Literal

# -----------------------------------------------------------------------------
# Helper type aliases
# -----------------------------------------------------------------------------
UserEmbMethod = Literal['none', 'additive_bias', 'cross_attention', 'FiLM']
OtherInfoMethod = Literal['none', 'TiSASRec', 'DIF-SR']
AggMethod = Literal['last', 'sum_attn_out', 'sum_ffn_out', 'weighted',
                    'adaptive_attn', 'adaptive_ffn']

# -----------------------------------------------------------------------------
# Self‑Attention block with optional user‑embedding / extra‑info injections
# -----------------------------------------------------------------------------
class SelfAttention(nn.Module):
    """Multi‑Head Self‑Attention with optional *User Embedding* and *Other Info*
    injection strategies.

    Parameters
    ----------
    d_model : int
        Token embedding dimension (input/output).
    n_heads : int
        Number of attention heads.
    user_emb_method : UserEmbMethod
        How to incorporate user embedding into attention.
    other_info_method : OtherInfoMethod
        Whether/how to inject delta_t / add_info into Key tensor (K).
    dropout : float, default 0.1
    """

    def __init__(self,
                 d_model: int = 256,
                 n_heads: int = 8,
                 user_emb_method: UserEmbMethod = 'none',
                 other_info_method: OtherInfoMethod = 'none',
                 dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, 'd_model must be divisible by n_heads'
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.user_emb_method = user_emb_method
        self.other_info_method = other_info_method
        self.causal = True  # default: causal masking
        # ---------- Projection matrices ----------
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        # User‑embedding specific params
        if user_emb_method == 'cross_attention':
            # Separate Query projection for user embedding → will be used as Q
            self.W_q_user = nn.Linear(d_model, d_model, bias=False)
        elif user_emb_method == 'FiLM':
            # a(u), b(u) : small 2‑layer MLP generating scale/shift
            self.film_scale = nn.Sequential(
                nn.Linear(d_model, d_model // 2), nn.ReLU(),
                nn.Linear(d_model // 2, d_model))
            self.film_shift = nn.Sequential(
                nn.Linear(d_model, d_model // 2), nn.ReLU(),
                nn.Linear(d_model // 2, d_model))

        # Other‑info params (delta_t or add_info)
        if other_info_method == 'TiSASRec':
            # learnable matrix multiplying delta_t → same dim as K
            self.W_delta = nn.Linear(1, d_model, bias=False)
        elif other_info_method == 'DIF-SR':
            # K concat add_info → adjust projection dim
            # assumption: add_info dim == d_model_add. We'll infer at runtime
            self.W_k_concat = None  # lazy init in forward

        self.dropout = nn.Dropout(dropout)

    # ---------------------------------------------------------------------
    # Utility
    # ---------------------------------------------------------------------
    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """[B, L, D] → [B, H, L, Dh]"""
        B, L, _ = x.shape
        return x.view(B, L, self.n_heads, self.d_head).transpose(1, 2)

    def _combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        """[B, H, L, Dh] → [B, L, D]"""
        B, H, L, Dh = x.shape
        return x.transpose(1, 2).contiguous().view(B, L, H * Dh)

    # ---------------------------------------------------------------------
    def forward(self,
                x: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                *,
                user_emb: Optional[torch.Tensor] = None,
                delta_t: Optional[torch.Tensor] = None,
                add_info: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute attention output.

        Parameters
        ----------
        x : Tensor [B, L, D]
        mask : Tensor [B, L] , 1 = valid, 0 = padding (optional)
        user_emb : Tensor [B, D]  (needed for additive_bias / cross_attention / FiLM)
        delta_t  : Tensor [B, L, 1]  (for TiSASRec)
        add_info : Tensor [B, L, d_add] (for DIF‑SR)
        """
        B, L, D = x.shape

        # -------------------- Projection --------------------
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # ----- other_information injection -----
        if self.other_info_method == 'TiSASRec' and delta_t is not None:
            K = K + self.W_delta(delta_t)  # broadcast add
        elif self.other_info_method == 'DIF-SR' and add_info is not None:
            if self.W_k_concat is None:
                d_add = add_info.size(-1)
                self.W_k_concat = nn.Linear(D + d_add, D, bias=False).to(x.device)
            K = self.W_k_concat(torch.cat([K, add_info], dim=-1))

        # ----- user_embedding injection -----
        bias = None  # [B, H, L, L]
        if self.user_emb_method == 'additive_bias' and user_emb is not None:
            # Compute bias term b_{i,j} = q_i · u   (simple projection)
            # shape → [B, H, L]
            u_proj = user_emb.view(B, 1, self.n_heads, self.d_head)  # [B,1,H,Dh]
            u_proj = u_proj.expand(-1, L, -1, -1).permute(0, 2, 1, 3)  # [B,H,L,Dh]
            bias = (self._split_heads(Q) * u_proj).sum(-1, keepdim=True)  # [B,H,L,1]
        elif self.user_emb_method == 'cross_attention' and user_emb is not None:
            # Replace a subset of Q with user‑based query (simpler: head‑0)
            Q_user = self.W_q_user(user_emb).view(B, 1, self.n_heads, self.d_head)
            Q_split = self._split_heads(Q)
            Q_split[:, 0:1, :, :] = Q_user.transpose(1, 2)  # head‑0 → user query
            Q = self._combine_heads(Q_split)

        # Split heads
        Qh, Kh, Vh = map(self._split_heads, (Q, K, V))  # [B, H, L, Dh]

        # Scaled dot‑product attention
        scores = torch.matmul(Qh, Kh.transpose(-2, -1)) / math.sqrt(self.d_head)  # [B,H,L,L]
        if bias is not None:
            scores = scores + bias
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)  # [B,1,1,L]
            scores = scores.masked_fill(mask == 0, float('-inf'))
        if self.causal:
            L = scores.size(-1)
            causal_mask = torch.triu(                        # 상삼각 1, else 0
                torch.ones(L, L, dtype=torch.bool, device=scores.device),
                diagonal=1                                    # k=1 → strictly future
            )
            scores = scores.masked_fill(causal_mask, float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        context = torch.matmul(attn, Vh)  # [B,H,L,Dh]
        out = self._combine_heads(context)

        if self.user_emb_method == 'FiLM' and user_emb is not None:
            scale = torch.sigmoid(self.film_scale(user_emb)).unsqueeze(1)
            shift = self.film_shift(user_emb).unsqueeze(1)
            out = scale * out + shift

        return self.W_o(out)

# -----------------------------------------------------------------------------
# Position‑wise Feed‑Forward Network (Transformer style)
# -----------------------------------------------------------------------------
class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# -----------------------------------------------------------------------------
# Aggregator
# -----------------------------------------------------------------------------
class Aggregator(nn.Module):
    """Aggregate token‑wise outputs into a single session vector."""
    def __init__(self, method: AggMethod = 'last', d_model: int = 0):
        super().__init__()
        self.method = method
        if method in {'weighted', 'adaptive_attn', 'adaptive_ffn'}:
            self.alpha = nn.Linear(d_model, 1)

    def forward(self,
                attn_out: torch.Tensor,
                ffn_out: torch.Tensor,
                valid_mask: torch.Tensor):
        """attn_out / ffn_out : [B, L, D]; valid_mask : [B, L]"""
        if self.method == 'last':
            idx = valid_mask.sum(dim=1) - 1  # [B]
            idx = idx.clamp(min=0)
            batch_idx = torch.arange(attn_out.size(0), device=attn_out.device)
            return attn_out[batch_idx, idx]
        elif self.method == 'sum_attn_out':
            return (attn_out * valid_mask.unsqueeze(-1)).sum(dim=1)
        elif self.method == 'sum_ffn_out':
            return (ffn_out * valid_mask.unsqueeze(-1)).sum(dim=1)
        elif self.method in {'weighted', 'adaptive_attn'}:
            weights = torch.softmax(self.alpha(attn_out).squeeze(-1), dim=1)
            return (attn_out * weights.unsqueeze(-1)).sum(dim=1)
        elif self.method == 'adaptive_ffn':
            weights = torch.softmax(self.alpha(ffn_out).squeeze(-1), dim=1)
            return (ffn_out * weights.unsqueeze(-1)).sum(dim=1)
        else:
            raise ValueError(f'Unknown aggregation method: {self.method}')

# -----------------------------------------------------------------------------
# Utility: prepare session sequence with (optional) user‑embedding & [SUM] token
# -----------------------------------------------------------------------------

def preprocess_encoder_input(session: torch.Tensor,
                             valid_mask: torch.Tensor,
                             usr_embedding: Optional[torch.Tensor] = None,
                             *,
                             user_embedding_first: bool = True,
                             sum_token: Optional[torch.Tensor] = None):
    """Return (new_seq, new_mask).
    Rules:
      - if user_embedding_first=True and usr_embedding is not None ⇒ prepend user token
      - if sum_token is not None ⇒ place right *after last valid session token* (before paddings)
      - original paddings remain at end
    """
    B, I, D = session.shape
    len_valid = valid_mask.sum(1)  # [B]

    extra = 0
    if user_embedding_first and usr_embedding is not None:
        extra += 1
    if sum_token is not None:
        extra += 1

    L_new = I + extra
    device = session.device
    seq = session.new_zeros(B, L_new, D)
    new_mask = torch.zeros(B, L_new, dtype=valid_mask.dtype, device=device)

    for b in range(B):
        offset = 0
        # user token
        if user_embedding_first and usr_embedding is not None:
            seq[b, 0] = usr_embedding[b]
            new_mask[b, 0] = 1
            offset = 1
        # session valid tokens
        lv = len_valid[b].item()
        if lv > 0:
            seq[b, offset:offset+lv] = session[b, :lv]
            new_mask[b, offset:offset+lv] = 1
        # sum token
        if sum_token is not None:
            seq[b, offset+lv] = sum_token
            new_mask[b, offset+lv] = 1
        # paddings already zero
    return seq, new_mask


# -----------------------------------------------------------------------------
# Main SessionEncoder module
# -----------------------------------------------------------------------------
class SessionEncoder(nn.Module):
    def __init__(self,
                 d_model: int,
                 n_heads: int,
                 d_ff: int,
                 n_layers: int = 1,
                 *,
                 user_embedding: UserEmbMethod = 'none',
                 user_embedding_first: bool = True,
                 sum_token: bool = True,
                 aggregation: AggMethod = 'last',
                 other_information: OtherInfoMethod = 'none',
                 dropout: float = 0.1):
        super().__init__()
        self.user_embedding = user_embedding
        self.user_embedding_first = user_embedding_first
        self.sum_token_flag = sum_token
        self.other_information = other_information

        # learnable [SUM] token if needed
        self.sum_token = nn.Parameter(torch.randn(d_model)) if sum_token else None

        # Stack of attention+FFN blocks (Transformer encoder style)
        layers = []
        for _ in range(n_layers):
            attn = SelfAttention(d_model, n_heads,
                                 user_emb_method=user_embedding,
                                 other_info_method=other_information,
                                 dropout=dropout)
            ffn = FeedForward(d_model, d_ff, dropout)
            layers.append(nn.ModuleDict({
                'attn': attn,
                'ffn': ffn,
                'norm1': nn.LayerNorm(d_model),
                'norm2': nn.LayerNorm(d_model),
                'drop': nn.Dropout(dropout)
            }))
        self.layers = nn.ModuleList(layers)

        self.aggregator = Aggregator(aggregation, d_model)

    # -----------------------------------------------------------------
    def forward(self,
                session: torch.Tensor,
                valid_mask: torch.Tensor,
                usr_embedding: Optional[torch.Tensor] = None,
                *,
                delta_t: Optional[torch.Tensor] = None,
                add_info: Optional[torch.Tensor] = None):
        """Encode a batch of sessions.

        Parameters
        ----------
        session : [B, I, D]
        valid_mask : [B, I]  (1 = valid, 0 = padding)
        usr_embedding : [B, D]  user‑level vector (optional)
        delta_t : [B, I, 1]  per‑token gap (optional, TiSASRec)
        add_info : [B, I, d_add]  extra token‑wise feature (optional, DIF‑SR)
        Returns
        -------
        session_vec : [B, D]
        """
        x, new_mask = preprocess_encoder_input(session, usr_embedding,
                                     user_embedding_first=self.user_embedding_first,
                                     sum_token=self.sum_token)
        # mask needs adjustment if we added tokens
        pad = x.new_ones(x.size()[:2], dtype=new_mask.dtype)
        offset = x.size(1) - new_mask.size(1)
        if offset > 0:
            pad[:, :-new_mask.size(1)] = 1  # assume added tokens valid
            pad[:, -new_mask.size(1):] = new_mask
            valid_mask = pad

        for layer in self.layers:
            # Self‑Attention
            res = x
            x = layer['norm1'](x + layer['drop'](
                layer['attn'](x, mask=valid_mask,
                                user_emb=usr_embedding,
                                delta_t=delta_t,
                                add_info=add_info)))
            # Feed‑Forward
            x = layer['norm2'](x + layer['drop'](layer['ffn'](x)))

        return self.aggregator(x, x, valid_mask)
