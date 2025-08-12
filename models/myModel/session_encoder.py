import math, torch, torch.nn as nn
from typing import Optional, Dict, Literal

# ─────────────────────────────────────────────────────────────────────────────
# Type aliases
UserEmbMethod  = Literal['none', 'additive_bias', 'cross_attention', 'FiLM']
OtherInfoMethod = Literal['none', 'TiSASRec', 'DIF-SR']
AggMethod = Literal['last', 'sum_attn_out', 'sum_ffn_out',
                    'weighted', 'adaptive_attn', 'adaptive_ffn']

# ─────────────────────────────────────────────────────────────────────────────
# Cached causal-mask (상삼각)
_causal_cache: Dict[int, torch.Tensor] = {}
def causal_mask(L, device):
    if L not in _causal_cache:
        _causal_cache[L] = torch.triu(torch.ones(L, L, dtype=torch.bool), 1)
    return _causal_cache[L].to(device)

# ─────────────────────────────────────────────────────────────────────────────
# Attention 블록
class SelfAttention(nn.Module):
    def __init__(self, d_model, n_heads,
                 *, user_emb_method: UserEmbMethod = 'none',
                 other_info_method: OtherInfoMethod = 'none',
                 dropout: float = .1, causal: bool = True):
        super().__init__()
        assert d_model % n_heads == 0
        self.dh = d_model // n_heads
        self.H  = n_heads
        self.user_emb_method = user_emb_method
        self.other_info_method = other_info_method
        self.causal = causal

        # 프로젝션
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        # user-specific
        if user_emb_method == 'cross_attention':
            self.W_q_user = nn.Linear(d_model, d_model, bias=False)
        elif user_emb_method == 'FiLM':
            hid = max(32, d_model // 2)
            self.film_scale = nn.Sequential(nn.Linear(d_model, hid), nn.ReLU(),
                                            nn.Linear(hid, d_model))
            self.film_shift = nn.Sequential(nn.Linear(d_model, hid), nn.ReLU(),
                                            nn.Linear(hid, d_model))
        # other-info
        if other_info_method == 'TiSASRec':
            self.W_delta = nn.Linear(1, d_model, bias=False)
        self.W_k_concat = None  # DIF-SR lazy

        self.drop = nn.Dropout(dropout)
        self.ln   = nn.LayerNorm(d_model)

    # util
    def _split(self, x):  # [B,L,D]→[B,H,L,Dh]
        B, L, _ = x.shape
        return x.view(B, L, self.H, self.dh).transpose(1, 2)
    def _merge(self, x):  # [B,H,L,Dh]→[B,L,D]
        B, H, L, Dh = x.shape
        return x.transpose(1, 2).contiguous().view(B, L, H*Dh)

    def forward(self, x, mask, *, user_emb=None,
                delta_t=None, add_info=None):
        Q, K, V = self.W_q(x), self.W_k(x), self.W_v(x)

        # --- other-info ---
        if self.other_info_method == 'TiSASRec' and delta_t is not None:
            K = K + self.W_delta(delta_t)
        elif self.other_info_method == 'DIF-SR' and add_info is not None:
            if self.W_k_concat is None:
                self.W_k_concat = nn.Linear(K.size(-1)+add_info.size(-1),
                                            K.size(-1), bias=False).to(x.device)
            K = self.W_k_concat(torch.cat([K, add_info], -1))

        # --- user embedding ---
        bias_term = None
        if self.user_emb_method == 'additive_bias' and user_emb is not None:
            # 단순 내적 bias
            bias_term = (user_emb @ self.W_q.weight).view(x.size(0), 1, self.H, self.dh)
        elif self.user_emb_method == 'cross_attention' and user_emb is not None:
            Qh = self._split(Q)
            uq = self.W_q_user(user_emb).view(x.size(0), 1, self.H, self.dh)
            Qh[:, 0:1] = uq  # head-0 교체
            Q = self._merge(Qh)

        Qh, Kh, Vh = map(self._split, (Q, K, V))
        scores = torch.matmul(Qh, Kh.transpose(-2, -1)) / math.sqrt(self.dh)
        if bias_term is not None:
            scores = scores + bias_term.unsqueeze(-1)

        scores = scores.masked_fill((mask==0).unsqueeze(1).unsqueeze(2), -1e4)
        if self.causal:
            scores = scores.masked_fill(causal_mask(scores.size(-1),
                                                    scores.device), -1e4)

        A = self.drop(torch.softmax(scores, -1))
        out = torch.matmul(A, Vh)
        out = self.W_o(self._merge(out))

        if self.user_emb_method == 'FiLM' and user_emb is not None:
            scale = torch.sigmoid(self.film_scale(user_emb)).unsqueeze(1)
            shift = self.film_shift(user_emb).unsqueeze(1)
            out = scale * out + shift
        return self.ln(out)

# ─────────────────────────────────────────────────────────────────────────────
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(),
                                 nn.Dropout(dropout),
                                 nn.Linear(d_ff, d_model), nn.Dropout(dropout))
        self.ln = nn.LayerNorm(d_model)
    def forward(self, x): return self.ln(x + self.net(x))

# ─────────────────────────────────────────────────────────────────────────────
class Aggregator(nn.Module):
    def __init__(self, method: AggMethod, d_model):
        super().__init__()
        self.method = method
        if method in {'weighted','adaptive_attn','adaptive_ffn'}:
            self.alpha = nn.Linear(d_model,1)

    def forward(self, attn_out, ffn_out, mask):
        if self.method == 'last':
            idx = (mask.sum(1) - 1).long()
            return attn_out[torch.arange(attn_out.size(0), device=attn_out.device), idx]
        if self.method == 'sum_attn_out':
            return (attn_out*mask.unsqueeze(-1)).sum(1)
        if self.method == 'sum_ffn_out':
            return (ffn_out*mask.unsqueeze(-1)).sum(1)
        if self.method in {'weighted','adaptive_attn'}:
            w = torch.softmax(self.alpha(attn_out).squeeze(-1)+(mask==0)*-1e9,1)
            return (attn_out*w.unsqueeze(-1)).sum(1)
        if self.method == 'adaptive_ffn':
            w = torch.softmax(self.alpha(ffn_out).squeeze(-1)+(mask==0)*-1e9,1)
            return (ffn_out*w.unsqueeze(-1)).sum(1)
        raise ValueError

# ─────────────────────────────────────────────────────────────────────────────
def preprocess_encoder_input(session:torch.Tensor,
                             valid_mask:torch.Tensor,
                             usr_embedding:Optional[torch.Tensor]=None,*,
                             user_embedding_first:bool=True,
                             sum_token:Optional[torch.Tensor]=None,
                             start_token:Optional[torch.Tensor]=None):
    """
    Returns
    -------
    seq        : [B,L,D]  – 앞/뒤 토큰 포함
    attn_mask  : [B,L]    – user/start, SUM, session‑valid = 1, padding = 0
    loss_mask  : [B,L]    – **오직 원본 session token**(예측 대상) = 1
    """
    B,I,D = session.shape
    device = session.device
    lv     = valid_mask.sum(1)             # [B]

    front_tok = None
    if user_embedding_first and usr_embedding is not None:
        front_tok = ('user', 1)
    elif start_token is not None:
        front_tok = ('start', 1)

    back_tok = ('sum', 1) if sum_token is not None else None

    L = I + (front_tok is not None) + (back_tok is not None)
    seq  = session.new_zeros(B,L,D)
    attn = torch.zeros(B,L, dtype=valid_mask.dtype, device=device)
    loss = torch.zeros_like(attn)

    for b in range(B):
        idx = 0
        # 앞 토큰
        if front_tok is not None:
            if front_tok[0]=='user':
                seq[b,idx] = usr_embedding[b]
            else:
                seq[b,idx] = start_token
            attn[b,idx]=1
            loss[b,idx] = 1
            idx+=1
        # session 토큰
        n = int(lv[b].item())
        if n>0:
            seq[b,idx:idx+n]  = session[b,:n]
            attn[b,idx:idx+n] = 1
            loss[b,idx:idx+n-1] = 1   # 예측 대상
        idx += n
        # SUM 토큰
        if back_tok is not None:
            seq[b,idx] = sum_token
            attn[b,idx] = 1
    return seq, attn, loss

class EncoderBlock(nn.Module):
    """
    LN → SelfAttention → Dropout → Residual → LN → FFN → Dropout → Residual
    """
    def __init__(self, d_model, n_heads, d_ff, dropout,
                 user_emb_method, other_info_method):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = SelfAttention(d_model, n_heads,
                                  user_emb_method=user_emb_method,
                                  other_info_method=other_info_method,
                                  dropout=dropout)
        self.drop1 = nn.Dropout(dropout)

        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x, attn_mask, *, user_emb=None,
                delta_t=None, add_info=None):
        # ① Self-Attention 서브층
        attn_out = x + self.drop1(
                self.attn(self.ln1(x), attn_mask,
                          user_emb=user_emb,
                          delta_t=delta_t, add_info=add_info)
            )                               # residual

        # ② FFN 서브층
        ffn_out = self.drop2(
                self.ffn(self.ln2(x))
            )                               # residual
        return ffn_out, attn_out
    
# ─────────────────────────────────────────────────────────────────────────────
class SessionEncoder(nn.Module):
    def __init__(self, 
                 d_model = 256, 
                 n_heads = 8, 
                 d_ff = 256, 
                 n_layers=1,
                 *, user_embedding:UserEmbMethod='none',
                 user_embedding_first=True,
                 sum_token=True,
                 cls_token=False,
                 aggregation:AggMethod='last',
                 other_information:OtherInfoMethod='none',
                 dropout=.1):
        super().__init__()
        self.user_embedding = user_embedding
        self.user_embedding_first = user_embedding_first
        self.other_information   = other_information
        self.sum_token = nn.Parameter(torch.randn(d_model)) if sum_token else None
        self.start_token = nn.Parameter(torch.randn(d_model)) if cls_token else None # [CLS]‑like

        # --- TiSASRec 스타일 블록 리스트 ---------------------------------
        self.layers = nn.ModuleList([
                        EncoderBlock(d_model, n_heads, d_ff, dropout,
                                    user_emb_method=user_embedding,
                                    other_info_method=other_information)
                        for _ in range(n_layers)])

        self.aggregator = Aggregator(aggregation, d_model)

    # ---------------------------------------------------------------------
    def forward(self, session, valid_mask, usr_embedding=None,
                *, delta_t=None, add_info=None):
        x, attn_mask, loss_mask = preprocess_encoder_input(session, valid_mask, usr_embedding,
                                           user_embedding_first=self.user_embedding_first,
                                           sum_token=self.sum_token,
                                           start_token=self.start_token)
        for layer in self.layers:
            ffn_out, attn_out = layer(x, attn_mask,
                      user_emb=usr_embedding,
                      delta_t=delta_t, add_info=add_info)
            x = ffn_out
        #x = self.layers[-1](x)  # 마지막 FFN 블록
        agg_vec = self.aggregator(attn_out, ffn_out, attn_mask)
        return x, agg_vec, loss_mask