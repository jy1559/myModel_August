"""
user_update.py
--------------
User state updater with several time–aware variants.

Variants
--------
1) default (TA-GRU) : concat(session_vec, Δt_emb) → GRU
2) no_time          : GRU over session_vec only
3) decay            : prev_state ● exp(-Δt/τ)  +  GRU(session_vec)
4) gate             : time-embedding used only to modulate GRU gates (MTAM)

Common features
---------------
* Learnable global `initial_state`
  ▸ `.freeze_initial()` / `.unfreeze_initial()`
* Log-bucket time-gap embedding (TiSASRec style)
"""

import math
import torch
import time
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Literal
from .input_embedding import DeltaTEmb

MethodType = Literal['default', 'no_time', 'decay', 'gate', 'tgru']

# --------------------------------------------------------------------- #
class UserStateUpdater(nn.Module):
    def __init__(self,
                 d_session=256,
                 hidden_size=256,
                 *,
                 method='default',
                 rnn='GRU',
                 dt_emb_module: DeltaTEmb | None = None,   
                 decay_tau=3600.,
                 device='cpu'):
        super().__init__()
        assert method in {'default', 'no_time', 'decay', 'gate'}
        self.method = method
        self.hidden_size = hidden_size
        self.use_lstm = (rnn == 'LSTM')
        self.decay_tau = decay_tau

        # -- RNN --------------------------------------------------------
        self.dtEmb = dt_emb_module
        dt_dim = self.dtEmb.table.embedding_dim if (method=='default') else 0
        rnn_inp_dim = d_session + dt_dim
        self.rnn = (nn.LSTM if rnn=='LSTM' else nn.GRU)(
            rnn_inp_dim, hidden_size, batch_first=True)
        if method == 'tgru':
            self.W_tau = nn.Linear(d_session + hidden_size, hidden_size)
            self.W_delta = nn.Linear(1, hidden_size, bias=False)   # δ_s
            self.W_g = nn.Linear(hidden_size * 2, hidden_size)     # combine δ_s , τ_s
        # -- global learnable initial state ----------------------------
        init = torch.zeros(hidden_size).to(device)  # [hidden_size]
        nn.init.xavier_uniform_(init.unsqueeze(0))
        self.initial_state = nn.Parameter(init)        # [hidden_size]

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------
    def freeze_initial(self, freeze: bool = True):
        """Freeze/unfreeze the global initial state parameter."""
        self.initial_state.requires_grad_(not freeze)

    def reset_state(self, batch_size: int, device=None):
        """Return initial state expanded to batch size."""
        h0 = self.initial_state.unsqueeze(0).expand(1, batch_size, -1).contiguous()
        h0 = h0 if device is not None else h0
        if self.use_lstm:
            c0 = torch.zeros_like(h0)
            return (h0, c0)
        return h0

    # -----------------------------------------------------------------
    def forward(self,
            session_vec: torch.Tensor,   # [B, D_sess]
            prev_state,                  # h_{t-1}  or (h,c)
            dt_sec: Optional[torch.Tensor] = None,   # [B]  (현재 세션의 가장 첫 delta_t (이전 세션과의 시간차))
            pad_mask: Optional[torch.Tensor] = None): #[B] (0/1, 현재 세션 가장 앞 부분 패딩 여부)
        """
        Parameters
        ----------
        session_vec : latest session representation
        prev_state  : previous user state (output of last call)
        delta_t_sec : gap (sec) between sessions (required for 'default', 'decay', 'gate')

        Returns
        -------
        state_t : updated state  (h or (h,c))
        """
        B = session_vec.size(0)
        device = session_vec.device
        # ╭── Build RNN input x_t ──────────────────────────────────────╮
        if self.method == 'default':
            assert dt_sec is not None, "'default' needs Δt"
            # dt_emb_module 은 InputEmbedding.dt_emb 이므로
            # ① 3-D 가 아니어도 처리할 작은 helper
            dt_emb = self.dtEmb(dt_sec, pad_mask) # [B,D_dt]
            x_t = torch.cat([session_vec, dt_emb], dim=-1).unsqueeze(1) #[B,1,D_sess+D_dt]
        elif self.method == 'no_time':
            x_t = session_vec.unsqueeze(1)
        elif self.method == 'decay':
            x_t = session_vec.unsqueeze(1)
        elif self.method == 'gate':
            assert dt_emb is not None, "'gate' needs Δt"
            x_t = session_vec.unsqueeze(1)
        # ╰─────────────────────────────────────────────────────────────╯

        # -- decay variant : pre-scale previous state ------------------
        if self.method == 'decay':
            decay = torch.exp(-dt_sec / self.decay_tau).view(1, B, 1)
            if self.use_lstm:
                h_prev, c_prev = prev_state
                prev_state = (h_prev * decay, c_prev * decay)
            else:
                prev_state = prev_state * decay
        if self.method == 'tgru':
            assert dt_sec is not None, "'tgru' requires Δt"
            # log-bucket + Linear → δ_s
            delta = dt_sec.clamp(min=1).log().unsqueeze(-1)   # [B,1]
            delta_feat = torch.tanh(self.W_delta(delta))           # δ_s  [B,D]

            # τ_s : tanh([x_s, h_{t-1}] W_tau)
            h_prev = prev_state[0] if self.use_lstm else prev_state  # [1,B,D]
            h_prev = h_prev.squeeze(0)
            tau_feat = torch.tanh(self.W_tau(
                torch.cat([session_vec, h_prev], dim=-1)))          # τ_s  [B,D]

            # temporal gate g_s
            g_s = torch.sigmoid(self.W_g(torch.cat([delta_feat, tau_feat], dim=-1)))

            # ▶ GRU 연산 (gate 내부에 시간 정보 X) ----------------
            x_t = session_vec.unsqueeze(1)
            out, state_t = self.rnn(x_t, prev_state)               # h̃_s
            if self.use_lstm:
                h_t, c_t = state_t
                h_t = g_s.unsqueeze(0) * h_prev + (1 - g_s).unsqueeze(0) * h_t
                state_t = (h_t, c_t)
            else:
                h_t = g_s.unsqueeze(0) * h_prev + (1 - g_s).unsqueeze(0) * state_t
                state_t = h_t
            return state_t
        # -- RNN (single-step) ----------------------------------------
        if pad_mask is not None and pad_mask.sum() < B:
            # ✦ 선택적 업데이트 ✦
            active = pad_mask.nonzero(as_tuple=True)[0]   # idx of valid rows
            x_active   = x_t[active]                      # [N_valid,1,D]
            h_prev_act = prev_state[..., active, :]       # keep dims
            out, h_new = self.rnn(x_active, h_prev_act)
            h_new = h_new.to(prev_state.dtype)
            
            # copy old state then scatter-update
            state_t = prev_state.clone()
            state_t[..., active, :] = h_new
        else:
            _, state_t = self.rnn(x_t, prev_state)

        # -- gate variant : blend with dt_emb --------------------------
        if self.method == 'gate':
            h_prev = prev_state[0] if self.use_lstm else prev_state
            alpha = torch.sigmoid(dt_sec).view(1, B, -1)       # [1,B,D]
            if self.use_lstm:
                h_t, c_t = state_t
                h_t = alpha * h_t + (1 - alpha) * h_prev
                state_t = (h_t, c_t)
            else:
                state_t = alpha * state_t + (1 - alpha) * h_prev

        return state_t
