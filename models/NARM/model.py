import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class SeqRecModel(nn.Module):
    """Neural Attentive Session Based Recommendation Model Class

    Args:
        n_items(int): the number of items
        hidden_size(int): the hidden size of gru
        embedding_dim(int): the dimension of item embedding
        batch_size(int): 
        n_layers(int): the number of gru layers

    """
    def __init__(self, n_items, cfg):
        super(SeqRecModel, self).__init__()
        embedding_dim = cfg.get("embed_dim", 256)
        hidden_size = cfg.get("hidden_dim", 128)
        self.n_items = n_items
        self.hidden_size = hidden_size
        self.n_layers=cfg.get("n_layers", 1)
        self.embedding_dim = embedding_dim
        self.emb = nn.Embedding(self.n_items + 1, self.embedding_dim, padding_idx = 0)
        self.emb_dropout = nn.Dropout(0.25)
        self.gru = nn.GRU(self.embedding_dim, self.hidden_size, self.n_layers)
        self.a_1 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.a_2 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_t = nn.Linear(self.hidden_size, 1, bias=False)
        self.ct_dropout = nn.Dropout(0.5)
        self.b = nn.Linear(self.embedding_dim, 2 * self.hidden_size, bias=False)
        #self.sf = nn.Softmax()
        self.device  = cfg.get("device", "cpu")

    def forward(self, batch):
        """
        batch["item_id"] : [B, S, I]  (0 = padding)
        ------------------------------------------------
        1) 0 패딩 제거 후 길이 < 2 세션 제거
        2) 남은 세션들을 padded matrix로 → [B′, L_max]
        3) pack_padded_sequence용 lengths 계산
        4) GRU 입력 형태 [L_max, B′] 로 transpose
        """
        # ─────────────────── 0. 세션 행렬 준비 ──────────────────────────
        item_ids = batch["item_id"].clamp_min(0).to(self.device)  # 0=padding
        eval_from = batch["eval_from"].to(self.device) 
        
        # ─ eval_from 이전 세션은 전부 0 패딩으로 처리 ─
        sess_idx = torch.arange(item_ids.size(1), device=self.device)  # [S]
        allow_sess = sess_idx.unsqueeze(0) >= eval_from.unsqueeze(1)   # [B,S]
        item_ids = item_ids * allow_sess.unsqueeze(-1)                 # [B,S,I]

        B, S, I = item_ids.shape
        sess_mat = item_ids.view(B * S, I)                             # [B*S, I]

        # ─────────────────── 1. 길이 / 필터 / 타깃 분리 ────────────────
        lengths_raw = (sess_mat != 0).sum(1)            # [B*S] 실제 길이 L
        valid_mask  = lengths_raw >= 2                  # 길이 2↑ 세션만
        sess_mat    = sess_mat[valid_mask]              # [B', I]
        lengths_raw = lengths_raw[valid_mask]           # [B']  (>=2)
        if lengths_raw.numel() == 0:
            raise RuntimeError("No valid sessions")

        # 타깃(마지막 아이템) 추출
        target_id = sess_mat[torch.arange(sess_mat.size(0), device=self.device),
                            lengths_raw - 1]           # [B']
        # 입력 길이 (= L-1)
        lengths_in = lengths_raw - 1                    # [B']
        L_in_max   = lengths_in.max().item()            # 최대 입력 길이

        # ─────────────────── 2. 입력 행렬 (padding 0) ──────────────────
        # 필요 열만 잘라 메모리↓
        input_mat  = sess_mat[:, :L_in_max]             # [B', L_in_max]
        # 마지막 col은 일부 행에서 0 일 수 있으나 pack이 무시

        # mask → attention & loss
        mask = (torch.arange(L_in_max, device=self.device)  # [L_in_max]
                .unsqueeze(0) < lengths_in.unsqueeze(1))  # [B',L_in_max]
        input_mat = input_mat * mask                       # pad 위치 0

        # ─────────────────── 3. GRU (pack) ───────────────────────────
        lens_sort, sort_idx = lengths_in.sort(descending=True)
        seq_sorted = input_mat[sort_idx].t()                # [L,B']
        embeds     = self.emb_dropout(self.emb(seq_sorted)) # [L,B',D]
        packed     = pack_padded_sequence(
            embeds, lens_sort.cpu(), enforce_sorted=False)

        h0 = self.init_hidden(lens_sort.size(0))            # [n_layers,B',H]
        gru_out, hidden = self.gru(packed, h0)              # GRU
        gru_out, _ = pad_packed_sequence(gru_out)           # [L,B',H]

        # 원 순서 복구
        _, rev_idx = sort_idx.sort()
        gru_out = gru_out[:, rev_idx]                       # [L,B',H]
        hidden  = hidden[:, rev_idx]
        mask    = mask[rev_idx]                             # [B',L]

        # ─────────────────── 4. NARM 어텐션 ──────────────────────────
        ht  = hidden[-1]                                    # [B',H]
        gru = gru_out.permute(1, 0, 2)                      # [B',L,H]

        q1 = self.a_1(gru.reshape(-1, self.hidden_size)).view_as(gru)
        q2 = self.a_2(ht).unsqueeze(1)                      # [B',1,H]
        alpha = self.v_t(torch.sigmoid(q1 + q2)).squeeze(-1) * mask
        c_loc = (alpha.unsqueeze(2) * gru).sum(1)           # [B',H]

        c_t = self.ct_dropout(torch.cat([c_loc, ht], 1))    # [B',2H]

        # ─────────────────── 5. 출력 dict ─────────────────────────────
        out = {
            "reps":   c_t,        # [B', 2H]
            "target": target_id,  # [B']
            "u_type": batch["u_type"]
                    .repeat_interleave(S)[valid_mask],   # [B']
        }
        return out

    def init_hidden(self, batch_size):
        return torch.zeros((self.n_layers, batch_size, self.hidden_size), requires_grad=True).to(self.device)