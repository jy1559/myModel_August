#!/usr/bin/env python
# tools/build_global_graph.py
# ---------------------------
# 전역 이종 그래프(global_graph.pt) 생성 스크립트
#
# 사용 예)
#   python tools/build_global_graph.py \
#          --dataset_folder /data/Datasets \
#          --dataset_names Steam Taobao \
#          --sampling_Ns   0 4

import json, argparse, collections, itertools, pathlib, torch
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────
def build_one(root: pathlib.Path, top_cooc: int):
    """
    단일 데이터셋·샘플링 폴더(root)에 대해 global_graph.pt 생성.
      • 노드 집합   : train ∪ val ∪ test (cold item/user 포함)
      • 간선(Weight): train 데이터에서만 계산
    """
    train_path = root / "interactions_train.jsonl"
    if not train_path.exists():
        print(f"  ✗  {train_path} 없음 → 스킵")
        return

    split_files = [root / f"interactions_{sp}.jsonl" for sp in ("train", "val", "test")]
    # ─────────────────────────────────────────────────────────────
    # 1) 전체 ID 범위 파악 (세 split 모두)
    # ─────────────────────────────────────────────────────────────
    uid2idx: dict[int, int] = {}        # ext uid → 0..U-1
    max_item_id = -1

    def _uid(u):
        if u not in uid2idx:
            uid2idx[u] = len(uid2idx)
        return uid2idx[u]

    for p in split_files:
        if not p.exists():
            continue
        for line in p.open():
            j = json.loads(line)
            _uid(j["uid"])
            for sess in j["sessions"]:
                for it, *_ in sess:
                    if it > 0:
                        max_item_id = max(max_item_id, it)

    V = max_item_id + 1
    U = len(uid2idx)
    N = V + U
    print(f"  • {root.name}: items={V:,}, users={U:,} (total nodes={N:,})")

    # ─────────────────────────────────────────────────────────────
    # 2) train split만으로 간선 카운트
    # ─────────────────────────────────────────────────────────────
    rel_edges = collections.defaultdict(collections.Counter)

    with train_path.open() as f:
        for line in tqdm(f, unit="lines", leave=False):
            j = json.loads(line)
            uid = uid2idx[j["uid"]]      # mapping 이미 존재
            for sess in j["sessions"]:
                items = [it for it, *_ in sess if it > 0]
                if len(items) < 1:
                    continue

                # user ↔ item
                for it in items:
                    rel_edges["u2i"][(uid, it)] += 1
                    rel_edges["i2u"][(it, uid)] += 1

                # in-session item transition
                for a, b in zip(items[:-1], items[1:]):
                    rel_edges["i_in"][(a, b)] += 1
                    rel_edges["i_out"][(b, a)] += 1

                # item-item co-occurrence
                for a, b in itertools.combinations(set(items), 2):
                    rel_edges["i_co"][(a, b)] += 1
                    rel_edges["i_co"][(b, a)] += 1

    # ─────────────────────────────────────────────────────────────
    # 3) co-occurrence top-k trimming (src 기준)
    # ─────────────────────────────────────────────────────────────
    if "i_co" in rel_edges and top_cooc:
        by_src = collections.defaultdict(list)
        for (src, tgt), w in rel_edges["i_co"].items():
            by_src[src].append((tgt, w))
        rel_edges["i_co"].clear()
        for src, lst in by_src.items():
            for tgt, w in sorted(lst, key=lambda x: -x[1])[:top_cooc]:
                rel_edges["i_co"][(src, tgt)] = w

    # ─────────────────────────────────────────────────────────────
    # 4) relation별 sparse 행렬 생성
    # ─────────────────────────────────────────────────────────────
    rel_names = ["i_in", "i_out", "i_co", "u2i", "i2u"]
    mats: list[torch.Tensor] = []

    for r in rel_names:
        rows, cols, vals = [], [], []
        deg = collections.Counter()
        for (src, tgt), w in rel_edges[r].items():
            if r.startswith("u"):              # u2i
                src_shift, tgt_shift = src + V, tgt
            elif r.endswith("u"):              # i2u
                src_shift, tgt_shift = src, tgt + V
            else:                              # i_in / i_out / i_co
                src_shift, tgt_shift = src, tgt
            rows.append(src_shift)
            cols.append(tgt_shift)
            vals.append(float(w))
            deg[src_shift] += w

        if rows:
            norm_vals = [v / deg[rw] for v, rw in zip(vals, rows)]
            idx = torch.tensor([rows, cols])
            val = torch.tensor(norm_vals, dtype=torch.float32)
            A = torch.sparse_coo_tensor(idx, val, (N, N)).coalesce()
        else:                                   # relation 간선이 전혀 없을 때
            A = torch.sparse_coo_tensor((N, N))
        mats.append(A)
        print(f"    {r:<5}: edges={len(rows):,}")

    # 저장
    # ─ 마지막 저장 부분만 수정하면 됩니다 ─
    out = {
        "rels":      mats,           # relation별 sparse 행렬
        "num_item":  V,              # 전체 item 수  (train∪val∪test)
        "num_user":  U,              # 전체 user 수
        "uid_map":   uid2idx,        # <원본 uid> → <0..U-1> 매핑
    }
    torch.save(out, root / "global_graph.pt")
    print(f"    ✓ saved → {root/'global_graph.pt'}\n")

# ─────────────────────────────────────────────────────────────
def main():
    dataset_folder = "/home/jy1559/Datasets"
    dataset_names = ["Retail_Rocket", "LFM-BeyMS", "MovieLens"]
    sampling_Ns = ["0", "4", "16"]
    top_cooc = 20
    base = pathlib.Path(dataset_folder)
    for dname in dataset_names:
        for sN in sampling_Ns:
            root = base / dname / "timesplit" / f"{sN}-Sampling"

            if root.exists():
                build_one(root, top_cooc)
            else:
                print(f"  ✗  {root} 존재 안 함 → 스킵")

if __name__ == "__main__":
    main()
