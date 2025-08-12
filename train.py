import argparse
import os
import random
import sys
import time
from pathlib import Path
import numpy as np
from typing import Dict, Tuple, List, Any

import torch
from optimizer import CosineAnnealingWarmUpRestarts  # 수정된 버전: CosineAnnealingWarmUpRestarts
import torch.nn.functional as F
from collections import defaultdict
from contextlib import nullcontext
from torch.amp import GradScaler, autocast
import pandas as pd, pathlib, datetime
from tqdm.auto import tqdm
import wandb
import csv
# local modules (경로는 프로젝트 구조에 맞게 수정)
import dataset                                  # Datasets.dataset 의 get_dataloaders 래퍼로 가정
from models.model import build_model                  # models/model.py
from loss import compute_loss                   # 수정된 버전: (loss, metric_dict) 반환
# ────────────────────────────────────────────────────────────────q
# Metric CSV
# ────────────────────────────────────────────────────────────────
def _get_metrics_csv_path(args):
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"{args.model}/{args.dataset_name}/{args.sampling_N}/{ts}_metrics.csv"
    path  = pathlib.Path("outputs/metrics") / fname
    path.parent.mkdir(parents=True, exist_ok=True)
    return path

_METRICS_CSV = None                # 전역으로 한 번만 생성

def _append_metrics(rows, args):
    """
    rows: List[Dict(epoch,split,label,metric,value)]
    → 한 CSV에 계속 append (헤더 자동)
    """
    global _METRICS_CSV
    if _METRICS_CSV is None:
        _METRICS_CSV = _get_metrics_csv_path(args)

    write_header = not _METRICS_CSV.exists()
    with _METRICS_CSV.open("a", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["epoch", "split", "label", "metric", "value"]
        )
        if write_header:
            w.writeheader()
        w.writerows(rows)

# ---------------------------------------------------------------------------
# 1. Negative‑sample table loader
# ---------------------------------------------------------------------------
class Sampler:
    """Return one positive + (k‑1) negatives for each target id."""

    def __init__(self, npz: torch.Tensor, device: torch.device | str = "cpu"):
        self.table = npz

    @torch.no_grad()
    def sample(self, target_ids: torch.Tensor, k: int = 64) -> torch.Tensor:
        """target_ids: [...],  returns [..., k] with target first column."""
        neigh = self.table[target_ids][:, : k - 1]
        return torch.cat([target_ids.unsqueeze(-1), neigh], dim=-1)  # [..., k]
    
# ────────────────────────────────────────────────────────────────
# Utils
# ────────────────────────────────────────────────────────────────
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if v.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
def pretty_stats(tag: str, stats: Dict[str, float], elapsed: float) -> str:
    """Return a nicely aligned summary line."""
    return (
        f"{tag:<5} │ "
        f"loss {stats['loss']:<7.4f} │ "
        f"HR@1 {stats['HR@1']*100:6.2f}% │ "
        f"HR@5 {stats['HR@5']*100:6.2f}% │ "
        f"HR@10 {stats['HR@10']*100:6.2f}% │ "
        f"{elapsed:6.1f}s"
    )

def parse_args() -> argparse.Namespace:
    """CLI arg 들 중 **자주 바꿀만한 것만** 노출한다. 필요하면 추가."""
    p = argparse.ArgumentParser("Train SeqRecModel")

    # data ----------------------------------------------------------------
    p.add_argument("--dataset_folder", type=str, default="/home/jy1559/Datasets")
    p.add_argument("--dataset_name",   type=str, default="Retail_Rocket")
    p.add_argument("--pop_threshold",   type=int, default=5)
    p.add_argument("--train_batch_th", type=int, default=10000,
                   help="Σ(session·inter²) upper bound for a training batch")
    p.add_argument("--val_batch_th",   type=int, default=12000)
    p.add_argument("--test_batch_th",  type=int, default=15000)
    p.add_argument("--use_bucket_batching", action="store_true")
    p.add_argument("--use_add_info",     type =bool, default=False)

    # model ---------------------------------------------------------------
    p.add_argument("--model",        type=str, default="myModel")
    p.add_argument("--sampling_N",   type=int, default=0)
    p.add_argument("--embed_dim",  type=int, default=256)
    p.add_argument("--hidden_dim",  type=int, default=128)
    p.add_argument("--n_layers",   type=int, default=2)
    p.add_argument("--n_heads",    type=int, default=8)
    p.add_argument("--d_ff",       type=int, default=1024)
    p.add_argument("--dropout",    type=float, default=0.1)
    p.add_argument("--max_len",    type=int, default=128)
    p.add_argument("--pe_method",  type=str,  default="learn")
    p.add_argument("--use_dt",     type =bool, default=False)
    p.add_argument("--use_llm",    type=str2bool, nargs="?", const=True, default=True, help="LLM embedding 사용 여부 (true/false, 1/0)",)
    p.add_argument("--dt_method",  type=str, default="bucket")
    p.add_argument("--num_bucket", type=int, default=32)
    p.add_argument("--bucket_size",type=int, default=30)

    # optimisation --------------------------------------------------------
    p.add_argument("--epochs",      type=int, default=30)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--weight_decay",type=float, default=1e-4)
    p.add_argument("--optimizer",   type=str,  default="adamw")
    p.add_argument("--scheduler",   type=str,  default="cosine_warmup_restarts",
                   choices=["cosine", "step", "none", "cosine_warmup_restarts", "cosine_warmup", "cosine_restarts"],)
    p.add_argument("--warmup_steps",type=int, default=1000)
    p.add_argument("--grad_clip",   type=float, default=1.0)
    p.add_argument("--use_amp",     type=str2bool, nargs="?", const=True, default=True,)
    p.add_argument("--accumulation_steps", type=int, default=1)

    # strategy / evaluation ----------------------------------------------
    p.add_argument("--candidate_emb", type=str, default="ID_64",
                   choices=["ID_64", "LLM_128", "INPUT_256"])
    p.add_argument("--train_strategy", type=str, default="everysess_allinter")
    p.add_argument("--log_pos_metrics", action="store_true")
    p.add_argument("--test_strategy",  type=str, default="everysess_lastinter")
    p.add_argument("--num_neg",        type=int, default=128,
                   help="negative samples per positive (candidate set size ‑ 1)")

    # misc ----------------------------------------------------------------
    p.add_argument("--seed",   type=int, default=42)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--wandb_off", action="store_true",
                   help="Turn off Weights & Biases logging")
    p.add_argument("--project", type=str, default="August")

    return p.parse_args()


# ────────────────────────────────────────────────────────────────
# Train / Eval functions
# ────────────────────────────────────────────────────────────────

def train_one_epoch(model: torch.nn.Module,
                    loader: torch.utils.data.DataLoader,
                    optimizer: torch.optim.Optimizer,
                    scheduler: Any,
                    scaler: GradScaler,
                    device: torch.device,
                    cfg: Dict[str, Any],
                    epoch: int,
                    args: argparse.Namespace,
                    wandb_enabled: bool = True) -> Dict[str, float]:
    model.train()
    model.strategy = args.train_strategy

    agg_loss: float = 0.0
    agg_metrics: Dict[str, float] = defaultdict(float)
    n_batches = len(loader)

    step_ctx = autocast if args.use_amp else nullcontext  # type: ignore

    optimizer.zero_grad(set_to_none=True)
    running: Dict[str, Dict[str, float]] = {}   # ← 새 dict
    pbar = tqdm(loader, desc=f"Train E{epoch}", dynamic_ncols=True)

    for step, batch in enumerate(pbar):
        if batch is None: continue
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

        with step_ctx('cuda'):
            outputs = model(batch)
            loss, metrics = compute_loss(
                batch,                       # Dict
                outputs,                     # Dict
                model,                       # model instance
                cfg,
                strategy=args.train_strategy,              # keyword-only
                log_pos_metrics=args.log_pos_metrics, # keyword-only
                candidate_emb = args.candidate_emb
            )
            loss = loss / args.accumulation_steps
        

        if args.use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step + 1) % args.accumulation_steps == 0:
            if args.grad_clip > 0:
                if args.use_amp:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            if args.use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            torch.cuda.empty_cache()              # ✦ 캐시 반환
            torch.cuda.ipc_collect()              # ✦ 단편화·미사용 블록 회수
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()

        # ─ Aggregate
        metrics_all = metrics["ALL"]
        agg_loss += loss.item() * args.accumulation_steps
        for k, v in metrics_all.items():        # HR@1, HR@5, HR@10
            agg_metrics[k] += v.item()
            
        # ─ W&B per‑batch log (optional)
        if wandb_enabled:
            log_dict = {
                "train/loss": loss.item() * args.accumulation_steps,
                **{f"train/{k}": v.item() for k, v in metrics_all.items()},
                "train/lr": optimizer.param_groups[0]['lr'],
                "epoch_progress": epoch + step / n_batches,
            }
            if args.log_pos_metrics:            # 위치-별 비교지표 추가
                for lbl, md in metrics.items():         # ALL 포함
                    if lbl not in running:
                        running[lbl] = {m: 0.0 for m in md}
                        running[lbl]["cnt"] = 0
                    for mname, val in md.items():
                        if 'NDCG' in mname:                      # 기본 train/ 로 이미 기록
                            continue
                        log_dict[f"compare_train/{lbl}_{mname}"] = val.item()
                        if lbl != "ALL": running[lbl][mname] += val.item() 
                    running[lbl]["cnt"] += 1
            wandb.log(log_dict)
        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            HR1=f"{metrics_all['HR@1']*100:.2f}%",
            HR5=f"{metrics_all['HR@5']*100:.2f}%",
            HR10=f"{metrics_all['HR@10']*100:.2f}%"
        )
    if args.log_pos_metrics:
        rows = []
        for lbl, agg in running.items():
            if lbl == "ALL": continue  # ALL 은 따로 기록
            cnt = max(agg.get("cnt", 1), 1)
            for mname, tot in agg.items():
                if mname == "cnt" or 'NDCG' in mname or 'loss' in mname: continue
                rows.append({
                    "epoch": epoch,
                    "split": "train",
                    "label": lbl,
                    "metric": mname,
                    "value": tot / cnt
                })
        _append_metrics(rows, args)
    # ─ epoch averaging
    out = {k: v / n_batches for k, v in agg_metrics.items()}
    out["loss"] = agg_loss / n_batches

    return out


def evaluate(model: torch.nn.Module,
             loader: torch.utils.data.DataLoader,
             device: torch.device,
             args: dict,
             *,                       # 이후는 키워드 전용
             split: str = "val",      # "val" | "test"
             epoch: int | None = None # validation 때만 step=epoch 전달
             ) -> Dict[str, float]:

    wandb_enabled = (not args['wandb_off'])
    model.eval()
    model.strategy = args['test_strategy']    # 그대로 유지
    step_ctx = autocast if args['use_amp'] else nullcontext  # type: ignore
    # ─ 누적 dict 초기화 ─
    running: Dict[str, Dict[str, float]] = {}   # label → metric 합계
    running_loss = 0.0
    n_batches     = len(loader)

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Eval ({split})"):
            if batch is None: continue
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            with step_ctx('cuda'):
                outputs = model(batch)

                # ─ compute_loss 호출부 (strategy / log_pos_metrics 추가) ─
                loss, metrics = compute_loss(
                    batch, outputs, model,
                    args,
                    strategy=args["test_strategy"],
                    log_pos_metrics=args["log_pos_metrics"],
                    candidate_emb = args["candidate_emb"]
                )

            # ─ 최초 패스에 레이블 dict 생성 ─
            if not running:
                for lbl in metrics:                    # "ALL", "sess_F" ...
                    running[lbl] = {m: 0.0 for m in metrics[lbl]}
                    running[lbl]["cnt"] = 0

            # ─ 누적 ─
            running_loss += loss.item()
            for lbl, md in metrics.items():
                if lbl not in running:                          # ★ 새 라벨 초기화
                    running[lbl] = {mname: 0.0 for mname in md}
                    running[lbl]["cnt"] = 0
                for mname, val in md.items():
                    if mname not in running[lbl]:               # ★ 새 지표 초기화
                        running[lbl][mname] = 0.0
                    running[lbl][mname] += val.item()
                running[lbl]["cnt"] += 1
                """for k, v in md.items():
                    running[lbl][k] += v.item()
                running[lbl]["cnt"] += 1"""

    # ─ 평균 계산 (ALL 기준만 반환) ─
    avg_all = {k: running["ALL"][k] / running["ALL"]["cnt"]
               for k in metrics["ALL"]}
    avg_all["loss"] = running_loss / n_batches

    # ─ W&B epoch-level 로깅 ─
    prefix = "val" if split == "val" else ("test" if split=="test" else "train")
    # ── scalar log dict ─────────────────────────────────────────
    log_dict = {f"{prefix}/{k}": v for k, v in avg_all.items()}

    rows = []
    if args["log_pos_metrics"]:
        for lbl, agg in running.items():
            cnt = max(agg.get("cnt", 1), 1)
            for mname, tot in agg.items():
                if mname in ("cnt",) or "NDCG" in mname or 'loss' in mname:
                    continue
                log_dict[f"compare_{prefix}/{lbl}_{mname}"] = tot / cnt
                rows.append({
                "epoch": epoch if epoch is not None else -1,
                "split": split,
                "label": lbl,
                "metric": mname,
                "value": tot / cnt,
                })
        _append_metrics(rows, args)


    return avg_all, log_dict


# ────────────────────────────────────────────────────────────────
# Scheduler builder (간단 버전)
# ────────────────────────────────────────────────────────────────

def build_scheduler(optimizer: torch.optim.Optimizer, args: argparse.Namespace, steps_per_epoch: int):
    if args.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * steps_per_epoch)
    elif args.scheduler == "cosine_warmup":
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=args.epochs * steps_per_epoch, T_mult=2, eta_min=1e-6)
    elif args.scheduler == "cosine_warmup_restarts":
        return CosineAnnealingWarmUpRestarts(optimizer, T_0=int(steps_per_epoch*6), T_mult=1, eta_max=args.lr, T_up=int(steps_per_epoch*0.1), gamma=0.9)
    elif args.scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=20000, gamma=0.6)
    else:
        return None


# ────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────

def main():
    wall_start_time = time.time()
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    # 1) W&B ----------------------------------------------------------------
    wandb_enabled = not args.wandb_off
    project_name = args.project + "_" + args.dataset_name  + "_"  + args.model + "_" + str(args.sampling_N)
    if args.candidate_emb == "LLM_128" and args.use_llm == False: 
        assert "candidate에서 LLM emb 활용하기 위해선 use_llm이 True여야함"
        print("use_llm이 False로 설정되어 LLM embedding을 사용하지 않습니다")
        sys.exit(0)
    if (args.model in ['H-RNN'] ) and args.embed_dim != args.hidden_dim: 
        print("H-RNN 모델은 embed_dim과 hidden_dim이 같아야 합니다, hidden_dim을 embed_dim으로 설정합니다")
        args.embed_dim = args.hidden_dim
    if wandb_enabled:

        wandb.init(project=project_name, config=vars(args))
        wandb.define_metric("epoch_progress")
        wandb.define_metric("train/*", step_metric="epoch_progress", overwrite=True)
        wandb.define_metric("compare_train/*", step_metric="epoch", overwrite=True)
        wandb.define_metric("epoch")
        wandb.define_metric("val/*", step_metric="epoch")
        wandb.define_metric("compare_val/*", step_metric="epoch")
        wandb.define_metric("mean_time", step_metric="epoch")
    if args.dataset_name == 'LFM-BeyMS': args.use_add_info = False
    if args.wandb_off:
        args.log_pos_metrics = False

    print(f"========================================{project_name}========================================")
    SAMPLING_NUM = {0: 0,  4:1, 16:2, 64:3}
    NUM_ITEM = {'Globo': [31070, 6355, -1, -1],     'LFM-BeyMS': [1068291, 452881, 192370], 'Retail_Rocket': [56257, 20875, -1, -1],
                'Steam':[35618, 15810, -1, -1],      'MovieLens':[48817, 5145, -1, -1],      'Taobao':[4070153, -1, 1197051, 542104]}
    num_items = NUM_ITEM[args.dataset_name][SAMPLING_NUM[args.sampling_N]]
    if args.dataset_name == 'Globo':
        add_info_num_cat = [('cat', 11), ('cat', 5), ('cat', 4), ('cat', 20), ('cat', 7), ('cat', 28)]
    elif args.dataset_name == 'LFM-BeyMS':
        add_info_num_cat = []
    elif args.dataset_name == 'Retail_Rocket':
        add_info_num_cat = [('cat', 3), ('num', -1)]
    elif args.dataset_name in ['Cloth', 'Home', 'Toy']:
        args.dataset_name = 'Amazon/'+ args.dataset_name
        add_info_num_cat = [('cat', 3), ('num', -1), ('cat', 2)]
    elif args.dataset_name == 'Steam':
        add_info_num_cat = [('num', -1), ('num', -1), ('num', -1)]
    elif args.dataset_name == 'MovieLens':
        add_info_num_cat = [('cat', 5)]
    elif args.dataset_name == 'Taobao':
        add_info_num_cat = [('cat', 4)]
    else:
        assert f'No dataset name {args.dataset_name}'
    args.sampling_N = 0 if args.sampling_N == 1 else f'{args.sampling_N}-Sampling'
    
    # 2) Dataloaders ---------------------------------------------------------
    train_loader, val_loader, test_loader = dataset.get_dataloaders(
        dataset_folder=args.dataset_folder,
        dataset_name=args.dataset_name,
        sampling_N = args.sampling_N,
        train_batch_th=args.train_batch_th,
        val_batch_th=args.val_batch_th,
        test_batch_th=args.test_batch_th,
        use_bucket_batching=args.use_bucket_batching,
        use_add_info=args.use_add_info,
    )
    stat_path = args.dataset_folder + '/' + args.dataset_name + '/timesplit/' + args.sampling_N + '/item_stats.npz'
    freq_arr  = np.load(stat_path)["train_counts"]          # np.int32
    args.item_freq = torch.as_tensor(freq_arr, device=device)
    # ─ hot / cold 경계 (상위 20%) ─
    valid = args.item_freq[1:].float()          # PAD(0) 제거
    args.pop_threshold = torch.quantile(valid, 0.80)

    negSample_path = args.dataset_folder + '/' + args.dataset_name + '/timesplit/' + args.sampling_N + '/negative_samples.npz'
    negSample  = np.load(negSample_path)["negatives"]          # np.int32
    negSam = torch.as_tensor(negSample, device=device)
    args.negSampler = Sampler(negSam, device=device)
    
    # 3) Model --------------------------------------------------------------
    model_cfg = vars(args)  # 간편: Namespace 통째 dict (모듈에서 필요한 key만 사용)
    model_cfg["add_info_specs"] = add_info_num_cat
    model = build_model(args.model, num_items, model_cfg).to(device)

    # 4) Optimiser / Scheduler / AMP scaler --------------------------------
    optim_class = {
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
    }[args.optimizer.lower()]
    optimizer = optim_class(model.parameters(), lr=args.lr if args.scheduler != 'cosine_warmup_restarts' else args.lr / 10, weight_decay=args.weight_decay)
    scheduler = build_scheduler(optimizer, args, len(train_loader))
    scaler = GradScaler('cuda', enabled=args.use_amp)

    # 5) Epoch loop ---------------------------------------------------------
    best_val_loss = float("inf")
    train_times = []
    val_times = []
    test_times = []
    val_best_metrics = {}
    times = []
    for epoch in range(args.epochs):
        start = time.time()
        train_stats = train_one_epoch(model, train_loader, optimizer, scheduler, scaler, device, vars(args), epoch, args, wandb_enabled)
        train_time = time.time() - start
        train_times.append(train_time)
        start = time.time()
        val_stats, log_dict  = evaluate(model, val_loader, device, vars(args), split="val", epoch=epoch)
        val_time = time.time() - start
        val_times.append(val_time)
        times.append(train_time + val_time)
        # logging --------------------------------------------------------
        if wandb_enabled:
            for lbl, val in val_stats.items():
                if lbl not in val_best_metrics:
                    val_best_metrics[lbl] = val
                else:
                    val_best_metrics[lbl] = max(val_best_metrics[lbl], val) if lbl != "loss" else min(val_best_metrics[lbl], val)
            wandb.log({**{f"val/train_{k}": v for k, v in train_stats.items()},
                       **{f"bestVal/{k}": v for k, v in val_best_metrics.items()},
                       **{f"val/{k}": v for k, v in val_stats.items()},
                       **{f"{k}": v for k, v in log_dict.items()},
                       "epoch": epoch,
                       "mean_time": sum(times)/len(times),})
            
        # save ckpt ------------------------------------------------------
        if val_stats["loss"] < best_val_loss:
            best_val_loss = val_stats["loss"]
            ckpt_path = Path("./outputs/best.pth")
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            """torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
            }, ckpt_path)"""
            """if wandb_enabled:
                wandb.save(str(ckpt_path))"""

         # ─ stdout neat table
        print("\n" + "─" * 90)
        print(f"Epoch {epoch:02d} │ " + pretty_stats("train", train_stats, train_time))
        print(f"         │ " + pretty_stats("val",   val_stats,   time.time() - start))
        print("─" * 90 + "\n")
    # 6) Final test ---------------------------------------------------------
    start = time.time()
    test_stats, log_dict = evaluate(model, test_loader, device, vars(args), split="test")
    test_time = time.time() - start
    test_times.append(test_time)
    if wandb_enabled:
        wandb.log({f"test/{k}": v for k, v in test_stats.items()})
        wandb.log({f"{k}": v for k, v in log_dict.items()})
        wandb.finish()

    print("\n===== TEST METRICS =====")
    for k, v in test_stats.items():
        print(f"{k}: {v:.4f}")
    print(f"========================================{project_name}========================================")
    total_time = time.time() - wall_start_time
    train_times_mean = np.mean(train_times)
    val_times_mean = np.mean(val_times)
    test_times_mean = np.mean(test_times)
    print(f"Total time: {total_time//60} min ({total_time % 60:.1f} sec)")
    print(f"Train mean time: {train_times_mean//60} min ({train_times_mean % 60:.1f} sec)")
    print(f"Validation mean time: {val_times_mean//60} min ({val_times_mean % 60:.1f} sec)")
    print(f"Train mean time: {test_times_mean//60} min ({test_times_mean % 60:.1f} sec)")

if __name__ == "__main__":
    main()
