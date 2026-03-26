#!/usr/bin/env python3
"""
adaptive_match_and_eval.py

One-script "official pipeline + LR" (Behavior A) + evaluation:

Per query:
  1) Read retrieval list (top-K) from preds/*.txt (query path + candidate paths)
  2) Always match top-1 to get inliers_top1
  3) LR predicts P(correct) from inliers_top1
  4) EASY if P(correct) >= threshold  -> stop, keep retrieval ranking
     HARD otherwise                   -> match remaining candidates up to K, rerank by inliers
  5) Save per-query matcher outputs to out-dir/qid.torch
       - EASY: list length 1
       - HARD: list length K (or fewer if fewer candidates exist)

At the end (over all evaluated queries), it prints:
  - Baseline R@{1,5,10,20}  (retrieval-only)
  - Adaptive R@{1,5,10,20}  (EASY uses retrieval order, HARD uses inliers-reranked order)
  - hard_rate
  - avg_matches_per_query (actual: 1 for easy, K for hard)
  - savings_vs_full (1 - avg_matches_per_query/K)
  - counts: evaluated, skipped

Optionally writes:
  - JSON summary with the metrics
  - JSONL log with one line per query (decision, p_correct, inliers_top1, matched_count)
"""

import os
import sys
import json
import argparse
from glob import glob
from pathlib import Path
from copy import deepcopy

import numpy as np
import torch
import joblib
from tqdm import tqdm

from util import read_file_preds, get_list_distances_from_preds

sys.path.append(str(Path(__file__).parent.joinpath("image-matching-models")))
from matching import get_matcher, available_models
from matching.utils import get_default_device

def remap_path(p: str, data_root: str | None) -> str:
    # normalize separators
    s = p.replace("\\", "/")

    # if it's a Windows path like D:/..., drop the drive prefix
    if len(s) >= 3 and s[1:3] == ":/":
        s = s.split(":/", 1)[1]  # keep path after "D:/"

    # If no data_root provided, just return normalized path
    if not data_root:
        return s

    # If path contains "/data/<dataset>/...", map everything after "/data/" to data_root
    # Examples:
    #   .../data/tokyo_xs/test/queries/xxx.jpg  ->  /content/vpr_data/tokyo_xs/test/queries/xxx.jpg
    if "/data/" in s:
        tail = s.split("/data/", 1)[1]          # "tokyo_xs/test/queries/..."
        return str(Path(data_root) / tail)

    # If it already looks like "tokyo_xs/..." (rare), map directly
    parts = s.strip("/").split("/")
    if len(parts) > 0 and parts[0].endswith("_xs"):
        return str(Path(data_root) / s.strip("/"))

    # Otherwise keep as-is
    return s

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--preds-dir", type=str, required=True, help="folder with retrieval preds *.txt")
    p.add_argument("--out-dir", type=str, required=True, help="output dir for adaptive matcher results (*.torch)")

    p.add_argument("--data-root", type=str, default=None, help="If preds paths are Windows/absolute, remap to this dataset root (e.g., /content/vpr_data)")

    p.add_argument("--matcher", type=str, required=True, choices=available_models)
    p.add_argument("--device", type=str, default=get_default_device(), choices=["cpu", "cuda"])
    p.add_argument("--im-size", type=int, default=512)
    p.add_argument("--num-preds", type=int, default=20)

    p.add_argument("--lr-model", type=str, required=True, help="joblib LogisticRegression")
    p.add_argument("--lr-json", type=str, required=True, help="json containing best_threshold")
    p.add_argument("--threshold", type=float, default=None,
                   help="override threshold; if not set uses best_threshold from lr-json")

    p.add_argument("--positive-dist-threshold", type=float, default=25.0)

    p.add_argument("--recall-values", type=str, default="1,5,10,20",
                   help="comma-separated recall@k values to compute (default: 1,5,10,20)")

    p.add_argument("--log-jsonl", type=str, default=None,
                   help="optional jsonl file path for per-query logs")
    p.add_argument("--out-json", type=str, default=None,
                   help="optional output json summary")

    p.add_argument("--strict", action="store_true",
                   help="fail on any error; otherwise skip problematic queries")

    return p.parse_args()


def hit_at_k(dists: np.ndarray, order: np.ndarray, k: int, pos_thr: float) -> int:
    kk = min(k, len(order), len(dists))
    if kk <= 0:
        return 0
    top = order[:kk]
    return 1 if np.any(dists[top] <= pos_thr) else 0


def main():
    args = parse_args()

    preds_dir = Path(args.preds_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    lr = joblib.load(args.lr_model)
    lr_params = json.loads(Path(args.lr_json).read_text())
    thr = float(args.threshold) if args.threshold is not None else float(lr_params["best_threshold"])

    matcher = get_matcher(args.matcher, device=args.device)

    K = int(args.num_preds)
    pos_thr = float(args.positive_dist_threshold)
    recall_values = sorted(set(int(x.strip()) for x in args.recall_values.split(",") if x.strip()))

    txt_files = glob(os.path.join(str(preds_dir), "*.txt"))
    txt_files.sort(key=lambda x: int(Path(x).stem))
    if not txt_files:
        raise RuntimeError(f"No preds txt files found in {preds_dir}")

    log_f = None
    if args.log_jsonl:
        lp = Path(args.log_jsonl)
        lp.parent.mkdir(parents=True, exist_ok=True)
        log_f = open(lp, "a", encoding="utf-8")

    # Aggregates
    n_eval = 0
    n_skip = 0
    hard_count = 0
    total_matches = 0

    base_hits = {k: 0 for k in recall_values}
    adapt_hits = {k: 0 for k in recall_values}

    for txt in tqdm(txt_files, desc="Adaptive match+eval"):
        qid = Path(txt).stem
        out_file = out_dir / f"{qid}.torch"

        try:
            dists_list = get_list_distances_from_preds(txt)
            q_path, pred_paths = read_file_preds(txt)
            # --- Remap Windows/absolute paths from preds.txt into Colab dataset root ---
            if args.data_root:
                q_path = remap_path(q_path, args.data_root)
                pred_paths = [remap_path(p, args.data_root) for p in pred_paths]

            # Normalize slashes (pred files sometimes contain backslashes)
            q_path = q_path.replace("\\", "/")
            pred_paths = [p.replace("\\", "/") for p in pred_paths]
            
            q_path = remap_path(q_path, args.data_root)
            pred_paths = [remap_path(pp, args.data_root) for pp in pred_paths]

            if n_eval == 0 and n_skip == 0:
                if not Path(q_path).exists():
                    raise FileNotFoundError(f"Query path after remap does not exist: {q_path}")
                if not Path(pred_paths[0]).exists():
                    raise FileNotFoundError(f"Pred0 path after remap does not exist: {pred_paths[0]}")

            if len(dists_list) == 0 or len(pred_paths) == 0:
                raise RuntimeError("Empty dists or pred_paths")

            kk = min(K, len(pred_paths), len(dists_list))
            dists = np.array(dists_list[:kk], dtype=np.float32)

            # Baseline metrics (retrieval order)
            retrieval_order = np.arange(kk, dtype=np.int64)
            for k in recall_values:
                base_hits[k] += hit_at_k(dists, retrieval_order, k, pos_thr)

            # If we already have saved results, load to avoid recompute
            cached = None
            if out_file.exists():
                cached = torch.load(out_file, weights_only=False)
                if not isinstance(cached, list) or len(cached) == 0:
                    cached = None

            # Always ensure we have top-1 match in the saved file
            if cached is None:
                # match top-1
                img0 = matcher.load_image(q_path, resize=args.im_size)
                img1 = matcher.load_image(pred_paths[0], resize=args.im_size)
                r0 = matcher(deepcopy(img0), img1)
                r0["all_desc0"] = r0["all_desc1"] = None
                results = [r0]
                torch.save(results, out_file)
            else:
                results = cached
                # If cached exists but does not contain top-1, recompute it
                if len(results) < 1 or ("num_inliers" not in results[0]):
                    img0 = matcher.load_image(q_path, resize=args.im_size)
                    img1 = matcher.load_image(pred_paths[0], resize=args.im_size)
                    r0 = matcher(deepcopy(img0), img1)
                    r0["all_desc0"] = r0["all_desc1"] = None
                    results = [r0]
                    torch.save(results, out_file)

            inliers_top1 = float(results[0].get("num_inliers", 0.0))

            # LR decision
            x = np.array([[inliers_top1]], dtype=np.float32)
            p_correct = float(lr.predict_proba(x)[0, 1])
            hard = (p_correct < thr)

            # If HARD, ensure we have full top-K matches saved (extend from 1 to K)
            if hard and len(results) < kk:
                # load query once
                img0 = matcher.load_image(q_path, resize=args.im_size)
                start = len(results)
                for pred_path in pred_paths[start:kk]:
                    img1 = matcher.load_image(pred_path, resize=args.im_size)
                    r = matcher(deepcopy(img0), img1)
                    r["all_desc0"] = r["all_desc1"] = None
                    results.append(r)
                torch.save(results, out_file)

            # Determine adaptive ranking order:
            # EASY -> retrieval order
            # HARD -> inliers rerank among matched items (should be kk)
            if hard:
                hard_count += 1
                mm = min(len(results), kk)
                inl = np.array([float(results[i].get("num_inliers", 0.0)) for i in range(mm)], dtype=np.float32)
                adapt_order = np.argsort(-inl)  # indices over [0..mm-1]
                # if for some reason mm < kk, append remaining retrieval order
                if mm < kk:
                    adapt_order = np.concatenate([adapt_order, np.arange(mm, kk, dtype=np.int64)], axis=0)
                matched = mm
            else:
                adapt_order = retrieval_order
                matched = 1

            # Adaptive metrics
            for k in recall_values:
                adapt_hits[k] += hit_at_k(dists, adapt_order, k, pos_thr)

            total_matches += matched
            n_eval += 1

            if log_f:
                log_f.write(json.dumps({
                    "qid": int(qid),
                    "hard": bool(hard),
                    "p_correct": p_correct,
                    "threshold": thr,
                    "inliers_top1": inliers_top1,
                    "matched": int(matched),
                    "out_file": str(out_file),
                }) + "\n")

        except Exception as e:
            if args.strict:
                raise
            n_skip += 1
            if log_f:
                log_f.write(json.dumps({
                    "qid": int(qid),
                    "error": str(e),
                    "out_file": str(out_file),
                }) + "\n")
            continue

    if log_f:
        log_f.close()

    if n_eval == 0:
        raise RuntimeError("No queries evaluated (all skipped?)")

    baseline = {f"R@{k}": base_hits[k] / n_eval for k in recall_values}
    adaptive = {f"R@{k}": adapt_hits[k] / n_eval for k in recall_values}

    hard_rate = hard_count / n_eval
    avg_matches_per_query = total_matches / n_eval
    savings_vs_full = 1.0 - (avg_matches_per_query / float(K))

    summary = {
        "preds_dir": str(preds_dir),
        "out_dir": str(out_dir),
        "matcher": args.matcher,
        "device": args.device,
        "im_size": args.im_size,
        "K": K,
        "positive_dist_threshold": pos_thr,
        "recall_values": recall_values,
        "lr_model": str(args.lr_model),
        "lr_threshold": thr,
        "evaluated_queries": n_eval,
        "skipped_queries": n_skip,
        "hard_rate": hard_rate,
        "avg_matches_per_query": avg_matches_per_query,
        "savings_vs_fullrerank": savings_vs_full,
        "baseline": baseline,
        "adaptive": adaptive,
    }

    # Print
    print("\n=== Adaptive Match + Eval (Official pipeline + LR) ===")
    print(f"preds_dir: {preds_dir}")
    print(f"out_dir:   {out_dir}")
    print(f"matcher:   {args.matcher}  device={args.device}  im_size={args.im_size}")
    print(f"LR:        {Path(args.lr_model).name}  thr={thr}")
    print(f"K:         {K}  pos_thr(m): {pos_thr}")
    print(f"Q eval:    {n_eval}  skipped: {n_skip}")
    print("baseline:", "  ".join([f"R@{k}={baseline[f'R@{k}']:.6f}" for k in recall_values]))
    print("adaptive:", "  ".join([f"R@{k}={adaptive[f'R@{k}']:.6f}" for k in recall_values]))
    print(f"hard_rate:             {hard_rate:.6f}")
    print(f"avg_matches_per_query: {avg_matches_per_query:.6f} (full={K})")
    print(f"savings_vs_full:       {savings_vs_full:.6f}")

    if args.out_json:
        outp = Path(args.out_json)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps(summary, indent=2))
        print(f"Saved: {outp}")


if __name__ == "__main__":
    main()
