#!/usr/bin/env python3
"""
build_csv.py

Build per-query CSV for LR training/validation.

Required output columns (and ONLY these):
  - query_id
  - inliers_top1
  - baseline_correct
  - reranked_correct

Inputs:
  - preds_dir: folder containing retrieval prediction txt files (e.g., 000.txt, 001.txt, ...)
  - inliers_dir: folder containing matcher outputs torch files (e.g., 000.torch, 001.torch, ...)
    Each torch file is a list[dict] with "num_inliers" for each candidate.

Logic:
  - baseline_correct: dist(top1) <= positive_dist_threshold
  - reranked_correct: dist(argmax(num_inliers over top-K)) <= positive_dist_threshold
  - inliers_top1: num_inliers for candidate rank-1 (index 0) in retrieval order

Notes:
  - Works with stems like "0" or "000".
  - If strict is off, skips problematic queries instead of crashing.
"""

import argparse
from pathlib import Path
from glob import glob
import math

import pandas as pd
import torch

from util import get_list_distances_from_preds


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--preds-dir", type=str, required=True,
                   help="Directory containing retrieval prediction txt files (e.g., .../preds)")
    p.add_argument("--inliers-dir", type=str, required=True,
                   help="Directory containing matcher outputs (e.g., .../preds_loftr or .../preds_superpoint-lg)")
    p.add_argument("--out-csv", type=str, required=True,
                   help="Output CSV path")
    p.add_argument("--num-preds", type=int, default=20,
                   help="How many candidates to consider (default: 20)")
    p.add_argument("--positive-dist-threshold", type=float, default=25.0,
                   help="Meters threshold for a prediction to be considered correct (default: 25m)")
    p.add_argument("--strict", action="store_true",
                   help="Fail on any missing/corrupt file instead of skipping.")
    return p.parse_args()


def main():
    args = parse_args()
    preds_dir = Path(args.preds_dir)
    inliers_dir = Path(args.inliers_dir)
    out_csv = Path(args.out_csv)

    if not preds_dir.exists():
        raise FileNotFoundError(f"preds_dir not found: {preds_dir}")
    if not inliers_dir.exists():
        raise FileNotFoundError(f"inliers_dir not found: {inliers_dir}")

    txt_files = glob(str(preds_dir / "*.txt"))
    if not txt_files:
        raise RuntimeError(f"No .txt files found in {preds_dir}")

    # Sort by integer stem ("000" -> 0)
    try:
        txt_files = sorted(txt_files, key=lambda x: int(Path(x).stem))
    except Exception:
        # fallback lexicographic
        txt_files = sorted(txt_files)

    rows = []
    skipped = 0
    K = int(args.num_preds)
    pos_thr = float(args.positive_dist_threshold)

    for txt_path_str in txt_files:
        txt_path = Path(txt_path_str)
        stem = txt_path.stem               # "000"
        try:
            qid = int(stem)                # 0
        except Exception:
            qid = stem                     # keep as string if not numeric

        torch_path = inliers_dir / f"{stem}.torch"
        if not torch_path.exists():
            msg = f"[Missing torch] {torch_path}"
            if args.strict:
                raise FileNotFoundError(msg)
            skipped += 1
            continue

        # distances in meters for each candidate in retrieval order
        try:
            dists = get_list_distances_from_preds(str(txt_path))
        except Exception as e:
            msg = f"[Bad preds txt] {txt_path}: {e}"
            if args.strict:
                raise RuntimeError(msg)
            skipped += 1
            continue

        if len(dists) == 0:
            if args.strict:
                raise RuntimeError(f"[Empty preds] {txt_path}")
            skipped += 1
            continue

        # matcher results: list of dicts with "num_inliers"
        try:
            results = torch.load(torch_path, weights_only=False)
        except Exception as e:
            msg = f"[Bad torch file] {torch_path}: {e}"
            if args.strict:
                raise RuntimeError(msg)
            skipped += 1
            continue

        if not isinstance(results, list) or len(results) == 0:
            if args.strict:
                raise RuntimeError(f"[Empty/invalid torch list] {torch_path}")
            skipped += 1
            continue

        L = min(K, len(dists), len(results))
        if L <= 0:
            if args.strict:
                raise RuntimeError(f"[No usable candidates] {txt_path}")
            skipped += 1
            continue

        # collect inliers
        inliers = []
        ok = True
        for i in range(L):
            if not isinstance(results[i], dict) or ("num_inliers" not in results[i]):
                ok = False
                break
            inliers.append(float(results[i]["num_inliers"]))
        if not ok:
            msg = f"[Missing num_inliers] {torch_path} (need num_inliers in each entry)"
            if args.strict:
                raise KeyError(msg)
            skipped += 1
            continue

        # baseline correctness (top-1 retrieval)
        baseline_correct = 1 if float(dists[0]) <= pos_thr else 0

        # reranked correctness (argmax inliers over top-L)
        best_idx = max(range(L), key=lambda i: inliers[i])
        reranked_correct = 1 if float(dists[best_idx]) <= pos_thr else 0

        rows.append({
            "query_id": qid,
            "inliers_top1": float(inliers[0]),
            "baseline_correct": int(baseline_correct),
            "reranked_correct": int(reranked_correct),
        })

    df = pd.DataFrame(rows)
    if len(df) > 0 and "query_id" in df.columns:
        # keep deterministic ordering if query_id numeric
        try:
            df = df.sort_values("query_id").reset_index(drop=True)
        except Exception:
            df = df.reset_index(drop=True)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    print(f"Saved CSV: {out_csv}")
    print(f"Rows: {len(df)} | Skipped: {skipped}")
    if len(df) > 0:
        print("Class balance (baseline_correct):")
        print(df["baseline_correct"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
