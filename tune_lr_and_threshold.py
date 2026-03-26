#!/usr/bin/env python3
"""
tune_lr_and_threshold.py  

LR predicts P(top-1 is correct) using ONLY inliers_top1.
- Select C by maximizing ROC-AUC on pooled validation 
- Select threshold t by maximizing classification accuracy on pooled validation 

Also computes adaptive R@1 + savings vs full rerank for each threshold (for plots/report).

Requires columns in CSVs:
  - inliers_top1
  - baseline_correct
  - reranked_correct   (must exist in validation CSVs to simulate policy)

Decision rule:
  EASY if P(correct) >= t
  HARD otherwise
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def parse_list_or_range(s: str) -> list[float]:
    """
    Accept either:
      - "0.05,0.15,0.25"
      - "0.00:1.00:0.10"  (start:stop:step, inclusive of start, inclusive of stop if hits exactly)
    """
    s = s.strip()
    if ":" in s:
        parts = s.split(":")
        if len(parts) != 3:
            raise ValueError(f"Bad thresholds range format: {s}")
        start, stop, step = map(float, parts)
        if step <= 0:
            raise ValueError("step must be > 0")
        vals = []
        x = start
        while x <= stop + 1e-12:
            vals.append(round(x, 10))
            x += step
        return vals
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def load_csvs(paths: list[str], require_reranked: bool) -> pd.DataFrame:
    dfs = []
    for p in paths:
        pth = Path(p)
        df = pd.read_csv(pth)

        needed = {"inliers_top1", "baseline_correct"}
        missing = needed - set(df.columns)
        if missing:
            raise ValueError(f"{pth} missing columns: {sorted(missing)}")

        if require_reranked:
            if "reranked_correct" not in df.columns:
                raise ValueError(f"{pth} missing reranked_correct (required for validation policy sim)")
            if df["reranked_correct"].isna().any():
                raise ValueError(f"{pth} has NaNs in reranked_correct")

        df = df.copy()
        df["baseline_correct"] = df["baseline_correct"].astype(int)
        if "reranked_correct" in df.columns:
            df["reranked_correct"] = df["reranked_correct"].astype(int)

        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


def build_X(df: pd.DataFrame) -> np.ndarray:
    return df[["inliers_top1"]].to_numpy(dtype=np.float32)


def compute_policy_metrics(val_df: pd.DataFrame, proba_correct: np.ndarray, thr: float) -> dict:
    """
    EASY if proba_correct >= thr
    HARD otherwise
    Adaptive correctness uses reranked_correct on HARD queries and baseline_correct on EASY.
    """
    baseline = val_df["baseline_correct"].to_numpy(dtype=int)
    reranked = val_df["reranked_correct"].to_numpy(dtype=int)

    hard = (proba_correct < thr)  # low confidence -> expensive rerank
    adaptive = np.where(hard, reranked, baseline)

    baseline_r1 = float(baseline.mean())
    fullrerank_r1 = float(reranked.mean())
    adaptive_r1 = float(adaptive.mean())

    hard_rate = float(hard.mean())
    savings_vs_full = 1.0 - (1.0 + 19.0 * hard_rate) / 20.0  # Option B cost model

    #threshold selection: classification accuracy for predicting baseline_correct
    pred_correct = (proba_correct >= thr).astype(int)
    acc = float((pred_correct == baseline).mean())

    return {
        "threshold": float(thr),
        "clf_acc": acc,
        "baseline_R@1": baseline_r1,
        "fullrerank_R@1": fullrerank_r1,
        "adaptive_R@1": adaptive_r1,
        "hard_rate": hard_rate,
        "savings_vs_fullrerank": float(savings_vs_full),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-csvs", nargs="+", required=True)
    ap.add_argument("--val-csvs", nargs="+", required=True)

    ap.add_argument("--out-model", type=str, required=True)
    ap.add_argument("--out-json", type=str, required=True)
    ap.add_argument("--out-table", type=str, required=True)  # per-C summary
    ap.add_argument("--out-curve", type=str, required=True)  # threshold curve for best_C

    ap.add_argument("--C-grid", type=str, default="0.01,0.03,0.1,0.3,1,3,10,30,100")
    ap.add_argument("--thresholds", type=str,
                    default="0.05,0.15,0.25,0.35,0.45,0.55,0.65,0.75,0.85,0.95",
                    help="Either comma list or start:stop:step. Use ~10 values for your report.")

    ap.add_argument("--solver", type=str, default="liblinear")
    ap.add_argument("--max-iter", type=int, default=2000)
    ap.add_argument("--class-weight", type=str, default="none", choices=["none", "balanced"])

    args = ap.parse_args()

    C_grid = parse_list_or_range(args.C_grid)
    thresholds = parse_list_or_range(args.thresholds)

    train_df = load_csvs(args.train_csvs, require_reranked=False)
    val_df = load_csvs(args.val_csvs, require_reranked=True)

    X_train = build_X(train_df)
    y_train = train_df["baseline_correct"].to_numpy(dtype=int)

    X_val = build_X(val_df)
    y_val = val_df["baseline_correct"].to_numpy(dtype=int)

    cw = None if args.class_weight == "none" else "balanced"

    perC_rows = []
    best = None

    for C in C_grid:
        lr = LogisticRegression(
            C=float(C),
            solver=args.solver,
            max_iter=args.max_iter,
            class_weight=cw,
        )
        lr.fit(X_train, y_train)

        proba_val = lr.predict_proba(X_val)[:, 1]  # P(correct)
        #pick C by ROC-AUC on validation
        try:
            auc = float(roc_auc_score(y_val, proba_val))
        except ValueError:
            auc = float("nan")

        perC_rows.append({"C": float(C), "val_roc_auc": auc})

        if best is None:
            best = {"C": float(C), "val_roc_auc": auc, "model": lr, "proba_val": proba_val}
        else:
            if (not np.isnan(auc)) and (np.isnan(best["val_roc_auc"]) or auc > best["val_roc_auc"]):
                best = {"C": float(C), "val_roc_auc": auc, "model": lr, "proba_val": proba_val}

    perC_df = pd.DataFrame(perC_rows).sort_values("C").reset_index(drop=True)
    Path(args.out_table).parent.mkdir(parents=True, exist_ok=True)
    perC_df.to_csv(args.out_table, index=False)

    # Sweep thresholds for best_C
    best_lr = best["model"]
    proba_val = best["proba_val"]

    curve_rows = [compute_policy_metrics(val_df, proba_val, float(thr)) for thr in thresholds]
    curve_df = pd.DataFrame(curve_rows).sort_values("threshold").reset_index(drop=True)
    Path(args.out_curve).parent.mkdir(parents=True, exist_ok=True)
    curve_df.to_csv(args.out_curve, index=False)

    # select threshold by max classification accuracy; tie-break by savings
    tol = 1e-12
    max_acc = curve_df["clf_acc"].max()
    cand = curve_df[curve_df["clf_acc"] >= max_acc - tol].copy()
    cand = cand.sort_values(["clf_acc", "savings_vs_fullrerank"], ascending=[False, False])
    best_row = cand.iloc[0].to_dict()

    # Save model
    out_model = Path(args.out_model)
    out_model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_lr, out_model)

    params = {
        "label": "baseline_correct (P(top1 is correct))",
        "decision_rule": "EASY if P(correct) >= t else HARD",
        "features": ["inliers_top1"],
        "best_C": float(best["C"]),
        "best_threshold": float(best_row["threshold"]),
        "val_roc_auc_at_best_C": float(best["val_roc_auc"]),
        "val_clf_acc": float(best_row["clf_acc"]),
        "val_baseline_R@1": float(best_row["baseline_R@1"]),
        "val_fullrerank_R@1": float(best_row["fullrerank_R@1"]),
        "val_adaptive_R@1": float(best_row["adaptive_R@1"]),
        "val_hard_rate": float(best_row["hard_rate"]),
        "val_savings_vs_fullrerank": float(best_row["savings_vs_fullrerank"]),
        "thresholds_swept": thresholds,
        "C_grid": C_grid,
        "class_weight": args.class_weight,
        "solver": args.solver,
        "max_iter": int(args.max_iter),
    }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(params, indent=2))

    print(f"Saved model: {out_model}")
    print(f"Saved params: {out_json}")
    print(f"Saved per-C summary: {args.out_table}")
    print(f"Saved best-C threshold curve: {args.out_curve}\n")

    print("Best selection :")
    print(f"  best_C: {params['best_C']}")
    print(f"  best_threshold: {params['best_threshold']}")
    print(f"  val_roc_auc: {params['val_roc_auc_at_best_C']}")
    print(f"  val_clf_acc: {params['val_clf_acc']}")
    print(f"  val_baseline_R@1: {params['val_baseline_R@1']}")
    print(f"  val_fullrerank_R@1: {params['val_fullrerank_R@1']}")
    print(f"  val_adaptive_R@1: {params['val_adaptive_R@1']}")
    print(f"  val_hard_rate: {params['val_hard_rate']}")
    print(f"  val_savings_vs_fullrerank: {params['val_savings_vs_fullrerank']}")


if __name__ == "__main__":
    main()
