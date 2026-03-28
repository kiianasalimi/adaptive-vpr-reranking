# **Visual Place Recognition – Extension 6.1: Adaptive Re-ranking**

This repository extends a Visual Place Recognition (VPR) pipeline with **Extension 6.1: adaptive re-ranking**.

## **Summary**

Baseline VPR produces a **top-K retrieval list** (we use **K \= 20**). A strong but expensive improvement is to run a local feature matcher (LoFTR or SuperPoint \+ LightGlue) between the query and **all top-20** candidates, then **rerank** by geometric consistency (number of inliers). This typically improves Recall@k but is computationally heavy.

**Adaptive re-ranking** reduces compute by performing full top-20 matching **only for hard queries**:

1. Always match **top-1** candidate → obtain inliers\_top1  
2. Logistic Regression (LR) predicts P(top1\_correct) from inliers\_top1  
3. If P(top1\_correct) \>= threshold: **EASY** → stop (keep retrieval order)  
4. Else: **HARD** → match remaining candidates up to K and rerank by inliers

We report:

* Baseline Recall@{1,5,10,20} (retrieval-only)  
* Adaptive Recall@{1,5,10,20} (gated re-ranking)  
* Compute proxies: hard\_rate, savings\_vs\_fullrerank

## **Methods used**

**VPR backbones (retrieval):**

* MixVPR  
* CosPlace

**Image matching methods (geometric verification):**

* LoFTR  
* SuperPoint + LightGlue

**Adaptive gate:**

* Logistic Regression with a single feature: inliers\_top1

## **Files added for the extension**

We added **three scripts**:

1. build\_csv.py  
   Builds per-query CSV datasets from retrieval outputs \+ matcher outputs.  
2. tune\_lr\_and\_threshold.py  
   Trains LR models and tunes LR regularization C and probability thresholds on validation.  
3. adaptive\_match\_and\_eval.py  
   Runs runtime adaptive matching and evaluation (baseline \+ adaptive Recall@k \+ compute proxies).

## **Constants (used everywhere)**

* K \= 20  
* Positive distance threshold \= **25m**

These must be consistent across CSV building, LR tuning, and evaluation.

# **0\) Setup**

## **0.1 Install dependencies (Colab / Linux)**

From repository root:
```bash
pip -q install pandas numpy tqdm scikit-learn==1.5.2 joblib opencv-python kornia einops
```
*Note: Torch is typically preinstalled in Colab. If not, install torch for your runtime CUDA version.*

## **0.2 Go to repo root**
```bash
cd /content/drive/MyDrive/VPR\_Project/Visual-Place-Recognition-Project
```
# **1\) Inputs required from previous parts**

Before running the extension, you must already have:

* Retrieval prediction files: .../preds/\*.txt (one per query)  
* Matcher outputs for building CSVs:  
  * .../preds\_loftr/\*.torch  
  * .../preds\_superpoint-lg/\*.torch

*Note: adaptive\_match\_and\_eval.py does not require precomputed matcher outputs (it runs matching on demand), but CSV building does require .torch outputs.*

# **2\) Build CSVs (Training \+ Validation)**

We build CSVs that contain only the required columns for LR and threshold tuning.

## **2.1 Column schema**

* **Training CSVs (minimal):** query\_id, inliers\_top1, baseline\_correct  
* **Validation CSVs (for threshold tuning):** query\_id, inliers\_top1, baseline\_correct, reranked\_correct

## **2.2 Build TRAIN CSVs (Sun and Night)**

You must produce 8 training CSV files:

* 4 pipelines for Sun (MixVPR/CosPlace × LoFTR/SuperPoint-LG)  
* 4 pipelines for Night (MixVPR/CosPlace × LoFTR/SuperPoint-LG)

**Example command (Sun, MixVPR × LoFTR):**
```bash
python build_csv.py \
  --preds-dir "logs/<RUN_SUN_MIXVPR>/preds" \
  --inliers-dir "logs/<RUN_SUN_MIXVPR>/preds_loftr" \
  --out-csv "csv/train/train_sun_mixvpr_loftr.csv" \
  --num-preds 20 \
  --positive-dist-threshold 25
```
**Example command (Sun, MixVPR × SuperPoint-LG):**
```bash
python build_csv.py \
  --preds-dir "logs/<RUN_SUN_MIXVPR>/preds" \
  --inliers-dir "logs/<RUN_SUN_MIXVPR>/preds_superpoint-lg" \
  --out-csv "csv/train/train_sun_mixvpr_superpoint-lg.csv" \
  --num-preds 20 \
  --positive-dist-threshold 25
```
Repeat analogously for:

* CosPlace × LoFTR  
* CosPlace × SuperPoint-LG

Repeat again for Night (changing train\_sun\_ to train\_night\_...).

## **2.3 Build VALIDATION CSVs (SF-XS validation)**

SF-XS validation is not split into Sun/Night. Produce 4 validation CSV files (one per pipeline):

* csv/validation/sf\_val\_mixvpr\_loftr.csv  
* csv/validation/sf\_val\_mixvpr\_superpoint-lg.csv  
* csv/validation/sf\_val\_cosplace\_loftr.csv  
* csv/validation/sf\_val\_cosplace\_superpoint-lg.csv

**Example (SF-XS val, CosPlace × LoFTR):**
```bash
python build_csv.py \
  --preds-dir "logs/<RUN_SFVAL_COSPLACE>/preds" \
  --inliers-dir "logs/<RUN_SFVAL_COSPLACE>/preds_loftr" \
  --out-csv "csv/validation/sf_val_cosplace_loftr.csv" \
  --num-preds 20 \
  --positive-dist-threshold 25
```
# **3\) Train LR models \+ tune thresholds**

We train two LRs:

1. **LR-Sun** trained on pooled Sun train CSVs (4 pipelines)  
2. **LR-Night** trained on pooled Night train CSVs (4 pipelines)

We tune:

* Logistic regression regularization C  
* Probability threshold used to gate EASY/HARD

**Outputs:**

* models/lr\_sun.joblib, models/lr\_sun.json  
* models/lr\_night.joblib, models/lr\_night.json  
* sweeps/\*.csv tables with sweep results

Create output folders:
```bash
mkdir -p models sweeps
```
## **3.1 Train \+ tune LR-Sun**
```bash
python tune_lr_and_threshold.py \
  --train-csvs \
    csv/train/train_sun_mixvpr_loftr.csv \
    csv/train/train_sun_mixvpr_superpoint-lg.csv \
    csv/train/train_sun_cosplace_loftr.csv \
    csv/train/train_sun_cosplace_superpoint-lg.csv \
  --val-csvs \
    csv/validation/sf_val_mixvpr_loftr.csv \
    csv/validation/sf_val_mixvpr_superpoint-lg.csv \
    csv/validation/sf_val_cosplace_loftr.csv \
    csv/validation/sf_val_cosplace_superpoint-lg.csv \
  --out-model models/lr_sun.joblib \
  --out-json models/lr_sun.json \
  --out-table sweeps/sun_best_per_C.csv \
  --C-grid "0.01,0.03,0.1,0.3,1,3,10,30,100" \
  --thresholds "0.00:1.00:0.01" \
  --objective "max_r1_then_savings"
```
## **3.2 Train \+ tune LR-Night**
```bash
python tune_lr_and_threshold.py \
  --train-csvs \
    csv/train/train_night_mixvpr_loftr.csv \
    csv/train/train_night_mixvpr_superpoint-lg.csv \
    csv/train/train_night_cosplace_loftr.csv \
    csv/train/train_night_cosplace_superpoint-lg.csv \
  --val-csvs \
    csv/validation/sf_val_mixvpr_loftr.csv \
    csv/validation/sf_val_mixvpr_superpoint-lg.csv \
    csv/validation/sf_val_cosplace_loftr.csv \
    csv/validation/sf_val_cosplace_superpoint-lg.csv \
  --out-model models/lr_night.joblib \
  --out-json models/lr_night.json \
  --out-table sweeps/night_best_per_C.csv \
  --C-grid "0.01,0.03,0.1,0.3,1,3,10,30,100" \
  --thresholds "0.00:1.00:0.01" \
  --objective "max_r1_then_savings"
```
# **4\) Adaptive matching \+ evaluation on test sets**

This is the "runtime" evaluation:

* Always match top-1  
* Match top-20 only for HARD queries  
* Evaluate baseline and adaptive Recall@{1,5,10,20}  
* Report compute proxies

Create results folders:
```bash
mkdir -p results/tokyo results/svox_sun results/svox_night results/sf_xs
```
## **4.1 Command template**
```bash
python adaptive_match_and_eval.py \
  --preds-dir "<RUN_TEST>/preds" \
  --out-dir "<RUN_TEST>/adaptive_<matcher>_<lr>" \
  --matcher <loftr|superpoint-lg> \
  --device cuda \
  --im-size 512 \
  --num-preds 20 \
  --lr-model "models/<lr>.joblib" \
  --lr-json "models/<lr>.json" \
  --positive-dist-threshold 25 \
  --recall-values "1,5,10,20" \
  --data-root "/content/vpr_data" \
  --log-jsonl "results/<dataset>/<name>_decisions.jsonl" \
  --out-json "results/<dataset>/<name>_summary.json"
```
**Notes:**

* Use a fresh \--out-dir for fair runtime (otherwise cached torch files can be reused).  
* \--data-root is used to remap image paths inside preds/\*.txt to your dataset location.

## **4.2 Example: Tokyo XS (MixVPR × LoFTR) with LR-Sun**
```bash
python adaptive_match_and_eval.py \
  --preds-dir "logs/tokyo_xs_mixvpr/<RUN>/preds" \
  --out-dir "logs/tokyo_xs_mixvpr/<RUN>/adaptive_loftr_lr_sun" \
  --matcher loftr \
  --device cuda \
  --im-size 512 \
  --num-preds 20 \
  --lr-model "models/lr_sun.joblib" \
  --lr-json "models/lr_sun.json" \
  --positive-dist-threshold 25 \
  --recall-values "1,5,10,20" \
  --data-root "/content/vpr_data" \
  --log-jsonl "results/tokyo/mixvpr_loftr_lr_sun_decisions.jsonl" \
  --out-json "results/tokyo/mixvpr_loftr_lr_sun_summary.json"
```
Repeat evaluation across:

* Backbones: MixVPR, CosPlace  
* Matchers: LoFTR, SuperPoint-LG  
* LR models: LR-Sun, LR-Night  
* Test datasets: Tokyo XS, SVOX Sun, SVOX Night, SF-XS (as required)

# **5\) Manual threshold testing (no code changes)**

The evaluation script reads the threshold from models/lr\_\*.json (key: best\_threshold). To test a custom threshold, create a temporary JSON.

**Example: LR-Night with threshold 0.45**
```bash
import json

src = "models/lr_night.json"
dst = "models/lr_night_thr_0p45.json"

with open(src, "r") as f:
    d = json.load(f)

d["best_threshold"] = 0.45

with open(dst, "w") as f:
    json.dump(d, f, indent=2)

print("Wrote:", dst)
```
Then run adaptive evaluation with:
```bash
--lr-json "models/lr_night_thr_0p45.json"
```
# **6\) Outputs**

**Models**

* models/lr\_sun.joblib  
* models/lr\_sun.json  
* models/lr\_night.joblib  
* models/lr\_night.json

**CSVs**

* csv/train/\*.csv  
* csv/validation/\*.csv

**Adaptive evaluation results**

* results/\<dataset\>/\*\_summary.json: Contains baseline/adaptive recalls and compute proxies.  
* results/\<dataset\>/\*\_decisions.jsonl: One JSON record per query: easy/hard decision, probability, matched count, etc.  
* \<RUN\>/adaptive\_\*/\*.torch: Per-query matcher outputs computed during runtime adaptive evaluation.
