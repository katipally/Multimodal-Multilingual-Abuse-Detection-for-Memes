# Step 7: Thresholds, Calibration, and Robustness

## 1. Goal of Step 7

Step 7 turns the evaluated models from Steps 4–6 into a **practical decision system**.

Starting from:
- **Text expert** (`models/text_expert`)
- **Vision expert** (`models/vision_expert`)
- **Multimodal fusion model** (`models/mm_fusion/fusion_model.pt`)
- **Evaluation metrics** from Step 6 (e.g. `Step_6/results_pilot.json`),

Step 7 will:
- Choose **decision thresholds** for the `abuse_hate` label.
- Optionally (but implemented here) fit **temperature scaling** for better calibration.
- Perform **error analysis** and simple **robustness checks**.
- Produce small machine-readable artifacts (thresholds JSON, error CSV) and a
  clear textual summary that can be used in later steps or deployment.

All of these pieces are implemented in `Step_7/step7_analysis.ipynb`.

---

## 2. Inputs and prerequisites

Before running Step 7 you should have:

From earlier steps:
- `Step_3/` packed shards and metadata (at least the pilot shard for now):
  - `Step_3/shards/{train,val,test}/*.tar` — for the pilot we use
    `Step_3/shards/train/shard-000000.tar`.
  - `Step_3/label_taxonomy.json`.
  - `Step_3/splits.json` (used for full-data splits later).
- Models:
  - `models/text_expert/` (Step 4 text-only expert).
  - `models/vision_expert/` (Step 4 image-only expert).
  - `models/mm_fusion/fusion_model.pt` (Step 5 fusion model checkpoint).
- Evaluation metrics (from Step 6):
  - `Step_6/results_pilot.json` for the pilot.
  - Later, `Step_6/results_full.json` (or similar) for full-data evaluation.

Environment:
- Same as Steps 4–6 (PyTorch + Transformers + WebDataset + NumPy):

```bash
pip install torch torchvision torchaudio
pip install transformers datasets webdataset accelerate timm sentencepiece
```

---

## 3. Step 7 folder structure

From the project root:

```text
Step_7/
  step7_analysis.ipynb     # Main notebook: thresholds, calibration, error analysis
  .gitkeep                 # Placeholder so Step_7/ is tracked in git

Step_7.md                  # This documentation file
```

All **code for Step 7** lives under `Step_7/`, while this markdown file
summarizes the goals, inputs, and how to run the analyses.

---

## 4. What Step 7 does

Step 7 operates on the same evaluation data used in Step 6 (pilot for now;
train/val/test splits later) and performs three main tasks:

1. **Model comparison and selection**
   - Re-runs evaluation for the three models (text, vision, fusion) to obtain
     logits and probabilities for `abuse_hate`.
   - Uses metrics from Step 6 (accuracy, macro-F1, Brier, ECE) to compare them.

2. **Threshold tuning for `abuse_hate`**
   - Sweeps over a grid of thresholds (e.g. 0.1–0.9) for each model.
   - For each threshold, computes:
     - Precision
     - Recall
     - F1
   - Stores the full threshold curves in CSV files.
   - Picks a **recommended threshold** per model (maximizing F1) and writes
     them to a small JSON file.

3. **Calibration and error analysis**
   - Fits a simple **temperature scaling** parameter per model to improve
     probability calibration.
   - Re-computes Brier and ECE after calibration.
   - Uses the calibrated probabilities and chosen thresholds to:
     - Build an error table (false positives/false negatives, etc.).
     - Save this as a CSV for manual inspection.

All of these are implemented end-to-end in `step7_analysis.ipynb` and
produce concrete artifacts under `Step_7/`.

---

## 5. Pilot procedure (50-example shard)

For the current **pilot** (50 Hateful Memes examples):

1. **Load evaluation data**
   - Use the pilot shard: `Step_3/shards/train/shard-000000.tar`.
   - Build a dataset of `(text, image, label)` examples and a DataLoader
     without shuffling (so evaluation is deterministic).

2. **Evaluate models and collect logits**
   - Run the three models on the evaluation loader:
     - Text expert: `AutoModelForSequenceClassification.from_pretrained("models/text_expert")`.
     - Vision expert: `AutoModelForImageClassification.from_pretrained("models/vision_expert")`.
     - Fusion model: re-create the `FusionModel` from Step 5 and load
       `models/mm_fusion/fusion_model.pt`.
   - For each model, gather logits and labels across the entire evaluation set.

3. **Fit temperature scaling per model**
   - For each model, treat its logits and labels as training data for a
     single-parameter temperature `T`:
     - Optimize `T` such that softmax(logits / T) minimizes cross-entropy on
       the evaluation set.
   - Use the fitted temperature to compute **calibrated probabilities**.
   - Recompute Brier and ECE for each model using the calibrated
     probabilities.

4. **Threshold sweep and selection**
   - Define a grid of thresholds, e.g. `0.1, 0.15, …, 0.9`.
   - For **each model**:
     - Apply the calibrated probabilities and each threshold in the grid.
     - Compute precision, recall, and F1 for `abuse_hate`.
     - Store the full curve in `Step_7/thresholds_curve_<model>_pilot.csv`.
     - Choose the threshold that maximizes F1.
   - Save a JSON summary of recommended thresholds, e.g.:

     ```json
     {
       "text_expert": {"abuse_hate": 0.55},
       "vision_expert": {"abuse_hate": 0.60},
       "mm_fusion": {"abuse_hate": 0.50}
     }
     ```

     in `Step_7/thresholds_pilot.json`.

5. **Error analysis and robustness (pilot)**
   - Using the **fusion model** (and its chosen threshold) as the primary
     decision maker, build a table where each row includes:
     - Example index
     - Text
     - True label (`abuse_hate`)
     - Text expert probability and prediction
     - Vision expert probability and prediction
     - Fusion probability and prediction
     - Error type for fusion (TP, TN, FP, FN)
   - Optionally include simple slices, such as:
     - Whether all three models agree.
     - Whether fusion is correct but both unimodal experts are wrong.
   - Save this table as `Step_7/errors_pilot.csv`.

6. **Summarize pilot findings**
   - The notebook prints a brief summary of:
     - Pre- and post-calibration metrics for each model.
     - Recommended thresholds.
     - Counts of each error type (e.g. number of false positives).

---

## 6. Extending to full data

Once you run Step 3 on the full dataset and Step 6 on full train/val/test
splits, Step 7 can be re-used with minimal changes:

1. **Use validation split for tuning**
   - Point the evaluation loader in `step7_analysis.ipynb` to the **validation**
     shards (e.g. `Step_3/shards/val/shard-*.tar`).
   - Re-run the notebook to:
     - Fit temperature scaling.
     - Sweep thresholds.
     - Produce `thresholds_full.json` (recommended thresholds based on val).

2. **Apply to test split**
   - Once thresholds and calibration parameters are fixed, evaluate on the
     **test** shards (e.g. `Step_3/shards/test/shard-*.tar`).
   - Generate a final error CSV for the test set.

3. **Finalize model choice**
   - Use the full-data results to confirm that the selected model (typically
     the fusion model) and thresholds meet your macro-F1 and calibration
     targets.

---

## 7. Outputs and quality targets

After running `Step_7/step7_analysis.ipynb` you should have:

- `Step_7/thresholds_curve_text_expert_pilot.csv`
- `Step_7/thresholds_curve_vision_expert_pilot.csv`
- `Step_7/thresholds_curve_mm_fusion_pilot.csv`
- `Step_7/thresholds_pilot.json` — recommended thresholds for `abuse_hate`.
- `Step_7/errors_pilot.csv` — detailed per-example error analysis.

For the **pilot run**, the main quality goals are:
- All analyses run end-to-end without errors.
- Threshold curves and error tables are produced for inspection.

For the **full run**, additional goals are:
- The chosen model (likely fusion) meets or exceeds target macro-F1 on the
  validation set.
- Calibration metrics (Brier, ECE) are reasonable after temperature scaling.
- Thresholds and calibration parameters are fixed and documented for use in
  any downstream deployment or serving code.

---

## 8. Summary

Step 7 turns the models and metrics from Steps 4–6 into a **decision-ready
multimodal classifier**:

- It chooses thresholds, improves calibration, and exposes error patterns.
- It produces concrete artifacts (JSON and CSV) that can be versioned
  alongside model checkpoints.
- It prepares you for any subsequent steps involving deployment,
  monitoring, or further robustness work.
