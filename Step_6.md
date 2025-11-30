# Step 6: Evaluation and Calibration

## 1. Goal of Step 6

Step 6 evaluates and analyzes the models trained in Steps 4 and 5:

- **Text expert** (`models/text_expert`)
- **Vision expert** (`models/vision_expert`)
- **Multimodal fusion model** (`models/mm_fusion/fusion_model.pt`)

The aims are to:
- Measure performance (accuracy, macro-F1) on held-out data.
- Inspect calibration (how well predicted probabilities match reality).
- Compare **unimodal vs fusion** models.
- Prepare a reusable evaluation pipeline that can be run on both the
  **pilot shard** and later on the **full dataset**.

---

## 2. Inputs and prerequisites

Before running Step 6 you should have:

From earlier steps:
- `Step_3/` with packed shards and metadata:
  - `Step_3/shards/{train,val,test}/*.tar` (for now, at least the pilot `train/shard-000000.tar`).
  - `Step_3/label_taxonomy.json`.
  - `Step_3/splits.json` (will be used for full-data evaluation later).
- Models from Steps 4 and 5:
  - `models/text_expert/` (text-only expert, Hugging Face `save_pretrained` format).
  - `models/vision_expert/` (image-only expert, Hugging Face `save_pretrained` format).
  - `models/mm_fusion/fusion_model.pt` (PyTorch `state_dict` for the fusion head + encoders).

Environment:
- Same environment as Steps 4 and 5 (PyTorch + Transformers + WebDataset):

```bash
pip install torch torchvision torchaudio
pip install transformers datasets webdataset accelerate timm sentencepiece
```

---

## 3. Step 6 folder structure

From the project root:

```text
Step_6/
  step6_eval.ipynb        # Main notebook: evaluates text, vision, and fusion models
  .gitkeep                # Placeholder so Step_6/ is tracked in git

Step_6.md                 # This documentation file
```

All **evaluation code** for Step 6 lives under `Step_6/`, while this
markdown file describes the goals, inputs, and how to run evaluations.

---

## 4. What Step 6 evaluates

For now, Step 6 focuses on the **binary `abuse_hate` label** defined in
`Step_3/label_taxonomy.json`.

For each available evaluation set, we will compute for:
- **Text expert** (text-only)
- **Vision expert** (image-only)
- **Fusion model** (text + image)

Metrics (per model):
- Accuracy
- Macro-F1 (average F1 over the 2 classes)
- Calibration metrics (optional, pilot-level):
  - Brier score (mean squared error of predicted probability for the true label).
  - Expected Calibration Error (ECE) with a small number of bins.

We also prepare hooks for **error analysis**, e.g. listing false positives
and false negatives for each model.

---

## 5. Data loading for evaluation

Step 6 uses the same WebDataset shards created in Step 3, but reads
**both text and image** for each example.

Conceptually, the loader does:

1. Open one or more shards via WebDataset (e.g. `Step_3/shards/train/shard-000000.tar`).
2. For each record, decode:
   - `txt` → combined text string.
   - `png` → PIL image.
   - `json` → metadata with `labels.abuse_hate`.
3. Build a dataset that returns `(text, image, label)`.
4. Use the **text expert tokenizer** and **vision expert image processor** to
   create model inputs:
   - `input_ids`, `attention_mask`, `pixel_values`, `labels`.

The same evaluation dataset is then fed to the three models, each using the
parts it needs.

---

## 6. Training/evaluation procedure on the pilot shard

For the **pilot run** (50 Hateful Memes examples):

1. **Select evaluation data**
   - Use the pilot shard: `Step_3/shards/train/shard-000000.tar`.
   - For now, treat all 50 examples as a small **evaluation set** to sanity-check
     the models. (Later, when more data is available, we will use proper
     train/val/test splits.)

2. **Load models**
   - Text expert: `AutoModelForSequenceClassification.from_pretrained("models/text_expert")`.
   - Vision expert: `AutoModelForImageClassification.from_pretrained("models/vision_expert")`.
   - Fusion model:
     - Recreate the same `FusionModel` architecture used in Step 5.
     - Load `state_dict` from `models/mm_fusion/fusion_model.pt`.

3. **Run evaluation**
   - For each model, run forward passes on the eval loader under `torch.no_grad()`.
   - Collect logits and labels.
   - Convert to probabilities (softmax) and predicted classes.
   - Compute accuracy, macro-F1, and basic calibration metrics.

4. **Save results**
   - Write a small JSON such as `Step_6/results_pilot.json` summarizing metrics for
     text, vision, and fusion models.
   - Optionally write a CSV or table of the most important errors
     (false positives/false negatives) for manual inspection.

The notebook `Step_6/step6_eval.ipynb` implements this pilot procedure end-to-end.

---

## 7. How to extend to full data later

Once you run **Step 3** on the full dataset and have proper splits in
`Step_3/splits.json` and multiple shards per split, you can reuse the same
Step 6 code with a few changes:

1. **Select shards by split**
   - Instead of hard-coding a single pilot shard, point the loader at patterns like:
     - Train: `Step_3/shards/train/shard-*.tar`
     - Val:   `Step_3/shards/val/shard-*.tar`
     - Test:  `Step_3/shards/test/shard-*.tar`
   - Optionally, use `Step_3/splits.json` to double-check which IDs belong to
     which split.

2. **Evaluate per split**
   - Run the same evaluation code on the **validation** set to choose models and
     hyperparameters.
   - Run once on the **test** set to report final numbers.

3. **Compare unimodal vs fusion at scale**
   - On the validation set, verify that the fusion model **matches or exceeds**
     the best unimodal expert on macro-F1 for `abuse_hate`.
   - Track calibration metrics per model; if needed, apply calibration methods
     (e.g. temperature scaling) in a later step.

4. **Update results artifacts**
   - Save summaries like `Step_6/results_full.json` containing per-split metrics
     and model comparisons.

The Step 6 notebook is written so that switching from **pilot** to **full-data**
mode mostly requires updating the shard patterns and enabling the extra splits.

---

## 8. Quality targets for Step 6

For the **pilot run** (50 examples):
- Evaluation completes without errors using all three models.
- Metrics are computed and written to a JSON file.
- At least one or two example error cases are inspected manually.

For the **full run** (all shards):
- Fusion model **matches or outperforms** the best unimodal expert on
  dev/validation macro-F1.
- Calibration curves and metrics look reasonable (no extreme overconfidence).
- Evaluation is reproducible via `Step_6/step6_eval.ipynb` and documented here.

---

## 9. Summary

Step 6 turns the models from Steps 4 and 5 into a **measured system**:

- You will know how good each expert and the fusion model are.
- You will have a clear comparison of text vs vision vs fusion.
- You will have a reusable evaluation pipeline that can be run on both
  pilot and full-scale data.

This lays the groundwork for any later steps (e.g. calibration tuning,
robustness checks, or deployment).
