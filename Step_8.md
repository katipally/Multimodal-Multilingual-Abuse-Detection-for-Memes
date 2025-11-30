# Step 8: Inference and Serving

## 1. Goal of Step 8

Step 8 provides a **simple inference interface** for the trained
multimodal system built in Steps 4–7.

It focuses on:
- Loading the **fusion model** (`mm_fusion`) together with
  **calibration temperatures** and **decision thresholds**.
- Exposing a clean `predict(text, image_path)` interface for new memes.
- Demonstrating how to switch from **pilot artifacts** to
  **full-data artifacts** once you run Steps 3–7 at scale.

---

## 2. Inputs and prerequisites

You should have completed Steps 1–7 for at least the pilot subset.
For Step 8, the key inputs are:

- Models:
  - `models/text_expert/` (Step 4 text expert).
  - `models/vision_expert/` (Step 4 vision expert).
  - `models/mm_fusion/fusion_model.pt` (Step 5 fusion checkpoint).
- Calibration and thresholds (from Step 7):
  - `Step_7/calibration_pilot.json` — temperature per model.
  - `Step_7/thresholds_pilot.json` — recommended `abuse_hate` threshold.

Later, after running Steps 3–7 on full data, you can generate and use
full-data equivalents, e.g. `calibration_full.json` and
`thresholds_full.json`, with the same structure.

Environment:
- Same as earlier steps (PyTorch + Transformers + PIL):

```bash
pip install torch torchvision torchaudio
pip install transformers webdataset accelerate timm sentencepiece
pip install pillow
```

---

## 3. Step 8 folder structure

From the project root:

```text
Step_8/
  step8_inference.ipynb    # Notebook: loads models and runs inference on new memes
  .gitkeep                 # Placeholder so Step_8/ is tracked in git

Step_8.md                  # This documentation file
```

All **inference code** for Step 8 lives under `Step_8/`, while this file
explains how to use it.

---

## 4. What the inference notebook does

`Step_8/step8_inference.ipynb` is structured to:

1. **Load configuration and artifacts**
   - Detect the project root.
   - Load `models/text_expert`, `models/vision_expert`, and
     `models/mm_fusion/fusion_model.pt`.
   - Load calibration temperatures and thresholds from JSON files.

2. **Construct the fusion inference model**
   - Re-create the `FusionModel` class used in Steps 5–7.
   - Apply the learned temperature `T` to fusion logits at inference time.

3. **Provide a single `predict` function**
   - `predict(text: str, image_path: str) -> Dict[str, Any]`:
     - Loads the image from disk.
     - Tokenizes the text using the text expert tokenizer.
     - Runs the fusion model to get logits.
     - Applies temperature scaling and softmax to get `p(hate)`.
     - Applies the chosen threshold from Step 7 to produce a binary label.
   - Returns a small dictionary containing:
     - `prob_hate` (float)
     - `label` (0 = non-hate, 1 = hate/abuse)
     - Optionally, the raw logits/probabilities for debugging.

4. **Demo cells**
   - A few example calls to `predict(...)` using sample images and
     texts (you can adapt these to your own test memes).

---

## 5. Pilot vs full-data usage

For the **pilot run** (current state):
- `step8_inference.ipynb` uses:
  - `Step_7/calibration_pilot.json`
  - `Step_7/thresholds_pilot.json`
- These were estimated on the 50-example pilot and are mainly for
  validating the end-to-end pipeline.

For the **full run** (after scaling Steps 3–7):
- Re-run Steps 6–7 on full train/val/test splits.
- Save full-data calibration and thresholds with the same format, e.g.:
  - `Step_7/calibration_full.json`
  - `Step_7/thresholds_full.json`
- Either:
  - Point the Step 8 notebook at these new JSON files, or
  - Add a simple flag / configuration cell (`mode = "pilot" | "full"`).

---

## 6. Quality targets for Step 8

After running Step 8, you should be able to:

- Load all necessary models and parameters without errors.
- Call `predict(text, image_path)` on arbitrary memes and get:
  - A probability for `abuse_hate`.
  - A decision using the tuned threshold for the fusion model.
- Switch between **pilot** and **full-data** artifacts by only changing
  a small config cell.

Step 8 does **not** retrain any models; it purely focuses on inference
and serving.

---

## 7. Summary

Step 8 provides the final user-facing interface for your multimodal
abuse/hate detector:

- It wraps the trained fusion model, calibration, and thresholds into
  a concise prediction function.
- It can be used interactively from a notebook or as a reference for
  building a future API or batch inference script.
- Once Steps 3–7 are run on full data, Step 8 becomes the main entry
  point for scoring new memes at scale.
