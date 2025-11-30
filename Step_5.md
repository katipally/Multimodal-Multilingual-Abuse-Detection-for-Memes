# Step 5: Multimodal Fusion Model

## 1. Goal of Step 5

Step 5 takes the **unimodal experts** trained in Step 4 and the **packed shards** from Step 3, and trains a
**multimodal fusion model** that jointly uses text (OCR + captions) and images for hate/abuse detection.

High-level objectives:
- Load and reuse the **text expert** (`models/text_expert`) and **vision expert** (`models/vision_expert`).
- Build a small **fusion head** that combines text and image representations.
- Train and evaluate the fusion model on the same unified labels as Step 4 (starting with `abuse_hate`).
- Verify that fusion is at least as good as, and ideally better than, the best unimodal expert (on dev/pilot).

This step is the bridge between unimodal experts (Step 4) and the full multimodal system (Step 6/7).

---

## 2. Inputs and prerequisites

Before running Step 5 you should have:

From earlier steps:
- `Datasets/` prepared as in `Step_1.md`.
- `Step_2/data_manifest.jsonl` and `Step_2/ocr.jsonl` created as in `Step_2.md`.
- `Step_3/` populated:
  - `Step_3/label_taxonomy.json`
  - `Step_3/splits.json`
  - `Step_3/shards/{train,val,test}/*.tar` (for now, at least a **train** shard for the pilot run)
- `Step_4/` completed:
  - `models/text_expert/` — text-only expert trained on packed text.
  - `models/vision_expert/` — image-only expert trained on packed images.

Environment:
- A Python environment with GPU is recommended for efficient training, but small pilots can run on CPU.
- The same core libraries as Step 4 are used:

```bash
pip install torch torchvision torchaudio
pip install transformers datasets webdataset accelerate timm sentencepiece
```

---

## 3. Step 5 folder structure

From the project root:

```text
Step_5/
  .gitkeep                 # Placeholder so Step_5/ is tracked in git
  # (Later) fusion scripts/notebooks will live here, e.g.:
  # step5_fusion.ipynb     # Notebook to train & evaluate multimodal fusion
  # train_fusion.py        # Optional script version (if needed)

Step_5.md                  # This documentation file
```

As with previous steps, all **code for Step 5** should live under `Step_5/`, while the
high-level description stays in `Step_5.md` at the repo root.

---

## 4. Fusion model design (pilot version)

For the current **pilot** (50 Hateful Memes examples), we implement a simple but correct
fusion model that can be scaled up later.

### 4.1 Encoders

We reuse the unimodal experts from Step 4:

- **Text encoder**:
  - Load from `models/text_expert` using Hugging Face.
  - For the pilot, we can use either:
    - The final **CLS representation** (hidden state), or
    - The **logits** as a compact text feature.
- **Vision encoder**:
  - Load from `models/vision_expert`.
  - Use the pooled image representation or logits similarly.

Initially, most of the encoder weights stay **frozen**; we train only the new fusion head and
optionally a few top layers (LoRA / partial fine-tuning) once the pipeline is stable.

### 4.2 Fusion head

A simple fusion strategy for the pilot:

1. Compute text feature `t` (e.g. 2D logits or a 768-dim pooled vector).
2. Compute image feature `v` (same idea).
3. Concatenate: `h = [t ; v]`.
4. Pass `h` through a small MLP:
   - Linear → GELU → Dropout → Linear → logits over `abuse_hate`.

This is easy to implement and debug, and can be upgraded later to token-level cross-attention
once the basic fusion pipeline is working.

### 4.3 Loss and targets

- Primary target for the pilot: `abuse_hate` (binary: hate/abuse vs non-hate).
- Loss: binary cross-entropy or 2-class cross-entropy.
- Metrics:
  - Accuracy
  - Macro-F1 (preferred for imbalanced classes)

---

## 5. Data loading for fusion

Step 5 uses the **same shards** created in Step 3, but now we need both modalities together.

Key idea:
- Use WebDataset to read `(txt, png, json)` from each example in a shard.
- The JSON metadata contains the unified labels (`labels.abuse_hate`, etc.).

Training pipeline (conceptually):

1. Open shards with WebDataset:
   - Example: `Step_3/shards/train/shard-000000.tar` for the pilot.
2. For each sample, decode:
   - `txt` → combined text string.
   - `png` → PIL image → image processor (from `vision_expert`).
   - `json` → metadata → `labels.abuse_hate`.
3. Tokenize text with the **text expert tokenizer**.
4. Form a batch: `(input_ids, attention_mask, pixel_values, label)`.

This will be implemented in a fusion notebook/script (e.g. `step5_fusion.ipynb`).

---

## 6. Training and evaluation procedure (pilot)

For the initial pilot (your 50-example Hateful Memes subset):

1. **Load unimodal experts**:
   - `text_expert` and `vision_expert` from `models/`.
2. **Build fusion model** on top of their representations/logits:
   - Use the text backbone from `text_expert` (CLS or pooled representation).
   - Use the 2-dimensional classification logits from `vision_expert` as compact image features.
3. **Train** for a small number of epochs (currently `EPOCHS = 1` for the pilot):
   - Use the same shard as in Step 4 for training and log fusion **training loss and accuracy**.
4. **(Optional, future extension)** Evaluate on a held-out subset or via simple train/dev split within the pilot:
   - Compute accuracy and macro-F1 for:
     - Text-only expert.
     - Vision-only expert.
     - Fusion model.
5. **Save fusion model**:
   - Under `models/mm_fusion/fusion_model.pt` as a PyTorch `state_dict` checkpoint.

On the full dataset (later), the same code will scale to many shards and full train/val/test splits.

---

## 7. Quality targets for Step 5

For the **pilot run** (small subset):
- Pipeline runs end-to-end without errors.
- Fusion model trains (loss decreases) and saves successfully.
- Basic metrics are computed (even if noisy due to small data).

For the **full run** (after Steps 2–3 are run on all datasets):
- Fusion model **matches or exceeds** the best unimodal expert on dev macro-F1 for `abuse_hate`.
- Training remains stable (no divergence) when using more data and (optionally) partial fine-tuning.

---

## 8. Summary

After completing Step 5, you will have:

- A working **multimodal fusion model** that consumes text (OCR+caption) and image features.
- A fusion checkpoint saved under `models/mm_fusion/`.
- A clear recipe for scaling up training to all packed shards.

The next steps (6–7) will focus on **calibration, robustness, and evaluation** of the full system.
