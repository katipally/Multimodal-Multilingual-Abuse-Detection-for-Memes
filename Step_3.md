# Step 3: Pack Each Example into Training Shards

## 1. Goal of Step 3

Step 3 takes the heterogeneous artifacts from Steps 1–2 (raw datasets, unified manifest, OCR results) and **packs each example into a single, model-ready bundle**.

This bundle will later be consumed by unimodal experts (Step 4) and the multimodal fusion model (Step 5).

Current scope:
- **Included**: the same image+text and text-only datasets from Steps 1–2
  - Hateful Memes
  - MAMI
  - MMHS150K
  - Memotion 1.0
  - Memotion 2.0
  - HateXplain (text-only)
  - OLID (text-only)
  - Diagnostics datasets (for evaluation only, can be optionally packed)
- **Excluded for now**: video / audio (e.g. HateMM). ASR segments will be added to the packing format later.

High-level objectives:
- Define a **unified label taxonomy** across all datasets.
- Build a **single text field** per example combining OCR text and any original caption/text.
- Preprocess images to the input size for the vision backbone (SigLIP-2, e.g. 384×384).
- Create **train/val/test splits** with no leakage and balanced label distributions.
- Serialize data into efficient **WebDataset-style shards** for training.

---

## 2. Environment and prerequisites

Before running Step 3:

- Steps 1 and 2 must be complete:
  - Datasets downloaded and inspected as per `Step_1.md`.
  - `Step_2/data_manifest.jsonl` built.
  - `Step_2/ocr.jsonl` generated for all relevant images (or at least a subset for pilot runs).
- A Python virtual environment should be active with standard dependencies.

Example setup (from repo root):

```bash
source venv/bin/activate
pip install webdataset pillow numpy torch
```

Notes:
- `webdataset` is used for creating and reading **tar-based shards** that work well with PyTorch `DataLoader`s.
- `torch` is needed for later training steps, and we may reuse transforms/utilities while packing.

Additional model-specific dependencies (e.g. tokenizers for mDeBERTa-v3, SigLIP-2 image processors) will be configured in later steps, but the packing format is designed to be **backbone-agnostic**.

---

## 3. Step 3 folder structure (after running scripts)

From the project root:

```text
Step_3/
  pack_examples.py         # Script to pack manifest + OCR into shards
  label_taxonomy.json      # Unified label schema across all datasets
  splits.json              # Mapping from example id → split (train/val/test)
  stats.json               # Packing statistics and quality metrics
  shards/
    train/
      shard-00000.tar
      shard-00001.tar
      ...
    val/
      shard-00000.tar
      ...
    test/
      shard-00000.tar
      ...
  .gitkeep                 # Placeholder so Step_3/ is tracked in git
```

Notes:
- `pack_examples.py` is the **only entry point** for Step 3.
- `label_taxonomy.json` and `splits.json` are **inputs** to `pack_examples.py`, but they may also be generated (or updated) by it.
- Shards are organized per split (`train/`, `val/`, `test/`).

---

## 4. Inputs to Step 3

Step 3 consumes artifacts from previous steps:

- `Step_2/data_manifest.jsonl`
  - One line per example, with at least:
    - `id`: global unique id (`<dataset>_<split>_<example_id>`)
    - `dataset`: dataset name
    - `split`: original split (if defined by the dataset)
    - `image_path`: absolute path to image file (or `null` for text-only)
    - `text_raw`: original caption/text (if any)
    - `labels_raw`: dataset-specific labels
- `Step_2/ocr.jsonl`
  - One line per image example, with OCR results:
    - `id`, `dataset`, `split`, `image_path`
    - `ocr.lines`: list of `{text, conf, bbox}` for detected textlines

Optional inputs (future extensions):
- ASR segments for video clips (Step 2B) to be incorporated later as part of the packed text field.

---

## 5. Label taxonomy and splits

### 5.1 Unified label taxonomy

Different datasets use different label schemes. Step 3 introduces a **unified taxonomy** so downstream models can share heads:

- `abuse_hate` (binary):
  - 1 = hateful or abusive content present
  - 0 = non-hateful / normal
- `offensive` (binary):
  - 1 = offensive language (even if not strictly hate)
  - 0 = non-offensive
- `target_type` (categorical):
  - `none` (no clear target)
  - `individual`
  - `group`
  - `other`
- `target_group` (multi-label string list, optional):
  - e.g. `"women"`, `"black"`, `"jewish"`, `"muslim"`, `"lgbtq"`, etc.
- Additional dataset-specific auxiliary labels:
  - Sentiment (from Memotion 1.0/2.0)
  - Fine-grained misogyny categories (from MAMI)
  - HateXplain rationales (can be carried through for analysis)

The exact mapping from each dataset’s `labels_raw` to this unified schema will be stored in:

- `Step_3/label_taxonomy.json`

This file describes:
- The **fields** in the unified schema.
- How each dataset’s original labels map into these fields.

### 5.2 Train/val/test splits

Step 3 also defines (or reuses) train/val/test splits:

- If a dataset already defines splits (e.g. Hateful Memes, OLID), those are respected.
- If not, Step 3 will create splits (e.g. 80/10/10) with:
  - No duplicate ids across splits.
  - Approximate per-class balance.

The final assignment will be recorded in:

- `Step_3/splits.json`:
  - `{ "<id>": "train" | "val" | "test", ... }`

---

## 6. `pack_examples.py`: packing pipeline

### 6.1 What `pack_examples.py` does

`pack_examples.py` reads the manifest and OCR outputs, merges them, and writes **WebDataset-style shards** containing:

- A canonical **text field** combining OCR text and any original caption/text.
- A preprocessed **image tensor** (for examples with images).
- Unified **labels** according to `label_taxonomy.json`.

For each example:

1. Load the manifest record by `id`.
2. Join with corresponding OCR record (if `image_path` is present).
3. Build the combined text:
   - Example pattern for English:
     - `[OCR] <joined OCR lines> [/OCR] [CAP] <text_raw> [/CAP] <lang=en>`
4. Optionally normalize or filter OCR lines (e.g. drop very low-confidence lines).
5. If `image_path` is set:
   - Load image.
   - Convert to RGB and resize to **384×384** (SigLIP-2 default).
6. Map `labels_raw` into the unified label schema.
7. Assign split using `splits.json` (or dataset-provided splits).
8. Write the example into the appropriate shard.

### 6.2 WebDataset shard format

Each shard is a `.tar` file with multiple examples. For an example with key `"<id>"`, shards will contain entries like:

- `<id>.jpg` or `<id>.png` — preprocessed image (if present)
- `<id>.txt` — combined text field
- `<id>.json` — unified labels and metadata

This format is directly compatible with `webdataset` and can be read in PyTorch like:

```python
import webdataset as wds

ds = (wds.WebDataset("Step_3/shards/train/shard-{00000..00010}.tar")
        .decode("pil")
        .to_tuple("jpg;png", "txt", "json"))
```

Shard size (number of examples per `.tar`) will be configurable (e.g. 5–10k examples per shard) to balance I/O efficiency and flexibility.

### 6.3 How to run packing

From the repo root:

```bash
# Pilot run (small subset, e.g. first 5k examples) for debugging
python Step_3/pack_examples.py \
  --manifest Step_2/data_manifest.jsonl \
  --ocr Step_2/ocr.jsonl \
  --output-dir Step_3/shards \
  --label-taxonomy Step_3/label_taxonomy.json \
  --splits Step_3/splits.json \
  --examples-limit 5000

# Full run (all examples)
python Step_3/pack_examples.py \
  --manifest Step_2/data_manifest.jsonl \
  --ocr Step_2/ocr.jsonl \
  --output-dir Step_3/shards \
  --label-taxonomy Step_3/label_taxonomy.json \
  --splits Step_3/splits.json
```

`pack_examples.py` will also write a summary report to `Step_3/stats.json` (see below).

---

## 7. Quality metrics and acceptance criteria

At the end of a run, `pack_examples.py` will compute and log:

- `total_examples`: number of manifest records considered.
- `packed_examples`: number successfully packed into shards.
- `failed_examples`: number of records skipped due to errors (missing files, invalid labels, etc.).
- `failure_rate`: `failed_examples / total_examples` (target: **< 0.5%**).
- Per-split and per-class counts, to check:
  - No duplicates across splits.
  - Per-class distributions within **±5%** across splits.

These metrics will be written to:

- `Step_3/stats.json`

and also printed to stdout.

---

## 8. Summary of Step 3 artifacts

After successfully running Step 3, you should have:

- `Step_3/label_taxonomy.json`
  - Definition of the unified label schema and dataset-specific mappings.
- `Step_3/splits.json`
  - Final train/val/test assignment for each example id.
- `Step_3/shards/`
  - WebDataset-style training shards for each split (`train`, `val`, `test`).
- `Step_3/stats.json`
  - Summary statistics and quality checks for the packing process.

These artifacts are the **direct inputs** to Step 4, where we train text-only and image-only experts on these packed shards.
