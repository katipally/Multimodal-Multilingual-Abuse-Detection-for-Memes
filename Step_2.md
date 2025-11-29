# Step 2: OCR for Image and Text Datasets

## 1. Goal of Step 2

Step 2 takes the raw datasets prepared in `Datasets/` (from **Step 1**) and builds a unified view plus OCR text for all examples that have images. This creates the bridge from heterogeneous dataset formats to a consistent multimodal representation.

Current scope:
- **Included**: image + text datasets and text-only datasets
  - Hateful Memes
  - MAMI
  - MMHS150K
  - Memotion 1.0
  - Memotion 2.0
  - HateXplain (text-only)
  - OLID (text-only)
  - Diagnostics datasets (kept in `Datasets/diagnostics`, not used in the manifest yet)
- **Excluded for now**: video / audio (e.g. HateMM). ASR for videos will be added later.

Outputs of Step 2:
- A **unified manifest** with one line per example: `Step_2/data_manifest.jsonl`.
- **OCR results** for every example with an `image_path`: `Step_2/ocr.jsonl`.
- Optional **QC images** with overlayed OCR boxes and text: `Step_2/ocr_qc/`.

---

## 2. Environment and prerequisites

Before running Step 2:

- Step 1 must be complete and all datasets placed under `Datasets/` as described in `Step_1.md`.
- A Python virtual environment should be active and have the required packages.

Example setup (from repo root):

```bash
source venv/bin/activate
pip install paddleocr paddlepaddle pillow numpy
```

**Note**: This project uses **PaddleOCR 3.x** (November 2025), which introduces a new API:
- Uses `ocr.predict(input=...)` instead of the deprecated `ocr.ocr(...)`
- Uses `use_textline_orientation` instead of deprecated `use_angle_cls`
- Results are returned as structured `OCRResult` objects with `.json` attribute

Other libraries (e.g. `pandas`) are installed as part of Step 1 / general project setup.

---

## 3. Step 2 folder structure (after running scripts)

From the project root:

```text
Step_2/
  build_data_manifest.py    # Script to build unified manifest from all datasets
  run_ocr.py                # Script to run OCR over all images in the manifest
  data_manifest.jsonl       # Unified manifest (one JSON object per line)
  ocr.jsonl                 # OCR results, one JSON object per image example
  ocr_qc/                   # (Optional) QC images with OCR overlays
    .gitkeep                # Placeholder so the folder is tracked in git
  .gitkeep                  # Placeholder so Step_2/ is tracked in git
```

Notes:
- `data_manifest.jsonl` and `ocr.jsonl` are **generated** artifacts.
- `ocr_qc/` will be **empty** until you run `run_ocr.py` with QC enabled (default).

---

## 4. `data_manifest.jsonl` format and how it is built

### 4.1 Purpose

The manifest provides a single, consistent place to look up every example across all datasets, with:
- A unified `id` string per example.
- Dataset name and split.
- Optional image path (if the example has an image).
- Raw text and labels as they appear in the original dataset (no unified label space yet).

This file is the input to:
- OCR (for image paths in Step 2).
- Later unimodal experts and fusion models (in later steps).

### 4.2 How to generate the manifest

From the repo root:

```bash
python Step_2/build_data_manifest.py
```

This reads relevant files under `Datasets/` and writes:

```text
Step_2/data_manifest.jsonl
```

### 4.3 Manifest schema

Each line is a single JSON object with at least the following fields:

- `id` (str)
  - A unique identifier for the example, typically of the form
    `"<dataset>_<split>_<example_id>"`.
- `dataset` (str)
  - Dataset name, e.g. `"hateful_memes"`, `"mami"`, `"mmhs150k"`, `"memotion1"`,
    `"memotion2"`, `"hatexplain"`, `"olid"`, etc.
- `split` (str)
  - Split name, e.g. `"train"`, `"val"`, `"test"`.
- `image_path` (str or null)
  - Absolute path to the image file for this example, if it has one.
  - `null` (or omitted) for pure text-only examples.
- `text_raw` (str or null)
  - Raw text for the example as provided by the dataset.
  - For meme datasets this may be the meme caption or associated text field.
- `labels_raw` (any JSON-serializable value)
  - Dataset-specific labels in their original format (integers, strings, lists, etc.).
  - No harmonization is performed yet; this will happen in later steps.

Example (Hateful Memes):

```json
{"id": "hateful_memes_train_42953", "dataset": "hateful_memes", "split": "train", "image_path": "/.../Datasets/hateful_memes/img/42953.png", "text_raw": "its their character not their color that matters", "labels_raw": 0}
```

---

## 5. `run_ocr.py` and `ocr.jsonl`

### 5.1 What `run_ocr.py` does

`run_ocr.py` iterates over `data_manifest.jsonl` and, for every record with a non-empty
`image_path` that exists on disk, runs OCR using PaddleOCR.

The script:
- Loads each image from `image_path`.
- Resizes it **in-memory** so that the longest side is at most **1024 px** (preserving aspect ratio).
- Runs PaddleOCR on the resized image.
- Saves recognized text lines and bounding boxes to `Step_2/ocr.jsonl`.
- Optionally writes QC overlay images to `Step_2/ocr_qc/<id>.png`.
- Prints basic quality statistics to stdout at the end.

### 5.2 How to run OCR

From the repo root:

```bash
python Step_2/run_ocr.py
```

**For testing** (process only a few images first):

```bash
python Step_2/run_ocr.py --limit 100 --progress 10
```

Optional arguments:

- `--manifest PATH`
  - Override the input manifest path (default: `Step_2/data_manifest.jsonl`).
- `--output PATH`
  - Override the OCR output path (default: `Step_2/ocr.jsonl`).
- `--no-qc`
  - Disable saving QC overlay images.
- `--limit N`
  - Process only the first N images (useful for testing).
- `--progress N`
  - Print progress every N images (default: 100).

### 5.3 `ocr.jsonl` schema

Each line in `ocr.jsonl` is a JSON object aligned to a manifest entry with an image:

- `id` (str)
  - Copied from the manifest.
- `dataset` (str)
  - Copied from the manifest.
- `split` (str)
  - Copied from the manifest.
- `image_path` (str)
  - Copied from the manifest.
- `ocr` (object)
  - `lines` (list of objects)
    - Each element represents a recognized text line:
      - `text` (str): recognized text.
      - `conf` (float): confidence score from PaddleOCR.
      - `bbox` (list[float]): `[x1, y1, x2, y2]` in image pixel coordinates of the
        **resized** image used for OCR.

Example line (structure only, values illustrative):

```json
{
  "id": "hateful_memes_train_42953",
  "dataset": "hateful_memes",
  "split": "train",
  "image_path": "/.../Datasets/hateful_memes/img/42953.png",
  "ocr": {
    "lines": [
      {"text": "its their character not their color that matters", "conf": 0.97, "bbox": [100.0, 50.0, 900.0, 150.0]}
    ]
  }
}
```

If OCR fails completely for an image, `lines` will be an empty list.

### 5.4 QC overlay images (`ocr_qc/`)

When run without `--no-qc`, `run_ocr.py` will also write PNG images to:

```text
Step_2/ocr_qc/<id>.png
```

Each QC image:
- Loads the original image from `image_path`.
- Draws red rectangles around detected text lines.
- Renders the recognized text and confidence on top of or near each box.

You can open these images to visually inspect OCR quality for specific examples.

---

## 6. Quality metrics and acceptance criteria

At the end of a run, `run_ocr.py` prints summary statistics:

- `Total examples with images`: number of manifest records with existing `image_path`.
- `Images with no OCR text`: how many had `lines == []`.
- `Mean OCR confidence over lines`: average confidence across all recognized lines.

These relate to the quality gates defined in the overall project spec:

- **OCR failure rate** (images with no text) should ideally be **< 10%**.
- **Mean confidence** should be **≥ 0.70**.

You can re-run OCR with improved models or parameters later if these thresholds are not met.

---

## 7. Summary of Step 2 artifacts

After successfully running both scripts, you should have:

- `Step_2/data_manifest.jsonl`
  - Unified list of all examples across image and text-only datasets.
- `Step_2/ocr.jsonl`
  - OCR results for all examples with images, aligned by `id`.
- `Step_2/ocr_qc/`
  - Optional visual QC overlays for spot-checking OCR performance.

These artifacts will be consumed by later steps to train unimodal experts and multimodal
fusion models for abuse/hate detection.
