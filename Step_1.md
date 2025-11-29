# Step 1 — Datasets (By Type)

This document summarizes how each dataset in **Step 1** was obtained and where it is stored in this repository. All paths are relative to the project root `DL_Proj/`.

---

## 1A) Memes / Images (image + overlaid text)

### 1. Hateful Memes (Facebook AI / NeurIPS 2020)

- **Source**: Kaggle mirror of the Facebook Hateful Memes dataset:
  - `https://www.kaggle.com/datasets/parthplc/facebook-hateful-meme-dataset`
- **How it was obtained**:
  1. Opened the Kaggle dataset page and accepted the dataset terms.
  2. Downloaded the main ZIP archive from Kaggle.
  3. Placed the archive into `Datasets/hateful_memes/` and unzipped it there.
- **Final location and structure**:
  - `Datasets/hateful_memes/`
    - `img/` — all meme images (`*.png`).
    - `train.jsonl` — training split.
    - `dev.jsonl` — development/validation split.
    - `test.jsonl` — test split.
- **Notes**:
  - The JSONL files contain at least: `text`, `img` (relative path to `img/`), and `label` (0 = non-hateful, 1 = hateful). We will add unified IDs and taxonomy-aligned labels at the manifest-building stage, not here.

### 2. MAMI (SemEval 2022, misogyny in memes)

- **Source**: Official GitHub repository:
  - `https://github.com/MIND-Lab/SemEval2022-Task-5-Multimedia-Automatic-Misogyny-Identification-MAMI-`
- **How it was obtained**:
  1. Visited the repository and followed the dataset download instructions (training, trial, test archives).
  2. Downloaded the provided ZIP files: `training.zip`, `trial.zip`, `test.zip`.
  3. Placed all archives in `Datasets/mami/` and unzipped them there.
- **Final location and structure**:
  - `Datasets/mami/`
    - `TRAINING/` — training images (`*.jpg`).
    - `test/` — test images (`*.jpg`).
    - `training.zip`, `trial.zip`, `test.zip` — original archives (kept for reference).
    - `test_labels.txt` — labels for the test split.
    - `readme.txt` — dataset readme from the authors.
- **Notes**:
  - This structure matches the official release and is suitable for building our unified manifest later.

### 3. MMHS150K (Twitter multimodal hate speech)

- **Source**: Official project page:
  - `https://gombru.github.io/2019/10/09/MMHS/`
- **How it was obtained**:
  1. Followed the "Download MMHS150K" link from the project page (CVC / Google Drive / Mega).
  2. Downloaded the full archive (~6 GB) and placed it under `Datasets/mmhs150k/`.
  3. Unzipped the archive in that folder.
- **Final location and structure**:
  - `Datasets/mmhs150k/`
    - `img_resized/` — 150K resized images from Twitter.
    - `img_txt/` — text crops / overlays extracted from images.
    - `MMHS150K_GT.json` — main ground-truth annotations (labels and metadata).
    - `splits/`
      - `train_ids.txt` — list of training IDs.
      - `val_ids.txt` — list of validation IDs.
      - `test_ids.txt` — list of test IDs.
    - `MMHS150K_readme.txt`, `hatespeech_keywords.txt` — additional metadata.
- **Notes**:
  - No reorganization was done; this matches the recommended layout from the authors.

### 4. Memotion 1.0 (SemEval 2020 Task 8: Memotion Analysis)

- **Source**: Kaggle "Memotion Dataset 7k" (SemEval 2020 Task 8):
  - `https://www.kaggle.com/datasets/williamscott701/memotion-dataset-7k`
- **How it was obtained**:
  1. Opened the Kaggle dataset page and accepted the terms of use.
  2. Downloaded the dataset ZIP archive.
  3. Placed the archive into `Datasets/memotion1/` and unzipped it there.
- **Final location and structure**:
  - `Datasets/memotion1/`
    - `images/` — ≈7k meme images.
    - `labels.csv` / `labels.xlsx` / `labels_pd_pickle` — label files for sentiment/emotion tasks.
    - `reference.csv` / `reference.xlsx` / `reference_df_pickle` — reference metadata.
- **Notes**:
  - The raw structure from Kaggle is preserved. Our ingestion code will read labels from `labels.csv` and map them into the unified abuse/target taxonomy.

### 5. Memotion 2.0 (Sentiment & emotion analysis of memes)

- **Source**: Memotion 2.0 task / CodaLab, as referenced in the paper:
  - `https://ceur-ws.org/Vol-3199/paper20.pdf`
  - Competition page: `https://competitions.codalab.org/competitions/35688`
- **How it was obtained**:
  1. Followed the competition/data links and downloaded the Memotion2 archives.
  2. Placed them in `Datasets/memotion2/` and unzipped.
- **Final location and structure**:
  - `Datasets/memotion2/`
    - `image folder/` — images for the Memotion2 task.
    - `Memotion2/`
      - `memotion_train.csv` — training labels.
      - `memotion_val.csv` — validation labels.
    - `memotion_test.csv`, `memotion2_test.zip`, `image folder.zip`, `Memotion2.zip`, `README.txt`, `format.py`, `Passwords.heic` — additional files from the release.
- **Notes**:
  - No restructuring was done. We will use `image folder/` plus the CSVs inside `Memotion2/` for ingestion.

---

## 1B) Text-Only Datasets (Auxiliary Supervision)

### 6. HateXplain

- **Source**: Official GitHub repository:
  - `https://github.com/hate-alert/HateXplain`
- **How it was obtained**:
  1. Downloaded/cloned the GitHub repo into `Datasets/hatexplain/`.
  2. Kept the full repository structure for reference and experiments.
- **Final location and structure**:
  - `Datasets/hatexplain/`
    - `Data/`
      - `dataset.json` — main dataset (posts, labels, targets, rationales).
      - `post_id_divisions.json` — train/dev/test split information.
      - Additional metadata files like `classes.npy`, `classes_two.npy`, `README.md`.
    - Plus original repo folders: `Models/`, `Preprocess/`, `eraserbenchmark/`, notebooks, scripts, etc.
- **Notes**:
  - For our pipeline we mainly need `Data/dataset.json` and `Data/post_id_divisions.json`. Everything else is kept for potential reference.

### 7. OLID / OffensEval 2019 (Offensive Language Identification Dataset)

- **Source**: Official OffensEval/OLID site:
  - `https://sites.google.com/site/offensevalsharedtask/olid`
- **How it was obtained**:
  1. Downloaded the OLID training and test TSVs and the label CSVs according to the instructions on the site.
  2. Placed all files in `Datasets/olid/`.
- **Final location and structure**:
  - `Datasets/olid/`
    - `olid-training-v1.0.tsv` — main training set.
    - `testset-levela.tsv`, `testset-levelb.tsv`, `testset-levelc.tsv` — test sets.
    - `labels-levela.csv`, `labels-levelb.csv`, `labels-levelc.csv` — gold labels for test.
    - `README.txt`, `olid-annotation.txt` — documentation.
- **Notes**:
  - This matches the official OLID release layout and is ready for ingestion into our unified manifest.

---

## 1D) Diagnostic Suites (Evaluation Only)

### 8. HateCheck (English functional tests)

- **Source**: Official HateCheck data repo:
  - `https://github.com/paul-rottger/hatecheck-data`
- **How it was obtained**:
  1. Downloaded/cloned the `hatecheck-data` repository into `Datasets/diagnostics/hatecheck/`.
- **Final location and structure**:
  - `Datasets/diagnostics/hatecheck/`
    - `test_suite_cases.csv` — main functional test cases.
    - `test_suite_annotations.csv`, `all_cases.csv`, `all_annotations.csv`.
    - `LICENSE`, `README.md`, `annotation_guidelines.pdf`, and template files.
- **Notes**:
  - We will use `test_suite_cases.csv` during Step 7 (functional evaluation). No changes to the original structure were made.

### 9. Multilingual HateCheck

- **Source**: Official Multilingual HateCheck repo:
  - `https://github.com/rewire-online/multilingual-hatecheck`
- **How it was obtained**:
  1. Downloaded/cloned the repository into `Datasets/diagnostics/multilingual_hatecheck/`.
- **Final location and structure**:
  - `Datasets/diagnostics/multilingual_hatecheck/`
    - `MHC Final Cases/` — final test suites for 10 languages.
    - `MHC Ingredients/` — templates/components for generating test cases.
    - `LICENSE`, `README.md`.
- **Notes**:
  - This will be used for multilingual functional testing (e.g., Hindi) in later steps.

---

## 1C) Video (Short Clips) — HateMM

### 10. HateMM (Hateful Video Dataset)

- **Source**: HateMM dataset introduced in the paper "HateMM: A Multi-Modal Dataset for Hate Video Classification" (access obtained according to the dataset's terms/availability).
- **How it was obtained**:
  1. Requested or downloaded the HateMM dataset following the instructions associated with the paper/official release.
  2. Organized the videos and annotations under `Datasets/video_custom/`.
- **Final location and structure**:
  - `Datasets/video_custom/`
    - `hate_videos/` — hateful video clips (e.g., `*.mp4`).
    - `non_hate_videos/` — non-hateful (control) video clips.
    - `HateMM_annotation.csv` — annotations mapping each video to hate / non-hate labels and possibly finer-grained metadata.
    - `readme.txt` — local notes about the dataset usage.
- **Notes**:
  - This folder is the basis for our multimodal **video** pipeline: later steps (Step 2B and beyond) will run VAD + Whisper ASR on these videos and integrate them into the unified manifest and multimodal model.

---

## Summary

By the end of Step 1, all datasets referenced in the project plan are:

- Downloaded from their official or widely-used mirrors (Hugging Face, Kaggle, GitHub, project pages).
- Placed under a consistent top-level `Datasets/` directory, with one subfolder per dataset.
- Kept as close as possible to their **original** author-provided structure, so that ingestion code can be written in a transparent and reproducible way.

The next step is to build a **unified `data_manifest.jsonl`** that:

- Indexes all examples across datasets.
- Maps dataset-specific labels into the project-wide taxonomy (abuse vs non-abuse, targets, etc.).
- Records media paths (images, videos) and raw text, ready for OCR/ASR extraction in Step 2 and packing in Step 3.
