# Step 4: Train Unimodal Experts (Text & Image)

## 1. Goal of Step 4

Step 4 takes the packed examples from **Step 3** and trains **unimodal experts**:

- A **text-only expert** over the combined text (OCR + captions).
- An **image-only expert** over meme images.

These experts will later be fused in **Step 5** into a multimodal model.

Current scope:
- Use the **packed shards** from `Step_3/shards/` (initially on the 50-example pilot).
- Implement training in a **Jupyter notebook** for easier experimentation:
  - Load WebDataset shards.
  - Fine-tune a text model (mDeBERTa‑v3‑base) on the text field.
  - Fine-tune a vision model (SigLIP‑like ViT) on the images.
  - Log losses and basic metrics (e.g. accuracy / F1 on a small dev split).

Audio-only expert (Whisper‑based) is deferred until we have ASR outputs from Step 2B.

---

## 2. Inputs and prerequisites

Before running Step 4:

- Steps 1–3 must be complete for the subset you want to train on:
  - `Datasets/` populated and documented in `Step_1.md`.
  - `Step_2/data_manifest.jsonl` and `Step_2/ocr.jsonl` created.
  - `Step_3/pack_examples.py` run to generate packed shards and stats.

Key inputs to Step 4:

- `Step_3/shards/`
  - WebDataset-style shards with keys:
    - `<id>.png` — preprocessed image (384×384) if available.
    - `<id>.txt` — combined text (`[OCR] ... [/OCR] [CAP] ... [/CAP] <lang=en>`).
    - `<id>.json` — metadata and unified labels.
- `Step_3/label_taxonomy.json`
  - Definition of the unified label schema (`abuse_hate`, `offensive`, `target_type`, etc.).
- `Step_3/splits.json`
  - Mapping from example `id` to split (`train` / `val` / `test`).

Environment:

- A Python environment with GPU is recommended but not strictly required for small pilots.
- Typical dependencies (installed in the notebook):

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121  # or CPU-only
pip install transformers datasets webdataset accelerate timm sentencepiece
```

(Exact commands and versions can be adjusted based on your platform.)

---

## 3. Step 4 folder structure

From the project root:

```text
Step_4/
  step4_unimodal_experts.ipynb  # Main notebook: trains text & vision experts
  .gitkeep                      # Placeholder so Step_4/ is tracked in git
```

And at the repo root:

```text
Step_4.md                       # This documentation file
```

The notebook reads data from `Step_3/shards/` and writes models under a `models/` folder
(which can be created relative to the repo root or `Step_4/`).

---

## 4. What the notebook does

`Step_4/step4_unimodal_experts.ipynb` is organized into the following sections:

1. **Setup and configuration**
   - Imports libraries (`torch`, `transformers`, `webdataset`, etc.).
   - Defines paths (root, `Step_3/shards/`, output `models/` directory).
   - Chooses model names, e.g.:
     - Text model: `microsoft/mdeberta-v3-base`.
     - Vision model: `google/siglip-base-patch16-384` (or similar SigLIP‑compatible model from Hugging Face).

2. **Load WebDataset shards**
   - Uses WebDataset to read `(txt, json)` pairs for text and `(png, json)` pairs for images from a single **training shard** (the 50-example pilot).
   - Builds simple in-memory lists of examples and then PyTorch `DataLoader`s for:
     - **Text-only** batches: tokenized combined text plus `abuse_hate` labels.
     - **Image-only** batches: processed image tensors plus `abuse_hate` labels.
   - For the current pilot run, all loaded examples are treated as **train**; later you can incorporate `Step_3/splits.json` for proper train/val/test splits.

3. **Text-only expert training**
   - Loads `microsoft/mdeberta-v3-base` with a classification head via `AutoModelForSequenceClassification`.
   - Performs binary classification on `labels.abuse_hate` (0/1) using cross-entropy loss.
   - Trains for a small number of epochs (currently `EPOCHS = 1` for the pilot) and logs the training loss.
   - Saves the fine-tuned text expert in Hugging Face format under `models/text_expert/` using `save_pretrained` (model and tokenizer).

4. **Image-only expert training**
   - Loads `google/siglip-base-patch16-384` via `AutoModelForImageClassification` with a 2-class head.
   - Uses images from the shards with basic augmentations (random horizontal flip) and the same `abuse_hate` labels.
   - Trains for a small number of epochs (currently `EPOCHS_V = 1` for the pilot) and logs the training loss.
   - Saves the fine-tuned vision expert in Hugging Face format under `models/vision_expert/` using `save_pretrained` (model and image processor).

5. **Logging and basic metrics**
   - For the current pilot run, tracks and prints **training loss per epoch** for both experts.
   - When scaling up to full shards, you can extend the notebook with proper train/validation splits and add accuracy and macro-F1 metrics on a held-out dev set.

The notebook is written so that you can start with **very small runs** (on your 50-example pilot) and later point it at larger shards when Steps 2–3 are run on the full dataset.

---

## 5. How to run Step 4

1. Start a Jupyter server from the repo root (inside your venv):

```bash
source venv/bin/activate
jupyter notebook
```

2. Open the notebook:

- `Step_4/step4_unimodal_experts.ipynb`

3. Run cells in order:

- **Setup & config**: verifies paths and imports libraries.
- **Data loading**: prints how many examples are in train/val.
- **Train text expert**: runs a small number of epochs and saves `models/text_cls.pt`.
- **Train image expert**: same for `models/vision_cls.pt`.

4. Inspect results:

- Check printed metrics at the end of each training section.
- Verify that model files exist under `models/`.

---

## 6. Quality targets for Step 4

From the overall project spec, the unimodal experts should:

- Outperform trivial baselines (e.g. majority-class) on dev macro-F1.
- Produce stable training curves (no divergence) with reasonable calibration.

At this stage, with only 50 examples packed, the goal is mostly to **validate the pipeline**.
When you later pack the full datasets in Step 3, you can re-run the same notebook with
more shards to train serious text and vision experts.
