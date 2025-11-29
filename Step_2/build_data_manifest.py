#!/usr/bin/env python
"""Build unified data_manifest.jsonl for Step 2.

Datasets (image+text + text-only):
- hateful_memes
- mami
- mmhs150k
- memotion1
- memotion2
- hatexplain (text-only)
- olid (text-only)

Output: Step_2/data_manifest.jsonl with records:
{ "id": str, "dataset": str, "split": str,
  "image_path": Optional[str], "text_raw": str,
  "labels_raw": Any (original label structure)}

This script assumes it is run from the repo root.
"""

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ROOT / "Datasets"
STEP2 = ROOT / "Step_2"


def _ensure_step2_dir() -> None:
    STEP2.mkdir(parents=True, exist_ok=True)


def _open_jsonl_write(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.open("w", encoding="utf-8")


def iter_hateful_memes() -> Iterable[Dict[str, Any]]:
    ds_root = DATASETS / "hateful_memes"
    for split_name, filename in [("train", "train.jsonl"), ("dev", "dev.jsonl"), ("test", "test.jsonl")]:
        jsonl_path = ds_root / filename
        if not jsonl_path.exists():
            continue
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                img_rel = obj.get("img")
                image_path = str((ds_root / img_rel).resolve()) if img_rel else None
                yield {
                    "id": f"hateful_memes_{split_name}_{obj.get('id')}",
                    "dataset": "hateful_memes",
                    "split": split_name,
                    "image_path": image_path,
                    "text_raw": obj.get("text", ""),
                    "labels_raw": obj.get("label"),
                }


def iter_mami() -> Iterable[Dict[str, Any]]:
    ds_root = DATASETS / "mami"

    # Train: TRAINING/training.csv + images
    train_csv = ds_root / "TRAINING" / "training.csv"
    train_img_root: Optional[Path] = None
    training_dir = ds_root / "TRAINING"
    # Heuristic: images are either directly under TRAINING or a subfolder named like "training" / "Training".
    if training_dir.exists():
        # Prefer nested image folder if present
        for sub in training_dir.iterdir():
            if sub.is_dir() and sub.name.lower().startswith("train"):
                train_img_root = sub
                break
        if train_img_root is None:
            train_img_root = training_dir

    if train_csv.exists() and train_img_root is not None:
        with train_csv.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            # If delimiter guess fails, fall back to default csv dialect
            if reader.fieldnames is None or reader.fieldnames == ["file_name,misogynous,shaming,stereotype,objectification,violence,Text Transcription"]:
                f.seek(0)
                reader = csv.DictReader(f)
            for row in reader:
                file_name = row.get("file_name") or row.get("file_name ")
                if not file_name:
                    continue
                img_path = (train_img_root / file_name).resolve()
                text = row.get("Text Transcription", "")
                labels = {
                    "misogynous": int(row.get("misogynous", 0)),
                    "shaming": int(row.get("shaming", 0)),
                    "stereotype": int(row.get("stereotype", 0)),
                    "objectification": int(row.get("objectification", 0)),
                    "violence": int(row.get("violence", 0)),
                }
                yield {
                    "id": f"mami_train_{file_name}",
                    "dataset": "mami",
                    "split": "train",
                    "image_path": str(img_path),
                    "text_raw": text,
                    "labels_raw": labels,
                }

    # Test: test/Test.csv + test_labels.txt + images
    test_txt = ds_root / "test_labels.txt"
    test_csv = ds_root / "test" / "Test.csv"
    test_img_root: Optional[Path] = None
    test_dir = ds_root / "test"
    if test_dir.exists():
        test_img_root = test_dir

    labels_by_name: Dict[str, List[int]] = {}
    if test_txt.exists():
        with test_txt.open("r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) != 6:
                    continue
                fn = parts[0]
                try:
                    vals = [int(x) for x in parts[1:]]
                except ValueError:
                    continue
                labels_by_name[fn] = vals

    if test_csv.exists() and test_img_root is not None:
        with test_csv.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                file_name = row.get("file_name")
                if not file_name:
                    continue
                img_path = (test_img_root / file_name).resolve()
                text = row.get("Text Transcription", "")
                raw_labels = labels_by_name.get(file_name)
                labels = None
                if raw_labels is not None and len(raw_labels) == 5:
                    labels = {
                        "misogynous": raw_labels[0],
                        "shaming": raw_labels[1],
                        "stereotype": raw_labels[2],
                        "objectification": raw_labels[3],
                        "violence": raw_labels[4],
                    }
                yield {
                    "id": f"mami_test_{file_name}",
                    "dataset": "mami",
                    "split": "test",
                    "image_path": str(img_path),
                    "text_raw": text,
                    "labels_raw": labels,
                }


def iter_mmhs150k() -> Iterable[Dict[str, Any]]:
    ds_root = DATASETS / "mmhs150k"
    gt_path = ds_root / "MMHS150K_GT.json"
    splits_root = ds_root / "splits"
    img_root = ds_root / "img_resized"
    if not gt_path.exists():
        return
    with gt_path.open("r", encoding="utf-8") as f:
        gt = json.load(f)

    id_to_split: Dict[str, str] = {}
    for split_name, fname in [("train", "train_ids.txt"), ("val", "val_ids.txt"), ("test", "test_ids.txt")]:
        p = splits_root / fname
        if not p.exists():
            continue
        with p.open("r", encoding="utf-8") as f_ids:
            for line in f_ids:
                ex_id = line.strip()
                if not ex_id:
                    continue
                id_to_split[ex_id] = split_name

    for ex_id, info in gt.items():
        split = id_to_split.get(ex_id, "unsplit")
        img_name = f"{ex_id}.jpg"
        img_path = (img_root / img_name).resolve()
        text = info.get("tweet_text", "")
        labels = {
            "labels": info.get("labels"),
            "labels_str": info.get("labels_str"),
        }
        yield {
            "id": f"mmhs150k_{ex_id}",
            "dataset": "mmhs150k",
            "split": split,
            "image_path": str(img_path),
            "text_raw": text,
            "labels_raw": labels,
        }


def iter_memotion1() -> Iterable[Dict[str, Any]]:
    ds_root = DATASETS / "memotion1"
    csv_path = ds_root / "labels.csv"
    img_root = ds_root / "images"
    if not csv_path.exists():
        return

    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_name = row.get("image_name")
            if not img_name:
                continue
            img_path = (img_root / img_name).resolve()
            text = row.get("text_corrected") or row.get("text_ocr") or ""
            labels = {
                "humour": row.get("humour"),
                "sarcasm": row.get("sarcasm"),
                "offensive": row.get("offensive"),
                "motivational": row.get("motivational"),
                "overall_sentiment": row.get("overall_sentiment"),
            }
            yield {
                "id": f"memotion1_{img_name}",
                "dataset": "memotion1",
                "split": "train",  # dataset does not define explicit splits
                "image_path": str(img_path),
                "text_raw": text,
                "labels_raw": labels,
            }


def iter_memotion2() -> Iterable[Dict[str, Any]]:
    ds_root = DATASETS / "memotion2"
    img_root_train = ds_root / "image folder" / "train_images"
    img_root_val = ds_root / "image folder" / "val_images"

    def _iter_split(csv_path: Path, img_root: Path, split_name: str) -> Iterable[Dict[str, Any]]:
        if not csv_path.exists() or not img_root.exists():
            return []
        with csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ex_id = row.get("Id")
                if not ex_id:
                    continue
                img_name = f"{ex_id}.jpg"
                img_path = (img_root / img_name).resolve()
                text = row.get("ocr_text", "")
                labels = {
                    "humour": row.get("humour"),
                    "sarcastic": row.get("sarcastic"),
                    "offensive": row.get("offensive"),
                    "motivational": row.get("motivational"),
                    "overall_sentiment": row.get("overall_sentiment"),
                    "classification_based_on": row.get("classification_based_on"),
                }
                yield {
                    "id": f"memotion2_{split_name}_{ex_id}",
                    "dataset": "memotion2",
                    "split": split_name,
                    "image_path": str(img_path),
                    "text_raw": text,
                    "labels_raw": labels,
                }

    train_csv = ds_root / "Memotion2" / "memotion_train.csv"
    val_csv = ds_root / "Memotion2" / "memotion_val.csv"

    for rec in _iter_split(train_csv, img_root_train, "train"):
        yield rec
    for rec in _iter_split(val_csv, img_root_val, "val"):
        yield rec

    # Test split has no labels; include if desired
    test_csv = ds_root / "memotion_test.csv"
    if test_csv.exists():
        with test_csv.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ex_id = row.get("Id")
                if not ex_id:
                    continue
                img_name = f"{ex_id}.jpg"
                # Test images are usually not provided locally; leave image_path empty if missing
                img_path = None
                if img_root_train.exists():
                    candidate = img_root_train / img_name
                    if candidate.exists():
                        img_path = str(candidate.resolve())
                if img_path is None and img_root_val.exists():
                    candidate = img_root_val / img_name
                    if candidate.exists():
                        img_path = str(candidate.resolve())
                text = row.get("ocr_text", "")
                labels = {
                    "classification_based_on": row.get("classification_based_on"),
                }
                yield {
                    "id": f"memotion2_test_{ex_id}",
                    "dataset": "memotion2",
                    "split": "test",
                    "image_path": img_path,
                    "text_raw": text,
                    "labels_raw": labels,
                }


def iter_hatexplain() -> Iterable[Dict[str, Any]]:
    ds_root = DATASETS / "hatexplain" / "Data"
    dataset_path = ds_root / "dataset.json"
    splits_path = ds_root / "post_id_divisions.json"
    if not dataset_path.exists() or not splits_path.exists():
        return
    with dataset_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    with splits_path.open("r", encoding="utf-8") as f:
        splits = json.load(f)

    id_to_split: Dict[str, str] = {}
    for split_name, ids in splits.items():
        for pid in ids:
            id_to_split[pid] = split_name

    for pid, info in data.items():
        split = id_to_split.get(pid, "unsplit")
        tokens = info.get("post_tokens", [])
        text = " ".join(tokens)
        labels = info.get("annotators", [])
        yield {
            "id": f"hatexplain_{pid}",
            "dataset": "hatexplain",
            "split": split,
            "image_path": None,
            "text_raw": text,
            "labels_raw": labels,
        }


def iter_olid() -> Iterable[Dict[str, Any]]:
    ds_root = DATASETS / "olid"
    train_tsv = ds_root / "olid-training-v1.0.tsv"
    test_tsv = ds_root / "testset-levela.tsv"
    test_labels_csv = ds_root / "labels-levela.csv"

    # Train
    if train_tsv.exists():
        with train_tsv.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                ex_id = row.get("id")
                if not ex_id:
                    continue
                text = row.get("tweet", "")
                labels = {
                    "subtask_a": row.get("subtask_a"),
                    "subtask_b": row.get("subtask_b"),
                    "subtask_c": row.get("subtask_c"),
                }
                yield {
                    "id": f"olid_train_{ex_id}",
                    "dataset": "olid",
                    "split": "train",
                    "image_path": None,
                    "text_raw": text,
                    "labels_raw": labels,
                }

    # Test (level a)
    id_to_label: Dict[str, str] = {}
    if test_labels_csv.exists():
        with test_labels_csv.open("r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) != 2:
                    continue
                id_to_label[row[0]] = row[1]

    if test_tsv.exists():
        with test_tsv.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                ex_id = row.get("id")
                if not ex_id:
                    continue
                text = row.get("tweet", "")
                label = id_to_label.get(ex_id)
                labels = {"subtask_a": label}
                yield {
                    "id": f"olid_test_{ex_id}",
                    "dataset": "olid",
                    "split": "test",
                    "image_path": None,
                    "text_raw": text,
                    "labels_raw": labels,
                }


def build_manifest(output_path: Path) -> None:
    _ensure_step2_dir()
    with _open_jsonl_write(output_path) as out_f:
        for iterator in [
            iter_hateful_memes,
            iter_mami,
            iter_mmhs150k,
            iter_memotion1,
            iter_memotion2,
            iter_hatexplain,
            iter_olid,
        ]:
            for record in iterator():
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build unified data_manifest.jsonl for Step 2.")
    parser.add_argument(
        "--output",
        type=str,
        default=str(STEP2 / "data_manifest.jsonl"),
        help="Path to output manifest JSONL",
    )
    args = parser.parse_args()
    output_path = Path(args.output)
    build_manifest(output_path)
    print(f"Wrote manifest to {output_path}.")


if __name__ == "__main__":
    main()
