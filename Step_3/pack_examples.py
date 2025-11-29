#!/usr/bin/env python
"""Pack manifest + OCR into WebDataset shards for training (Step 3).

This script reads:
- Step_2/data_manifest.jsonl  (unified examples from all datasets)
- Step_2/ocr.jsonl            (OCR results per image example, if available)

and writes WebDataset-style shards under Step_3/shards/ with, for each example:
- <id>.png   : preprocessed image (if image_path is not null)
- <id>.txt   : combined text field (OCR + caption/text)
- <id>.json  : labels and metadata (unified schema)

The unified label schema is defined in Step_3/label_taxonomy.json. If that file does
not exist yet, this script will create a default version the first time it runs.
Similarly, Step_3/splits.json will be written on first run, mapping example ids to
train/val/test splits (mirroring the manifest's `split` field).

You can run a small pilot first (e.g. for 5k examples) before packing the full set.
"""

import argparse
import json
import sys
import time
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
from PIL import Image

try:
    import webdataset as wds
except ImportError as e:  # pragma: no cover
    raise RuntimeError(
        "webdataset is not installed. Install with:\n"
        "  pip install webdataset\n"
    ) from e


ROOT = Path(__file__).resolve().parents[1]
STEP2 = ROOT / "Step_2"
STEP3 = ROOT / "Step_3"


def _iter_manifest(manifest_path: Path) -> Iterable[Dict[str, Any]]:
    """Yield records from a JSONL manifest file."""
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _load_ocr_index(ocr_path: Path) -> Dict[str, List[Dict[str, Any]]]:
    """Load OCR results into an index keyed by example id.

    Each value is the `ocr.lines` list for that example.
    """
    index: Dict[str, List[Dict[str, Any]]] = {}
    if not ocr_path.exists():
        print(f"[WARN] OCR file not found, continuing without OCR: {ocr_path}")
        return index

    with ocr_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            rec_id = rec.get("id")
            ocr = rec.get("ocr") or {}
            lines = ocr.get("lines") or []
            if rec_id:
                index[rec_id] = lines
    return index


def _build_text(ocr_lines: List[Dict[str, Any]], text_raw: Optional[str]) -> str:
    """Build combined text field from OCR lines and raw caption/text.

    Pattern (for now, assuming English):
    [OCR] <joined OCR lines> [/OCR] [CAP] <text_raw> [/CAP] <lang=en>
    """
    parts: List[str] = []

    texts_ocr: List[str] = []
    for ln in ocr_lines or []:
        t = (ln.get("text") or "").strip()
        try:
            conf = float(ln.get("conf", 0.0))
        except Exception:
            conf = 0.0
        if not t:
            continue
        # Drop extremely low confidence lines
        if conf < 0.3:
            continue
        texts_ocr.append(t)

    if texts_ocr:
        parts.append("[OCR] " + " ".join(texts_ocr) + " [/OCR]")

    if text_raw is not None:
        tr = str(text_raw).strip()
        if tr:
            parts.append("[CAP] " + tr + " [/CAP]")

    if parts:
        parts.append("<lang=en>")

    return " ".join(parts).strip()


def _map_labels(dataset: str, labels_raw: Any) -> Dict[str, Any]:
    """Map dataset-specific `labels_raw` into a unified label schema.

    This is a minimal initial mapping that is safe for all datasets and can be
    extended later without breaking consumers.

    - `abuse_hate`: binary hate/abuse flag when clearly available.
    - `offensive`: binary offensive-language flag when clearly available.
    - `target_type`: categorical (none/individual/group/other/unknown).
    - `target_group`: multi-label list of coarse target group tags (optional).
    - `aux`: carries the original `labels_raw` and dataset name.
    """
    result: Dict[str, Any] = {
        "abuse_hate": None,
        "offensive": None,
        "target_type": "unknown",
        "target_group": [],
        "aux": {
            "dataset": dataset,
            "labels_raw": labels_raw,
        },
    }

    # Example: Hateful Memes uses 0/1 labels where 1 is hateful
    if dataset == "hateful_memes":
        try:
            v = int(labels_raw)
        except Exception:
            v = labels_raw
        if v in (0, 1):
            result["abuse_hate"] = v
            result["offensive"] = v
            result["target_type"] = "group" if v == 1 else "none"

    # TODO: Extend mappings for other datasets (MAMI, MMHS150K, Memotion, HateXplain, OLID)

    return result


def _ensure_default_taxonomy(taxonomy_path: Path) -> Dict[str, Any]:
    """Ensure a default label taxonomy file exists and return its contents.

    If `taxonomy_path` already exists, it is loaded and returned.
    Otherwise, a minimal default taxonomy is written and returned.
    """
    if taxonomy_path.exists():
        with taxonomy_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    taxonomy: Dict[str, Any] = {
        "schema": {
            "abuse_hate": {
                "type": "binary",
                "values": [0, 1],
                "description": "Hate/abuse present (1) or not (0)",
            },
            "offensive": {
                "type": "binary",
                "values": [0, 1],
                "description": "Offensive language present",
            },
            "target_type": {
                "type": "categorical",
                "values": ["none", "individual", "group", "other", "unknown"],
                "description": "Target granularity (if any)",
            },
            "target_group": {
                "type": "multilabel",
                "description": "Target groups (e.g. women, black, jewish, muslim, lgbtq)",
            },
            "aux": {
                "type": "object",
                "description": "Dataset-specific raw labels and metadata",
            },
        },
        "datasets": {
            "hateful_memes": {
                "notes": "labels_raw in {0,1} mapped to abuse_hate/offensive; target_type='group' if 1 else 'none'",
            },
        },
    }

    taxonomy_path.parent.mkdir(parents=True, exist_ok=True)
    with taxonomy_path.open("w", encoding="utf-8") as f:
        json.dump(taxonomy, f, indent=2, ensure_ascii=False)

    return taxonomy


def _open_shard_writers(output_dir: Path, shard_size: int) -> Dict[str, wds.ShardWriter]:
    """Create a dict of ShardWriter objects per split (lazily)."""
    writers: Dict[str, wds.ShardWriter] = {}

    def get_writer(split: str) -> wds.ShardWriter:
        if split not in writers:
            split_dir = output_dir / split
            split_dir.mkdir(parents=True, exist_ok=True)
            pattern = str(split_dir / "shard-%06d.tar")
            writers[split] = wds.ShardWriter(pattern, maxcount=shard_size)
        return writers[split]

    # Attach the accessor to the dict for convenience
    writers["__get_writer__"] = get_writer  # type: ignore
    return writers


def _close_writers(writers: Dict[str, wds.ShardWriter]) -> None:
    for key, writer in list(writers.items()):
        if key == "__get_writer__":
            continue
        writer.close()


def run_packing(
    manifest_path: Path,
    ocr_path: Path,
    output_dir: Path,
    taxonomy_path: Path,
    splits_path: Path,
    examples_limit: Optional[int] = None,
    shard_size: int = 5000,
    image_size: int = 384,
    progress_interval: int = 1000,
    include_datasets: Optional[List[str]] = None,
) -> None:
    """Main packing routine."""
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    print("Ensuring label taxonomy exists...")
    _ensure_default_taxonomy(taxonomy_path)

    print("Loading OCR index (this may take a while for large files)...")
    ocr_index = _load_ocr_index(ocr_path)
    print(f"Loaded OCR for {len(ocr_index)} examples.")

    include_set = set(include_datasets) if include_datasets else None

    writers = _open_shard_writers(output_dir, shard_size)
    get_writer = writers["__get_writer__"]  # type: ignore

    # Stats
    total_seen = 0
    total_packed = 0
    total_failed = 0
    split_counts: Dict[str, int] = {}
    split_packed: Dict[str, int] = {}
    dataset_packed: Dict[str, int] = {}
    splits_map: Dict[str, str] = {}

    start_time = time.time()

    for record in _iter_manifest(manifest_path):
        total_seen += 1

        example_id = record.get("id")
        dataset = record.get("dataset", "unknown")
        split = record.get("split") or "train"

        if not example_id:
            total_failed += 1
            continue

        splits_map[example_id] = split
        split_counts[split] = split_counts.get(split, 0) + 1

        if include_set is not None and dataset not in include_set:
            # Skip but do not treat as failure; it's a user-chosen subset
            continue

        if examples_limit is not None and total_packed >= examples_limit:
            break

        try:
            image_path = record.get("image_path")
            text_raw = record.get("text_raw")
            labels_raw = record.get("labels_raw")

            # Image (optional)
            img_bytes: Optional[bytes] = None
            if image_path:
                img_path = Path(image_path)
                if img_path.exists():
                    with Image.open(img_path) as img:
                        img = img.convert("RGB")
                        img = img.resize((image_size, image_size), Image.BICUBIC)
                        buf = BytesIO()
                        img.save(buf, format="PNG")
                        img_bytes = buf.getvalue()
                else:
                    # Missing image file; treat as a failure for this example
                    raise FileNotFoundError(f"Image not found: {img_path}")

            # OCR lines (may be empty)
            ocr_lines = ocr_index.get(example_id, [])

            # Combined text
            combined_text = _build_text(ocr_lines, text_raw)

            # Labels
            labels = _map_labels(dataset, labels_raw)

            meta = {
                "id": example_id,
                "dataset": dataset,
                "split": split,
                "labels": labels,
            }

            sample: Dict[str, Any] = {"__key__": example_id}
            if img_bytes is not None:
                sample["png"] = img_bytes
            sample["txt"] = (combined_text or "").encode("utf-8")
            sample["json"] = json.dumps(meta, ensure_ascii=False).encode("utf-8")

            writer = get_writer(split)
            writer.write(sample)

            total_packed += 1
            split_packed[split] = split_packed.get(split, 0) + 1
            dataset_packed[dataset] = dataset_packed.get(dataset, 0) + 1

        except Exception as e:
            total_failed += 1
            print(f"[WARN] failed to pack id={example_id}: {e}", file=sys.stderr)

        if progress_interval and total_seen % progress_interval == 0:
            elapsed = time.time() - start_time
            rate = total_seen / elapsed if elapsed > 0 else 0.0
            print(
                f"Processed {total_seen} records, packed={total_packed}, "
                f"failed={total_failed} - {rate:.1f} rec/s"
            )

    _close_writers(writers)

    elapsed = time.time() - start_time
    failure_rate = float(total_failed) / float(total_seen) if total_seen > 0 else 0.0

    stats = {
        "total_manifest_records": total_seen,
        "packed_examples": total_packed,
        "failed_examples": total_failed,
        "failure_rate": failure_rate,
        "split_counts": split_counts,
        "split_packed": split_packed,
        "dataset_packed": dataset_packed,
        "elapsed_seconds": elapsed,
    }

    stats_path = STEP3 / "stats.json"
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    # Only create splits.json automatically if it does not exist yet
    if not splits_path.exists():
        with splits_path.open("w", encoding="utf-8") as f:
            json.dump(splits_map, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("STEP 3 PACKING SUMMARY")
    print("=" * 60)
    print(f"Total manifest records seen: {total_seen}")
    print(f"Packed examples: {total_packed}")
    print(f"Failed examples: {total_failed}")
    print(f"Failure rate: {failure_rate:.4f}")
    print(f"Elapsed time: {elapsed/60:.1f} minutes ({elapsed:.0f} seconds)")
    print(f"Stats written to: {stats_path}")
    if not splits_path.exists():
        print(f"Splits mapping written to: {splits_path}")
    else:
        print(f"Splits mapping already existed at: {splits_path}")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pack manifest + OCR into WebDataset shards for Step 3."
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=str(STEP2 / "data_manifest.jsonl"),
        help="Path to input manifest JSONL (from Step 2)",
    )
    parser.add_argument(
        "--ocr",
        type=str,
        default=str(STEP2 / "ocr.jsonl"),
        help="Path to OCR JSONL (from Step 2)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(STEP3 / "shards"),
        help="Directory to write WebDataset shards",
    )
    parser.add_argument(
        "--label-taxonomy",
        type=str,
        default=str(STEP3 / "label_taxonomy.json"),
        help="Path to label taxonomy JSON file (created if missing)",
    )
    parser.add_argument(
        "--splits",
        type=str,
        default=str(STEP3 / "splits.json"),
        help="Path to splits JSON file (created if missing)",
    )
    parser.add_argument(
        "--examples-limit",
        type=int,
        default=None,
        help="Limit number of packed examples (for pilot runs)",
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=5000,
        help="Maximum number of examples per shard",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=384,
        help="Output image size (square, pixels)",
    )
    parser.add_argument(
        "--progress",
        type=int,
        default=1000,
        help="Print progress every N manifest records",
    )
    parser.add_argument(
        "--include-datasets",
        type=str,
        nargs="*",
        default=None,
        help=(
            "Optional list of dataset names to include (e.g. hateful_memes mami). "
            "If omitted, all datasets are considered."
        ),
    )

    args = parser.parse_args()

    run_packing(
        manifest_path=Path(args.manifest),
        ocr_path=Path(args.ocr),
        output_dir=Path(args.output_dir),
        taxonomy_path=Path(args.label_taxonomy),
        splits_path=Path(args.splits),
        examples_limit=args.examples_limit,
        shard_size=args.shard_size,
        image_size=args.image_size,
        progress_interval=args.progress,
        include_datasets=args.include_datasets,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
