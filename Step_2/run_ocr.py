#!/usr/bin/env python
"""Run OCR over all images in Step_2/data_manifest.jsonl using PaddleOCR 3.x API.

PaddleOCR 3.x (Nov 2025) uses:
- ocr.predict(input=...) instead of ocr.ocr(...)
- Result objects with .json attribute containing rec_texts, rec_scores, rec_boxes

Output: Step_2/ocr.jsonl with fields:
    {"id", "dataset", "split", "image_path", "ocr": {"lines": [...], "stats": {...}}}

Optionally writes QC overlay images to Step_2/ocr_qc/<id>.png.
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
STEP2 = ROOT / "Step_2"
QC_DIR = STEP2 / "ocr_qc"


@dataclass
class OcrLine:
    text: str
    conf: float
    bbox: List[float]  # [x_min, y_min, x_max, y_max]


def _load_paddleocr():
    """Load PaddleOCR with the modern 3.x API settings."""
    try:
        from paddleocr import PaddleOCR
    except ImportError as e:
        raise RuntimeError(
            "PaddleOCR is not installed. Install with:\n"
            "  pip install paddleocr paddlepaddle pillow numpy\n"
        ) from e

    # PaddleOCR 3.x API - disable extra preprocessing for speed
    ocr = PaddleOCR(
        use_doc_orientation_classify=False,  # Skip document orientation classification
        use_doc_unwarping=False,             # Skip text image unwarping
        use_textline_orientation=False,      # Skip text line orientation classification
        lang="en",                           # English model
    )
    return ocr


def _iter_manifest(manifest_path: Path) -> Iterable[Dict[str, Any]]:
    """Iterate over records in the manifest JSONL file."""
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            yield json.loads(line)


def _count_manifest_images(manifest_path: Path) -> int:
    """Count records with image_path in the manifest."""
    count = 0
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("image_path"):
                count += 1
    return count


def _resize_image_if_needed(img: Image.Image, max_side: int = 1024) -> Image.Image:
    """Resize image so long side <= max_side, preserving aspect ratio."""
    w, h = img.size
    long_side = max(w, h)
    if long_side > max_side:
        scale = max_side / float(long_side)
        new_size = (int(w * scale), int(h * scale))
        img = img.resize(new_size, Image.LANCZOS)
    return img


def _run_ocr_on_image(ocr_engine, image_path: Path) -> List[OcrLine]:
    """Run OCR on a single image using PaddleOCR 3.x predict() API."""
    lines: List[OcrLine] = []

    try:
        # Load and optionally resize image
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            img = _resize_image_if_needed(img, max_side=1024)
            img_array = np.array(img)

        # PaddleOCR 3.x API: use predict() method
        results = ocr_engine.predict(input=img_array)

        if not results:
            return lines

        # results is a list of Result objects (one per input image)
        for res in results:
            # Get the JSON representation of results
            # Structure: res.json = {'res': {'rec_texts': [...], 'rec_scores': [...], ...}}
            res_json = res.json if hasattr(res, 'json') else {}
            res_dict = res_json.get("res", {}) if isinstance(res_json, dict) else {}

            # Extract recognized texts, scores, and boxes
            rec_texts = res_dict.get("rec_texts", [])
            rec_scores = res_dict.get("rec_scores", [])
            rec_boxes = res_dict.get("rec_boxes", [])

            # Build OcrLine objects
            for i, text in enumerate(rec_texts):
                conf = rec_scores[i] if i < len(rec_scores) else 0.0
                # rec_boxes is list of [x_min, y_min, x_max, y_max]
                if i < len(rec_boxes):
                    box = rec_boxes[i]
                    if isinstance(box, np.ndarray):
                        box = box.tolist()
                    bbox = [float(x) for x in box[:4]] if len(box) >= 4 else [0.0, 0.0, 0.0, 0.0]
                else:
                    bbox = [0.0, 0.0, 0.0, 0.0]

                lines.append(OcrLine(text=str(text), conf=float(conf), bbox=bbox))

    except Exception as e:
        # Log error but don't crash the whole pipeline
        print(f"  [WARN] OCR failed for {image_path}: {e}", file=sys.stderr)
        return []

    return lines


def _draw_qc(image_path: Path, lines: List[OcrLine], qc_path: Path) -> None:
    """Draw OCR boxes and text on image for QC visualization."""
    try:
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            draw = ImageDraw.Draw(img)
            for ln in lines:
                x1, y1, x2, y2 = ln.bbox
                draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
                label = f"{ln.text[:30]}... ({ln.conf:.2f})" if len(ln.text) > 30 else f"{ln.text} ({ln.conf:.2f})"
                draw.text((x1, max(0, y1 - 12)), label, fill="red")
            qc_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(qc_path)
    except Exception:
        pass  # QC failures should not crash main pipeline


def run_ocr(
    manifest_path: Path,
    output_path: Path,
    qc: bool = True,
    limit: Optional[int] = None,
    progress_interval: int = 100,
) -> None:
    """Run OCR on all images in the manifest and write results to output_path."""
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    QC_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading PaddleOCR model...")
    ocr_engine = _load_paddleocr()
    print("PaddleOCR model loaded.")

    # Count total for progress
    print("Counting images in manifest...")
    total_images = _count_manifest_images(manifest_path)
    if limit:
        total_images = min(total_images, limit)
    print(f"Will process up to {total_images} images.")

    processed = 0
    no_text = 0
    conf_sum = 0.0
    conf_count = 0
    start_time = time.time()

    with output_path.open("w", encoding="utf-8") as out_f:
        for record in _iter_manifest(manifest_path):
            image_path = record.get("image_path")
            if not image_path:
                continue

            img_path = Path(image_path)
            if not img_path.exists():
                continue

            processed += 1
            if limit and processed > limit:
                break

            # Run OCR
            lines = _run_ocr_on_image(ocr_engine, img_path)

            if not lines:
                no_text += 1

            for ln in lines:
                conf_sum += ln.conf
                conf_count += 1

            # Write result
            ocr_payload = {
                "id": record.get("id"),
                "dataset": record.get("dataset"),
                "split": record.get("split"),
                "image_path": image_path,
                "ocr": {
                    "lines": [asdict(ln) for ln in lines]
                },
            }
            out_f.write(json.dumps(ocr_payload, ensure_ascii=False) + "\n")

            # QC image
            if qc and lines:
                qc_path = QC_DIR / f"{record.get('id')}.png"
                _draw_qc(img_path, lines, qc_path)

            # Progress reporting
            if processed % progress_interval == 0:
                elapsed = time.time() - start_time
                rate = processed / elapsed if elapsed > 0 else 0
                eta = (total_images - processed) / rate if rate > 0 else 0
                print(f"  Processed {processed}/{total_images} images "
                      f"({processed/total_images*100:.1f}%) - "
                      f"{rate:.1f} img/s - ETA: {eta/60:.1f} min")

    # Final summary
    elapsed = time.time() - start_time
    failure_rate = float(no_text) / float(processed) if processed > 0 else 0.0
    avg_conf = float(conf_sum) / float(conf_count) if conf_count > 0 else 0.0

    print("\n" + "=" * 60)
    print("OCR SUMMARY")
    print("=" * 60)
    print(f"Total images processed: {processed}")
    print(f"Images with no OCR text: {no_text} (failure rate: {failure_rate:.3f})")
    print(f"Total text lines detected: {conf_count}")
    print(f"Mean OCR confidence: {avg_conf:.3f}")
    print(f"Total time: {elapsed/60:.1f} minutes ({elapsed:.0f} seconds)")
    print(f"Output written to: {output_path}")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run OCR on images in Step_2/data_manifest.jsonl (PaddleOCR 3.x)"
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=str(STEP2 / "data_manifest.jsonl"),
        help="Path to input manifest JSONL",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(STEP2 / "ocr.jsonl"),
        help="Path to output OCR JSONL",
    )
    parser.add_argument(
        "--no-qc",
        action="store_true",
        help="Disable saving QC overlay images",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of images to process (for testing)",
    )
    parser.add_argument(
        "--progress",
        type=int,
        default=100,
        help="Print progress every N images",
    )
    args = parser.parse_args()

    run_ocr(
        Path(args.manifest),
        Path(args.output),
        qc=not args.no_qc,
        limit=args.limit,
        progress_interval=args.progress,
    )


if __name__ == "__main__":
    main()
