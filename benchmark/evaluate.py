"""
VideoSTE-Bench Evaluation Runner
==================================
Runs all 4 evaluation metrics on a set of edited videos and produces
a comprehensive evaluation report.

Usage:
  python benchmark/evaluate.py \
      --results_dir outputs/textvace_v3_inference/train_samples \
      --records data/processed/parsed_records.json \
      --raw_dir data/raw \
      --output benchmark/results/v3_train.json

  python benchmark/evaluate.py \
      --results_dir outputs/textvace_inference/unseen_final \
      --records data/inference_processed/inference_records.json \
      --raw_dir "" \
      --output benchmark/results/v2_unseen.json \
      --record_format inference
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from benchmark.metrics.text_accuracy import evaluate_text_accuracy
from benchmark.metrics.background_preservation import evaluate_background_preservation
from benchmark.metrics.temporal_consistency import evaluate_temporal_consistency
from benchmark.metrics.vlm_quality import evaluate_vlm_quality
from benchmark.metrics.gt_similarity import evaluate_gt_similarity


def find_edited_video(results_dir, vid_id):
    """Find the edited video file for a given sample ID."""
    candidates = [
        os.path.join(results_dir, f"{vid_id}_generated.mp4"),
        os.path.join(results_dir, f"{vid_id}.mp4"),
        os.path.join(results_dir, f"{vid_id}_anchored.mp4"),
        os.path.join(results_dir, "videos", f"{vid_id}_generated.mp4"),
        os.path.join(results_dir, "videos", f"{vid_id}.mp4"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def load_records(records_path, record_format, raw_dir):
    """Load and normalize records to a common format."""
    with open(records_path) as f:
        records = json.load(f)

    normalized = []
    for rec in records:
        if record_format == "train":
            gt_video = os.path.join(raw_dir, rec.get("edited_video", ""))
            normalized.append({
                "id": rec["id"],
                "original_video": os.path.join(raw_dir, rec["original_video"]),
                "mask_video": os.path.join(raw_dir, rec["mask_video"]),
                "gt_video": gt_video if os.path.exists(gt_video) else None,
                "source_text": rec["source_text"],
                "target_text": rec["target_text"],
            })
        elif record_format == "inference":
            normalized.append({
                "id": rec["id"],
                "original_video": rec["video_path"],
                "mask_video": rec["mask_path"],
                "gt_video": None,
                "source_text": rec.get("original_text", ""),
                "target_text": rec.get("replacement_text", ""),
            })
    return normalized


def main():
    parser = argparse.ArgumentParser(description="VideoSTE-Bench Evaluation")
    parser.add_argument("--results_dir", required=True,
                        help="Directory containing edited videos (*_generated.mp4)")
    parser.add_argument("--records", required=True,
                        help="Path to records JSON file")
    parser.add_argument("--raw_dir", default="data/raw",
                        help="Base directory for original/mask videos")
    parser.add_argument("--record_format", choices=["train", "inference"], default="train",
                        help="Format of the records file")
    parser.add_argument("--output", default="benchmark/results/eval_results.json",
                        help="Output path for results JSON")
    parser.add_argument("--skip_vlm", action="store_true",
                        help="Skip VLM evaluation (faster)")
    parser.add_argument("--skip_clip", action="store_true",
                        help="Skip CLIP frame consistency (faster)")
    parser.add_argument("--vlm_model", default="qwen3-vl:32b-instruct",
                        help="Ollama model for VLM evaluation")
    parser.add_argument("--ocr_interval", type=int, default=6,
                        help="Run OCR every N frames")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Load records
    records = load_records(args.records, args.record_format, args.raw_dir)
    print(f"Loaded {len(records)} records from {args.records}")

    # Initialize models (reuse across samples)
    print("Initializing EasyOCR...")
    import easyocr
    ocr_engine = easyocr.Reader(["en", "ch_sim"], gpu=True, verbose=False)

    clip_model, clip_processor = None, None
    if not args.skip_clip:
        print("Loading CLIP model...")
        from transformers import CLIPModel, CLIPProcessor
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to("cuda").eval()
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # Evaluate each sample
    all_results = []
    for i, rec in enumerate(records):
        vid_id = rec["id"]
        edited_path = find_edited_video(args.results_dir, vid_id)

        if edited_path is None:
            print(f"[{i+1}/{len(records)}] SKIP {vid_id}: no edited video found")
            continue

        if not os.path.exists(rec["original_video"]) or not os.path.exists(rec["mask_video"]):
            print(f"[{i+1}/{len(records)}] SKIP {vid_id}: missing original/mask video")
            continue

        print(f"[{i+1}/{len(records)}] Evaluating {vid_id}: "
              f"\"{rec['source_text']}\" -> \"{rec['target_text']}\"")

        sample_result = {"id": vid_id, "source_text": rec["source_text"],
                         "target_text": rec["target_text"]}

        # --- Metric 1: Text Accuracy ---
        t0 = time.time()
        text_acc = evaluate_text_accuracy(
            edited_path, rec["mask_video"], rec["target_text"],
            ocr_engine=ocr_engine, sample_interval=args.ocr_interval,
        )
        sample_result["text_accuracy"] = text_acc
        print(f"  Text Accuracy: word={text_acc['word_accuracy']:.3f}, "
              f"char={text_acc['char_accuracy']:.3f}, "
              f"conf={text_acc['ocr_confidence']:.3f} ({time.time()-t0:.1f}s)")

        # --- Metric 2: Background Preservation ---
        t0 = time.time()
        bg_pres = evaluate_background_preservation(
            rec["original_video"], edited_path, rec["mask_video"],
        )
        sample_result["background_preservation"] = bg_pres
        print(f"  Background: PSNR={bg_pres['bg_psnr']:.2f}, "
              f"SSIM={bg_pres['bg_ssim']:.4f} ({time.time()-t0:.1f}s)")

        # --- Metric 3: Temporal Consistency ---
        t0 = time.time()
        temp_con = evaluate_temporal_consistency(
            edited_path, rec["mask_video"],
            use_clip=not args.skip_clip,
            clip_model=clip_model, clip_processor=clip_processor,
        )
        sample_result["temporal_consistency"] = temp_con
        clip_str = f", CLIP={temp_con.get('clip_frame_consistency', 0):.4f}" if not args.skip_clip else ""
        print(f"  Temporal: SSIM={temp_con['text_temporal_ssim']:.4f}"
              f"{clip_str} ({time.time()-t0:.1f}s)")

        # --- Metric 4: VLM Quality ---
        if not args.skip_vlm:
            t0 = time.time()
            vlm_score = evaluate_vlm_quality(
                edited_path, rec["target_text"],
                model=args.vlm_model,
            )
            sample_result["vlm_quality"] = vlm_score
            seen = vlm_score['seen_texts'][0] if vlm_score['seen_texts'] else '?'
            print(f"  VLM Quality: score={vlm_score['vlm_score']:.1f}/10, "
                  f"seen=\"{seen}\" ({time.time()-t0:.1f}s)")

        # --- Metric 5: GT Similarity (only if GT exists) ---
        if rec.get("gt_video") is not None:
            t0 = time.time()
            gt_sim = evaluate_gt_similarity(
                edited_path, rec["gt_video"], rec["mask_video"],
            )
            sample_result["gt_similarity"] = gt_sim
            print(f"  GT Similarity: PSNR={gt_sim['gt_psnr']:.2f}, "
                  f"SSIM={gt_sim['gt_ssim']:.4f}, "
                  f"Text-PSNR={gt_sim['gt_text_psnr']:.2f}, "
                  f"Text-SSIM={gt_sim['gt_text_ssim']:.4f} ({time.time()-t0:.1f}s)")

        all_results.append(sample_result)

    # --- Aggregate results ---
    print(f"\n{'='*60}")
    print(f"AGGREGATE RESULTS ({len(all_results)} samples)")
    print(f"{'='*60}")

    if len(all_results) == 0:
        print("No samples evaluated!")
        return

    summary = {}

    # Text Accuracy
    summary["text_accuracy"] = {
        "word_accuracy": float(np.mean([r["text_accuracy"]["word_accuracy"] for r in all_results])),
        "char_accuracy": float(np.mean([r["text_accuracy"]["char_accuracy"] for r in all_results])),
        "ocr_confidence": float(np.mean([r["text_accuracy"]["ocr_confidence"] for r in all_results])),
        "detection_rate": float(np.mean([r["text_accuracy"]["detection_rate"] for r in all_results])),
    }
    print(f"\n[Text Accuracy]")
    print(f"  Word Accuracy:   {summary['text_accuracy']['word_accuracy']:.4f}")
    print(f"  Char Accuracy:   {summary['text_accuracy']['char_accuracy']:.4f}")
    print(f"  OCR Confidence:  {summary['text_accuracy']['ocr_confidence']:.4f}")
    print(f"  Detection Rate:  {summary['text_accuracy']['detection_rate']:.4f}")

    # Background Preservation
    summary["background_preservation"] = {
        "bg_psnr": float(np.mean([r["background_preservation"]["bg_psnr"] for r in all_results])),
        "bg_ssim": float(np.mean([r["background_preservation"]["bg_ssim"] for r in all_results])),
    }
    print(f"\n[Background Preservation]")
    print(f"  BG-PSNR: {summary['background_preservation']['bg_psnr']:.2f}")
    print(f"  BG-SSIM: {summary['background_preservation']['bg_ssim']:.4f}")

    # Temporal Consistency
    tc_keys = ["text_temporal_ssim"]
    if not args.skip_clip:
        tc_keys.append("clip_frame_consistency")
    summary["temporal_consistency"] = {}
    for k in tc_keys:
        vals = [r["temporal_consistency"].get(k, 0) for r in all_results]
        summary["temporal_consistency"][k] = float(np.mean(vals))
    print(f"\n[Temporal Consistency]")
    print(f"  Text Temporal SSIM:     {summary['temporal_consistency']['text_temporal_ssim']:.4f}")
    if not args.skip_clip:
        print(f"  CLIP Frame Consistency: {summary['temporal_consistency']['clip_frame_consistency']:.4f}")

    # VLM Quality
    if not args.skip_vlm:
        vlm_results = [r for r in all_results if "vlm_quality" in r and r["vlm_quality"]["vlm_score"] > 0]
        if vlm_results:
            summary["vlm_quality"] = {
                "vlm_score": float(np.mean([r["vlm_quality"]["vlm_score"] for r in vlm_results])),
            }
            print(f"\n[VLM Quality Score]")
            print(f"  VLM Score:  {summary['vlm_quality']['vlm_score']:.2f}/10")

    # GT Similarity (only for paired samples)
    gt_results = [r for r in all_results if "gt_similarity" in r]
    if gt_results:
        summary["gt_similarity"] = {
            "gt_psnr": float(np.mean([r["gt_similarity"]["gt_psnr"] for r in gt_results])),
            "gt_ssim": float(np.mean([r["gt_similarity"]["gt_ssim"] for r in gt_results])),
            "gt_text_psnr": float(np.mean([r["gt_similarity"]["gt_text_psnr"] for r in gt_results])),
            "gt_text_ssim": float(np.mean([r["gt_similarity"]["gt_text_ssim"] for r in gt_results])),
        }
        print(f"\n[GT Similarity] ({len(gt_results)} paired samples)")
        print(f"  GT-PSNR:      {summary['gt_similarity']['gt_psnr']:.2f}")
        print(f"  GT-SSIM:      {summary['gt_similarity']['gt_ssim']:.4f}")
        print(f"  GT-Text-PSNR: {summary['gt_similarity']['gt_text_psnr']:.2f}")
        print(f"  GT-Text-SSIM: {summary['gt_similarity']['gt_text_ssim']:.4f}")

    # Save results
    output_data = {
        "summary": summary,
        "n_samples": len(all_results),
        "per_sample": all_results,
    }
    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
