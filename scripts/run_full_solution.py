#!/usr/bin/env python3
"""
Drift-Sense – Full Solution Runner
==================================
Single entry point for anyone to run and understand the complete system.

Produces:
  1. Generated Reference & Search images (10× scale)
  2. Classical Top-20 candidates visualization
  3. Siamese ranking scores
  4. Decision (Classical vs Siamese)
  5. Final localization result
  6. Metrics summary
"""

from __future__ import annotations
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_generator import SyntheticWaferGenerator, GeneratorConfig
from src.classical import ClassicalPeakDetector, DetectorConfig
from src.ranking_cnn.siamese_model import SiameseRanker

PS = 96
CLASSICAL_CONF = 0.85          # if classical score > this → trust classical


def crop(img, cx, cy, size):
    h, w = img.shape
    half = size // 2
    x1 = max(0, min(int(round(cx)) - half, w - size))
    y1 = max(0, min(int(round(cy)) - half, h - size))
    patch = img[y1:y1+size, x1:x1+size]
    if patch.shape != (size, size):
        patch = cv2.resize(patch, (size, size))
    return patch


def run_case(case_id, style, difficulty, out, gen, detector, model, device):
    print(f"\n{'='*65}")
    print(f"CASE {case_id:02d} | {style.upper()} | {difficulty.upper()}")
    print(f"{'='*65}")

    # ---------- STAGE 1: Generation ----------
    pair = gen.generate_pair(style=style, difficulty=difficulty)
    ref, search, gt = pair["reference"], pair["search"], pair["gt_center"]

    print("\n[1] DATA GENERATION")
    print(f"    Reference & Search : 1000×1000 | Scale relation : 10×")
    print(f"    Ground Truth       : ({gt[0]:.1f}, {gt[1]:.1f})")

    Image.fromarray(ref).save(out / "stage1_reference" / f"{case_id:02d}_ref.png")
    Image.fromarray(search).save(out / "stage2_search" / f"{case_id:02d}_search.png")

    # ---------- STAGE 2: Classical Top-20 ----------
    t0 = time.perf_counter()
    peaks = detector.detect(ref, search)
    t_classical = (time.perf_counter() - t0) * 1000

    print("\n[2] CLASSICAL MATCHING (Top-20)")
    print(f"    Candidates         : {len(peaks)}")
    print(f"    Classical #1       : ({peaks[0].x:.1f}, {peaks[0].y:.1f})  score={peaks[0].score:.3f}")
    print(f"    Runtime            : {t_classical:.1f} ms")

    vis20 = cv2.cvtColor(search, cv2.COLOR_GRAY2BGR)
    for i, p in enumerate(peaks):
        color = (0, 0, 255) if i == 0 else (0, 255, 255)
        cv2.circle(vis20, (int(p.x), int(p.y)), 11, color, 2)
        cv2.putText(vis20, str(i+1), (int(p.x)+12, int(p.y)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    cv2.circle(vis20, (int(gt[0]), int(gt[1])), 16, (0, 255, 0), 2)
    cv2.imwrite(str(out / "stage3_classical_top20" / f"{case_id:02d}_top20.png"), vis20)

    # ---------- STAGE 3: Siamese Ranking ----------
    print("\n[3] SIAMESE CNN RANKING")
    tmpl = cv2.resize(ref, (PS, PS)).astype(np.float32) / 255
    ref_t = torch.from_numpy(tmpl[None, None]).to(device)

    scores = []
    for p in peaks:
        cand = crop(search, p.x, p.y, PS).astype(np.float32) / 255
        cand_t = torch.from_numpy(cand[None, None]).to(device)
        with torch.no_grad():
            s = float(model.score(ref_t, cand_t).cpu())
        scores.append(s)

    best_idx = int(np.argmax(scores))
    print(f"    Top-5 Siamese scores: {[f'{s:.3f}' for s in sorted(scores, reverse=True)[:5]]}")
    print(f"    Best CNN rank       : {best_idx+1}  (score={scores[best_idx]:.3f})")

    # save ranking text
    with open(out / "stage4_siamese_ranking" / f"{case_id:02d}_scores.txt", "w") as f:
        f.write(f"Case {case_id} | {style} | {difficulty}\n")
        f.write(f"GT: ({gt[0]:.1f}, {gt[1]:.1f})\n\n")
        for i, (p, s) in enumerate(zip(peaks, scores)):
            f.write(f"Rank {i+1:2d}  pos=({p.x:.1f},{p.y:.1f})  classical={p.score:.3f}  siamese={s:.3f}\n")

    # ---------- STAGE 4: Decision ----------
    if peaks[0].score > CLASSICAL_CONF:
        final = peaks[0]
        method = "classical"
        print(f"\n[4] DECISION → CLASSICAL (score {peaks[0].score:.3f} > {CLASSICAL_CONF})")
    else:
        final = peaks[best_idx]
        method = "siamese"
        print(f"\n[4] DECISION → SIAMESE (classical score {peaks[0].score:.3f} ≤ {CLASSICAL_CONF})")

    # ---------- STAGE 5: Final Result ----------
    err = np.hypot(final.x - gt[0], final.y - gt[1])
    ok = err <= 5.0
    print("\n[5] FINAL LOCALIZATION")
    print(f"    Prediction  : ({final.x:.1f}, {final.y:.1f})")
    print(f"    Ground Truth: ({gt[0]:.1f}, {gt[1]:.1f})")
    print(f"    Error       : {err:.2f} px")
    print(f"    Method      : {method}")
    print(f"    Result      : {'CORRECT' if ok else 'WRONG'}")

    vis = cv2.cvtColor(search, cv2.COLOR_GRAY2BGR)
    cv2.circle(vis, (int(gt[0]), int(gt[1])), 16, (0, 255, 0), 2)
    cv2.drawMarker(vis, (int(final.x), int(final.y)), (0, 0, 255),
                   markerType=cv2.MARKER_CROSS, markerSize=22, thickness=2)
    cv2.imwrite(str(out / "stage5_final_results" / f"{case_id:02d}_final.png"), vis)

    return {"error": err, "correct": ok, "method": method, "difficulty": difficulty}


def main():
    print("\n" + "="*65)
    print("  DRIFT-SENSE  |  Full Solution Runner")
    print("  Transparent pipeline – every stage is saved and shown")
    print("="*65)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = ROOT / "models" / "siamese_final_96.pt"
    if not model_path.exists():
        model_path = ROOT / "outputs" / "models" / "siamese_final_96.pt"

    model = SiameseRanker().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True), strict=False)
    model.eval()
    print(f"Model loaded: {model_path.name}  |  Device: {device}")

    detector = ClassicalPeakDetector(DetectorConfig(top_k=20))
    gen = SyntheticWaferGenerator(GeneratorConfig(seed=42))

    out = ROOT / "outputs"
    for sub in ["stage1_reference", "stage2_search", "stage3_classical_top20",
                "stage4_siamese_ranking", "stage5_final_results"]:
        (out / sub).mkdir(parents=True, exist_ok=True)

    # Generate a good mix of cases
    cases = []
    for diff in ["easy", "medium", "hard", "trap"]:
        for style in ["dram", "finfet"]:
            for _ in range(2):
                cases.append((style, diff))

    results = []
    for i, (style, diff) in enumerate(cases):
        results.append(run_case(i, style, diff, out, gen, detector, model, device))

    # ---------- STAGE 6: Metrics ----------
    correct = sum(1 for r in results if r["correct"])
    mean_err = np.mean([r["error"] for r in results])
    classical = sum(1 for r in results if r["method"] == "classical")
    siamese = sum(1 for r in results if r["method"] == "siamese")

    summary = f"""
=================================================================
DRIFT-SENSE  FINAL METRICS SUMMARY
=================================================================
Total cases          : {len(results)}
Correct              : {correct}/{len(results)}  ({100*correct/len(results):.1f}%)
Mean error           : {mean_err:.2f} px
Classical decisions  : {classical}
Siamese decisions    : {siamese}
-----------------------------------------------------------------
Per difficulty:
"""
    for d in ["easy", "medium", "hard", "trap"]:
        sub = [r for r in results if r["difficulty"] == d]
        if sub:
            c = sum(1 for r in sub if r["correct"])
            summary += f"  {d:8s} : {c}/{len(sub)} correct\n"
    summary += "=================================================================\n"
    summary += "Green circle = Ground Truth | Red cross = Final prediction\n"
    summary += "All stage images are saved inside the outputs/ folder.\n"

    print(summary)
    with open(out / "metrics_summary.txt", "w") as f:
        f.write(summary)
    print(f"Metrics written to: {out / 'metrics_summary.txt'}")


if __name__ == "__main__":
    main()