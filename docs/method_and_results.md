# Drift-Sense – Technical Method & Results

## 1. Problem

Given a high-resolution Reference image and a lower-resolution Search image (approximately 10× larger physical field of view), recover the centre coordinates `(x, y)` of the Reference pattern inside the Search image.

Semiconductor layouts (DRAM and FinFET) are highly periodic. Many locations look almost identical, so ordinary single-peak template matching frequently fails.

---

## 2. Solution Overview

```
Reference
   ↓
Exact 10× down-scaling → template          (from problem statement)
   ↓
Normalized Cross-Correlation on Search     (standard CV – Lewis / OpenCV)
   ↓
Top-20 candidate peaks                     (because patterns are periodic)
   ↓
Siamese CNN ranking                        (Bromley / Chopra style)
   ↓
Confidence decision rule                   (our experimental choice)
   ↓
Final centre (x, y)
```

**Decision rule (simple words):**  
If classical score > 0.85 → trust classical.  
Otherwise → use the candidate preferred by the Siamese network.

---

## 3. Math We Used (Simple Words + Source)

### 3.1 Scale normalization
Template = Resize(Reference, 1/10)  
Source: problem statement (≈10× magnification difference).

### 3.2 Normalized Cross-Correlation (NCC)
Measures how well bright/dark patterns match after removing average brightness.

\[
\text{NCC}(u,v) = \frac{\sum (T-\bar{T})(S_{uv}-\bar{S}_{uv})}
{\sqrt{\sum (T-\bar{T})^2 \cdot \sum (S_{uv}-\bar{S}_{uv})^2}}
\]

Source: Lewis (1995) “Fast Normalized Cross-Correlation”; Gonzalez & Woods; OpenCV `TM_CCOEFF_NORMED`.

### 3.3 Top-20 peaks
Keep the 20 strongest local maxima (with non-maximum suppression).  
Reason: on periodic layouts the true location is often not rank-1 but is almost always inside Top-20.

### 3.4 Siamese similarity
Shared encoder \(f\) for Reference patch and Candidate patch (96×96).  
Score from concatenated features and absolute difference.

Source: Bromley et al. (1993/94) Siamese networks; Chopra et al. (2005) similarity learning.

### 3.5 Ranking loss
\[
\mathcal{L} = \max(0,\; m - (s_{\text{positive}} - s_{\text{negative}}))
\]

Forces the true candidate to score higher than hard negatives by a margin.  
Source: standard margin ranking / metric-learning losses (similar spirit to FaceNet-style ranking).

Full details: see `docs/Mathematical_Formulation.md`.

---

## 4. Synthetic Data Generator

- Styles: DRAM and FinFET  
- Exact 10× scale, independent SEM-like noise, rotation, blur, edge bloom  
- Difficulty: easy / medium / hard / trap  
- Trap mode deliberately makes classical Top-1 wrong while GT stays in Top-20  
- Every pair stores ground-truth centre

---

## 5. Classical Peak Detector

1. Resize Reference by exactly 10×  
2. NCC against Search  
3. Top-20 peaks + non-maximum suppression  
4. Return centres and scores  

Runtime: typically 15–25 ms per 1000×1000 pair.

---

## 6. Siamese Ranking CNN

- Shared encoder, 96×96 patches  
- Ranking / margin loss on hard negatives (classical peaks + near-GT shifts)  
- Used only when classical confidence is low  

---

## 7. Results

| Difficulty | Accuracy |
|------------|----------|
| Easy | 100% |
| Medium | 100% |
| Hard | 100% |
| Trap (hard periodic) | ~70–75% |
| **Overall** | **≈ 93–94%** |
| Mean error | **< 1.0 pixel** |

Runtime: classical 15–25 ms; full hybrid typically < 70 ms.

---

## 8. Design Rationale

| Choice | Reason | Source of idea |
|--------|--------|----------------|
| Exact 10× scale | Matches problem statement | Problem statement |
| NCC | Robust, standard template matching | Lewis / OpenCV |
| Top-20 peaks | True location rarely lost on periodic layouts | Multi-hypothesis matching practice |
| Siamese ranking | Compare candidate to **this** Reference | Bromley / Chopra |
| Ranking loss | Make true match rank first | Metric-learning literature |
| Confidence rule | Protect easy cases | Our experiments |

---

## 9. Limitations

- Extremely strong periodic distractors can still cause residual errors  
- No real SEM images were available for domain testing  
- Further gains would need real-wafer fine-tuning or richer features  

---

## 10. Reproducibility

- Source: `src/`  
- Weights: `models/siamese_final_96.pt`  
- Runner: `python scripts/run_full_solution.py`  
- Inference: `python scripts/inference.py <ref> <search>`  
- Math details: `docs/Mathematical_Formulation.md`  

---

## 11. One-Sentence Summary

We solve navigation-error recovery with a scale-aware hybrid method: classical multi-peak template matching (standard NCC) generates candidates, a Siamese network (classic similarity architecture) ranks them on hard cases, and a confidence rule selects the final centre with high accuracy and full transparency.
