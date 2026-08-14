# Drift-Sense Submission Checklist

## Official Requirements vs Our Status

| # | Required Item | Status | Location / Action |
|---|---------------|--------|-------------------|
| 1 | README.md with complete setup instructions | Ready | `README.md` |
| 2 | Dataset Generator (standalone .py) | Need to expose cleanly | Use `src/data_generator` + small wrapper |
| 3 | **Localization Inference Script** (takes ref + search paths → outputs x,y) | **Created** | `scripts/inference.py` |
| 4 | DL Model Weights (.pt) | Ready | `models/siamese_final_96.pt` |
| 5 | Training Script / Notes | Ready | `training/` |
| 6 | requirements.txt | Ready | `requirements.txt` |
| 7 | Citation / Supporting References | Need short list | Add to `docs/` |
| 8 | Process / Method Document | **Created** | `docs/Complete_Process_Document.md` |
| 9 | Transparent full runner (optional but useful) | Ready | `scripts/run_full_solution.py` |
| 10 | Sample outputs / visuals | Generate by running the scripts | `outputs/` |

## Critical File for Scoring

```bash
python scripts/inference.py <reference.png> <search.png>
```
Must print only: `x,y`

Test this on a fresh environment before final upload.

## PPT / PDF (i4C Template) – Content Mapping

| Slide | Content we already have |
|-------|-------------------------|
| 1 Team Details | Fill manually |
| 2 Problem Statement | Use Complete_Process_Document §1 |
| 3 Idea Description | Hybrid classical + Siamese, 10× scale handling |
| 4 Proposed Solution | Pipeline diagram + generator + algorithm |
| 5 Innovation | Multi-peak + Siamese ranking + confidence rule |
| 6 Results | 93–94% overall, visuals from outputs/ |
| 7 Technology | Python, PyTorch, OpenCV, RTX 3050 |
| 8 GitHub & Video | Your repo link + optional demo video |
| 9 References | SEM noise, template matching, Siamese networks |

## About the outputs/ folder

The `outputs/` folder is **generated when you run** `run_full_solution.py`.  
You do **not** need to pre-copy old outputs.  
Judges / reviewers will run the script themselves and obtain fresh stage images.

Pre-generated sample images are optional evidence only.
