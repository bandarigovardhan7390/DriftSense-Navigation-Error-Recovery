# Drift-Sense: Navigation-Error Recovery

**Applied Materials / SEMICON Hackathon Solution**

Scale-aware hybrid localization for semiconductor wafer navigation-error recovery.

---

## Quick Start (for any user)

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the complete transparent solution:
   ```bash
   python scripts/run_full_solution.py
   ```

3. Open the `outputs/` folder. You will find:
   - `stage1_reference/` – generated high-resolution Reference images
   - `stage2_search/` – generated wide-FOV Search images
   - `stage3_classical_top20/` – Top-20 classical candidates marked
   - `stage4_siamese_ranking/` – Siamese scores for every candidate
   - `stage5_final_results/` – final localization (Green = GT, Red = Prediction)
   - `metrics_summary.txt` – accuracy and method statistics

---

## What the System Does

1. **Data Generation**  
   Creates realistic DRAM / FinFET image pairs with exact 10× scale relationship and known ground-truth centre.

2. **Classical Matching**  
   Down-scales the Reference by 10× and runs normalized cross-correlation to produce the Top-20 most similar locations.

3. **Siamese Ranking**  
   A lightweight Siamese CNN compares the Reference against each of the 20 candidates and produces a similarity score.

4. **Decision Rule**  
   - If classical confidence is high (> 0.85) → use classical result  
   - Otherwise → use the best Siamese-ranked candidate  

5. **Output**  
   Final centre coordinates `(x, y)` of the recovered location.

---

## Results (locked evaluation)

| Difficulty | Accuracy |
|------------|----------|
| Easy       | 100%     |
| Medium     | 100%     |
| Hard       | 100%     |
| Trap (hard periodic) | ~70–75% |
| **Overall** | **~93–94%** |
| Mean error | **< 1 px** |

---

## Folder Structure

```
DriftSense_Submission/
├── README.md
├── requirements.txt
├── models/
│   └── siamese_final_96.pt          # final trained weights
├── src/                             # source code
│   ├── data_generator/
│   ├── classical/
│   ├── ranking_cnn/
│   └── pipeline/
├── scripts/
│   └── run_full_solution.py         # main entry point
├── training/                        # how the model was trained
├── docs/
│   └── method_and_results.md
└── outputs/                         # created when you run the script
```

---

## Notes for Judges / Reviewers

- Green circle = Ground Truth location  
- Red cross = System prediction  
- The pipeline is fully transparent: every intermediate stage is saved as an image.  
- Classical matching is the primary method on easy/medium data.  
- Siamese CNN is activated only when classical confidence is lower (hard periodic cases).  
- Training code and description are provided in the `training/` folder for reproducibility.

---

## Contact / Team

| Field | Details |
|-------|---------|
| **Team Name** | Quantized Minds |
| **Member 1** | Bandari Govardhan – 24215a0409 |
| **Member 2** | Pakala Karthik – 24215a0411 |
| **Member 3** | Gandham Karunakar – 25215a0412 |
| **College** | BV Raju Institute of Technology |
| **Contact** | 6305557390 / 24215A0409@gmail.com |
| **Guide** | Dr. U. Gnaneshwara Chary |
| **Project** | Drift-Sense: Navigation-Error Recovery |
| **Hackathon** | SEMICON / Applied Materials |
