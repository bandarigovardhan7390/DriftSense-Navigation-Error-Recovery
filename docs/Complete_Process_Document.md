# Drift-Sense: Complete Process Document
## From Data Generation to Final Localization – Full Explanation

**Project:** Navigation-Error Recovery for Semiconductor Wafer Inspection  
**Hackathon:** SEMICON / Applied Materials  
**Document purpose:** Explain every step of our solution in clear, simple English so that any reviewer, guide, or new team member can understand what we built and why.

---

## 1. What Problem Are We Solving?

### The real-world situation

In a semiconductor factory, inspection tools take pictures of wafers at different magnifications.

- First, the tool captures a **Reference image**.  
  This is a high-magnification, high-detail picture of a small area (approximately 1 µm × 1 µm).  
  It shows the exact pattern the engineers care about.

- Later, because of mechanical drift or navigation error, the stage may not land in exactly the same place.  
  The tool then captures a **Search image**.  
  This is a wider view (approximately 10 µm × 10 µm) at lower magnification.

Both images are usually 1000 × 1000 pixels, but they represent different physical sizes.  
The pattern that filled the Reference image now appears roughly **10 times smaller** inside the Search image.

### Our task

Find the centre coordinates `(x, y)` of that small pattern inside the large Search image.

### Why it is hard

DRAM and FinFET layouts are made of repeating cells and lines.  
Many places look almost the same.  
A simple “find the best matching patch” method often picks a wrong but very similar location.

This problem is called **navigation-error recovery**.  
Solving it correctly is important for automated inspection, defect review, and high-throughput metrology.

---

## 2. Overall Approach We Chose

We did **not** rely only on classical template matching.  
We did **not** rely only on a deep neural network.

We built a **hybrid system** with three parts:

1. **Classical computer vision**  
   Quickly finds the 20 most similar locations in the Search image.

2. **Siamese neural network**  
   Looks at the Reference and each of those 20 candidates and decides which one is the true match.

3. **Confidence decision rule**  
   - If classical matching is very confident → trust classical.  
   - If classical matching is uncertain → trust the neural network.

This design is fast on easy images and more robust on difficult periodic images.

---

## 3. Step-by-Step Process (What Happens When the Pipeline Runs)

### Stage 1 – Generate Realistic Image Pairs

Because the organizers did not give us real wafer images, we built our own **synthetic data generator**.

**What the generator creates:**

| Item | Description |
|------|-------------|
| Style | DRAM (grid of circular contacts) or FinFET (horizontal/vertical line structures) |
| Reference | 1000 × 1000 high-detail close-up |
| Search | 1000 × 1000 wider view (physical area ~10× larger) |
| Scale relationship | Exactly 10× |
| Noise | Independent noise on Reference and Search (shot noise, blur, contrast change, illumination gradient, edge brightening) |
| Geometry | Small random rotation and scale jitter |
| Ground truth | Exact centre coordinates of the target are saved for every pair |

**Difficulty levels we support:**

- **Easy** – clean, high contrast, almost no distraction  
- **Medium** – moderate noise and small geometric change  
- **Hard** – stronger noise, blur, and contrast variation  
- **Trap** – deliberately creates a strong competing distractor so that classical matching often ranks the wrong peak as #1, while the true location remains inside the Top-20

**Why we need a generator:**  
We can create hundreds of test cases with known answers.  
We can also create the exact failure cases that simple methods cannot solve, and then measure whether our hybrid system recovers them.

---

### Stage 2 – Classical Matching (Find Top-20 Candidates)

1. Take the Reference image and resize it down by exactly **10×**.  
   This produces a small template that matches the scale of the Search image.

2. Slide this template across the entire Search image using **Normalized Cross-Correlation** (a standard similarity measure).

3. Instead of keeping only the single best location, we keep the **Top-20 strongest peaks**.  
   We use non-maximum suppression so that nearby peaks do not all come from the same region.

**Why Top-20 instead of Top-1?**  
On periodic layouts the true location is frequently not the strongest peak, but it is almost always among the top 20.  
Keeping 20 candidates gives the next stage a chance to find the correct answer.

**Speed:** This stage usually takes 15–25 milliseconds on a normal computer.

---

### Stage 3 – Siamese CNN Ranking

When classical matching is not highly confident, we ask a small neural network for help.

**How the Siamese network works:**

- It has two identical branches that share the same weights.
- One branch receives a 96 × 96 patch of the Reference.
- The other branch receives a 96 × 96 patch cut from one candidate location in the Search image.
- The network compares the two patches and outputs a **similarity score**.

We repeat this for all 20 candidates and choose the one with the highest score.

**How we trained it:**

- We created many hard “trap” cases where classical Top-1 is wrong.
- For each case we took the true location as a **positive** example.
- We took other high-ranking classical peaks and nearby shifted locations as **hard negatives**.
- We trained the network with a **ranking loss**: the score of the true candidate must be higher than the score of every hard negative by a margin.

This teaches the network to notice the small differences that classical correlation misses.

---

### Stage 4 – Final Decision Rule

```
If the best classical correlation score > 0.85:
    Use the classical result
Else:
    Use the candidate preferred by the Siamese network
```

**Why this rule exists:**

- On easy and medium images, classical matching is already correct and very confident.  
  Using the network in those cases is unnecessary and could introduce small errors.
- On hard and trap images, classical confidence drops.  
  The network is then allowed to re-rank the candidates and often recovers the correct location.

This hybrid rule protects accuracy on easy data while improving difficult cases.

---

### Stage 5 – Output and Visualization

The system returns the final centre coordinates `(x, y)`.

For explanation and debugging we also save:

- Reference image  
- Search image  
- Search image with Top-20 candidates marked  
- Ranking scores for every candidate  
- Final result image with:
  - **Green circle** = true ground-truth location  
  - **Red cross** = our prediction  

When green and red overlap, the system is correct.

---

## 4. Journey of Development (What We Actually Did)

1. **Understood the official problem**  
   10× scale difference + highly periodic DRAM/FinFET layouts.

2. **Built a synthetic generator**  
   Both architecture styles, realistic noise, exact scale, and ground truth.

3. **Implemented classical template matching**  
   Already achieved near-perfect accuracy on easy and medium data.

4. **Created hard trap cases**  
   Classical Top-1 deliberately wrong, but true location still inside Top-20.  
   We collected hundreds of such cases.

5. **First CNN attempts failed**  
   A simple classification-style network hurt correct classical answers and produced near-zero scores on FinFET patterns.

6. **Added a safety fallback**  
   The network was no longer allowed to override classical when its confidence was low.

7. **Switched to a Siamese architecture**  
   The network now explicitly compares Reference versus Candidate instead of just classifying a patch.

8. **Improved training**  
   Ranking loss, larger 96×96 context, strong near-ground-truth hard negatives, and more data.  
   Recovery on hard trap cases rose from near 0% to approximately 88% in offline tests.

9. **Locked the best model and decision rule**  
   Final system: classical when confident, Siamese when classical is uncertain.

10. **Built a transparent runner**  
    Anyone can run one script and see every stage with images and numbers.

We deliberately tested failure cases and report both successes and remaining limitations honestly.

---

## 5. Results We Achieved

### Quantitative results (mixed evaluation)

| Difficulty | Accuracy | Notes |
|------------|----------|-------|
| Easy | 100% | Classical is sufficient |
| Medium | 100% | Classical is sufficient |
| Hard | 100% | Hybrid works well |
| Trap (hard periodic) | ~70–75% | CNN recovers many classical failures |
| **Overall** | **≈ 93–94%** | Mean error usually less than 1 pixel |

### Speed

- Classical stage: typically 15–25 ms per 1000 × 1000 pair  
- Full hybrid (including ranking of 20 patches): typically under 70 ms on GPU or a modest CPU

### What the numbers mean

On normal data the system is essentially perfect.  
On the hardest periodic traps it still recovers the majority of classical mistakes.  
We do not claim 100% on every possible adversarial case; we report the real measured numbers.

---

## 6. Why This Solution Is Suitable for the Hackathon

- It directly produces the required centre coordinates `(x, y)`.
- It explicitly handles the official 10× scale relationship.
- It addresses periodic ambiguity with multi-peak candidates plus learned ranking.
- The classical stage is fast, explainable, and already very strong.
- The neural network is used only when needed and has a safety fallback.
- Everything is reproducible: generator, training notes, weights, and inference script are provided.
- The pipeline is transparent: every intermediate stage can be visualized.

---

## 7. What Is Included in the Submission

| Item | Purpose |
|------|---------|
| Standalone dataset generator | Create Reference + Search pairs with ground truth |
| Standalone inference script | Takes two image paths and prints only `x,y` |
| Final model weights | `siamese_final_96.pt` |
| Full source code | Generator, classical detector, Siamese network, pipeline |
| Training notes | How the network was trained |
| Transparent runner | Shows every stage with images and metrics |
| This process document | Complete explanation in simple English |
| README | How any person can run the solution |

---

## 8. Honest Limitations

- Extremely strong periodic distractors can still produce residual errors (the remaining ~25–30% of trap cases).
- Performance on real SEM images has not been measured because no real wafer data was supplied.
- Further improvement would require domain adaptation to real images or richer features (larger context, multi-scale cues, etc.).

We prefer an honest, working system over an over-claimed one.

---

## 9. One-Sentence Summary

We solve navigation-error recovery with a scale-aware hybrid method: classical multi-peak template matching generates candidates, a Siamese network ranks them on hard cases, and a confidence rule selects the final centre coordinate with high accuracy and full transparency.

---

## 10. How to Demonstrate the Solution to Anyone

1. Open a terminal in the project folder.
2. Run:
   ```bash
   python scripts/run_full_solution.py
   ```
3. Open the `outputs/` folder.
4. Walk through the stages:
   - Stage 1 – generated Reference and Search
   - Stage 2 – classical Top-20 candidates
   - Stage 3 – Siamese scores
   - Stage 4 – decision (classical or Siamese)
   - Stage 5 – final green/red result image
5. Show the metrics summary file.

This is the clearest way to prove that the system works and that every design choice is intentional.

---

*End of Complete Process Document*
