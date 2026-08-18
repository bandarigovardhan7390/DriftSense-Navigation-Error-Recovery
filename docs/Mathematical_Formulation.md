# Mathematical Formulation & Sources (Simple Words)

This document explains the math we used, in plain language, and where each idea comes from.

---

## 1. Scale Normalization

**What we do**  
Shrink the Reference image by exactly 10 times so its pattern size matches the Search image.

**Why**  
The problem statement itself says the magnification difference is approximately 10×. We follow that number.

**Formula (simple)**  
Template T = Resize(Reference, scale = 1/10)

No special paper is needed for this step — it is direct use of the given physical relationship.

---

## 2. Normalized Cross-Correlation (NCC)

**What we do**  
Slide the small template over the Search image. At every position we compute a similarity score.

**In simple words**  
NCC measures how well the bright and dark patterns match, after removing average brightness. Lighting changes do not confuse it much.

**Formula**
$$
\text{NCC}(u,v)=\frac{\sum (T-\bar{T})(S_{uv}-\bar{S}_{uv})}{\sqrt{\sum (T-\bar{T})^{2}\cdot\sum (S_{uv}-\bar{S}_{uv})^{2}}}
$$

- **T** = Template (Reference shrunk 10×)
- **Sᵤᵥ** = Search patch at position (u,v)
- **T̄, S̄ᵤᵥ** = Average brightness
- Score close to **+1** = very similar match

**Where this comes from**
- Standard computer-vision method for template matching
- Classic reference: Lewis, J.P. (1995). “Fast Normalized Cross-Correlation”
- Also in textbooks: Gonzalez & Woods, *Digital Image Processing*
- We use the OpenCV implementation `TM_CCOEFF_NORMED` (same math)

We did not invent this formula. We used the well-known, trusted version.

---

## 3. Top-20 Peaks (instead of only Top-1)

$$
\{(x_k,y_k)\}_{k=1}^{20}=\text{Top-20 local maxima of NCC map}
$$

**What we do**  
After computing the correlation map we keep the 20 strongest local peaks (with non-maximum suppression).

**Why in simple words**  
On DRAM and FinFET images many places look almost the same. The single best match is often wrong.  
But the true location is almost always still among the top 20.  
So we turn a hard “find the one correct place” problem into an easier “rank these 20 candidates” problem.

**Symbol meaning**
- **(xₖ, yₖ)** = Centre coordinates of the k-th candidate
- **k = 1 to 20** = We keep the first 20 strongest matches
- **NCC map** = Full similarity image from template matching
- **Local maxima** = Peaks higher than their nearby neighbours

This multi-peak / multi-hypothesis idea is common when patterns are repetitive.

---

## 4. Siamese Similarity Score

$$
\mathbf{f}_R = f(R_{96}), \quad \mathbf{f}_k = f(P_k)
$$

$$
s_k = g\big([\mathbf{f}_R \;\|\; \mathbf{f}_k \;\|\; |\mathbf{f}_R - \mathbf{f}_k|]\big)
$$

**What we do**  
For each of the 20 candidates we cut a 96×96 patch.  
We also resize the Reference to 96×96.  
A Siamese network looks at both patches and gives a similarity score.

**In simple words**  
The network answers: “How well does this candidate match **this exact Reference**?”

**Symbol meaning**
- **R₉₆** = Reference resized to 96×96
- **Pₖ** = Candidate patch (96×96) from Search image
- **f(·)** = Shared encoder (same weights for both patches)
- **f_R, f_k** = Feature vectors of Reference and candidate
- **|f_R − f_k|** = Absolute difference between features
- **g(·)** = Small network that outputs one similarity score
- **sₖ** = Final similarity score (higher = better match)

**Where this comes from**
- Original Siamese networks: Bromley et al. (1993/1994) – signature verification
- Similarity learning: Chopra et al. (2005) and later metric-learning papers
- We use the same basic idea: shared weights, compare two inputs, output similarity

---

## 5. Ranking Loss (how we trained)

$$
\mathcal{L} = \max\big(0,\; m - (s_{\text{positive}} - s_{\text{negative}})\big)
$$

**What we do**  
During training we show the network:
- Positive = true correct location
- Hard negatives = other strong but wrong candidates (including places only a few pixels away)

**In simple words**  
The loss says: “The correct candidate must score higher than the wrong ones by at least a small margin.”

**Symbol meaning**
- **s_positive** = Siamese score of the correct candidate
- **s_negative** = Siamese score of a wrong candidate
- **m** = Margin (minimum gap we want)
- **L** = Loss value  
  - If correct score is already higher by margin m → loss = 0 (good)  
  - If wrong score is higher or too close → loss > 0 (network is corrected)

**Where this comes from**
- Standard ranking / margin losses in metric learning
- Similar ideas appear in FaceNet (Schroff et al.) and many Siamese training papers
- We chose a simple margin form because it directly matches our goal: make the true match rank first among the Top-20

---

## 6. Final Decision Rule

**What we do**

```
If best classical NCC score > 0.85:
    use classical result
Else:
    use the candidate with highest Siamese score
```

**Why in simple words**  
On easy images classical is already correct and very confident — we leave it alone.  
On hard images classical becomes unsure — we let the network help.

This rule is an engineering decision based on our experiments.

---

## 7. Complete Flow (one view)

```
Reference + Search
        ↓
Shrink Reference by 10×              ← from problem statement
        ↓
Normalized Cross-Correlation         ← Lewis / OpenCV / standard CV
        ↓
Keep Top-20 peaks                    ← because patterns are periodic
        ↓
Siamese network scores them          ← Bromley / Chopra style
        ↓
Ranking loss during training         ← standard metric-learning idea
        ↓
Confidence rule                      ← our experimental decision
        ↓
Final (x, y)
```

---

## 8. What we can honestly tell judges

- We did not invent Normalized Cross-Correlation — we used the standard method.
- We did not invent Siamese networks — we used the classic comparison idea.
- Our contribution is the **complete system design**:
  - exact 10× handling
  - Top-20 instead of Top-1
  - Siamese trained on hard periodic failures
  - confidence rule that protects easy cases
  - full transparency of every stage

---

## Short reference list

1. Lewis, J.P. (1995). Fast Normalized Cross-Correlation.  
2. Gonzalez, R.C. & Woods, R.E. Digital Image Processing (NCC / template matching chapters).  
3. Bromley et al. (1993/1994). Signature Verification using a “Siamese” Time Delay Neural Network.  
4. Chopra, S., Hadsell, R., LeCun, Y. (2005). Learning a Similarity Metric Discriminatively, with Application to Face Verification.  
5. Schroff, F. et al. (FaceNet) – ranking / embedding losses for similarity.

(These are the sources of the mathematical ideas we used; our system design and experimental choices are our own work.)
