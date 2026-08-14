# Training Notes – Siamese Ranking CNN

## Final Model
- File: `models/siamese_final_96.pt`
- Input size: 96 × 96
- Architecture: Siamese encoder (shared weights) + ranking head
- Loss: Margin ranking loss
- Training data: Synthetic trap cases (classical Top-1 wrong, GT inside Top-20) + strong near-GT hard negatives

## Key Training Decisions
1. Switched from ordinary binary classification to a true Siamese comparison of Reference vs Candidate.
2. Used ranking / margin loss so the correct candidate is forced to score higher than hard negatives.
3. Increased context from 64×64 to 96×96.
4. Emphasised very close hard negatives (±1 to ±10 pixels from GT) because these are the main failure mode of classical matching.
5. Kept classical matching as the primary method; CNN is only used when classical confidence is lower.

## Main Training Script
The principal training script used for the final model is:

- `train_siamese_final.py` (located in this folder or in the original project `scripts/`)

It builds ranking triplets from the collected trap cases, trains for ~25 epochs with AdamW, and saves the best weights according to ranking loss.

## Reproducibility
Anyone can re-train by:
1. Generating trap cases with the provided generator (`difficulty="trap"`)
2. Running the training script with the same hyperparameters
3. Evaluating recovery on a held-out set of trap cases

The final submitted system uses the locked weights `siamese_final_96.pt` and does not require re-training to run.
