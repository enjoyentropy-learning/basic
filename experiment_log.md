# Experiment Log: SymbolicLogicEngine on Duplicate Detection

## Problem
Detect whether a 9-symbol sequence has duplicates. One-hot encoded input (batch, 9, 9).
Valid (permutation) → 1, Invalid (has duplicates) → 0.
Goal: exact 0/1 outputs with zero MSE loss.

## Total dataset: ~725,760 samples (362,880 valid permutations + same number invalid)

---

## Experiment 1: Baseline (mean pooling, no scaling)
**Config:** d_model=9, depth=3, float64, train_frac=0.003 (~2K samples), AdamW lr=0.001, 2500 epochs + L-BFGS 300 steps
**Architecture:** SymbolicLogicEngine with mean pooling, no attention scaling
**Parameters:** 1,009
**Log file:** test_plt_output.log

| Metric | Post-AdamW | Post-LBFGS |
|---|---|---|
| Train loss | 1.43e-03 | 6.44e-05 |
| Train acc | 99.99% | 100% |
| Val loss | 16.7 | 1,706 |
| Val acc | 99.96% (263 err) | 99.94% (459 err) |
| Val pred range | [-3,379, +11] | [-34,726, +50] |

**Observations:**
- L-BFGS helps train loss but destroys validation (100x worse val loss)
- Train loss never reaches zero
- Unbounded polynomial outputs on unseen data
- All val errors are false positives (invalid sequences predicted as valid)

---

## Experiment 2: Baseline with ~10K training data
**Config:** Same as Exp 1 but train_frac=0.014 (~10K samples)
**Log file:** test_plt_output.log (overwritten)

| Metric | Post-AdamW | Post-LBFGS |
|---|---|---|
| Train loss | 1.43e-03 | 1.61e-04 |
| Train acc | 99.99% | 100% |
| Val loss | 16.7 | 1,706 |
| Val acc | 99.96% (263 err) | 99.94% (459 err) |
| Val pred range | [-3,379, +11] | [-34,726, +50] |

**Observations:**
- More data improved post-AdamW val loss dramatically vs Exp 1 (16.7 vs 69,487)
- L-BFGS still hurts generalization badly
- Decided to drop L-BFGS going forward

---

## Experiment 3: Flatten pooling + attention scaling (d_model=9)
**Changes:** Replaced mean pooling with flatten+linear readout (81→1). Added attention scaling by 1/sqrt(d_model).
**Config:** d_model=9, depth=3, float64, train_frac=0.014 (~10K), AdamW lr=0.001, 2500 epochs, no L-BFGS
**Parameters:** 1,081
**Log file:** test_plt_output_v2.log

| Metric | Value |
|---|---|
| Train loss | 1.18e-03 |
| Train acc | 100% |
| Val loss | **9.6** |
| Val acc | **99.98% (152 err)** |
| Val pred range | [-2,583, +26] |

**Observations:**
- Best val loss so far (9.6 vs 16.7 baseline)
- 42% fewer val errors (152 vs 263)
- Flatten pooling preserves positional structure, helps generalization
- Polynomial blowup still present but reduced
- Train loss still doesn't reach zero

---

## Experiment 4: Embedding layer (d_model=32)
**Changes:** Added Linear(9→32) embedding layer to decouple d_model from vocab_size.
**Config:** d_model=32, depth=3, float64, train_frac=0.014 (~10K), AdamW lr=0.001, 2500 epochs
**Parameters:** 12,961
**Log file:** test_plt_output_v3.log

| Metric | Value |
|---|---|
| Train loss | 1.20e-03 |
| Train acc | 100% |
| Val loss | 530 |
| Val acc | 99.87% (920 err) |
| Val pred range | [-2.5, +18,641] |

**Observations:**
- 12x more parameters but val loss 55x worse than Exp 3
- Massively overfitting — more capacity memorizes training patterns that don't generalize
- Max val prediction +18,641 (vs +26 in Exp 3)
- **Reverted** — d_model=9 without embedding is better

---

## Experiment 5: More data + more epochs + cosine LR schedule
**Changes:** train_frac=0.14 (~100K samples), 5000 epochs, cosine annealing LR (0.001→1e-6)
**Config:** d_model=9, depth=3, float64, AdamW lr=0.001, cosine annealing, full-batch
**Parameters:** 1,081
**Log file:** test_plt_output_v5.log

| Metric | Value |
|---|---|
| Train loss | 8.73e-04 |
| Train acc | 100% |
| Val loss | **0.0367** |
| Val acc | **99.996% (26 err)** |
| Val pred range | [-91.4, +19.7] |

**Observations:**
- **Massive val improvement:** loss went from 9.6 (Exp 3) → 0.037 — a 260x improvement
- **Only 26 val errors** (down from 152 in Exp 3, 2629 in Exp 1)
- Val pred range dramatically tighter: [-91, +20] vs [-2,583, +26]
- Train loss still doesn't reach zero (8.73e-04), similar to Exp 3
- Gradients very small (~1e-4) — stuck at a plateau, not converging further
- All 26 val errors are false positives on highly-repeated sequences (e.g. [3,3,2,3,2,2,3,1,2])
- More data was the biggest lever — constrains the polynomial to generalize

---

## Key Insight
det(X)^2 = 1 for valid permutations, 0 for invalid. This is a degree-18 polynomial.
The PLT architecture with 3 layers can represent up to degree-27 polynomials.
So the architecture CAN represent the exact solution — the challenge is optimization finding it.

## Current Best: Experiment 5
- Flatten pooling + attention scaling, d_model=9, ~100K training data, 5000 epochs + cosine LR
- Val loss: 0.037, Val acc: 99.996% (26 errors out of 624K)
